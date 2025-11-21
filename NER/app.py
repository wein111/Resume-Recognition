import io
import argparse
import torch
import os
import json
import datetime
from transformers import BertTokenizerFast, BertForTokenClassification
from flask import Flask, jsonify, request
from NER.server.utils import preprocess_data, predict, idx2tag

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

MAX_LEN = 500
NUM_LABELS = 21
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'bert-base-uncased'
STATE_DICT = torch.load("model-state.bin", map_location=DEVICE)
TOKENIZER = BertTokenizerFast("./vocab/vocab.txt", lowercase=True)

model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', state_dict=STATE_DICT['model_state_dict'], num_labels=NUM_LABELS)
model.to(DEVICE)


def merge_entities_by_offsets(entities, full_text):
    """
    将 BIO 格式的实体列表合并成完整实体

    Args:
        entities: 预测返回的实体列表，格式如:
                  [{'entity': 'B-Name', 'start': 0, 'end': 1, 'text': 'A'}, ...]
        full_text: 原始完整文本

    Returns:
        合并后的实体列表，格式如:
        [{'label': 'Name', 'text': 'Ayush Srivastava', 'start': 0, 'end': 16}]
    """
    if not entities:
        return []

    merged = []
    current_entity = None

    for entity in entities:
        label = entity.get('entity', '')  # 使用 'entity' 键

        if not label:
            continue

        # 跳过 'O' 标签
        if label == 'O':
            if current_entity:
                merged.append(current_entity)
                current_entity = None
            continue

        # 解析 BIO 标签
        if label.startswith('B-'):
            # 保存之前的实体
            if current_entity:
                merged.append(current_entity)

            # 开始新实体
            entity_type = label[2:]  # 去掉 'B-' 前缀
            current_entity = {
                'label': entity_type,
                'text': entity.get('text', ''),
                'start': entity.get('start'),
                'end': entity.get('end')
            }

        elif label.startswith('I-'):
            entity_type = label[2:]  # 去掉 'I-' 前缀

            if current_entity and current_entity['label'] == entity_type:
                # 继续当前实体：更新结束位置
                current_entity['end'] = entity.get('end')
                # 从原文提取完整文本（确保准确性）
                if current_entity['start'] is not None and current_entity['end'] is not None:
                    current_entity['text'] = full_text[current_entity['start']:current_entity['end']]
            else:
                # I- 标签但没有对应的 B- 开头，视为新实体（容错处理）
                if current_entity:
                    merged.append(current_entity)

                current_entity = {
                    'label': entity_type,
                    'text': entity.get('text', ''),
                    'start': entity.get('start'),
                    'end': entity.get('end')
                }
        else:
            # 没有 B- 或 I- 前缀，直接作为单独实体
            if current_entity:
                merged.append(current_entity)

            current_entity = {
                'label': label,
                'text': entity.get('text', ''),
                'start': entity.get('start'),
                'end': entity.get('end')
            }

    # 添加最后一个实体
    if current_entity:
        merged.append(current_entity)

    print(f"✅ 合并完成：{len(entities)} 个 token → {len(merged)} 个实体")
    return merged


@app.route('/predict', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        # 1) 读取 PDF 并提取纯文本
        data = io.BytesIO(request.files.get('resume').read())
        resume_text = preprocess_data(data)

        # 2) 调用模型预测得到 token 级实体（含 start/end/text）
        entities = predict(model, TOKENIZER, idx2tag, DEVICE, resume_text, MAX_LEN)

        # 3) 用 offset 合并 BIO 实体；如果没有 offset 则回退到 BIO 拼 text
        formatted = merge_entities_by_offsets(entities, resume_text)

        # 4) 保存到带时间戳的 outputs 文件夹
        os.makedirs("outputs", exist_ok=True)
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("outputs", f"entities.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=4)

        print(f"✅ result save to  {output_path}")

        # 5) 返回清晰的 JSON 响应（包含文件路径 + 合并后实体）
        return jsonify({
            "message": "Extraction successful",
            "output_file": output_path,
            "entities": formatted
        })


if __name__ == '__main__':
    app.run()