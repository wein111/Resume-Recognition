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
TOKENIZER = BertTokenizerFast("NER/vocab/vocab.txt", lowercase=True)

model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', state_dict=STATE_DICT['model_state_dict'], num_labels=NUM_LABELS)
model.to(DEVICE)


def merge_entities_by_offsets(entities, full_text):
    """
    Merge BIO-formatted entity list into complete entities.

    Args:
        entities: List of predicted entities, e.g.:
                  [{'entity': 'B-Name', 'start': 0, 'end': 1, 'text': 'A'}, ...]
        full_text: Original full text.

    Returns:
        Merged entity list, e.g.:
        [{'label': 'Name', 'text': 'Ayush Srivastava', 'start': 0, 'end': 16}]
    """
    if not entities:
        return []

    merged = []
    current_entity = None

    for entity in entities:
        label = entity.get('entity', '')

        if not label:
            continue

        if label == 'O':
            if current_entity:
                merged.append(current_entity)
                current_entity = None
            continue

        if label.startswith('B-'):
            if current_entity:
                merged.append(current_entity)

            entity_type = label[2:]
            current_entity = {
                'label': entity_type,
                'text': entity.get('text', ''),
                'start': entity.get('start'),
                'end': entity.get('end')
            }

        elif label.startswith('I-'):
            entity_type = label[2:]

            if current_entity and current_entity['label'] == entity_type:
                current_entity['end'] = entity.get('end')
                if current_entity['start'] is not None and current_entity['end'] is not None:
                    current_entity['text'] = full_text[current_entity['start']:current_entity['end']]
            else:
                if current_entity:
                    merged.append(current_entity)

                current_entity = {
                    'label': entity_type,
                    'text': entity.get('text', ''),
                    'start': entity.get('start'),
                    'end': entity.get('end')
                }
        else:
            if current_entity:
                merged.append(current_entity)

            current_entity = {
                'label': label,
                'text': entity.get('text', ''),
                'start': entity.get('start'),
                'end': entity.get('end')
            }

    if current_entity:
        merged.append(current_entity)

    print(f"Merged: {len(entities)} tokens -> {len(merged)} entities")
    return merged


@app.route('/predict', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        data = io.BytesIO(request.files.get('resume').read())
        resume_text = preprocess_data(data)

        entities = predict(model, TOKENIZER, idx2tag, DEVICE, resume_text, MAX_LEN)

        formatted = merge_entities_by_offsets(entities, resume_text)

        os.makedirs("outputs", exist_ok=True)
        #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("outputs", f"entities.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=4)

        print(f"Result saved to {output_path}")

        return jsonify({
            "message": "Extraction successful",
            "output_file": output_path,
            "entities": formatted
        })


if __name__ == '__main__':
    app.run()