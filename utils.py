import re
import json
import logging
import numpy as np
from tqdm import trange, tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def convert_goldparse(dataturks_JSON_FilePath):
    """
    Convert Dataturks-style annotated JSON lines into a list of:
        (text, {"entities": [(start, end, label), ...]})
    Used as NER training format similar to spaCy.
    """
    try:
        training_data = []

        # Read JSONL file (each line is one resume annotation)
        with open(dataturks_JSON_FilePath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)

            # Original text (replace newline for consistency)
            text = data['content'].replace("\n", " ")

            entities = []
            data_annotations = data['annotation']

            if data_annotations is not None:
                # Each annotation block contains points + label(s)
                for annotation in data_annotations:
                    point = annotation['points'][0]     # annotation span info
                    labels = annotation['label']        # may be a string or list

                    # Ensure labels is always a list
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        point_start = point['start']
                        point_end = point['end']
                        point_text = point['text']

                        # Fix leading/trailing whitespace offsets
                        lstrip_diff = len(point_text) - len(point_text.lstrip())
                        rstrip_diff = len(point_text) - len(point_text.rstrip())

                        if lstrip_diff != 0:
                            point_start += lstrip_diff
                        if rstrip_diff != 0:
                            point_end -= rstrip_diff

                        # Dataturks end index is inclusive → convert to Python exclusive
                        entities.append((point_start, point_end + 1, label))

            training_data.append((text, {"entities": entities}))

        return training_data

    except Exception as e:
        logging.exception(
            "Unable to process " + dataturks_JSON_FilePath +
            "\n" + "error = " + str(e)
        )
        return None


def trim_entity_spans(data: list) -> list:
    """
    Remove leading/trailing whitespace from entity spans.
    Ensures each entity's (start, end) indices point to non-space characters.
    """
    invalid_span_tokens = re.compile(r'\s')  # matches any whitespace

    cleaned_data = []

    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []

        for start, end, label in entities:
            valid_start = start
            valid_end = end

            # Move start forward until it's not a whitespace
            while valid_start < len(text) and invalid_span_tokens.match(text[valid_start]):
                valid_start += 1

            # Move end backward until it's not a whitespace
            while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                valid_end -= 1

            valid_entities.append([valid_start, valid_end, label])

        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data


def get_label(offset, labels, previous_label='O'):
    """
    Assign BIO tag to a token based on its character-level offset.

    Args:
        offset: (start_char, end_char) of the token.
        labels: list of gold entity spans (start, end, label).
        previous_label: BIO tag of the previous token (for deciding B/I).

    Returns:
        A BIO-formatted label string, e.g., "B-Name", "I-Name", or "O".
    """

    # Special tokens have offset (0,0), always assign 'O'
    if offset[0] == 0 and offset[1] == 0:
        return 'O'

    # Loop over all annotated entity spans
    for label in labels:
        ent_start, ent_end, ent_type = label

        # Case 1: Token fully inside entity span
        if offset[0] >= ent_start and offset[1] <= ent_end:
            # If previous token is same entity, continue with I- prefix
            if previous_label in (f"I-{ent_type}", f"B-{ent_type}"):
                return f"I-{ent_type}"
            else:
                return f"B-{ent_type}"

        # Case 2: Token partially overlaps with entity (boundary cases)
        elif offset[0] < ent_end and offset[1] > ent_start:
            # If token start is inside entity
            if offset[0] >= ent_start:
                if previous_label in (f"I-{ent_type}", f"B-{ent_type}"):
                    return f"I-{ent_type}"
                else:
                    return f"B-{ent_type}"

    # No overlap → label is 'O'
    return 'O'



tags_vals = [
    "O",
    "B-Name", "I-Name",
    "B-Degree", "I-Degree",
    "B-Skills", "I-Skills",
    "B-College Name", "I-College Name",
    "B-Email Address", "I-Email Address",
    "B-Designation", "I-Designation",
    "B-Companies worked at", "I-Companies worked at",
    "B-Graduation Year", "I-Graduation Year",
    "B-Years of Experience", "I-Years of Experience",
    "B-Location", "I-Location"
]

tag2idx = {t: i for i, t in enumerate(tags_vals)}
idx2tag = {i: t for i, t in enumerate(tags_vals)}



def process_resume(data, tokenizer, tag2idx, max_len, is_test=False):
    tok = tokenizer.encode_plus(
        data[0],
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_offsets_mapping=True
    )

    curr_sent = {'orig_labels': [], 'labels': []}

    if not is_test:
        labels = data[1]['entities']
        labels = sorted(labels, key=lambda x: x[0])  # ← 按起始位置排序，不要reverse

        previous_label = 'O'
        for off in tok['offset_mapping']:
            label = get_label(off, labels, previous_label)  # ← 传入previous_label
            curr_sent['orig_labels'].append(label)
            curr_sent['labels'].append(tag2idx.get(label, tag2idx['O']))  # ← 使用.get()防止KeyError
            previous_label = label

        # 对于padding部分，应该用-100而不是0
        padding_length = max_len - len(tok['input_ids'])
        curr_sent['labels'] = curr_sent['labels'] + ([-100] * padding_length)  # ← 改为-100
    else:
        padding_length = max_len - len(tok['input_ids'])
        curr_sent['labels'] = [-100] * max_len

    curr_sent['input_ids'] = tok['input_ids']
    curr_sent['token_type_ids'] = tok['token_type_ids']
    curr_sent['attention_mask'] = tok['attention_mask']

    return curr_sent



class ResumeDataset(Dataset):
    def __init__(self, resume, tokenizer, tag2idx, max_len, is_test=False):
        self.resume = resume
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.resume)

    def __getitem__(self, idx):
        data = process_resume(
            self.resume[idx], self.tokenizer, self.tag2idx, self.max_len, self.is_test)
        return {
            'input_ids': torch.tensor(data['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(data['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(data['labels'], dtype=torch.long),
            'orig_label': data['orig_labels']
        }


def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters


def get_special_tokens(tokenizer, tag2idx):
    vocab = tokenizer.get_vocab()
    pad_tok = vocab["[PAD]"]
    sep_tok = vocab["[SEP]"]
    cls_tok = vocab["[CLS]"]
    o_lab = tag2idx["O"]

    return pad_tok, sep_tok, cls_tok, o_lab


def annot_confusion_matrix(valid_tags, pred_tags):
    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    header = sorted(list(set(valid_tags + pred_tags)))

    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    mat_formatted = [header[i] + "\t\t\t" +
                     str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content


def flat_accuracy(valid_tags, pred_tags):
    return (np.array(valid_tags) == np.array(pred_tags)).mean()


def compute_label_weights(train_dataloader, num_labels, device):
    """计算类别权重以处理不平衡问题"""
    print("Computing class weights...")
    all_labels = []

    for batch in train_dataloader:
        labels = batch['labels'].numpy().flatten()
        # 过滤掉-100 (padding)
        labels = labels[labels != -100]
        all_labels.extend(labels.tolist())

    unique_labels = np.unique(all_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=all_labels
    )

    # 创建完整的权重向量
    class_weights = torch.ones(num_labels)
    for label, weight in zip(unique_labels, weights):
        class_weights[label] = weight

    # 打印权重信息
    print("\nClass weights:")
    for i, w in enumerate(class_weights):
        if w > 1:  # 只打印重要的权重
            print(f"  {idx2tag[i]}: {w:.2f}")

    return class_weights.to(device)


def train_and_val_model(
        model,
        tokenizer,
        optimizer,
        epochs,
        idx2tag,
        tag2idx,
        max_grad_norm,
        device,
        train_dataloader,
        valid_dataloader
):
    pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, tag2idx)

    class_weights = compute_label_weights(train_dataloader, len(tag2idx), device)

    loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    epoch = 0
    for _ in trange(epochs, desc="Epoch"):
        epoch += 1

        # Training loop
        print("Starting training loop.")
        model.train()
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        batch_loop = tqdm(train_dataloader, desc="Train batches", leave=False)
        for step, batch in enumerate(batch_loop):
            model.zero_grad()

            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            # ========== 修改: 不传入labels，使用自定义损失 ==========
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                # labels=b_labels,  # ← 注释掉
            )

            # ========== 修改: 手动计算加权损失 ==========
            logits = outputs.logits  # 或 outputs[0]
            loss = loss_fct(logits.view(-1, len(tag2idx)), b_labels.view(-1))

            # Backward pass
            loss.backward()

            # Compute train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # Subset out unwanted predictions on CLS/PAD/SEP tokens
            preds_mask = (
                    (b_input_ids != cls_tok)
                    & (b_input_ids != pad_tok)
                    & (b_input_ids != sep_tok)
                    & (b_labels != -100)  # ← 新增: 过滤-100
            )

            tr_logits = logits.cpu().detach().numpy()  # ← 改用logits
            tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            preds_mask_np = preds_mask.cpu().detach().numpy()
            tr_batch_preds = np.argmax(tr_logits[preds_mask_np.squeeze()], axis=1)
            tr_batch_labels = tr_label_ids.to("cpu").numpy()
            tr_preds.extend(tr_batch_preds)
            tr_labels.extend(tr_batch_labels)

            # Compute training accuracy
            tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
            tr_accuracy += tmp_tr_accuracy

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )

            # Update parameters
            optimizer.step()

        tr_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps

        # Print training loss and accuracy per epoch
        print(f"Train loss: {tr_loss}")
        print(f"Train accuracy: {tr_accuracy}")


        train_pred_tags = [idx2tag[i] for i in tr_preds]
        o_ratio = train_pred_tags.count('O') / len(train_pred_tags)
        print(f"Training O-tag ratio: {o_ratio:.2%}")

        # Validation loop
        print("Starting validation loop.")
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        predictions_nested = []
        true_labels_nested = []
        predictions_flat = []
        true_labels_flat = []

        for batch in valid_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask
                )
                logits = outputs.logits  # 或 outputs[0]


                tmp_eval_loss = loss_fct(logits.view(-1, len(tag2idx)), b_labels.view(-1))

            preds_mask = (
                    (b_input_ids != cls_tok)
                    & (b_input_ids != pad_tok)
                    & (b_input_ids != sep_tok)
                    & (b_labels != -100)  # ← 新增
            )

            logits = logits.cpu().detach().numpy()
            label_ids = b_labels.cpu().numpy()
            preds_mask = preds_mask.cpu().numpy()

            val_batch_preds = np.argmax(logits, axis=2)

            batch_size = b_input_ids.size(0)
            for i in range(batch_size):
                valid_mask = preds_mask[i]
                valid_preds = val_batch_preds[i][valid_mask]
                valid_labels = label_ids[i][valid_mask]

                pred_tags_sent = [idx2tag[p] for p in valid_preds]
                label_tags_sent = [idx2tag[l] for l in valid_labels]

                predictions_nested.append(pred_tags_sent)
                true_labels_nested.append(label_tags_sent)
                predictions_flat.extend(pred_tags_sent)
                true_labels_flat.extend(label_tags_sent)

            val_batch_preds_flat = val_batch_preds[preds_mask]
            val_batch_labels_flat = label_ids[preds_mask]
            tmp_eval_accuracy = flat_accuracy(val_batch_labels_flat, val_batch_preds_flat)

            eval_loss += tmp_eval_loss.item()  # ← 改为.item()
            eval_accuracy += tmp_eval_accuracy
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps

        cl_report = classification_report(true_labels_nested, predictions_nested)
        conf_mat = annot_confusion_matrix(true_labels_flat, predictions_flat)

        print(f"Validation loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")
        print(f"Classification Report:\n {cl_report}")
        print(f"Confusion Matrix:\n {conf_mat}")

        val_o_ratio = predictions_flat.count('O') / len(predictions_flat)
        print(f"Validation O-tag ratio: {val_o_ratio:.2%}")

