import torch
import numpy as np
from pdfminer.high_level import extract_text
from collections import Counter


def preprocess_data(data):
    text = extract_text(data)
    text = text.replace("\n", " ")
    text = text.replace("\f", " ")
    return text


def tokenize_resume(text, tokenizer, max_len):
    tok = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_offsets_mapping=True
    )

    curr_sent = dict()

    curr_sent['input_ids'] = tok['input_ids']
    curr_sent['token_type_ids'] = tok.get('token_type_ids', [0] * max_len)
    curr_sent['attention_mask'] = tok['attention_mask']

    final_data = {
        'input_ids': torch.tensor(curr_sent['input_ids'], dtype=torch.long),
        'token_type_ids': torch.tensor(curr_sent['token_type_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(curr_sent['attention_mask'], dtype=torch.long),
        'offset_mapping': tok['offset_mapping']
    }

    return final_data



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
idx2tag = {i: t for i, t in enumerate(tags_vals)}
resticted_lables = [ "O", "B-Email Address", "I-Email Address"]


def predict(model, tokenizer, idx2tag, device, test_resume, max_len):
    model.eval()
    data = tokenize_resume(test_resume, tokenizer, max_len)

    input_ids = data['input_ids'].unsqueeze(0).to(device)
    input_mask = data['attention_mask'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            token_type_ids=None,
            attention_mask=input_mask,
        )

    logits = outputs.logits
    logits = logits.cpu().detach().numpy()
    label_ids = np.argmax(logits, axis=2)

    entities = []
    for label_id, offset in zip(label_ids[0], data['offset_mapping']):
        curr_id = idx2tag[label_id]
        curr_start, curr_end = offset
        if curr_start == curr_end:
            continue
        if curr_id not in resticted_lables:
            if len(entities) > 0 and entities[-1]['entity'] == curr_id and curr_start - entities[-1]['end'] in [0, 1]:
                entities[-1]['end'] = curr_end
            else:
                entities.append({'entity': curr_id, 'start': curr_start, 'end': curr_end})

    for ent in entities:
        ent['text'] = test_resume[ent['start']:ent['end']]

    return entities


def merge_entities_by_offsets(entities, source_text, verbose=False):
    """
    Merge contiguous/nearby entities with the same label.
    Keeps offsets stable and rebuilds merged text spans from source_text.
    """
    if not entities:
        return []

    ordered = sorted(entities, key=lambda e: (e["start"], e["end"]))
    merged = [dict(ordered[0])]

    for curr in ordered[1:]:
        prev = merged[-1]
        same_label = prev.get("entity") == curr.get("entity")
        touching = curr["start"] <= prev["end"] + 1
        if same_label and touching:
            prev["end"] = max(prev["end"], curr["end"])
            prev["text"] = source_text[prev["start"]:prev["end"]]
            if verbose:
                print(f"Merged {prev['entity']} @ {prev['start']}:{prev['end']}")
        else:
            merged.append(dict(curr))

    for ent in merged:
        ent["text"] = source_text[ent["start"]:ent["end"]]

    return merged


def entity_label_counts(entities):
    """Return per-entity-label counts preserving first-seen order."""
    counts = Counter(ent.get("entity", "UNKNOWN") for ent in entities)
    ordered = {}
    for ent in entities:
        label = ent.get("entity", "UNKNOWN")
        if label not in ordered:
            ordered[label] = counts[label]
    return ordered
