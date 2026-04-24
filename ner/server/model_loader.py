import os
from typing import Tuple

import torch
from transformers import BertForTokenClassification, BertTokenizerFast

from ner.server.utils import idx2tag


def load_resume_ner_model(
    model_path: str, vocab_path: str, device: torch.device
) -> Tuple[BertForTokenClassification, BertTokenizerFast]:
    """Load fine-tuned BERT token classifier and tokenizer (same contract as ``ner.app``)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model weights: {model_path}")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")

    state = torch.load(model_path, map_location=device)
    tokenizer = BertTokenizerFast(vocab_path, lowercase=True)
    num_labels = len(idx2tag)
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        state_dict=state["model_state_dict"],
        num_labels=num_labels,
    )
    model.to(device)
    return model, tokenizer
