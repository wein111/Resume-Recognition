import argparse
import numpy as np
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from transformers import AdamW
from utils import trim_entity_spans, convert_goldparse, ResumeDataset, tag2idx, idx2tag, get_hyperparameters, train_and_val_model


parser = argparse.ArgumentParser(description='Train Bert-NER')
args = parser.parse_args().__dict__
output_path = "./"

# Hyperparameters & Config
MAX_LEN = 500
EPOCHS = 5
MAX_GRAD_NORM = 1.0   # gradient clipping
MODEL_NAME = 'bert-base-uncased'

# Load tokenizer (custom vocab)
TOKENIZER = BertTokenizerFast('./vocab/vocab.txt', lowercase=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load and preprocess data
data = trim_entity_spans(convert_goldparse('data/Resumes.json'))  # fix overlapping entity spans

total = len(data)
train_data, val_data = data[:180], data[180:]

# Build dataset objects
train_d = ResumeDataset(train_data, TOKENIZER, tag2idx, MAX_LEN)
val_d = ResumeDataset(val_data, TOKENIZER, tag2idx, MAX_LEN)


# DataLoaders
train_sampler = RandomSampler(train_d)   # shuffle training data
train_dl = DataLoader(train_d, sampler=train_sampler, batch_size=8, num_workers=0, pin_memory=True)

val_dl = DataLoader(val_d, batch_size=4, num_workers=0, pin_memory=True)


# Load BERT model for token classification
model = BertForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(tag2idx))   # output layer size = number of tags
model.to(DEVICE)

# AdamW works better for transformers
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)

# Training + Validation loop
train_and_val_model(
    model,
    TOKENIZER,
    optimizer,
    EPOCHS,
    idx2tag,
    tag2idx,
    MAX_GRAD_NORM,
    DEVICE,
    train_dl,
    val_dl
)

# Save trained model
torch.save(
    {
        "model_state_dict": model.state_dict()
    },
    f'{output_path}/model-state.bin',
)
