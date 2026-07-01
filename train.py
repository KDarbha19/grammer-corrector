import os
import numpy as np
from datasets import load_dataset
from transformers import(
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import torch
from difflib import SequenceMatcher
from datasets import Dataset

device = torch.device("mps")

#Loading dataset
dataset = load_dataset("jhu-clsp/jfleg")

def pick_best_correction(sentence, corrections):
#Score each correction by similarity to original
#pick best one so its easier for model to understand
    def similarity(a,b):
        return SequenceMatcher(None, a, b). ratio()

    scores = [similarity(sentence, c) for c in corrections]
    best_idx = scores.index(max(scores))
    return corrections[best_idx]

def process_dataset(split):
    inputs = []
    targets = []
    
    for example in dataset[split]:
        sentence = example['sentence'].strip()
        corrections = [c.strip() for c in example['corrections']]

        best = pick_best_correction(sentence, corrections)

        #Skip if sentence and correction are identical (nothing for model to learn)
        if best == sentence:
            continue

        #t5 expects a task prefix, tell t5 what to do
        inputs.append("fix grammar: " + sentence)
        targets.append(best)

    return inputs, targets
    
train_inputs, train_targets = process_dataset('validation')
test_inputs, test_targets = process_dataset('test')

print(f"Train pairs : {len(train_inputs)}")
print(f"Test pairs  : {len(test_inputs)}")
print(f"\nSample input  : {train_inputs[0]}")
print(f"Sample target : {train_targets[0]}")

MODEL_NAME = "t5-small"
print(f"\nLoading {MODEL_NAME}")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to(device)

#Tokenize, Sentences are rarely over 60 words, corrections are similar length to input
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 128

def tokenize(inputs, targets):
    tokenized = tokenizer(
        inputs,
        text_target = targets,
        max_length = MAX_INPUT_LEN,
        max_target_length = MAX_TARGET_LEN,
        truncation = True,
        padding = "max_length"
    )
    #replace padding token id with -100 so loss ignores the pad tokens
    tokenized["labels"] = [
        [(t if t!= tokenizer.pad_token_id else -100) for t in label]
        for label in tokenized["labels"]
    ]
    return tokenized

print("\nTokenizing")
train_tokenized = tokenize(train_inputs, train_targets)
test_tokenized = tokenize(test_inputs, test_targets)

print(f"Input token sample: {train_tokenized['input_ids'][0][:10]}")
print(f"Label token sample: {train_tokenized['labels'][0][:10]}")


#Convert tokenized dictionaries to HuggingFace Dataset objects
train_dataset = Dataset.from_dict(train_tokenized)
test_dataset = Dataset.from_dict(test_tokenized)

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Test dataset: {len(test_dataset)} samples")
print(f"Columns: {train_dataset.column_names}")