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
"""
os.makedirs("model", exist_ok=True)

training_args = TrainingArguments(
    output_dir = "model/checkpoints",
    num_train_epochs = 15,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    warmup_steps = 50,
    weight_decay = 0.01,
    learning_rate = 3e-4,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    logging_steps = 10,
    fp16 = False,
    use_cpu = False,
    dataloader_pin_memory = False,
    save_total_limit = 1,
    report_to = "none",
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model = model,
    padding = True,
    label_pad_token_id = -100,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    processing_class = tokenizer,
    data_collator = data_collator,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

print('\nStarting Training')
trainer.train()

model.save_pretrained("model/final")
tokenizer.save_pretrained("model/final")
print("\nModel saved to model/final/")"""

print("\n--- QUICK SANITY CHECK ---")
loaded_tokenizer = T5Tokenizer.from_pretrained("model/final")
loaded_model = T5ForConditionalGeneration.from_pretrained("model/final")
loaded_model = loaded_model.to(device)

test_sentences = [
    "i goes to the store yesterday and buyed some milk",
    "She don't know nothing about it",
    "Their going to there house over they're",
    "He runned very fastly to school this morning",
    "i think this are very good product and i buyed it again"
]

for sentence in test_sentences:
    inputs = loaded_tokenizer(
        "fix grammar: " + sentence,
        return_tensors="pt",
        max_length=128,
        truncation=True,
    ).to(device)
    outputs = loaded_model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        early_stopping=True,
    )
    result = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input  : {sentence}")
    print(f"Output : {result}")
    print()
