from flask import Flask, request, jsonify, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import difflib

app = Flask(__name__)

#Load model once when the server starts- not every req
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Loading model on {device}")
tokenizer = T5Tokenizer.from_pretrained("model/final").to(device)
model = T5ForConditionalGeneration.from_pretrained("model/final").to(device)
model.eval() #puts model in inference mode, disables dropout
print("Model loaded")

#correction function
def correct_grammar(text):
    inputs = tokenizer(
        "fix grammar: " + text,
        return_tensors = "pt",
        max_length = 128,
        truncation = True
    ).to(device)