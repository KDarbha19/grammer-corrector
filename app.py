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

    with torch.no_grad(): # disables gradient tracking, saves memory and faster speed
        outputs = model.generate(
            **inputs,
            max_new_tokens = 128,
            num_beams = 4,
            early_stopping = True,
            no_repeat_ngram_size = 2
        )

    return tokenizer.decode(outputs[0],  skip_special_tokens = True)

#compare orginal and corrected text word by word, highlight differences
def generate_diff(original, corrected):
    original_words = original.split()
    corrected_words = corrected.split()

    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    result = []

    for opcode, i1,i2,j1,j2 in matcher.get_opcodes():
        if opcode == 'equal':
            result.append(' '.join(original_words[i1:i2]))
        elif opcode in ('replace', 'insert'):
            #words changes or added, wrap in green
            result.append(
                f'<span class="changed">{" ".join(corrected_words[j1:j2])}</span>'
            )
        elif opcode == 'delete':
            #words removed, wrap in red
            result.append(
                f'<span class="deleted">{" ".join(original_words[i1:i2])}</span>'
            )
    return ' '.join(result)

#routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/correct', methods=['POST'])
def correct():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if len(text) > 500:
        return jsonify({'error': 'Text too long - keep it under 500 characters'}), 400
    
    corrected = correct_grammar(text)
    highlighted = generate_diff(text, corrected)

    return jsonify({
        'original' : text,
        'corrected' : corrected,
        'highlighted' : highlighted
    })

if __name__ == '__main__':
    app.run(debug=True)