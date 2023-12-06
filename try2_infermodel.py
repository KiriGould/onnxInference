import onnxruntime as ort
from transformers import BartTokenizer
import numpy as np

# Load the ONNX model
model_path = 'bart-large-cnn.onnx' #download bart-large-cnn.onnx from HuggingFace
try:
    session = ort.InferenceSession(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

path_tokenizer = "./tokenizer_cache"

# Load the tokenizer
tokenizer = BartTokenizer.from_pretrained(path_tokenizer)

text = "Your input text here."

def preprocess_input(user_input, tokenizer, max_length=124):
    # Tokenize the input
    inputs = tokenizer.encode_plus(user_input, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
    return inputs['input_ids'], inputs['attention_mask']

# Tokenize and preprocess the input text
input_ids, attention_mask = preprocess_input(text, tokenizer)

# Convert to numpy arrays
input_ids_np = input_ids.numpy()
attention_mask_np = attention_mask.numpy()

# Get input names from the model
input_names = [input.name for input in session.get_inputs()]

# Run the model
outputs = session.run(None, {input_names[0]: input_ids_np, input_names[1]: attention_mask_np})

# Process the output
for batch in outputs:
    # Convert logits to token IDs
    token_ids = batch.flatten()
    # Decode and print the output text
    output_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(output_text)
