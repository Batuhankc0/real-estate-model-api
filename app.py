import spacy
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import logging

# Keep logging simple
logging.basicConfig(level=logging.INFO)

# --- 1. Initialize the Flask App and Load the Model ---
app = Flask(__name__)
model_path = Path("./real_estate_model")

# Load the model only once when the application starts. This is very important!
try:
    nlp = spacy.load(model_path)
    logging.info(f"Model '{model_path}' loaded successfully.")
except Exception as e:
    logging.error(f"ERROR: Model could not be loaded! Details: {e}")
    nlp = None

# --- 2. Endpoint for the simple HTML Interface ---
# This page will be shown when users visit the website.
@app.route('/')
def home():
    return render_template('index.html')

# --- 3. Main API Endpoint for Predictions ---
# This endpoint will handle requests from the JavaScript.
@app.route('/predict', methods=['POST'])
def predict():
    if not nlp:
        return jsonify({"error": "Model is not loaded, please check the server logs."}), 500

    # Get the incoming JSON data
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({"error": "Please send a sentence with the 'sentence' key."}), 400

    text = data['sentence']
    logging.info(f"Received Sentence: '{text}'")

    # Process the sentence with the model
    doc = nlp(text)

    # Extract the intent
    intent = None
    if doc.cats:
        # Get the intent with the highest score
        top_intent = max(doc.cats, key=doc.cats.get)
        score = doc.cats[top_intent]
        intent = {"name": top_intent, "score": f"{score:.2f}"}

    # Extract the entities
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    # Return the result in JSON format
    response = {
        "sentence": text,
        "intent": intent,
        "entities": entities
    }
    
    return jsonify(response)

# These lines are important for Gunicorn to find the application.
if __name__ == '__main__':
    # This part is only for testing locally.
    # Render will not run this block because it uses Gunicorn.
    app.run(host='0.0.0.0', port=5000, debug=True)
