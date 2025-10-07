import spacy
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import logging

# Loglamayı basit tut
logging.basicConfig(level=logging.INFO)

# --- 1. Flask Uygulamasını ve Modeli Başlat ---
app = Flask(__name__)
model_path = Path("./final_real_estate_model")

# Modeli sadece bir kez, uygulama başlarken yükle. Bu çok önemlidir!
try:
    nlp = spacy.load(model_path)
    logging.info(f"Model '{model_path}' başarıyla yüklendi.")
except Exception as e:
    logging.error(f"HATA: Model yüklenemedi! Detaylar: {e}")
    nlp = None

# --- 2. Basit HTML Arayüzü için Endpoint ---
# Kullanıcılar web sitesini ziyaret ettiğinde bu sayfa gösterilecek.
@app.route('/')
def home():
    return render_template('index.html')

# --- 3. Tahmin Yapan Ana API Endpoint'i ---
# JavaScript'ten gelen istekleri bu endpoint karşılayacak.
@app.route('/predict', methods=['POST'])
def predict():
    if not nlp:
        return jsonify({"error": "Model yüklenemedi, lütfen sunucu loglarını kontrol edin."}), 500

    # Gelen JSON verisini al
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({"error": "Lütfen 'sentence' anahtarıyla bir cümle gönderin."}), 400

    text = data['sentence']
    logging.info(f"Gelen Cümle: '{text}'")

    # Cümleyi model ile işle
    doc = nlp(text)

    # Niyeti çıkar
    intent = None
    if doc.cats:
        # En yüksek skora sahip niyeti ve skorunu al
        top_intent = max(doc.cats, key=doc.cats.get)
        score = doc.cats[top_intent]
        intent = {"name": top_intent, "score": f"{score:.2f}"}

    # Varlıkları çıkar
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_})

    # Sonucu JSON formatında döndür
    response = {
        "sentence": text,
        "intent": intent,
        "entities": entities
    }
    
    return jsonify(response)

# Gunicorn'un uygulamayı bulabilmesi için bu satırlar önemlidir.
if __name__ == '__main__':
    # Bu kısım sadece lokalde test etmek içindir.
    # Render, Gunicorn kullandığı için bu bloğu çalıştırmaz.
    app.run(host='0.0.0.0', port=5000, debug=True)
