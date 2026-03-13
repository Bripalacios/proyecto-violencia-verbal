import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv(override=False)

app = Flask(__name__)
CORS(app)

print("Cargando modelo de análisis de sentimiento...")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="finiteautomata/beto-sentiment-analysis"
)

print("Modelo listo.")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Servidor backend de detector de violencia activo"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    api_key_exists = bool(os.environ.get("DEEPGRAM_API_KEY"))
    return jsonify({
        "status": "ok",
        "deepgram_api_key_configured": api_key_exists
    }), 200


@app.route("/detect", methods=["POST"])
def detect_violence():

    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = sentiment_model(text)[0]

    label = result["label"]
    score = result["score"]

    risk = "LOW"

    if label == "NEG" and score > 0.85:
        risk = "HIGH"
    elif label == "NEG":
        risk = "MEDIUM"

    return jsonify({
        "text": text,
        "sentiment": label,
        "confidence": score,
        "violence_risk": risk
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8081))
    host = os.environ.get("HOST", "127.0.0.1")
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    print(f"Servidor corriendo en http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)