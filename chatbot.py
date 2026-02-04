from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "").lower()

    if "ddos" in msg:
        reply = "DDoS is a flooding attack that overwhelms network resources."
    elif "severity" in msg:
        reply = "Severity is based on packet rate and attack confidence."
    elif "confidence" in msg:
        reply = "Confidence represents IDS certainty based on dataset patterns."
    else:
        reply = "I can explain IDS alerts, attack types, confidence, and severity."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(port=5001)
