from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

from preprocessing import preprocess_image
from feature_extraction import extract_features

app = Flask(__name__)

# Load model
model = joblib.load("currency_model.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "Currency Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "currency": "Unknown",
            "confidence": 0
    })

    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:

        image = preprocess_image(filepath)

        features = extract_features(image)

        probabilities = model.predict_proba([features])[0]

        classes = model.named_steps["svm"].classes_

        index = np.argmax(probabilities)

        prediction = int(classes[index])

        confidence = float(probabilities[index] * 100)

        print("--------------------------------")
        print("Prediction :", prediction)
        print("Confidence :", confidence)
        print("--------------------------------")

        # Optional confidence threshold
        if confidence < 40:
            return jsonify({
                "currency": "Unknown",
                "confidence": round(confidence, 2)
            })

        return jsonify({
            "currency": f"₹{prediction}",
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        print(e)

        return jsonify({
            "currency": "Unknown",
            "confidence": 0
        })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )