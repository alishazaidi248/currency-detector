from flask import Flask, request, jsonify
import os
import joblib

from preprocessing import preprocess_image
from feature_extraction import extract_features

app = Flask(__name__)

# ===========================
# Load ML Model
# ===========================

model = joblib.load("currency_model.pkl")

# ===========================
# Upload Folder
# ===========================

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===========================
# Label Mapping
# IMPORTANT:
# Match these indexes with the labels
# used while training train_model.py
# ===========================

label_map = {
    0: "₹5",
    1: "₹10",
    2: "₹20",
    3: "₹50",
    4: "₹100",
    5: "₹200",
    6: "₹500"
}

# ===========================
# Home
# ===========================

@app.route("/")
def home():
    return jsonify({
        "message": "Currency Detection API Running Successfully"
    })

# ===========================
# Prediction API
# ===========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "message": "No image uploaded"
            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "message": "Empty filename"
            }), 400

        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        # ----------------------
        # Preprocess
        # ----------------------

        image = preprocess_image(filepath)

        # ----------------------
        # Feature Extraction
        # ----------------------

        features = extract_features(image)

        # ----------------------
        # Prediction
        # ----------------------

        prediction = model.predict([features])[0]

        probabilities = model.predict_proba([features])[0]

        confidence = float(max(probabilities) * 100)

        currency = label_map.get(
            int(prediction),
            "Unknown"
        )

        # Delete uploaded image
        os.remove(filepath)

        return jsonify({

            "success": True,

            "currency": currency,

            "confidence": round(confidence, 2)

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "message": str(e)

        }), 500


# ===========================
# Run Server
# ===========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )