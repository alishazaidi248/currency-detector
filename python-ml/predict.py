import joblib
from preprocessing import preprocess_image
from feature_extraction import extract_features

# Load trained model
model = joblib.load("currency_model.pkl")

# CHANGE THIS TO ANY TEST IMAGE
image_path = r"../dataset/train/100rs/100 Rs Front.jpeg"

# Preprocess image
image = preprocess_image(image_path)

# Extract HOG features
features = extract_features(image)

# Predict
prediction = model.predict([features])[0]

# Get confidence
confidence = max(model.predict_proba([features])[0]) * 100

print(f"Detected Currency : ₹{prediction}")
print(f"Confidence : {confidence:.2f}%")