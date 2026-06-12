import joblib
import numpy as np

from preprocessing import preprocess_image
from feature_extraction import extract_features

model = joblib.load("currency_model.pkl")

image_path = input("Image Path: ")

image = preprocess_image(image_path)

features = extract_features(image)

prob = model.predict_proba([features])[0]

classes = model.named_steps["svm"].classes_

index = np.argmax(prob)

print("\nDetected :",classes[index])

print("Confidence : {:.2f}%".format(prob[index]*100))

print()

for c,p in zip(classes,prob):

    print(f"₹{c} -> {p*100:.2f}%")