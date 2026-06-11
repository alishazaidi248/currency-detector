import os
import cv2
import joblib

from preprocessing import preprocess_image
from feature_extraction import extract_features

from sklearn.svm import SVC

# Dataset location
dataset_path = r"../dataset/train"

X = []
y = []

# Read every currency folder
for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    # Example:
    # folder = "10rs"
    # label = 10

    label = int(folder.replace("rs", ""))

    for file in os.listdir(folder_path):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(folder_path, file)

        try:

            image = preprocess_image(image_path)

            features = extract_features(image)

            X.append(features)

            y.append(label)

        except Exception as e:

            print("Skipped:", image_path)
            print(e)

print("Images Loaded :", len(X))

model = SVC(kernel="linear", probability=True)

model.fit(X, y)

joblib.dump(model, "currency_model.pkl")

print("Model saved successfully!")