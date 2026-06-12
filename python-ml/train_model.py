import os
import joblib

from preprocessing import preprocess_image
from feature_extraction import extract_features

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

dataset_path = r"../dataset/train"

X = []
y = []

count = 0

for folder in sorted(os.listdir(dataset_path)):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    label = int(folder.replace("rs",""))

    print(f"Loading {folder}...")

    for file in os.listdir(folder_path):

        if not file.lower().endswith((".jpg",".jpeg",".png")):
            continue

        image_path = os.path.join(folder_path,file)

        try:

            image = preprocess_image(image_path)

            features = extract_features(image)

            X.append(features)

            y.append(label)

            count += 1

        except Exception as e:

            print(image_path)
            print(e)

print("\nImages Loaded :",count)

model = Pipeline([

    ("scaler",StandardScaler()),

    ("svm",SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced"
    ))

])

print("Training...")

model.fit(X,y)

joblib.dump(model,"currency_model.pkl")

print("Model Saved Successfully!")