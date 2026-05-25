import cv2
import os
import numpy as np

from sklearn.svm import SVC

dataset_path = r"C:\Users\alish\OneDrive\Documents\..MCA\sem_2\cv\currency-detector-system\dataset\train"

data = []
labels = []

for img_name in os.listdir(dataset_path):

    img_path = os.path.join(dataset_path, img_name)

    image = cv2.imread(img_path)

    image = cv2.resize(image, (100, 100))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flattened = gray.flatten()

    data.append(flattened)

    if "10" in img_name:
        labels.append(0)

    elif "500" in img_name:
        labels.append(1)

data = np.array(data)
labels = np.array(labels)

model = SVC(kernel='linear')

model.fit(data, labels)

print("Model Trained Successfully")