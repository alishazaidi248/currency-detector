import cv2

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Cannot open image")

    image = cv2.resize(image, (128, 128))

    return image