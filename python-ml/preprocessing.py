import cv2

def preprocess_image(image_path):
    """
    Reads an image and performs basic preprocessing.
    Returns the processed grayscale image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise Exception(f"Cannot open image: {image_path}")

    # Resize
    image = cv2.resize(image, (224, 224))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    return blur