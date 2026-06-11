import cv2
from preprocessing import preprocess_image

image = preprocess_image(
    r"../dataset/train/10rs/10 Rs Front.jpeg"
)

cv2.imshow("Processed Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()