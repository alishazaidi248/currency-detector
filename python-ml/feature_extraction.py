from skimage.feature import hog
import cv2
import numpy as np

def extract_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    # Mean RGB
    mean_b = np.mean(image[:,:,0])
    mean_g = np.mean(image[:,:,1])
    mean_r = np.mean(image[:,:,2])

    # Mean HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mean_h = np.mean(hsv[:,:,0])
    mean_s = np.mean(hsv[:,:,1])
    mean_v = np.mean(hsv[:,:,2])

    # Simple color histogram (16 bins each)
    hist_b = cv2.calcHist([image],[0],None,[16],[0,256]).flatten()
    hist_g = cv2.calcHist([image],[1],None,[16],[0,256]).flatten()
    hist_r = cv2.calcHist([image],[2],None,[16],[0,256]).flatten()

    hist = np.concatenate((hist_b,hist_g,hist_r))

    hist = hist / np.sum(hist)

    features = np.concatenate((
        hog_features,
        [mean_b,mean_g,mean_r],
        [mean_h,mean_s,mean_v],
        hist
    ))

    return features