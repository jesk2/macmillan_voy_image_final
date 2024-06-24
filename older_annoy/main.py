import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#Load image
image = cv.imread('chess.jpg')

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv.ORB_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(gray, None)
# Draw keypoints on image
output_image = cv.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

# display original with kps drawn
plt.figure(figsize=(10, 8))
plt.imshow(cv.cvtColor(output_image, cv.COLOR_BGR2RGB))
plt.title("ORB feature detection")
plt.show()
