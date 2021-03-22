import cv2
import numpy as np

src = np.zeros((300, 300, 3), dtype = np.uint8)

#BGR 순임

src
src[150:160, 150:160] = [255, 1, 1]
src[160:170, 160:170] = [255, 255, 1]
src[170:180, 170:180] = [255, 255, 255]

cv2.imshow('src', src)

cv2.waitKey()
cv2.destroyAllWindows()