import numpy as np
import cv2

src = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
div_src = cv2.divide(src, 3)
cv2.imwrite('fruits_div3.jpg', div_src)
