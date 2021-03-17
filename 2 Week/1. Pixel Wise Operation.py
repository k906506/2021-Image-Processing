import numpy as np
import cv2

src = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE)
add_src = cv2.add(src, 128)
sub_src = cv2.subtract(src, 128)

cv2.imshow('fruits', src)
cv2.imshow('function add 128', add_src)
cv2.imshow('add 128', src + 128)
cv2.imshow('function sub 128', sub_src)
cv2.imshow('sub 128', src - 128)

cv2.waitKey()
cv2.destroyAllWindows()
