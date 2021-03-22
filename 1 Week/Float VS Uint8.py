import cv2
import numpy as np

src1 = np.zeros((200, 200))
src2 = np.ones((200, 200))

src3 = np.zeros((200, 200), dtype = np.uint8)
src4 = np.ones((200, 200), dtype = np.uint8)
src5 = np.full((200, 200), 100, dtype = np.uint8)

src3[:, 100:200] = 100
cv2.imshow('src1', src1)
cv2.imshow('src2', src2)
cv2.imshow('src3', src3)
cv2.imshow('src4', src4)
cv2.imshow('src5', src5)

cv2.waitKey()
cv2.destroyAllWindows()