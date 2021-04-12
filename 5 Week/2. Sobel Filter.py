import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.filtering import my_filtering

def get_sobel():
    derivative = np.array([[- 1, 0, 1]])
    blur = np.array([[1], [2], [1]])

    x = np.dot(blur, derivative)
    y = np.dot(derivative.T, blur.T)

    return  x, y

def main():
    sobel_x, sobel_y = get_sobel()

    src = cv2.imread('edge_detection_img.png', cv2.IMREAD_GRAYSCALE)
    dst_x = my_filtering(src, sobel_x, 'zero')
    dst_y = my_filtering(src, sobel_y, 'zero')

    cv2.imshow('dst_x', abs(dst_x))
    cv2.imshow('dst_y', abs(dst_y))
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()