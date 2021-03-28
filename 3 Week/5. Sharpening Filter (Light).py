import cv2
import numpy as np

def my_sharpening_filter_3x3(src):
    standard = np.zeros((3,3))
    standard[1, 1] = 3
    mask = np.full((3, 3), 1/9)
    mask = standard - mask
    dst = cv2.filter2D(src, -1, mask)
    return dst

if __name__ == "__main__":
    src = cv2.imread("Lena.png", cv2.IMREAD_GRAYSCALE)
    dst = my_sharpening_filter_3x3(src)

    cv2.imshow("original", src)
    cv2.imshow("sharpening filter", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
