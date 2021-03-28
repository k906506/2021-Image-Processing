import cv2
import numpy as np

def my_average_filter_3x3(src):
    mask = np.full((3, 3), 1/12)
    dst = cv2.filter2D(src, -1, mask)
    return dst

if __name__ == "__main__":
    src = cv2.imread("Lena.png", cv2.IMREAD_GRAYSCALE)
    dst = my_average_filter_3x3(src)

    cv2.imshow("original", src)
    cv2.imshow("average filter", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()