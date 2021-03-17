import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist_gray(mini_img): #  mini_img의 histogram 계산
    h, w = mini_img.shape[:2]
    hist = np.zeros((256,), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            intensity = mini_img[row, col]
            hist[intensity] += 1
    return hist

def my_hist_stretch(src, hist):
    (h, w) = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    min = 256
    max = -1

    for i in range(len(hist)):
        if hist[i] != 0 and i < min:
            min = i
        if hist[i] != 0 and i > max:
            max = i

    hist_stretch = np.zeros(hist.shape, dtype=np.uint8)
    for i in range(min, max+1):
        j = int((255-0)/(max-min) * (i-min) + 0)
        hist_stretch[j] = hist[i]

    for row in range(h):
        for col in range(w):
            dst[row, col] = (255-0)/(max-min) * (src[row, col] - min) + 0

    return dst, hist_stretch

def main():
    src = cv2.imread('fruits.jpg', cv2.IMREAD_GRAYSCALE) # 기본 이미지의 값 저장
    src_div3 = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE) # 1/3 값 저장

    hist = my_calcHist_gray(src) # 기본 이미지의 histogram 저장
    hist_div3 = my_calcHist_gray(src_div3) # 1/3 이미지의 histogram 저장

    dst, hist_stretch = my_hist_stretch(src_div3, hist_div3) # 1/3 이미지에 histogram stretch 적용

    binX = np.arange(len(hist_stretch))
    plt.bar(binX, hist_div3, width=0.5, color='g')
    plt.title('divide 3 image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    plt.bar(binX, hist_stretch, width=0.5, color='g')
    plt.title('stretching image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    plt.bar(binX, hist, width=0.5, color='g')
    plt.title('original image')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

    cv2.imshow('div 1/3 image', src_div3)
    cv2.imshow('stretched image', dst)
    cv2.imshow('original image', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()