import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. Histogram 계산.
def my_calcHist(src):
    h, w = src.shape[:2]
    hist = np.zeros((256,), dtype=np.uint64)
    for row in range(h):
        for col in range(w):
            intensity = src[row, col]  # 좌표의 intensity
            hist[intensity] += 1  # count
    return hist

# 2. Histogram / 총 픽셀 수 -> 정규화를 위한 과정
def my_normalize_hist(hist, pixel_num):
    return hist/pixel_num

# 3. 2에서 구한 값들을 누적 -> 정규화를 위한 과정
def my_PDF2CDF(pdf):
    for i in range(1, len(pdf)):
        pdf[i] = pdf[i] + pdf[i - 1]
    return pdf

# 4. gray_level(pixel_value)를 곱함.
def my_denormalize(normalized, gray_level):
    return normalized * gray_level

# 5. Histogram equaliztion을 수행.
def my_calcHist_equalization(denormalized, hist):
    hist_equal = np.zeros((256,), dtype=np.uint64)
    for i in range(len(hist)):
        hist_equal[denormalized[i]] += hist[i]
    return hist_equal

# 5. Histogram equaliztion을 수행.
def my_equal_img(src, output_gray_level):
    (h, w) = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            dst[i][j] = output_gray_level[src[i][j]]

    return dst

# input_image의  equalization된 histogram & image 를 return
def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    binX = np.arange(len(denormalized_output))
    plt.plot(binX, denormalized_output)
    plt.title('mapping function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()

    ### dst : equalization 결과 image
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal

if __name__ == '__main__':
    src = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)

    plt.figure(figsize=(8, 5))
    cv2.imshow('original', src)
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()

    plt.figure(figsize=(8, 5))
    cv2.imshow('equalizetion after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()

