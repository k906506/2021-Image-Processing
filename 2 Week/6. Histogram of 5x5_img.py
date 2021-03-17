import numpy as np
import matplotlib.pyplot as plt

def my_calcHist_gray_mini_img(mini_img):
    h, w = mini_img.shape[:2]
    hist = [0 for _ in range(10)]
    for row in range(h):
        for col in range(w):
            intensity = mini_img[row, col]
            hist[intensity] += 1
    return hist

def main():
    src = np.array([[3, 1, 3, 5, 4], [9, 8, 3, 5, 6],
                    [2, 2, 3, 8, 7], [5, 4, 6, 5, 4],
                    [1, 0, 0, 2, 6]], dtype=np.uint8)
    hist = my_calcHist_gray_mini_img(src)
    binX = np.arange(len(hist))
    plt.bar(binX, hist, width=0.8, color='g')
    plt.title('histogram')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel num')
    plt.show()

if __name__ == "__main__":
    main()