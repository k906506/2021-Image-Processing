import cv2
import numpy as np


def get_hist(src):
    hist = np.zeros((256,))
    h, w = src.shape

    for row in range(h):
        for col in range(w):
            hist[src[row, col]] += 1

    return hist


def threshold(src, value):
    h, w = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if src[row, col] <= value:
                dst[row, col] = 0
            else:
                dst[row, col] = 255

    return dst


def get_threshold(src, type='rice'):
    #####################################################
    # TODO                                              #
    # get_threshold 완성                                 #
    #####################################################
    hist = get_hist(src)
    intensity = np.array([i for i in range(256)])
    h, w = src.shape

    if type == 'rice':
        # 한 줄로 작성하세요
        # p = ni / m*n
        p = hist / (h*w)
    '''
    else:
        # 여러줄로 작성하셔도 상관 없습니다.
        p = ???
    '''
    
    k_opt_warw = []
    k_opt_warb = []
    for k in range(256):
        # 각각 한 줄로 작성하세요
        q1 = np.sum(p[:k+1]) # 0 ~ k까지의 sum
        q2 = np.sum(p[k+1:]) # k+1 ~ L-1까지의 sum

        # 굳이 할 필요 없는 경우
        if q1 == 0 or q2 == 0:
            k_opt_warw.append(np.inf)
            k_opt_warb.append(0)
            continue

        # 각각 한 줄로 작성하세요 (m1, m2, mg, var1, var2)
        i1 = [i for i in range(k+1)]
        i2 = [i for i in range(k+1, 256)]
        m1 = np.sum(np.multiply(i1, p[:k+1])) / q1
        m2 = np.sum(np.multiply(i2, p[k+1:])) / q2

        mg = m1*q1 + m2*q2

        i3 = [i**2 for i in range(k+1)]
        i4 = [i**2 for i in range(k + 1, 256)]
        var1 = np.sum(np.multiply(i3, p[:k+1])) / q1 - np.multiply(m1, m1)
        var2 = np.sum(np.multiply(i4, p[k+1:])) / q2 - np.multiply(m2, m2)

        # varg = np.sum(np.square(intensity - mg)*p)

        # 실수(float)라 약간의 오차가 있을 수 있음
        assert np.abs((q1 + q2) - 1) < 1E-6
        assert np.abs((q1 * m1 + q2 * m2) - mg) < 1E-6

        # 각각 한 줄로 작성하세요 (varw, varb)
        varw = q1 * var1 + q2 * var2
        varb = q1 * q2 * (np.square(m1-m2))

        k_opt_warw.append(varw)
        k_opt_warb.append(varb)

    k_opt_warw = np.array(k_opt_warw)
    k_opt_warb = np.array(k_opt_warb)

    print(k_opt_warw)
    print()
    print()
    print(k_opt_warb)
    # 2개의 결과가 같아야 함
    assert k_opt_warw.argmin() == k_opt_warb.argmax()

    dst = threshold(src, k_opt_warw.argmin())
    return dst, k_opt_warw.argmin()


def rice_main():
    src = cv2.imread('rice.png', cv2.IMREAD_GRAYSCALE)
    val, _ = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    print('< cv2.threshold >')
    print(val)
    dst, threshold_value = get_threshold(src)
    print('< get_threshold >')
    print(threshold_value)

    cv2.imshow('original', src)
    cv2.imshow('threshold', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

'''
def meat_main():
    meat = cv2.imread('meat.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('mask.TIFF', cv2.IMREAD_GRAYSCALE)
    #####################################################
    # TODO                                              #
    # meat_main 완성                                     #
    # 이 부분은 결과가 잘 나오도록 각자 알아서 구현해보세요       #
    #####################################################


    dst, val = get_threshold(src, 'meat')
    #tip : 4칙연산(그냥 사칙연산 혹은 cv2.사칙연산 잘 사용하기)
    final = ???

    cv2.imshow('dst', dst)
    cv2.imshow('final', final)

    cv2.waitKey()
    cv2.destroyAllWindows()
'''

def main():
    rice_main()
    #meat_main()


if __name__ == '__main__':
    main()