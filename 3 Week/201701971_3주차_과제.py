import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero padding'):
    (h, w) = src.shape  # 그림의 사이즈
    (p_h, p_w) = pad_shape  # 늘리고자하는 사이즈
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #up
        for i in range(p_h):
            for j in range(w):
                pad_img[i][j + p_w] = src[0][j]
        #down
        for i in range(h, h + p_h):
            for j in range(w):
                pad_img[i + p_h][j + p_w] = src[h-1][j]
        #left
        for i in range(h + 2*p_h):
            for j in range(p_w):
                pad_img[i][j] = pad_img[i][p_w]
        #right
        for i in range(h + 2*p_h):
            for j in range(w, w + p_w):
                pad_img[i][j + p_w] = pad_img[i][w + p_w - 1]

    else:
        print('zero padding')

    return pad_img

def my_filtering(src, ftype, fshape):
    (h, w) = src.shape
    dst = np.zeros((h, w))
    rangeH = fshape[0] // 2
    rangeW = fshape[1] // 2
    pad_type = 'repetition' # 패딩 타입 설정

    if ftype == 'average':
        print('average filtering')
        mask = np.full((fshape[0], fshape[1]), 1 / int(fshape[0] * fshape[1]))

        for i in range(rangeH, h-rangeH):
            for j in range(rangeW, w-rangeW):
                squareValue = src[i-rangeH:i+rangeH+1, j-rangeW:j+rangeW+1]
                # 따로 선언하지 않고 위의 코드처럼 진행해도 OK.
                dst[i][j] = np.sum(squareValue * mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        standard = np.zeros((fshape[0], fshape[1]))
        standard[fshape[0]//2, fshape[1]//2] = 2
        mask = np.full((fshape[0], fshape[1]), 1 / int(fshape[0] * fshape[1]))
        mask = standard - mask

        for i in range(rangeH, h - rangeH):
            for j in range(rangeW, w - rangeW):
                squareValue = src[i - rangeH:i + rangeH + 1, j - rangeW:j + rangeW + 1]
                # 따로 선언하지 않고 위의 코드처럼 진행해도 OK.
                dst[i][j] = np.sum(squareValue * mask)
                if dst[i][j] > 255:
                    dst[i][j] = 255
                elif dst[i][j] < 0:
                    dst[i][j] = 0

    dst = dst[rangeH:h - rangeH, rangeW:w - rangeW]
    dst = my_padding(dst, (fshape[0] // 2, fshape[1] // 2), pad_type)

    dst = (dst + 0.5).astype(np.uint8)

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # padding test
    zero_test = my_padding(src, (20, 20))
    rep_test = my_padding(src, (20, 20), 'repetition')

    # 3x3 filter
    size33_dst_average = my_filtering(src, 'average', (3, 3))
    size33_dst_sharpening = my_filtering(src, 'sharpening', (3, 3))

    # 7x9 filter
    size79_dst_average = my_filtering(src, 'average', (7, 9))
    size79_dst_sharpening = my_filtering(src, 'sharpening', (7, 9))

    # 11x13 filter
    size1113_dst_average = my_filtering(src, 'average', (11, 13))
    size1113_dst_sharpening = my_filtering(src, 'sharpening', (11, 13))
    
    cv2.imshow('original', src)

    # padding test
    cv2.imshow('zero padding test', zero_test.astype(np.uint8))
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))

    # 3x3 filter
    cv2.imshow('3x3 average filter', size33_dst_average)
    cv2.imshow('3x3 sharpening filter', size33_dst_sharpening)

    # 7x9 filter
    cv2.imshow('self_size average filter', size79_dst_average)
    cv2.imshow('self_size sharpening filter', size79_dst_sharpening)

    # 11x13 filter
    cv2.imshow('11x13 average filter', size1113_dst_average)
    cv2.imshow('11x13 sharpening filter', size1113_dst_sharpening)

    cv2.waitKey()
    cv2.destroyAllWindows()
