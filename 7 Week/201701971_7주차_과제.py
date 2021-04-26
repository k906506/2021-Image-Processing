import cv2
import numpy as np

def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                           #
    ###############################################
    (h, w) = B.shape
    (s_h,s_w) = S.shape

    dst = np.zeros((h, w))
    rangeH = int(s_h/2)
    rangeW = int(s_w/2)

    for i in range(h):
        for j in range(w):
            if B[i][j] == 1: # 흰색일 경우
                for x in range(i-rangeH, i+rangeH+1):
                    for y in range(j-rangeW, j+rangeW+1):
                        if x >= 0 and x < h and y >= 0 and y < w: # 기존 이미지의 범위 이내인 경우
                            dst[x][y] = 1

    return dst


def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                            #
    ###############################################
    (h, w) = B.shape
    (s_h, s_w) = S.shape

    dst = np.zeros((h, w))
    rangeH = int(s_h / 2)
    rangeW = int(s_w / 2)

    for i in range(h):
        for j in range(w):
            if B[i][j] == 1:  # 흰색일 경우
                check = True
                for x in range(i - rangeH, i + rangeH + 1):
                    for y in range(j - rangeW, j + rangeW + 1):
                        if (x < 0 or x >= h) or (y < 0 or y >= w): # 기존 이미지의 범위를 벗어난 경우
                            check = False
                            break
                        if B[x][y] == 0: # 기존 이미지의 범위이지만 S의 범위에 0이 있는 경우
                            check = False
                            break
                if check == True:
                    dst[i][j] = 1

    return dst


def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################
    # erosion -> dilation
    erosion_image = erosion(B, S)
    dst = dilation(erosion_image, S)

    return dst

def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################
    # dilation -> erosion
    dilation_image = dilation(B, S)
    dst = erosion(dilation_image, S)

    return dst

if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)

