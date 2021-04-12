import cv2
import numpy as np

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    # bilinear interpolation 적용
    for row in range(h_dst):
        for col in range(w_dst):
            # 참고로 꼭 한줄로 구현해야 하는건 아닙니다 여러줄로 하셔도 상관없습니다.(저도 엄청길게 구현했습니다.)
            orow = int(row / scale) # 변경 후의 row 좌표 (값을 알고 있는 점 1)
            ocol = int(col / scale) # 변경 후의 col 좌표 (값을 알고 있는 점 2)

            # 선형보간법을 적용시켜 특정 좌표의 위치를 찾음.
            r1 = float(row/scale) - orow
            r2 = 1 - r1
            c1 = float(col/scale) - ocol
            c2 = 1 - c1

            n_orow = orow + 1 # (값을 알고 있는 점 3)
            if n_orow > h-1: # 기존의 사진보다 큰 경우
                n_orow -= 1
            n_ocol = ocol + 1 # (값을 알고 있는 점 4)
            if n_ocol > w-1: # 기존의 사진보다 큰 경우
                n_ocol -= 1

            # 네 개의 사각형의 넓이를 이용하여 특정 좌표를 계산
            dst[row][col] = src[orow][ocol]*r2*c2 + src[n_orow][ocol]*r1*c2 + src[orow][n_ocol]*c1*r2 + src[n_orow][n_ocol]*r1*c1

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/11
    #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 2배로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


