import cv2
import numpy as np
import sys, os
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from filtering import my_filtering


# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    ###########################################
    # TODO                                    #
    # apply_lowNhigh_pass_filter 완성          #
    # Ix와 Iy 구하기                            #
    ###########################################
    y, x = np.mgrid[-int(fsize / 2):int(fsize / 2) + 1, -int(fsize / 2):int(fsize / 2) + 1]

    # 2차 gaussian mask의 x에 대한 미분 -> np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))/sigma**2의 x에 대한 미분
    DoG_x = (-x / sigma ** 2) * (np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
    DoG_x = DoG_x - (DoG_x.sum() / fsize ** 2)

    # 2차 gaussian mask의 y에 대한 미분 -> np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))/sigma**2의 y에 대한 미분
    DoG_y = (-y / sigma ** 2) * (np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
    DoG_y = DoG_y - (DoG_y.sum() / fsize ** 2)

    Ix = my_filtering(src, DoG_x, 'repetition')
    Iy = my_filtering(src, DoG_y, 'repetition')
    return Ix, Iy


# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    return magnitude


# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    ###################################################
    # TODO                                            #
    # calcAngle 완성                                   #
    # angle     : ix와 iy의 angle                      #
    # e         : 0으로 나눠지는 경우가 있는 경우 방지용     #
    # np.arctan 사용하기(np.arctan2 사용하지 말기)        #
    ###################################################
    e = 1E-6
    angle = np.arctan(Iy / (Ix + e))  # 0으로 나눠지는거 방지
    return angle


# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    h, w = magnitude.shape
    largest_magnitude = np.zeros((h, w))

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            ang_tan = np.tan(angle[i][j])

            if ang_tan >= 0 and ang_tan <= 1:  # 0 ~ 45도 && 180 ~ 225도
                p1 = magnitude[i][j + 1] * (1 - ang_tan) + magnitude[i + 1][j + 1] * ang_tan
                p2 = magnitude[i][j - 1] * (1 - ang_tan) + magnitude[i - 1][j - 1] * ang_tan
            elif ang_tan > 1:  # 45 ~ 90도 && 225 ~ 270도
                ang_tan = np.tan((np.pi / 2) - angle[i][j])
                p1 = magnitude[i + 1][j] * (1 - ang_tan) + magnitude[i + 1][j + 1] * ang_tan
                p2 = magnitude[i - 1][j] * (1 - ang_tan) + magnitude[i - 1][j - 1] * ang_tan
            elif ang_tan < -1:  # 90 ~ 135도 && 270 ~ 315도
                ang_tan = np.tan(angle[i][j] - (np.pi / 2))
                p1 = magnitude[i + 1][j] * (1 - ang_tan) + magnitude[i + 1][j - 1] * ang_tan
                p2 = magnitude[i - 1][j] * (1 - ang_tan) + magnitude[i - 1][j + 1] * ang_tan
            else:  # 135 ~ 180도 && 315 ~ 360도
                ang_tan = np.tan(np.pi - angle[i][j])
                p1 = magnitude[i][j - 1] * (1 - ang_tan) + magnitude[i + 1][j - 1] * ang_tan
                p2 = magnitude[i][j + 1] * (1 - ang_tan) + magnitude[i - 1][j + 1] * ang_tan

            p3 = magnitude[i][j]
            if p3 >= p2 and p3 >= p1:  # 강한 엣지만 남김
                largest_magnitude[i][j] = p3

    return largest_magnitude


# double_thresholding 수행
def double_thresholding(src):
    dst = src.copy()

    # dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)

    (h, w) = dst.shape
    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    # high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고
    # low threshold값은 (high threshold * 0.4)로 구한다
    low_threshold_value = high_threshold_value * 0.4

    ######################################################
    # TODO                                               #
    # double_thresholding 완성                            #
    # dst     : double threshold 실행 결과 이미지           #
    ######################################################

    # 강한 엣지는 255로, 약한 엣지는 100으로 설정.
    for i in range(h):
        for j in range(w):
            if dst[i][j] >= int(high_threshold_value):
                dst[i][j] = 255
            elif dst[i][j] < int(low_threshold_value):
                dst[i][j] = 0
            else:
                dst[i][j] = 100

    # 강한 엣지와 연결된 약한 엣지를 탐색.
    for i in range(h):
        for j in range(w):
            if dst[i][j] == 255:
                mid = deque()
                mid.append((i, j))
                while mid: # DFS 진행
                    x, y = mid.popleft()
                    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
                    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
                    for k in range(8):
                        nx = x + dx[k]
                        ny = y + dy[k]
                        if nx >= 0 and nx < h and ny >= 0 and ny < w and dst[nx][ny] == 100:  # 약한 엣지인 경우
                            mid.append((nx, ny))
                            dst[nx][ny] = 255  # 강한 엣지로 만들어 준다.

    # 강한 엣지로 변환되지 않은 약한 엣지는 0으로.
    for i in range(h):
        for j in range(w):
            if dst[i][j] != 255:
                dst[i][j] = 0

    return dst


def my_canny_edge_detection(src, fsize=3, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG 를 사용하여 1번 filtering
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)

    # Ix와 Iy 시각화를 위해 임시로 Ix_t와 Iy_t 만들기
    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    magnitude_t = magnitude
    cv2.imwrite("magnitude.png", magnitude_t)
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # non-maximum suppression 수행
    largest_magnitude = non_maximum_supression(magnitude, angle)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # double thresholding 수행
    dst = double_thresholding(largest_magnitude)
    return dst


def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()