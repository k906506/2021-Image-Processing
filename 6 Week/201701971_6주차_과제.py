import cv2
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from filtering import my_filtering
from padding import my_padding

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

    # 2차 gaussian mask의 x에 대한 미분 -> (x*(np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))))/sigma**2의 x에 대한 미분
    DoG_x = (-x / sigma ** 2) * (np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
    DoG_x = DoG_x - (DoG_x.sum() / fsize ** 2)

    # 2차 gaussian mask의 y에 대한 미분 -> (x*(np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))))/sigma**2의 y에 대한 미분
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
    magnitude = np.sqrt(Ix**2 + Iy**2)
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
    angle = np.arctan(Iy / (Ix + e)) # 0으로 나눠지는거 방지
    return angle

# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    largest_magnitude = my_padding(magnitude, (1, 1), 'repetition') # 가장자리도 작업을 진행하기 위해 row+2, col+2 -> size up

    h, w = magnitude.shape
    for row in range(h):
        for col in range(w):
            i = row + 1
            j = col + 1
            ang_tan = np.tan(angle[row][col])
            if ang_tan >= 0 and ang_tan < 1: # 0 ~ 45도 && 180 ~ 225도
                p1 = largest_magnitude[i][j+1] * (1 - ang_tan) + largest_magnitude[i-1][j+1] * ang_tan
                p2 = largest_magnitude[i][j-1] * (1 - ang_tan) + largest_magnitude[i+1][j-1]
            elif ang_tan >= 1: # 45 ~ 90도 && 225 ~ 270도
                p1 = largest_magnitude[i-1][j+1] * (1 - ang_tan) + largest_magnitude[i-1][j] * ang_tan
                p2 = largest_magnitude[i+1][j-1] * (1 - ang_tan) + largest_magnitude[i+1][j] * ang_tan
            elif ang_tan <= -1: # 90 ~ 135도 && 270 ~ 315도
                p1 = largest_magnitude[i-1][j] * (1 - ang_tan) + largest_magnitude[i-1][j-1] * ang_tan
                p2 = largest_magnitude[i+1][j] * (1 - ang_tan) + largest_magnitude[i+1][j+1] * ang_tan
            else: # 135 ~ 180도 && 315 ~ 360도
                p1 = largest_magnitude[i][j-1] * (1 - ang_tan) + largest_magnitude[i-1][j-1] * ang_tan
                p2 = largest_magnitude[i+1][j+1] * (1 - ang_tan) + largest_magnitude[i+1][j+1] * ang_tan

            p3 = largest_magnitude[i][j]
            if p3 < p2 or p3 < p1: # 최대값이 아닌 경우 0으로 만든다.
                largest_magnitude[i][j] = 0

    h, w = largest_magnitude.shape
    largest_magnitude = largest_magnitude[1:h-1, 1:w-1] # 가장자리 작업을 위해 늘려준 사이즈를 다시 줄여준다.

    return largest_magnitude


# double_thresholding 수행
def double_thresholding(src):
    dst = src.copy()

    #dst => 0 ~ 255
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
    for i in range(h):
        for j in range(w):
            if dst[i][j] >= int(high_threshold_value):
                dst[i][j] = 255
            elif dst[i][j] < int(low_threshold_value):
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
    cv2.imwrite("after thresholding.png", dst)
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