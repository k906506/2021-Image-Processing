import numpy as np

def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[-int(msize/2):int(msize/2)+1, -int(msize/2):int(msize/2)+1]

    # 2차 gaussian mask 생성
    gaus2D = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x**2 + y**2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D