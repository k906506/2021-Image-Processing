import numpy as np
import cv2

src = np.array([[3, 1, 3, 5, 4], [9, 8, 3, 5, 6],
               [2, 2, 3, 8, 7], [5, 4, 6, 5, 4],
               [1, 0, 0, 2, 6]], dtype=np.uint8)

src_visible = (src/9 * 255).astype(np.uint8)
cv2.imwrite('5x5_img.png', src_visible)

binX = np.arange
