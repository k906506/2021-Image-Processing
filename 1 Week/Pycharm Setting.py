import cv2
import numpy as np
import matplotlib.pyplot as plt

src = np.zeros((512, 512), dtype = np.uint8)
plt.plot([0, 1, 2, 3, 4])
plt.show()

cv2.imshow("src", src)
cv2.waitKey()
cv2.destroyAllWindows()