import numpy as np
import matplotlib.pyplot as plt

arr1 = np.array([0, 1, 2, 3, 2, 1])
binX = np.arange(len(arr1))
plt.bar(binX, arr1, width=0.8, color='g')
plt.title('plt_plot test')
plt.xlabel('x')
plt.ylabel('y')
plt.show()