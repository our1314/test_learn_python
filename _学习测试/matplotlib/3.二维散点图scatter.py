import numpy as np
import matplotlib.pyplot as plt

pts = np.random.randn(4,2)
plt.figure()
plt.scatter(pts[:,0],pts[:,1])#散点图
plt.show()