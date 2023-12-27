import numpy as np
import scipy
from matplotlib import pyplot as plt

x = np.array([23, 24, 24, 25, 25])
y = np.array([13, 12, 13, 12, 13])
tck, u = scipy.interpolate.splprep([x,y], s=0, per=False)
unew = np.arange(0, 1.00, 0.005)
xi,yi = scipy.interpolate.splev(unew, tck)
plt.plot(xi,yi)
plt.show()

