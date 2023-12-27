import numpy as np
from math import *

rad = np.arange(0, 2*pi, pi/18, dtype=np.float32)
for r in rad:
    cosr = cos(r)
    sinr = sin(r)
    theta = atan2(cosr, sinr)
    print(r,pi/2-theta)
    pass
