import numpy as np
import matplotlib.pyplot as plt

#
mean = 10
sigma = 5
x = np.random.randn(100)
x = x*sigma+mean
x = x.tolist()#type:list
x = float(x)
x.append(-20.0)

#过滤出异常点
miu = np.mean(x)
sigma = np.std(x)
ee = x-miu
ee = ee[np.abs(ee-miu)>4*sigma]
print(f'miu={miu},sigma={sigma},{ee}')

#绘制直方图
plt.hist(x,bins=50)#直方图，绘制每个数值出现的次数，可以直接表达概率分布情况
plt.show()