"""
网格状图像分割的关键是找到网格间距、网格方向、相位，即可实现对其分割，通过傅里叶变换
https://blog.csdn.net/greatwgb/article/details/17913559
https://blog.csdn.net/weixin_43394528/article/details/119895945
"""
from math import *
import numpy as np
import matplotlib.pyplot as plt

#1、生成方波采样信号
#region 注释：
# a.采样定理告诉我们，采样频率要大于信号频率的两倍
# b.N个采样点，经过FFT之后，就可以得到N个点的FFT结果。为了方便进行FFT运算，通常N取2的整数次方。
#endregion

T = 4*pi #信号周期
N = 2**10 #采样点的数量
Fs = N/T #采样频率（一个周期内的采样点数量）

x = np.linspace(start=-2*pi,stop=-2*pi+T,num=N)
y = [1.0 if np.sin(i)>0 else -1.0 for i in x]
y = np.float64(y)

#2、傅里叶变换(正弦波的频率、幅值、相位)
f = np.fft.fft(y)#fft的结果数量与采样点的数量一样,每一个点对应一个频率的正弦波，其模即为正弦波的幅值。

#3、将傅里叶结果的各种频率波形进行合成。
#region 利用fft结果合成原信号
# a.每个点对应一个频率的正弦波
# b.需要知道其频率、相位、幅度，即可绘制除此波形
# c.幅度为复数的模，相位是复数的atan2(y,x)，频率稍麻烦，第n个点的频率为Fn=(n-1)*Fs/N(其中Fs为采样频率)
#endregion
yy = np.zeros_like(y)
AA = []
FF = []
for i in range(1,len(f)+1):
    a,b = np.real(f[i-1]),np.imag(f[i-1])#提取实部和虚部

    Fn = (i-1)*Fs/N #正弦波的频率（需要采样频率）
    An = 2*sqrt(a**2+b**2)/(N) #正弦波的幅值
    Pn = atan2(b,a) #正弦波的相位
    yy = yy + An*np.cos(2*pi*Fn*x+Pn)#根据原始抽样点叠加
    
    #region
    AA.append(An)
    FF.append(Fn)
    #endregion

    #绘制每一个正弦波
    plt.subplot(3,1,2)
    plt.plot(x,An*np.cos(2*pi*Fn*x+Pn),'-r')

    if i==N/2:
        break
idx = np.argmax(AA)

print(y)
print("sum(y)=", np.sum(y))
print(f)
print(AA[idx], 1/FF[idx], 2*pi)

#绘制原始信号
plt.subplot(3,1,1)
plt.plot(x,y,'-r')

#绘制合成的正弦波
plt.subplot(3,1,3)
plt.plot(x,yy,'-r')

plt.show()

