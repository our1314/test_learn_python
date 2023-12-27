import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
# from numpy import fft,ifft

x=np.linspace(0,1,1400,dtype=float)
y=7*np.sin(2*np.pi*200*x) + 5*np.sin(2*np.pi*400*x)+3*np.sin(2*np.pi*600*x)

# plt.figure()
# plt.plot(x,y)
# plt.title("原始波形")

plt.figure()
plt.plot(x[:50],y[:50])
plt.title('原始部分波形（前50组样本）')
plt.show()

fft_y = fft(y)
print(len(fft_y))
print(fft_y[0:5])

x = np.arange(400)
abs_y = np.abs(fft_y)
angle_y = np.angle(fft_y)

plt.figure()
plt.plot(x,abs_y)   
plt.title('双边振幅谱（未归一化）')
 
plt.figure()
plt.plot(x,angle_y)   
plt.title('双边相位谱（未归一化）')
plt.show()