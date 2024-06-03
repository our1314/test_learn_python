import cv2
from matplotlib import pyplot as plt
import numpy as np


def style_transfer(image, ref):
    out = np.zeros_like(ref)
    _, _, ch = image.shape
    for i in range(ch):
        print(i)
        hist_img, _ = np.histogram(image[:, :, i], 256)
        hist_ref, _ = np.histogram(ref[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)
        cdf_ref = np.cumsum(hist_ref)

        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))  # 找出tmp中最小的数，得到这个数的索引
            out[:, :, i][ref[:, :, i] == j] = idx
    return out


img = cv2.imread('D:/work/files/data/image/a1.png')
ref = cv2.imread('D:/work/files/data/image/b.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

out = style_transfer(img, ref)

hist_img = cv2.calcHist([img], [0], None, [255], [0, 255])
hist_ref = cv2.calcHist([ref], [0], None, [255], [0, 255])
hist_out = cv2.calcHist([out], [0], None, [255], [0, 255])

plt.subplot(231)
plt.title("img")
plt.imshow(img)
plt.subplot(234)
plt.plot(hist_img)

plt.subplot(232)
plt.title("ref")
plt.imshow(ref)
plt.subplot(235)
plt.plot(hist_ref)

plt.subplot(233)
plt.title("out")
plt.imshow(out)
plt.subplot(236)
plt.plot(hist_out)

plt.show()
plt.pause()
pass