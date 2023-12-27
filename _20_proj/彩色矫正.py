#将图像的RGB通道的均值调整为一样。
import os
from random import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

path = 'D:/desktop/tmp2.png'#input('输入图像路径：')
src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
dir_image = "D:\work\proj\抽检机\program\ChouJianJi\data\ic"
files_all = os.listdir(dir_image)
images_path = [os.path.join(dir_image, f) for f in files_all if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
# images_path.append(path)
#shuffle(images_path)#随机排序
images_path.insert(0, path)

imgs = np.array([cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR) for f in images_path])

b = imgs[:,:,:,0]
g = imgs[:,:,:,1]
r = imgs[:,:,:,2]
mb = b.mean()
mg = g.mean()
mr = r.mean()

print(mb,mg,mr)

for img in imgs:
    src = img.copy()

    bb = img[:,:,0]
    gg = img[:,:,1]
    rr = img[:,:,2]

    b_ = int(mb-bb.mean())
    g_ = int(mg-gg.mean())
    r_ = int(mr-rr.mean())

    print(mb, mg, mr)
    print(bb.mean(), gg.mean(), rr.mean())
    print(b_, g_, r_)

    bb = bb+b_
    gg = gg+g_
    rr = rr+r_

    img = np.stack([bb,gg,rr], axis=2)
    img = img.astype(np.uint8)
    dis = np.hstack((src, img))
    cv2.imshow("dis", dis)
    cv2.waitKey()

