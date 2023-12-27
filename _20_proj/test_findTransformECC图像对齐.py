import os
from random import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = 'D:/desktop/tmp2.png'#input('输入图像路径：')
src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray

dir_image = 'D:/work/proj/抽检机/program/ChouJianJi/data/ic' #"D:/work/files/deeplearn_datasets/其它数据集/抽检机缺陷检测/8050"
files_all = os.listdir(dir_image)
images_path = [os.path.join(dir_image, f) for f in files_all if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]
#imgs = [cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR) for f in images_path]

for p in images_path:
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    cv2.imshow("tmp", tmp)
    cv2.imshow("src", img)
    cv2.waitKey()

    H = np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-8)
    r,H = cv2.findTransformECC(tmp, img, H, motionType=cv2.MOTION_EUCLIDEAN, criteria=criteria)
    print(H)
    pass

