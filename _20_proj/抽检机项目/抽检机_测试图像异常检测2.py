import sys
sys.path.append("D:/work/program/python/DeepLearning/test_learn_python")
import os
from random import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
# from model1 import Model1
# from model2 import Model2
import torch
from torchvision import transforms
from _05_anomalydetect.PatchCore.patchcore import PatchCore
import torchvision.transforms.functional as F
from PIL import Image
from math import *
import faiss
from our1314.work import GetAllFiles
from queue import Queue
import gc


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_img = [
        transforms.Resize(300),
        # transforms.RandomRotation(rotate_degrees, transforms.InterpolationMode.BILINEAR),
        #transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
        #transforms.RandomHorizontalFlip(h_flip_p),
        #transforms.RandomVerticalFlip(v_flip_p),
        #transforms.RandomGrayscale(gray_p),
        # transforms.RandomAffine(rotate_degrees, 
        #                         translate=(translate, translate),
        #                         scale=(1.0-scale, 1.0+scale),
        #                         interpolation=transforms.InterpolationMode.BILINEAR),
        
        #transforms.GaussianBlur(kernel_size=(7,7),sigma=(0.1,2.0)),#随机高斯模糊          

        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
transform = transforms.Compose(transform_img)

dir_image = "D:/work/files/deeplearn_datasets/choujianji/roi-mynetseg/test/test"
files_all = GetAllFiles(dir_image)
#shuffle(files_all)
files_all = [f for f in files_all if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.bmp')]

#faiss_index = faiss.IndexFlatL2(1024)
patchcore = PatchCore(torch.device("cuda:1"))
cnt_queue = 30

#1、向队列填充特征
queue_feas = Queue()
for i,path in enumerate(files_all):
    image = transform(Image.open(path).convert('RGB'))#type:torch.Tensor
    with torch.no_grad():
        image = image.unsqueeze(0)
        input_image = image.to(torch.device("cuda:1"))
        fea = patchcore._embed(input_image)#提取特征
        fea = np.array(fea)
        s = int(sqrt(len(fea)))
        fea = fea.reshape(-1,s,s,1024)
        fea = torch.tensor(fea)
        queue_feas.put([path,fea])
    
    if i >= cnt_queue:
        path_query, fea_query = queue_feas.get()
        features = [fea[1] for fea in queue_feas.queue]
        fea_lib = torch.cat(list(features),dim=0)

        #region 提取特征序列进行可视化
        
        # dis = cv2.imread(path_query)
        # dis = cv2.resize(dis, (224,224))
        # ss = 224/28
        # for r in range(fea_lib.shape[1]):
        #     for c in range(fea_lib.shape[2]):
                

        #         jkjk = fea_lib[:2,r,c]
        #         xxxx = jkjk.flatten().numpy()
        #         plt.clf()
        #         plt.plot(xxxx)
        #         plt.ion()
        #         plt.show()


        #         pt1=(int(c*ss),int(r*ss))
        #         pt2=(int((c+1)*ss),int((r+1)*ss))
        #         dis = cv2.rectangle(dis, pt1, pt2, (0,0,255), 2)
        #         cv2.imshow("dis", dis)
        #         cv2.waitKey(1)

        #         if r==28:
        #             jj=0
        #             pass
        #endregion

        off = fea_query - fea_lib
        d = torch.norm(off,dim=-1)
        
        #region 分布图
        # dis = cv2.imread(path_query)
        # dis =cv2.resize(dis, (224,224))
        # cv2.imshow("dis",dis)
        # cv2.waitKey(1)
        # for r in range(28):
        #     for c in range(28):
        #         x,y=c*8,r*8
        #         dis = cv2.rectangle(dis, (x,y),(x+8,y+8),color=(0,0,255),thickness=2)
        #         cv2.imshow("dis",dis)
        #         cv2.waitKey(1)
        #         fff = d[:,r,c].numpy()
        #         print('mean=',np.mean(fff),'std=',np.std(fff))
        #         plt.clf()
        #         plt.plot(fff)
        #         plt.show()
        #endregion
        
        #region torch
        d,_ = torch.sort(d,dim=0)
        d = d[:10,:,:]
        d = torch.mean(d,dim=0)
        d = d - torch.mean(d)+0.3
        d = d/2
        d = d.numpy()
        d = cv2.resize(d, (224,224), cv2.INTER_LINEAR)
        #endregion
        
        #region numpy
        # d = d.numpy()
        # d = np.sort(d,axis=0)
        # d = d[:10,:,:]
        # d = np.mean(d,axis=0)
        # d = d-np.mean(d)+0.3
        # d = d/2
        # d = cv2.resize(d, (224,224), cv2.INTER_LINEAR)
        #endregion

        

        # d[d>=0.7]=1
        # d[d<0.7]=0
        d[d>1]=1
        d[d<0]=0
        d = (d*255).astype("uint8") 
        # d = np.expand_dims(d, axis=2)
        # k,d = cv2.threshold(d,0,255,cv2.THRESH_OTSU)

        #region 可视化
        xx = d.flatten()
        plt.plot(range(xx.shape[0]),xx, '-r')
        plt.show()
        #endregion

        print(path)
        #cv2.imwrite(f"D:/desktop/eee/{os.path.basename(path_query)}", d)
        cv2.imshow("dis", d)
        cv2.waitKey()


imgs = [transform(Image.open(f).convert('RGB')) for f in files_all[-100:]]#读取队列内的所有图像
imgs = torch.stack(imgs,dim=0)#合并为张量

with torch.no_grad():
    input_image = imgs.to(torch.device("cuda:1"))
    feas = patchcore._embed(input_image)#提取batchsize的特征
    feas = np.array(feas)
    s = int(sqrt(len(feas)/cnt_queue))
    feas = feas.reshape(-1,s,s,1024)

for fea in feas:#将特征填充进队列
    queue_feas.put(fea)

#2、遍历图像与队列特征进行异常检测
for i,path in enumerate(files_all[:-100]):
    # imgs = [transform(Image.open(f).convert('RGB')) for f in images_queue]#读取队列内的所有图像
    # imgs = torch.stack(imgs,dim=0)#合并为张量
    img = transform(Image.open(path).convert('RGB'))#type:torch.Tensor
    img = img.unsqueeze(0)

    #region patchcore提取特征
    with torch.no_grad():
        input_image = img.to(torch.device("cuda:1"))
        feas = patchcore._embed(input_image)
        feas = np.array(feas)
        feas = feas.reshape(-1,s,s,1024)
        #feas_train = feas.reshape(-1,1024)
        #faiss_index.add(feas_train)
    #endregion

    #region 遍历图像，计算异常分
    """
    如果以当前特征与所有特征距离的平均值作为异常分可能有问题。
    尝试以最近的几个(10个)近邻的均值作为异常分。
    """
    qfeas = np.stack(list(queue_feas.queue),axis=0)
    dis = []
    off = feas - qfeas
    for b in range(off.shape[0]):
        for y in range(off.shape[1]):
            for x in range(off.shape[2]):
                dis.append(np.linalg.norm(off[b,y,x]))
    dis = np.reshape(dis, off.shape[:3])
    dis = np.sort(dis,axis=0)
    dis = dis[:10,:,:]
    dd = np.mean(dis,axis=0)
    print(dis[:,0,0])

    # f = np.sort(ff,axis=3)
    # f = f[:,:,:,:10]
    # dd = np.mean(f,axis=3)-0.5
    dd = dd-0.5
    # dd = np.array(dd)
    # dd = dd.reshape(s,s)-0.5
    #dd = (dd - np.min(dd)) / (np.max(dd) - np.min(dd))

    dd = dd + 0.29 - np.mean(dd)

    dd = cv2.resize(dd, (224,224), cv2.INTER_LINEAR)
    cv2.imshow("dis", dd)
    cv2.waitKey(1)
    #endregion

    dd = dd*255
    dd = dd.astype("int32")
    cv2.imwrite(f"D:/desktop/eee/{os.path.basename(path)}", dd)
    #del imgs,feas
    gc.collect()


#鼠标点选计算异常
src = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)#type:np.ndarray
num = 224//s
#clf = LocalOutlierFactor(n_neighbors=40, contamination=0.01)#异常检测器
clf = LocalOutlierFactor(n_neighbors=10, contamination=1e-6, novelty=False)

def onmouse(*p):
    event, x, y, flags, param=p
    if event == cv2.EVENT_LBUTTONDOWN:
        dis = src.copy()

        #1、绘制鼠标位置
        #cv2.circle(dis, (x,y), 5, (0,0,255), 2)
        cv2.line(dis,(x,y),(x,y),(0,0,255),1)
        cv2.imshow("dis", dis)

        #3、异常检测
        # 1、收集异常点所在的图像索引
        # 2、所在图像索引上的异常坐标的聚类
        #X = imgs[:,y,x,:]
        y = int(y/num)
        x = int(x/num)
        X = feas[:,y,x,:]
        pred = clf.fit_predict(X)
        #pred = clf.score_samples(X)
        ng_index = [i for i,p in enumerate(pred) if p<0]
        p1 = [np.sqrt(np.sum(np.power(X[0]-p,2))) for p in X]
        p1 = [np.round(d,2) for d in p1]
        
        print(p1)
        print(np.max(p1[1:]),np.min(p1[1:]),np.mean(p1),np.std(p1))
        #print(ng_index)
        
        plt.clf()#清屏
        plt.ion()#不会阻塞线程
        plt.axis([0,50,0,2])
        plt.plot(p1, c='blue')
        plt.show()

cv2.namedWindow('dis')
cv2.setMouseCallback("dis",onmouse)
cv2.imshow('dis', src)
cv2.waitKey()

"""
总结：总体来说实现了思路，目前存在的问题如下
1、大量异常通常出现在图像未准确对齐的图像上
2、检测时间较长
解决方案：
1、正常测试时，图像想过应该比现在好很多，至少能保证不会出现过于模糊的图像。
2、图像对齐初步依靠模板匹配解决。
3、检测时间长的问题尝试使用深度学习训练的特征提取器提取特征进行解决。
第1、2在改造的机器上比较容易解决，现在优先测试第三个问题。
"""
