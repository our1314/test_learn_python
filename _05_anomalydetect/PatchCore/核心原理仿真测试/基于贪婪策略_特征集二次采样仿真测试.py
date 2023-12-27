"""
目标是将一个大型的特征集合进行缩小，缩小后的特征集合尽量尽量分布均匀才能代码原始特征集。
而随机采样效果不是很好。
"""
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt

#region 0、生成样本点
xx = np.array([range(200)])
yy = np.array([range(200)])
xx,yy = np.meshgrid(xx,yy)
xx,yy = xx.reshape(-1),yy.reshape(-1)
pts = np.stack([xx,yy],axis=0).T
#endregion

#region 1、随机采样
idx = np.random.choice(len(pts), 1000)
subpts1 = pts[idx]

plt.subplot(1,2,1)
plt.plot(subpts1[:,0],subpts1[:,1],".r")
#plt.axis([0,200,0,200])#
#endregion

#region 2、贪婪策略
"""
1、随机采集10个点，计算其与集合的平均距离
2、选择其中的最大值
3、计算选择点与集合的
"""

#按batch计算向量的欧氏距离
def _compute_batchwise_differences(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
    """Computes batchwise Euclidean distances using PyTorch. 使用pytorch计算batchwise的欧氏距离"""
    a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)#bmm为计算两个tensor的矩阵乘法
    b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
    a_times_b = matrix_a.mm(matrix_b.T)
    return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()#sqrt((a-b)²)

#1、随机选择一个点p1，计算p1与集合所有点的距离集合d1
#2、找到d1中距离最大的点p2，并加入二次采样点集
#3、计算p2与所有点的距离集合d2
#4、用d1与d2中更小的距离更新d1，并重复第2步
select_idx = []
pts = torch.tensor(pts)
idx = np.random.choice(range(len(pts)),1)
d1 = _compute_batchwise_differences(pts,pts[idx])
d1 = torch.mean(d1, axis=-1).reshape(-1, 1)#求均值
for i in tqdm.tqdm(range(1000),desc="jj"):
    idx = torch.argmax(d1).item()
    select_idx.append(idx)

    d2 = _compute_batchwise_differences(pts,pts[idx:idx+1])
    d1 = torch.cat([d1,d2],dim=-1)
    d1 = torch.min(d1,dim=1).values.reshape(-1,1)
    
    #region 逐点显示(需要打断点)
    # subpts = pts[idx]
    # plt.subplot(1,2,2)
    # plt.plot(subpts[0],subpts[1],".r")
    # plt.ion()
    # plt.show()
    # pass
    #endregion
 
subpts = pts[select_idx]
plt.subplot(1,2,2)
plt.plot(subpts[:,0],subpts[:,1],".r")
plt.show()

#endregion
pass



 