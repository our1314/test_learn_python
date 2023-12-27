import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 
import random
from sklearn.metrics import pairwise_distances

#https://blog.csdn.net/qq_45759229/article/details/124855867

#region 测试np.unique的作用

a = np.unique([1,3,2,2,2,3,3,4,4,4,4])#对于一维数组或者列表，np.unique()函数 去除其中重复的元素，并按元素由小到 返回一个新的无元素重复的元组或者列表。
print(a)
#endregion

n_sample=10
n_feature=2
n_cluster=3

# 轮廓系数是对聚类质量进行评价
X = np.random.rand(n_sample,n_feature)
Y = np.random.choice(range(n_cluster), size=n_sample) # random assign cluster label
# X = np.array([[1,2],[1,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11]],dtype=float)
# Y = np.array([0,1,2,0,1,2,0,1,2,1],dtype=int)
X = np.array([[0,9],[2,3],[2,4],[3,3],[3,4],[5,9]],dtype=float)
Y = np.array([0,1,1,1,1,0],dtype=int)

score = silhouette_score(X, Y, metric='euclidean')
print("sklearn calculate silhouette:",score)

# use dict to store all cluster data and label, so it is easy to code 
cluster_dict={}
label_set=np.unique(Y)
for i in label_set:
    cluster_dict[i]=X[Y==i,:]
#print(cluster_dict)  ## 

s=0
for i in cluster_dict:
    for idx in range(len(cluster_dict[i])): ##
        temp=cluster_dict[i][idx].reshape(1,-1) #
        temp_a=pairwise_distances(temp,cluster_dict[i]) #same cluster
        a=np.sum(temp_a)/(len(cluster_dict[i])-1) # mean
        b=np.inf
        for j in cluster_dict:
            if(j!=i):
                temp_b=pairwise_distances(temp,cluster_dict[j]) # not same custer
                temp_b=np.sum(temp_b)/len(cluster_dict[j]) #mean
                b=min(b,temp_b) #
        s_sample=(b-a)/max(a,b)
        #print(s_sample) ## each silhouette score 
        s=s+s_sample
print("==============")
print("manual implementation of silhouette:",s/X.shape[0])

