import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.colors as mcolors
from math import *


colors=list(mcolors.TABLEAU_COLORS.keys()) #颜色变化
#生成数据
clu = 6
x1 = np.random.choice(6,clu).astype(float)
y1 = np.random.choice(6,clu).astype(float)

x = x1
y = y1
# for i in range(100):
#     x = np.append(x, x1 + np.random.randn(clu)*1.1)
#     y = np.append(y, y1 + np.random.randn(clu)*1.1)

# plt.subplot(1,2,1)
# plt.plot(x, y, '.r')

K = []
score = []
data = np.stack([x,y],axis=1)

result = DBSCAN(eps=sqrt(1) + 1e-2, min_samples=1).fit(data)
pred = result.labels_

# plt.subplot(1,2,2)
for p in range(max(pred)+1):
    pts = data[pred==p]
    plt.plot(pts[:,0], pts[:,1], '.')

#plt.axis([0,6,0,6])#
plt.axis('equal')
plt.grid()
plt.show()


for k in range(2,clu*2):
    kmeans_result = KMeans(n_clusters=k, random_state=0).fit(data)
    cluster_labels = kmeans_result.labels_
    score.append(silhouette_score(data, cluster_labels))
    K.append(k)
    plt.subplot(1,2,2)

print(np.array(score).argmax()+2)
plt.plot(K,score)
plt.show()

pass



