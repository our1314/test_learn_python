# 代码来至 https://blog.csdn.net/helloword111222/article/details/121279599
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.RandomState(42)

# 生成训练数据
X = 0.3*rng.randn(100,2)#从一个正太分布中随机采样
X_train = np.r_[X-2,X+2]#行方向堆叠两个矩阵（垂直方向拼接）

# 生成均匀分布数据
X_outliers = rng.uniform(low=-4, high=4, size=(20,2))#从一个均匀分布[low,high)中随机采样(左闭右开)

X = np.r_[X_train,X_outliers]

plt.subplot(121)
#显示训练数据
b1 = plt.scatter(X_train[:,0], X_train[:,1], c='white', s=20, edgecolors='k')#绘制散点图
b2 = plt.scatter(X_outliers[:,0], X_outliers[:,1], c='black', s=20, edgecolors='k')#绘制散点图
plt.axis('tight') # https://blog.csdn.net/qq_36439087/article/details/121527402
plt.xlim(-5,5)
plt.ylim(-5,5)
#plt.show()

#训练模型
clf = IsolationForest(n_estimators=100, max_samples=100, contamination=0.01, random_state=rng)
clf = LocalOutlierFactor(n_neighbors=80, novelty=True)
clf.fit(X_train)

xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(122)
plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
#plt.show()

y_pred = clf.score_samples(X)+2.5

print(y_pred)

okindex = []
ngindex = []
for i,value in enumerate(y_pred):
    if value > -0.13:
        okindex.append(i)
    else:
        ngindex.append(i)

ok=X[okindex]
ng=X[ngindex]
# 显示训练数据
b1 = plt.scatter(ok[:,0], ok[:,1], c='green', s=20, edgecolors='k')#绘制散点图
b2 = plt.scatter(ng[:,0], ng[:,1], c='red', s=20, edgecolors='k')#绘制散点图
plt.axis('tight') # https://blog.csdn.net/qq_36439087/article/details/121527402
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()
