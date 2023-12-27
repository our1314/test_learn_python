import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 输入数据
data = np.genfromtxt('C:/Users/pc/Documents/WeChat Files/wxid_1hosdyyri73b22/FileStorage/File/2023-09/K-means.txt')

# 设置待测试的K值范围
k_values = range(2, 15)

best_k = 0
best_score = -1
best_cluster_labels = None

# 遍历不同的K值
for k in k_values:
    # 进行K-means聚类
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

    # 获取聚类标签
    cluster_labels = kmeans.labels_

    # 计算轮廓系数
    score = silhouette_score(data, cluster_labels)

    # 打印当前K值和对应的轮廓系数
    print(f"K={k}, Silhouette Score: {score}")

    # 更新最佳的K值和轮廓系数
    if score > best_score:
        best_score = score
        best_k = k
        best_cluster_labels = cluster_labels

# 使用最佳K值进行聚类
kmeans = KMeans(n_clusters=best_k, random_state=0).fit(data)

# 可视化聚类结果的轮廓系数
plt.figure(figsize=(8, 5))
plt.plot(list(k_values),
         [silhouette_score(data, KMeans(n_clusters=k, random_state=0).fit(data).labels_) for k in k_values], marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette coefficient')
plt.title('Silhouette coefficient for different values of K')
plt.show()

# 可视化聚类结果
fig, ax = plt.subplots()
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i in range(best_k):
    cluster_points = data[best_cluster_labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i])
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', color='black')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Clustering Results (K={best_k})")
plt.show()
