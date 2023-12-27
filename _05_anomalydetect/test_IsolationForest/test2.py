import numpy as np
from sklearn.model_selection import train_test_split

n_samples, n_outliers = 120, 40
rng = np.random.RandomState(0)
a = rng.randn(1, 2)

covariance = np.array([[0.5, -0.1], [0.7, 0.4]])#协方差
cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical 球型
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))#外点

X = np.concatenate([cluster_1, cluster_2, outliers])#生成训练数据
y = np.concatenate([np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)])#生成标签

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


import matplotlib.pyplot as plt
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
handles, labels = scatter.legend_elements()
plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.title("Gaussian inliers with \nuniformly distributed outliers")
plt.show()

from sklearn.ensemble import IsolationForest
clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_train)


import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay #决策边界显示
disp = DecisionBoundaryDisplay.from_estimator( clf, X, response_method="predict",alpha=0.5,)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
disp.ax_.set_title("Binary decision boundary \nof IsolationForest")
plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.show()