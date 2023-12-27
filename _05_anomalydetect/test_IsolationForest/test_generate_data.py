from pyod.utils.data import generate_data
from pyod.utils.example import visualize, data_visualize

X_train, Y_train, X_test, Y_test = generate_data(n_train=5, n_test=2, contamination=0.1)

print(X_train)
print(X_train[:,0])

print(Y_train)
print(Y_train[:,0])

data_visualize(X_train=X_train, y_train=Y_train)
