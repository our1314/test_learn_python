import matplotlib.pyplot as plt
import pandas as pd

path = "D:/work/files/deeplearn_datasets/异常检测/international-airline-passengers.csv"
data = pd.read_csv(path)
x = data[['data']]
plt.plot(x)
plt.show()
pass