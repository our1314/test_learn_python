"""
参考：https://github.com/facebookresearch/faiss/wiki/Getting-started
Faiss为稠密向量提供高效相似度搜索和聚类，支持十亿级别向量的搜索，是目前最为成熟的近似近邻搜索库。
"""

import faiss                   # make faiss available
import numpy as np

#1、生成数据
d = 64                           # dimension 维度
nb = 100000                      # database size 数据库的数量
nq = 9000                       # nb of queries 查询数据的数量
#np.random.seed(1234)             # make reproducible 
data_lib = np.random.random((nb, d)).astype('float32')
data_query = np.random.random((nq, d)).astype('float32')

#2、使用
faiss_index = faiss.IndexFlatL2(d)   # build the index 创建索引器时显示指定数据维度(后续查询维度固定)
faiss_index.add(data_lib)                  # add vectors to the index 添加数据到索引器
distance_topk,index_topk = faiss_index.search(data_query, k=1)#计算每条数据(每一行)在数据库中距离最近的k个样本。

#3、打印结果
print("dis_distance_topktopk(topk个距离):", distance_topk.shape)
print("index_topk(topk个距离对应的索引):", index_topk.shape)
pass