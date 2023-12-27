from numpy import array,dot
a=array([[1,2],
         [1,2]])
b=array([[5,6],
         [5,6]])
print(a@b)
print(dot(a,b))
#这是python3.5后的新运算符,它与numpy.dot（）的作用是一样的，矩阵乘法（就是线性代数里学的）