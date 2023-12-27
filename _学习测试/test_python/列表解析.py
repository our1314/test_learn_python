#  以下两种语法效果一样 https://www.cnblogs.com/liu-shuai/p/6098227.html
#  1、普通方法
L = []
for i in range(1, 11):
    L.append(i ** 2)#双星号表示求幂
print(L)

#  2、列表解析：根据已有列表，高效创建新列表的方式。
L = [i ** 2 for i in range(1, 11)]
print(L)
