import numpy as np;
a = np.array([10,20,30,40]);
b = np.arange(4);
print(a,b);
print (a + b);
#对每个数求sin
c = 10 * np.sin(a);
print (c);
#判断有哪些值小于某个数 返回一个列表
print(a == 20);
e = np.array([[1,1],[0,1]]);
f = np.arange(4).reshape((2,2));
print(e);
print(f);
#逐个相乘
print(e * f);
#矩阵乘法
print(np.dot(e,f));
print(e.dot(f));
#随机生成 shape
a = np.random.random((2,4));
#求和 求最小 求最大 等方法 axis = 1 行数列数求和 axis = 0 列数中求和
print(np.sum(a,axis=0));
