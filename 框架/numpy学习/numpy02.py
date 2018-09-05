import numpy as np;
#创建 定义类型
a = np.array([2,23,4],dtype = np.float64);
#无逗号
print (a);
print (a.dtype);
#定义二维array
b = np.array([[2,23,4],
             [2,3,4]]);
print (b);
#全部为0的矩阵 3行4列 或者全部为1
c = np.zeros((3,4));
print (c);
#定义起始 截止 步长 reshape将矩阵大小变换
d = np.arange(1,9,1).reshape(2,4);
print (d);
e = np.array([[1,2],[3,4]]);
#从轴的方向看过去
sum0 = np.sum(e,axis=0);
sum1 = np.sum(e,axis=1);
print (sum0);
print (sum1);