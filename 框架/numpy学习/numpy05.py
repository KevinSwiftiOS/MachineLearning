import numpy as np;
#缩影
a = np.arange(3,15).reshape((3,4));
print(a);
#对位置进行索引 输出第二行
print(a[2]);
print(a[1][1]);
#输出第二行 第一列
print(a[1,0]);
#第二行所有数
print(a[2:]);
#第一行 第1列到第二列的数
print(a[1,1:2]);
#for循环进行定位
for row in a:
    #迭代行
    print(row);
#迭代列
for column in a.transpose():
    print(column);
#迭代一个个元素 弄成了一行
print(a.flatten());
#一个迭代器
for item in a.flat:
    print(item);