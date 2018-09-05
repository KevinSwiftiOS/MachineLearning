import numpy as np;
a = np.arange(14,2,-1).reshape((3,4));
print (a);
#计算最小值的索引 最前面一位为0 最后面为1
print (a.argmax());
#平均值
print(a.mean());
#中位数
print(np.median(a));
#累加 第二个为第一个加第二个 第三个为1 2 3相加
print(a.cumsum());
#两个数之间的差
print(np.diff(a));
#非零的数 #输出行数和列数
print(np.nonzero(a));
#排序 逐行进行排序
print(np.sort(a));
#矩阵的转置
print(a.transpose());
print(a.transpose().dot(a));
#clip功能 小于5的数等于5 大于9的数为9
print(np.clip(a,5,9));
#平均数
print(np.mean(a,axis=0));