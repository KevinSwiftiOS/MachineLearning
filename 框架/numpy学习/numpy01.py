#!/usr/bin/python
# -*- coding: UTF-8 -*-
import  numpy as np
'''
学习网站https://blog.csdn.net/cxmscb/article/details/54583415
首先需要创建数组才能对其进行其它操作。
我们可以通过给array函数传递Python的序列对象创建数组，如果传递的是多层嵌套的序列，将创建多维数组(下例中的变量c):

'''
a = np.array([1,2,3,4]);
b = np.array((5,6,7,8));
c = np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]]);
print c;
#获得数组的大小
print a.shape;
#通过改变shape 动态调整大小
c.shape = 4,3
print c;
c.shape = 4,-1;
print  c;
#arange 函数相当于 python的range 函数 指定开始值 步长 和终步
d = np.arange(0,1,0.1);
print d;
#linspace函数通过指定开始值 终值和元素个数来创建一维数组
e = np.linspace(0,1,12);
print e;
# 从字节序列中产生数组
s = "abcdefgh";
print np.fromstring(s,dtype=np.int8);
#将数组下标转换为对应的值
def func(i):
    return (i % 4) + 1;
print np.fromfunction(func,(10,));
#创建二维数组表示九九乘法表
def func2(i,j):
    return (i + 1) *(j + 1);
print np.fromfunction(func2,(9,9));

x = np.array([[1,2,6,3],[2,3,4,6]],dtype=np.float32);
print x.astype(np.int64);
#矢量和标量的运算 将标量广播到矢量的各个元素上
print x * 2;
print x[1][0:3];
#数组的转置或者转换只会返回元数据的一个视图 不会对元数据进行修改
k = np.arange(9); #0.1.8
print k;
#每个维度是3的
m = k .reshape((3,3));
print m.T;
print np.dot(m,m.T);
n = np.arange(9);
print np.square(n);
print n.mean();
f = np.array([[1,2,3],[1,2,3]]);
g = np.array([[1,1],[1,1],[1,1]]);
print np.dot(f,g);
#随机数生成
print np.random.randint(0,2,size = 100000);
#指定元素的个数
b = np.array([1, 2, 3, 4, 5, 6]);
print b[np.newaxis];
print np.transpose(b);
X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
print X[:,1][:,np.newaxis];
print X[0,1];