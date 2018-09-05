#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np#for array compute
from numpy import *
import random

def pla():
    #构造矩阵 默认范围类型是float64
    W=np.ones(4)#initial all weight with 1
    count=0
    dataset=[[1,0.10723,0.64385, 0.29556    ,1],
            [1 ,0.2418, 0.83075, 0.42741,   1],
            [1 ,0.23321 ,0.81004 ,0.98691,  1],
            [1 ,0.36163, 0.14351 ,0.3153,   -1],
            [1, 0.46984, 0.32142, 0.00042772,   -1],
            [1, 0.25969, 0.87208 ,0.075063, -1],
            ]

    while True:
        count+=1
        iscompleted=True
        for i in range(0,len(dataset)):
            #从-1开始赋值
            X=dataset[i][:-1];
            #返回两个数组的点积 一维数组返回向量积 二维数组返回矩阵的乘积
            Y=np.dot(W,X)#matrix multiply
             #表明都在一边时则一直分割下去 否则线性分割不正确 要进行线的调整
            if sign(Y)==sign(dataset[i][-1]):
                continue
            else:
                iscompleted=False
                W=W+(dataset[i][-1])*np.array(X);
        if iscompleted:
            break
    print("final W is :",W)
    print("count is :",count)
    return W;

def main():
    pla()
'''
一个模块被另一个程序第一次引入时，其程序将运行
如果我们想在模块被引入时，模块中的某一个程序块不被运行，可以使用_name_属性来使该程序仅在模块自身运行时执行
'''
if __name__ == '__main__':
    main()