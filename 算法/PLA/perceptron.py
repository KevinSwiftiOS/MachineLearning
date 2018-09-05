#PLA算法的实现
import numpy as np;
from  numpy import *;
#初始化w和b
w = [0,0];
b = 0;
def dataSet():
    #数据集的建立
    T = np.array([[3,3],[4,3],[1,1]]);
    y = [1,1,-1];
    return T,y;
#计算最小值
def calMinL(x,y):
    global w,b;
    #np中的二维矩阵乘法 一维是内积
    return  y * (np.dot(x,w) + b);


def update(x,y):
    #跟新w b
    global w,b;
    for i in range(len(x)):
        w[i] += y * x[i];
    b += y;
def perceptron(T,y):
    global w,b;
    #迭代次数
    iteration = 0;
    #判断是否找到损失函数的最小值
    findMinL = False;
    #样本点个数
    #1表示第一维的长度 0表示第二维的长度
    sampleNumber = T.shape[0];
    while(not findMinL):
        for i in range(sampleNumber):
            #是负数的边一侧的
            if calMinL(T[i],y[i]) <= 0:
                if iteration == 0:
                    print(1)
                    print("{} {} {}".format(iteration,w,b));
                else:
                    print(2)
                    print('{} X{} {} {}'.format(iteration, i + 1, w, b));
                iteration += 1;
                update(T[i],y[i]);
                break;
            elif i == sampleNumber - 1:
                #表示已经找到了最后一个
                print(3)
                print('{} X{} {} {}'.format(iteration, i + 1, w, b));
                iteration += 1;
                findMinL = True;
    print (4);
    print('{}    {} {}'.format(iteration, w, b))
if __name__ == '__main__':
    T,y = dataSet();
    perceptron(T,y);
