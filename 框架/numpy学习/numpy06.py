import numpy as np;
#进行array合并
# a = np.array([1,1,1]);
# b = np.array([2,2,2]);
# #上下合并 shape表示这个是2行3列的
# print(np.vstack((a,b)).shape);
# #左右合并 为[1,1,1,2,2,2];
# print(np.hstack((a,b)));
# c = a.reshape(a.size,1);
# print(c);
#
# #后面加一个维度
# a= a[:,np.newaxis];
# b = b[:,np.newaxis];
# #或者前面加一个维度
# #进行多个array的合并 在哪个维度进行合并
# print(np.concatenate((a,b,b,a),axis=1));
a = np.arange(12).reshape((3,4));
sum0 = np.sum(a,axis=0);
sum1 = np.sum(a,axis=1);
print (a);
#1行0列
print(sum0);
print(sum1);
