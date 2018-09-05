#分割array
import numpy as np;
a = np.arange(12).reshape((3,4));
print(a);
#从横向分割 对列进行操作 分割成4块
print(np.split(a,4,axis=1));
#从行数分割
print(np.split(a,3,axis=0));
#进行不等量的分割 4列可以分割成3列
print(np.array_split(a,3,axis=1));
#分割函数 横向分
print(np.vsplit(a,3));
#纵向分
print(np.hsplit(a,4));