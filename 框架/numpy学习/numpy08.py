#赋值
import numpy as np;
a = np.arange(4);
print(a);
b = a;
c = a;
d = b;
#都是指向同一个内存空间
print (b is a);
a[0] = 11;
print(b);
print (d is a);
d[1:3] = [22,33];
print (a);
#浅拷贝 b与a没有关联
b = a.copy();
a[3] = 55;
print (a);
print (b);
#查看地址是否相同
print(id(a) == id(b));