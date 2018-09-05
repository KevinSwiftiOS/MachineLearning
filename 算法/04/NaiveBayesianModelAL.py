 #!/usr/bin/python
 #coding:utf-8
dim = 10;
# **代表乘方的意思
print  1 / (0.01 ** dim);
#做可视化
from matplotlib import pyplot as plt
import  numpy as np
max_dim = 10;
ax = plt.axes(xlim = (0,max_dim),ylim = (0,1 / (0.01 ** max_dim)));
x = np.linspace(0,max_dim,1000);
y = 1 / (0.01 ** x);
plt.plot(x,y,lw = 2);
plt.show();