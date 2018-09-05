#merge合并
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
#plot data;
#Series
data = pd.Series(np.random.randn(1000),index=np.arange(1000));
#累加
data = data.cumsum();
data.plot();
dataFrame = pd.DataFrame(
    np.random.randn(1000,4),
    index=np.arange(1000),
    columns=list("ABCD")
    )
#输出前5个数据
print(dataFrame.head());
#以散点图的形式画出
ax = dataFrame.plot.scatter(x = 'A',y = 'B',color = 'DarkBlue',label = 'Class1');
#将之下这个data花在上一个ax中
dataFrame.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
dataFrame.cumsum();
dataFrame.plot();
plt.show();