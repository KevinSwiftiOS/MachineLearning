import numpy as np;
import pandas as pd;
dates = pd.date_range('20130101',periods=6);
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d']);
#concat 合并
#定义资料集
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
#进行上下合并 竖向合并 1为横向 重新排序
res = pd.concat([df1,df2,df3],axis=0,ignore_index=True);
print(res);
#join功能 inner outer的join
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'],index=[1,2,3]);
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'],index=[2,3,4]);
#不一样的进行处理 默认是outer的join inner寻找相同的部分
res = pd.concat([df1,df2],join='inner',ignore_index=True);
print(res);
#join_axes 没有的部分不会进行考虑 只会考虑df1的index
res = pd.concat([df1,df2],axis=1,join_axes=[df1.index]);
print(res);
#append功能 默认竖向加数据
df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
print(df1.append([df2,df2],ignore_index=True));
s1 = pd.Series([1,2,3,4],index=['a','b','c','d']);
print(df1.append(s1,ignore_index=True));
