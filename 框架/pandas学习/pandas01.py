#pandas更像是字典话的numpy
import numpy as np;
import pandas as pd;
#序列更像是列表
s = pd.Series([1,3,6,np.nan,44,1]);
print (s);
#创建dataForm 一个序列
dates = pd.date_range('20160101',periods=6);
print(dates);
#一行是index 列是abcd
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d']);
print (df);
df1 = pd.DataFrame(np.random.randn(6,4).reshape((6,4)));
print(df1);
#用一列代替每行数据 每一列的不同形式
print(df1.dtypes);
print(df.index);
print(df.columns);
#打印value
print(df.values);
#描述方差数据
print(df.describe());
#转置
print(df.T);
#进行排序 倒序排列
print(df.sort_index(axis=0,ascending=False));
#对values进行排序
print(df.sort_values(by = 'd'));