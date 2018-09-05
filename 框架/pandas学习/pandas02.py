import numpy as np;
import pandas as pd;
dates = pd.date_range('20130101',periods=6);
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d']);
print(df);
#选择第一列
print(df.a);
#进行切片选择
print(222);
print(df[0:3]);
print(333);
print(df['20130102':'20130104']);
#select by label:loc 即行号
print(df.loc['20130102']);
#保存行的 以列打印出
print(df.loc[:,['a','b']]);
#保留某行
print(df.loc['20130102',['a','b']]);
#从位置iloc 位置postion
print(444);
print(df.iloc[3]);
print(df.iloc[3:5,1:3]);

#逐个不连续的删选
print(df.iloc[[1,3,5],1:3]);
print(df.ix[:3,['a','c']]);
#进行是或否的删选
print(df[df.a > 8]);

