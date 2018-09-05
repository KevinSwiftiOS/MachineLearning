import numpy as np;
import pandas as pd;
dates = pd.date_range('20130101',periods=6);
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d']);
#处理丢失的数据
df.iloc[0,1] = np.nan;
#丢失掉nan的值 any 表示有nan 就丢掉一行
print(df.dropna(axis=0,how='any'));
#丢失掉nan的值 any 表示一行全部nan 就丢掉一行
print(df.dropna(axis=0,how='all'));
#将nan填上
print(df.fillna(value=33333));
#表示有没有nan
print(df.isnull());
#表格太大 查找是否丢失了数据
print(np.any(df.isnull()) == True);