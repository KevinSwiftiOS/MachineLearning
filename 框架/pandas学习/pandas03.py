import numpy as np;
import pandas as pd;
dates = pd.date_range('20130101',periods=6);
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['a','b','c','d']);
#附上另外的值
df.iloc[2,2] = 111;
print(df);
df.loc['20130101','b'] = 222;
print(df);
#混合更改 第一中>0的数
df[df.a > 0] = 33;
print(df);
#只对第一行改
df.b[df.a > 4] = 0;
print(df);
df['d'] = 55555;
#加上一列
df['e'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130101',periods=6));
print(df);