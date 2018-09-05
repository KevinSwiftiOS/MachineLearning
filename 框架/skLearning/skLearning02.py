#利用sklearn中强大的数据库
from __future__ import print_function;
from sklearn import datasets;
from sklearn.linear_model import LinearRegression;
import matplotlib.pyplot as plt;
#导入数据 被给x和y赋值
loaded_data = datasets.load_boston();
data_X = loaded_data.data;
data_Y = loaded_data.target;
#定义模型
model = LinearRegression();
model.fit(data_X,data_Y);

#用前4个进行预测 同时打印出真实值
print(model.predict(data_X[:4,:]));
print(data_Y[:4]);
#创建虚拟数据 进行可视化 建立100个sample 有一个future 和一个target
X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=50)
plt.scatter(X,y);
plt.show();