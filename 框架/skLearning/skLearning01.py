#导入模块
from sklearn import  datasets;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import  KNeighborsClassifier;
#加载数据集 属性标签放在X中 类别标签放在y中
iris = datasets.load_iris();
iris_X = iris.data
iris_Y = iris.target;
print(iris_X[:2,:]);
print(iris_Y);
#分为训练集和测试集 其中test_size = 0.3 表示测试集占总数的0.3
x_train,x_test,y_train,y_test = train_test_split(iris_X,iris_Y,test_size=0.3);
#看到分割号的数据集 顺序也会被打乱
print(y_train);
#建立模型-训练-预测
knn = KNeighborsClassifier();
#进行训练
knn.fit(x_train,y_train);
#进行预测
print(knn.predict(x_test));
print(y_test);
