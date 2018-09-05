#交叉验证的方法
from sklearn.datasets import  load_iris;
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import  KNeighborsClassifier;
import matplotlib.pyplot as plt;
#进行交叉验证
from sklearn.cross_validation import  cross_val_score;
#加载数据集
iris = load_iris();
X = iris.data;
Y = iris.target;
#分割数据集
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3);
#建立模型
knn = KNeighborsClassifier();
#训练模型
knn.fit(X_train,Y_train);
#将标准率打印出来
print(knn.score(X_test,Y_test));

#使用k折交叉验证模块 cv=5 分成5分 4:1 1份是测试集
scores = cross_val_score(knn,X,Y,cv=5,scoring='accuracy');
#将五次预测的准确率打印出来
print(scores);
print(scores.mean());
#建立测试参数集
k_range = range(1,31);
k_scores = [];
#由迭代的方式计算不同的参数对模型的影响 并且返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k);
    scores = cross_val_score(knn,X,Y,cv = 10,scoring="accuracy");
    k_scores.append(scores.mean());
#进行可视化
plt.plot(k_range,k_scores);
plt.xlabel("Value of K for KNN");
plt.ylabel("Cross-Validated Accuracy");
plt.show();
#以平均方差进行判断
k_range = range(1,31);
k_scores = [];
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k);
    #记得乘上-1 因为计算出来是负数
    loss = -cross_val_score(knn,X,Y,scoring="mean_squared_error");
    k_scores.append(loss.mean());
plt.plot(k_range,k_scores);
plt.xlabel("Value of K for knn");
plt.ylabel("Cross-Validated MSE");
plt.show();