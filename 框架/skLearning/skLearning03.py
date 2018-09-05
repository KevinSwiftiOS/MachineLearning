#资料的偏差和跨度会影响机器学习 进行正规化 即标准化提升成效
from sklearn import preprocessing;
import  numpy as np;
from sklearn.model_selection import  train_test_split;
#生成适合做classification资料的模块
from sklearn.datasets.samples_generator import make_classification;
from sklearn.svm import  SVC;
import  matplotlib.pyplot as plt;
#建立array
a = np.array([[10,2.7,3.6],
              [-100,5,-2],
              [120,20,40]],
             dtype=np.float64);
#标准化后的输出
print(preprocessing.scale(a));
#生成适合做分类的数据 生成具有2中属性的300笔数据
X,Y = make_classification(n_samples=300,n_features=2,n_redundant=0,
                          n_informative=2,random_state=22,n_clusters_per_class=1,
                          scale=100);

#数据的可视化
plt.scatter(X[:,0],X[:,1],c=Y);
plt.show();
#数据标准化前只有0.47的准确率
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3);
clf = SVC();
clf.fit(X_train,Y_train);
print(clf.score(X_test,Y_test));
X = preprocessing.scale(X);
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3);
clf = SVC();
clf.fit(X_train,Y_train);
print(clf.score(X_test,Y_test));