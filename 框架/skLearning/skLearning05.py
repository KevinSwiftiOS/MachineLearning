#克服过拟合问题
from sklearn.learning_curve import  learning_curve #学习曲线模块
from sklearn.datasets import  load_digits;
from sklearn.svm import SVC;
import matplotlib.pyplot as plt;
import numpy as np;
digits = load_digits(); #数据集 都是数字
X = digits.data;
Y = digits.target;
#样本由小到大分成5轮检视学习曲线
train_sizes,train_loss,test_lost = learning_curve(
    SVC(gamma=0.001),X,Y,cv=10,scoring='mean_squared_error',
    train_sizes=[0.1,0.25,0.5,0.75,1]
);

#平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss,axis=1);
test_loss_mean = -np.mean(test_lost,axis=1);
#进行可视化图像
plt.plot(train_sizes,train_loss_mean,'o-',color = 'r',
         label = "Training");
plt.plot(train_sizes,test_loss_mean,'o-',color = 'g',
         label = "Cross-Validation");
plt.xlabel("Training examples");
plt.ylabel("Loss");
plt.legend(loc = "best");
plt.show();

