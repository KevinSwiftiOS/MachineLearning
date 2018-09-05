#线性回归模型
import numpy as np;
from metric import r2_score;
class LinearRegression:
    #初始化参数
    def __init__(self):
        self.a_ = None;
        self.b_ = None;
    def fit(self,x_train,y_train):
        #x_train 训练数据集
        #return 训练后的模型

        assert  x_train.ndim == 1; #暂时只能处理一维的数据 必须要1维才能进行下去

        assert len(x_train) == len(y_train);   # x_train 的大小必须和y_train大小相同
        x_mean = np.mean(x_train);
        y_mean = np.mean(y_train);
        denominator = 0.0;
        numerator = 0.0;
        #通过循环计算分子和分母
        #打包成元祖后的列表
        for x,y in zip(x_train,y_train):
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean) ** 2;
            #通过向量化运算提高性能
        self.a_ = numerator / denominator;
        self.b_ = y_mean - self.a_ * x_mean;
        return self;
    def predict(self,x_predict):
        #返回表示x_predict的结果向量
        assert  x_predict.ndim == 1;  '数据集必须得是一纬的'
        assert self.a_ is not None and self.b_ is not  None; 'must fit before predict!'
        return np.array([self._predict(x) for x in x_predict]);

    def _predict(self,x_single):
        #计算单个点的预测结果并且返回
        return self.a_ * x_single + self.b_;
    def score(self,x_test,y_test):
        #评估模型准确度 采用R2标准
        y_predict = self.predict(x_test);
        return r2_score(y_test,y_predict);
    #用来表示对象的可打印字符串
    def __repr__(self):
        """LinearRegression()"""
class MultipleLinearRegression:
    def __init__(self):
        self.coef_ = None;
        self.bias_ = None;
        self._theta = None;
    def fit_normal(self,X_train,y_train):
        print(2233);
        print(X_train.shape[0]);
        print(y_train.shape[0]);
        assert X_train.shape[0] == y_train.shape[0],'两者的大小必须相同'
        #将两者合并
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        #inv为求逆矩阵 T表示转置矩阵
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train);
        self.bias_ = self._theta[0];
        self.coef_ = self._theta[1:];
        return self;
    def predict(self,X_predict):
        assert self.bias_ is not None and self.coef_ is not None;'must fit before predict ！'
        assert  X_predict.shape[1] == len(self.coef_)
        X_b = np.hstack([np.ones((len(X_predict),1)),X_predict]);
        return X_b.dot(self._theta);
    def score(self,X_test,y_test):
        y_predict = self.predict(X_test);
        return r2_score(y_test,y_predict);
    def __repr__(self):
        """MultipleLinearRegression()"""