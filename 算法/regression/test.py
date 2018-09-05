import matplotlib.pyplot as plt
from sklearn import datasets
from model_selection import train_test_split
from  regression.linear_regression import MultipleLinearRegression
from regression.logistic_regression import LogisticRegression
from metric import mean_squared_error
from metric import mean_absolute_error
from metric import root_mean_squared_error
from metric import r2_score
import pandas as pd;
if __name__ == '__main__':
    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=666)
    reg = MultipleLinearRegression()
    reg.fit_normal(X_train, y_train)
    print(reg.coef_, reg.bias_)
    print(reg.score(X_test, y_test))


    #燕尾挖数据做测试

    iris = pd.read_csv('../iris.data', header=None)
    iris_data = iris.loc[:, :].values
    x_data = iris_data[:100, :2]
    y_data = iris_data[:100, 4]
    y_data[y_data == 'Iris-setosa'] = 0
    y_data[y_data == 'Iris-versicolor'] = 1
    x_train, y_train, x_test, y_test = train_test_split(x_data, y_data)
    plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='red')
    plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='blue')
    plt.show()
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    print(log_reg.score(x_test, y_test))
    print(y_test)
    print(log_reg.predict(x_test))
    print(log_reg.predict_proba(x_test))
