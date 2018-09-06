import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
import  numpy as np
# 加载鸢尾花数据
iris = datasets.load_iris()

# 查看特征名称
print("feature_names: {0}".format(iris.feature_names))
# 查看目标标签名称
print("target_names: {0}".format(iris.target_names))

# 查看元数据（特征矩阵）形状
print("data shape: {0}".format(iris.data.shape))
# 查看元数据（特征矩阵）前五条
print("data top 5:\n {0}".format(iris.data[: 5]))
# 查看目标标签的类别标识
print("target unique: {0}".format(np.unique(iris.target)))
print("target top 5:\n {0}".format(iris.target[: 5]))
sepal_length_list = iris.data[:, 0] # 花萼长度
sepal_width_list = iris.data[:, 1] # 花萼宽度

# 构建 setosa、versicolor、virginica 索引数组
setosa_index_list = iris.target == 0 # setosa 索引数组
versicolor_index_list = iris.target == 1 # versicolor 索引数组
virginica_index_list = iris.target == 2 # virginica 索引数组

plt.scatter(sepal_length_list[setosa_index_list], 
            sepal_width_list[setosa_index_list], color="red", marker='o', label="setosa")
plt.scatter(sepal_length_list[versicolor_index_list], 
            sepal_width_list[versicolor_index_list], color="blue", marker="x", label="versicolor")
plt.scatter(sepal_length_list[virginica_index_list], 
            sepal_width_list[virginica_index_list],color="green", marker="+", label="virginica")
# 设置 legend
plt.legend(loc="best", title="iris type")
# 设定横坐标名称
plt.xlabel("sepal_length (cm)")
# 设定纵坐标名称
x = iris.data
y = iris.target

clf = linear_model.LogisticRegression()
clf.fit(x,y)
#待遇测的样本
wait_predict_sample = x[np.newaxis,0]
print("wait_predict_sample:{0}".format(wait_predict_sample))
#预测
print("predict:{0}".format(clf.form))