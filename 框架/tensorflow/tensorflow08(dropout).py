# -- coding: utf-8 --
#用dropout 解决神经网络中的过拟合问题
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
#激励函数：将线性方程变为非线性方程
# 添加定义神经层 输入值 输入大小 输出大小 激励函数
def add_layer(inputs,in_size,out_size,layer_name,activation_function = None):
    #定义随机变量
    Weights = tf.Variable(tf.random_normal([in_size,out_size]));
    #bias 不推荐0
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1);
    #定义未激活的值
    Wx_plus_b = tf.matmul(inputs,Weights) + biases;
    #dropout 舍弃多少
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob);


    #如果激励函数为none的话 默认就输出wx_plus_b
    if activation_function is None:
        outputs = Wx_plus_b;
    else:
        outputs = activation_function(Wx_plus_b);
        tf.summary.histogram(layer_name +'/outputs/',outputs);
    return outputs;


#加载数据
digits = load_digits();
X = digits.data;
y = digits.target;
y = LabelBinarizer().fit_transform(y);
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=3);
#添加层 表示舍弃多少 保留下多少 dropout通过舍弃机制等 来达到避免过拟合
keep_prob = tf.placeholder(tf.float32);
xs = tf.placeholder(tf.float32,[None,64]);
ys = tf.placeholder(tf.float32,[None,10]);
#add layer
l1 = add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh);
prediction = add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax);
#定义相似度 和训练
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.Session() as sess:
    merged = tf.summary.merge_all();
    train_writer = tf.summary.FileWriter("Desktop/logs/train",sess.graph);
    test_writer = tf.summary.FileWriter("Desktop/logs/test",sess.graph);
    sess.run(tf.global_variables_initializer());
    for i in range(500):
        sess.run(train_step,feed_dict={
            xs:X_train,
            ys:y_train,
            keep_prob:0.5
        });
        if i % 50 == 0:
            # record loss
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)