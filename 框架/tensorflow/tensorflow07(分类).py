# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True);
#激励函数：将线性方程变为非线性方程
# 添加定义神经层 输入值 输入大小 输出大小 激励函数
def add_layer(inputs,in_size,out_size,activation_function = None):
    #定义随机变量
    Weights = tf.Variable(tf.random_normal([in_size,out_size]));
    #bias 不推荐0
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1);
    #定义未激活的值
    Wx_plus_b = tf.matmul(inputs,Weights) + biases;
    #如果激励函数为none的话 默认就输出wx_plus_b
    if activation_function is None:
        outputs = Wx_plus_b;
    else:
        outputs = activation_function(Wx_plus_b);
    return outputs;
#计算训练集和验证集的误差
def compute_accuracy(v_xs, v_ys):
    #全局变量
    global prediction
#预测值
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
#与真实值的差别
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
#导入数据 通过x_data和y_data  并不是严格的一元二次函数 还需加上Noise
#linespace 通过指定开始课结束 和步长300 指定类型
#搭建神经网络 图像是32*32 是784
xs = tf.placeholder(tf.float32,[None,784]);
#每张图片表示一个数字 代表哪一个分类 是0~9
ys = tf.placeholder(tf.float32,[None,10]);
#softmax主要用于分类 输入784个特征 输出10个特征分类
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax);
#交叉熵为预测函数 表示相似程度
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]));
#train 采用梯度下降方法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy);
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer());
    #开始训练 每次取100张图片
    for i in range(1000):
     batch_xs,batch_ys = mnist.train.next_batch(100);
     sess.run(train_step,feed_dict={
        xs:batch_xs,
        ys:batch_ys
    });
     if i % 50 == 0:
         print(compute_accuracy(
             mnist.test.images, mnist.test.labels));

