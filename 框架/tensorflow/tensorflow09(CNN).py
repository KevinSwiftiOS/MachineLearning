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
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_brob: 1});
#与真实值的差别
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
#定义weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1);
    return tf.Variable(initial);
#定义bias
def bias_variable(shape):
    inital = tf.constant(0.1,shape = shape);
    return tf.Variable(inital);
#定义卷积层
def conv2d(x,W):
    #con2d方法 stride为步长 让水平，垂直方向跨度为1 padding用same 不会确实边缘信息的值
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME");
#定义pooling
def max_poo_2x2(x):
    #ksize表示窗口大小 strides表示步长 进行压缩
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME');
#定义输入 和keep_brob
xs = tf.placeholder(tf.float32,[None,784]); #28*28;
ys = tf.placeholder(tf.float32,[None,10]);
keep_brob = tf.placeholder(tf.float32);
#随后改变形状 channel 由于是黑白 因此为-1
x_image = tf.reshape(xs,[-1,28,28,1]);
#建立卷积层 patch为5*5  输出为32的 输入为1个
W_conv1 = weight_variable([5,5, 1,32]);
b_conv1 = bias_variable([32]);
#定义第一个卷积层 同时进行激活 和前面的w * x + b相同
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1); #输出依然是28 * 28 * 32
#定义卷积 两个一步 因此输出为14*14*32
h_pool1 = max_poo_2x2(h_conv1);
#32个输入 随后输出是64
W_conv2 = weight_variable([5,5,32,64]);
b_conv2 = bias_variable([64]);
#定义第一个卷积层
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2);#输出为14*13*64
h_pool2 =  max_poo_2x2(h_conv2); #输出为7 * 7 * 64
#建立全连接层 用reshape 将3维变成一维
h_pool2_flat = tf.reshape(h_pool2,[-1,7 * 7 * 64]);
W_fc1 = weight_variable([7 * 7 * 64,1024]);
b_fc1 = bias_variable([1024]);
#进行全连接
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1);
#考虑过拟合
h_fc1_drop = tf.nn.dropout(h_fc1,keep_brob);
#进行分类 输入是1024 输出是10个类
W_fc2 = weight_variable([1024,10]);
b_fc2 = bias_variable([10]);
#进行预测
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2);
#使用交叉熵
cross_entropy=tf.reduce_mean(
    -tf.reduce_sum(ys*tf.log(prediction),
    reduction_indices=[1]));
#优化器 用adam
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy);
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100);
        sess.run(train_step,feed_dict={
            xs:batch_xs,
            ys:batch_ys,
            keep_brob:0.5

        });
        if(i % 50 == 0):
            print(compute_accuracy(
                mnist.test.images[:1000], mnist.test.labels[:1000]))