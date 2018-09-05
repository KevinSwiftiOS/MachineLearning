# -- coding: utf-8 --
import tensorflow as tf
import numpy as np
#输出结果可视化
import matplotlib.pyplot as plt
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
#导入数据 通过x_data和y_data  并不是严格的一元二次函数 还需加上Noise
#linespace 通过指定开始课结束 和步长300 指定类型
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis];
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32);
y_data = np.square(x_data) - 0.5 + noise;
#用占位符表示输入 表示输入只有一个特征1 只会占用一个节点
xs = tf.placeholder(tf.float32, [None, 1]);
ys = tf.placeholder(tf.float32, [None, 1]);
#定义隐藏层  定义输入数据 输入大小 1 输出大小 10 激励函数 placeHolder只产生一个节点 而不占用多个
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu);
#定义输出层 输入为隐藏层 输出为1 无激励函数
prediction = add_layer(l1,10,1,activation_function=None);
#计算误差 需求和 压缩到1围
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]));
#提高学习效率 来减小误差 梯度线性下降
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss);
#使用变量都需要进行初始化
init = tf.global_variables_initializer();



#生成图片框
fig = plt.figure();
ax = fig.add_subplot(1,1,1);
#把真实数据显示出来
ax.scatter(x_data,y_data);
#连续画图
plt.ion();
plt.show();

with tf.Session() as sess:
    sess.run(init);
    #开始训练train_step
    for i in range(1000):
        sess.run(train_step,feed_dict = {
            xs:x_data,
            ys: y_data
        });
        if i % 50 == 0:
            print sess.run(loss,feed_dict={
                xs:x_data,
                ys:y_data
            });
            prediction_value = sess.run(prediction,feed_dict={
                xs:x_data
            });
            try:
                ax.lines.remove(lines[0]);
            except Exception:
                pass
            lines = ax.plot(x_data,prediction_value,'r-',lw=5);
            #去除掉第一条

            plt.pause(0.1);