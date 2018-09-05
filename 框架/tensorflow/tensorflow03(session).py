# -- coding: utf-8 --
#定义变量 variable
import tensorflow as tf
#赋值
state = tf.Variable(0,name = "counter");
#打印输出
print (state.name);
#定义常量
one = tf.constant(1);
print (one);

#进行累加
new_value = tf.add(state,one);
#进行赋值 把累加好的new_value 都赋值进state去
update = tf.assign(state,new_value);

#如果定义了variable 一定要进行初始化
init = tf.initialize_all_variables();

#利用with机制打开session
with tf.Session() as sess:
    #首先要跑初始化
    sess.run(init);
    for _ in range(3):
        #累加3次
      res = sess.run(update);
      print sess.run(state);
sess = tf.InteractiveSession();
x = tf.Variable([1.0,2.0]);
a = tf.constant([3.0,3.0]);
x.initializer.run();
sub = tf.subtract(x,a);
print(sub.eval());
sess.close();
#fetch取回
input1 = tf.constant(3.0);
input2 = tf.constant(2.0);
input3 = tf.constant(5.0);
intermed = tf.add(input2,input3);
mul = tf.multiply(input1,intermed);
with tf.Session() as sess:
    #跑两个例子 第一个是乘 第二个是加
    print(sess.run([mul,intermed]));