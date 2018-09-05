# -- coding: utf-8 --
import tensorflow as tf
#placeHolder的学习 从外部传入data 需要用到placeHolder
#type一般为float32的形式
input1 = tf.placeholder(tf.float32);
input2 = tf.placeholder(tf.float32);
#乘法将input1 和 input2 进行乘法 输出为output

output = tf.multiply(input1,input2);
# 将实际值用feed_dic输入
with tf.Session() as sess:
    print(sess.run(output,feed_dict={
        input1:[7.],
        input2:[2,]
    }));

