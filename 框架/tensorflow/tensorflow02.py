# -- coding: utf-8 --
#session的学习
import  tensorflow as tf
import numpy as np
"""
实现两个矩阵的相乘
第一个是1行2列的矩阵
第二个是2行1列的矩阵
"""
matrix1 = tf.constant([[1,2],[3,4]]);
matrix2 = tf.constant([[1,2],[3,4]
                       ]);

#在nummpy中 也可以通过dot进行相乘 dot是进行点积乘
product = tf.matmul(matrix1,matrix2);
# 通过手动建立session会话机制
sess = tf.Session();
result = sess.run(product);
print(result);
#手动关闭 规范
sess.close();

#通过with机制session自动会关闭
with tf.Session() as sess:
    result2 = sess.run(product);
    print (result);
