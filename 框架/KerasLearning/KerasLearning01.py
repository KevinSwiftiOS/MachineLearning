#!/usr/bin/python
# -*- coding: UTF-8 -*-
#keras学习
from keras.models import Sequential;
from keras.layers.core import Dense,Dropout,Activation;
from keras.optimizers import SGD;
from keras.datasets import mnist;
import numpy;
#选择模型 采用函数模型
model = Sequential();
#构建网络层
#输入层 500张28*28 即 500 784
model.add(Dense(500,input_shape=(784,)));
#激活函数是tanh
model.add(Activation('tanh'));
#采用50%的dropout
model.add(Dropout(0.5));
#输出结果是10个示例
model.add(Dense(10));
model.add(Activation('softmax'));
#编译
#优化函数 设定学习率
sgd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True);
#使用交叉熵作为loss函数
model.compile(loss='categorical_crossentropy', optimizer=sgd)
#进行训练
'''
   第四步：训练
   .fit的一些参数
   batch_size：对总的样本数进行分组，每组包含的样本数量
   epochs ：训练次数
   shuffle：是否把数据随机打乱之后再进行训练
   validation_split：拿出百分之多少用来做交叉验证
   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
'''
(x_train,y_train),(x_test,y_test) = mnist.load_data();
#将28*28维的变成784维
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] * x_train.shape[2]);
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] * x_test.shape[2]);
y_train = (numpy.arange(10) == y_train[:, None]).astype(int)
y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

model.fit(x_train,y_train,batch_size=200,epochs=50,shuffle=True,validation_split=0.3)
model.evaluate(x_test,y_test,batch_size=200,verbose=0)
#进行输出
print("test set");
scores = model.evaluate(x_test,y_test,batch_size=200,verbose=0)
print("");
print("The test loss is %f" % scores)
#进行预测
result = model.predict(x_test,batch_size=200,verbose=0);
result_max = numpy.argmax(result,axis=1);
test_max = numpy.argmax(y_test,axis=1);
result_bool = numpy.equal(result_max,test_max);
true_num = numpy.sum(result_bool);
print("");
print("The accuracy of the model is %f" %(true_num / len(result_bool)))