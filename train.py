# import sys, os
# sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
# from net.lenet5 import LeNet5
# from dataset.mnist import load_mnist
# from layers.utils import save_model,load_model,imgreshape
# """

# """

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
# x_train = x_train[:10000]
# t_train = t_train[:10000]
# x_test = x_test[:1000]
# t_test = t_test[:1000]

# # lenet5 = LeNet5(x_train,t_train,x_test,t_test,epochs=10,optimizer='adam',weight_decay_lambda=0.01,learning_rate_decay=0.95)
# # lenet5.train()

# # save_model(lenet5,"lenet5.pkl")
# lenet5 = load_model("lenet5.pkl")
# test = imgreshape(x_test,(32,32))
# y = lenet5.predict(test)
# print(y[:100] == t_test[:100])



from net.model import Model
from dataset.mnist import load_mnist
from layers.convolution import Conv2D
from layers.dense import Dense
from layers.maxpool import MaxPooling
from layers.relu import Relu
from layers.softmax_ce import SoftmaxWithLoss
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False) # 加载数据
x_train = x_train[:1000]
t_train = t_train[:1000]
x_test = x_test[:100]
t_test = t_test[:100]
"""
构建一个网络，网络结构为：
Conv2D-Relu-MaxPooling-Dense-SoftmaxWithLoss
"""
model = Model(x_train = x_train,t_train = t_train,epochs=5,batch_size=100)
model.add(Conv2D(kernels_num=16, kernel_h=3, kernel_w=3, stride=2,padding=2))
model.add(Relu())
model.add(MaxPooling(stride=1,pool_h=3,pool_w=3))

model.add(Dense(hidden_size=1024))

model.add(SoftmaxWithLoss(),loss=True)
model.init_weights() # 初始化权重
model.train() #开始训练
model.test(x_test,t_test) # 计算测试集精度