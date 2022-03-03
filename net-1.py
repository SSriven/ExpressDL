import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from layers.convolution import Conv2D
from layers.dense import Dense
from layers.maxpool import MaxPooling
from layers.relu import Relu
from layers.softmax_ce import SoftmaxWithLoss
from layers.dropout import Dropout
from dataset.mnist import load_mnist
from net.model import Model
from layers.batchnormalization import BatchNormalization
import matplotlib.pyplot as plt
"""
构建一个网络,网络结构为：
    conv - BatchNorm -relu - pool-
    conv - BatchNorm -relu - pool-
    dense - relu -dropout - dense - SoftmaxWithLoss

"""

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_train = x_train[:5000]
t_train = t_train[:5000]
x_test = x_test[:1000]
t_test = t_test[:1000]

"""
模型一
"""
model1 = Model(x_train=x_train,t_train=t_train, weight_decay_lambda=0.01,
              batch_size=100, optimizer="adam", learning_rate_decay=0.95,epochs=30,verbose=True)

model1.add(Conv2D(kernels_num=16, kernel_h=3, kernel_w=3, stride=2,padding=2, name="Conv_1", weights_init_type="he")) # (28+4-3)/2+1=15
model1.add(BatchNormalization(gamma=1,beta=0))
model1.add(Relu(name='Relu_1'))
model1.add(MaxPooling(name="MaxPooling_1", stride=1,pool_h=3,pool_w=3)) # (15-3)/1+1=13

model1.add(Conv2D(kernels_num=32, kernel_h=3, kernel_w=3, stride=2,padding=1, name="Conv_2", weights_init_type="he")) # (13+2-3)/2+1=7
model1.add(BatchNormalization(gamma=1,beta=0))
model1.add(Relu(name='Relu_2'))
model1.add(MaxPooling(name="MaxPooling_2", stride=1,pool_h=3,pool_w=3)) # (7-3)/1+1=5


model1.add(Dense(hidden_size=1024,name="Dense_1",weights_init_type="he")) #(100,1024)
model1.add(Relu(name='Relu_3'))
model1.add(Dropout(drop_ratio=0.5,name="Dropout_1"))

model1.add(Dense(hidden_size=10, name="Dense_2",weights_init_type="he"))#(100,10)
# model1.add(Relu(name='Relu_4'))
# model1.add(Dropout(drop_ratio=0.5,name="Dropout_2"))

loss_layer = SoftmaxWithLoss()
model1.add(loss_layer, loss=True)

model1.init_weights()
# model1.desc()
model1.train()
model1.test(x_test,t_test)



"""
模型2
"""
model2 = Model(x_train=x_train,t_train=t_train, weight_decay_lambda=0.01,
              batch_size=100, optimizer="adam", learning_rate_decay=0.95,epochs=30,verbose=True)

model2.add(Conv2D(kernels_num=16, kernel_h=3, kernel_w=3, stride=2,padding=2, name="Conv_1", weights_init_type="he")) # (28+4-3)/2+1=15
# model2.add(BatchNormalization(gamma=1,beta=0))
model2.add(Relu(name='Relu_1'))
model2.add(MaxPooling(name="MaxPooling_1", stride=1,pool_h=3,pool_w=3)) # (15-3)/1+1=13

model2.add(Conv2D(kernels_num=32, kernel_h=3, kernel_w=3, stride=2,padding=1, name="Conv_2", weights_init_type="he")) # (13+2-3)/2+1=7
# model2.add(BatchNormalization(gamma=1,beta=0))
model2.add(Relu(name='Relu_2'))
model2.add(MaxPooling(name="MaxPooling_2", stride=1,pool_h=3,pool_w=3)) # (7-3)/1+1=5


model2.add(Dense(hidden_size=1024,name="Dense_1",weights_init_type="he")) #(100,1024)
model2.add(Relu(name='Relu_3'))
model2.add(Dropout(drop_ratio=0.5,name="Dropout_1"))

model2.add(Dense(hidden_size=10, name="Dense_2",weights_init_type="he"))#(100,10)
# model1.add(Relu(name='Relu_4'))
# model1.add(Dropout(drop_ratio=0.5,name="Dropout_2"))

loss_layer = SoftmaxWithLoss()
model2.add(loss_layer, loss=True)

model2.init_weights()
# model2.desc()
model2.train()
model2.test(x_test,t_test)

# 绘制图形
loss_history_1 = np.array(model1.loss_history)
acc_history_1 = np.array(model1.acc_history)

loss_history_2 = np.array(model2.loss_history)
acc_history_2 = np.array(model2.acc_history)

iters = np.arange(int(model1.epochs * model1.iter_per_epochs)-1)
plt.subplot(121)
plt.plot(iters,loss_history_1[1:],label="dropout")
plt.plot(iters,loss_history_2[1:],color="red",linestyle="--",label="without dropout")
plt.xlabel("iters")
plt.ylabel("loss")

epochs = np.arange(model1.epochs)
plt.subplot(122)
plt.plot(epochs,acc_history_1,label="dropout")
plt.plot(epochs,acc_history_2,color="red",linestyle="--",label="without dropout")
plt.xlabel("epoch")
plt.ylabel("acc")

plt.show()

