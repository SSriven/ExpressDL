# ExpressDL简介

ExpressDL是一款极度轻量级的深度学习框架。使用ExpressDL，您可以简单、自由、快速地搭建您的网络。

# 快速入门

```python
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
```

```
输出：
开始训练...
 epoch: 1 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 6.916107, train acc: 0.096000, lr: 1.000000e-02 
 epoch: 2 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 6.888526, train acc: 0.185000, lr: 1.000000e-02 
 epoch: 3 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 6.821256, train acc: 0.225000, lr: 1.000000e-02 
 epoch: 4 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 6.605810, train acc: 0.313000, lr: 1.000000e-02 
 epoch: 5 / 5 100% ▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋  loss: 5.852512, train acc: 0.243000, lr: 1.000000e-02 
训练结束
time：0:00:14
=============== Final Test Accuracy ===============
test acc:0.26
```

# Modules

## layers

所有layer都继承Layer类

***前向传播：Layer.forward(x)***

| 参数 | 说明     | 类型 |      |
| ---- | -------- | ---- | ---- |
| x    | 输入数据 | 张量 | 必选 |

***反向传播：Layer.backward(dout)***

| 参数 | 说明                     | 类型 |      |
| ---- | ------------------------ | ---- | ---- |
| dout | 下一层反向传播回来的数据 | 张量 | 必选 |

***初始化参数：Layer.init_weights(input_shape)***

| 参数        | 说明           | 类型 |      |
| ----------- | -------------- | ---- | ---- |
| input_shape | 输入数据地形状 | 元组 | 必选 |

### 全连接

**Dense(hidden_size,weights_init_type="normal",name="Dense")**

示例：

```python
from layers.dense import Dense
import numpy aas np
x = np.random.randn(100,256,6,6)
dense = Dense(hidden_size=4096) #创建一个Dense对象
dense.init_weights(x.shape) #初始化参数
out = dense.forward(x) #前向传播
print("forward:",out.shape)
dout = np.random.randn(*out.shape)
dout = dense.backward(dout) #反向传播
print("backward:",dout.shape)
```

```
输出：
forward: (100, 4096)
backward: (100, 256, 6, 6)
```

| 参数              | 说明                                                         | 类型         |      |
| ----------------- | ------------------------------------------------------------ | ------------ | ---- |
| hidden_size       | 隐藏单元数                                                   | int          | 必选 |
| weights_init_type | 权重初始化方式，默认为“normal”，可以选择“he”或者“xavier”，当激活函数为sigmoid建议使用xavier，当激活函数为relu建议使用he | float/string | 可选 |
| name              | 为该层取一个昵称，默认为“Dense”                              | string       | 可选 |

### 卷积

**Conv2D(kernels_num ,kernel_h = 5,kernel_w = 5,stride = 1,padding = 0,weights_init_type="normal",name="Conv")**

示例：

```python
from layers.convolution import Conv2D
import numpy aas np
con = Conv2D(kernels_num=64,kernel_h=3,kernel_w=3,stride=1,padding=1)
imgs = np.random.randint(low=0,high=256,size=(1000,32,3,3))
con.init_weights(imgs.shape)
print("forward:\n")
print("input_img_shape:",imgs.shape)
out = con.forward(imgs)
print("out_shape:",out.shape)
print("backward:\n")
dout = np.random.randint(low=0,high=256,size=out.shape)
dout = con.backward(dout)
print("dout_shape:",dout.shape)
```

输出：

```
forward:

input_img_shape: (1000, 32, 3, 3)
out_shape: (1000, 64, 3, 3)
backward:

dout_shape: (1000, 32, 3, 3)
```

| 参数              | 说明                                                         | 类型         |      |
| ----------------- | ------------------------------------------------------------ | ------------ | ---- |
| kernels_num       | 卷积核数量                                                   | int          | 必选 |
| kernel_h          | 卷积核高，默认为5                                            | int          | 可选 |
| kernel_w          | 卷积核宽，默认为5                                            | int          | 可选 |
| stride            | 卷积核滑动步长，默认为1                                      | int          | 可选 |
| padding           | 填充，默认为0                                                | int          | 可选 |
| weights_init_type | 权重初始化方式，默认为“normal”，可以选择“he”或者“xavier”，当激活函数为sigmoid建议使用xavier，当激活函数为relu建议使用he | float/string | 可选 |
| name              | 为该层取一个昵称，默认为“Conv2D”                             | string       | 可选 |

### 池化

**MaxPooling(pool_h = 2,pool_w = 2,stride = 1,padding = 0,name="MaxPooling")**

示例：

```python
from layers.maxpool import MaxPooling
import numpy aas np
maxpool = MaxPooling(stride=2)
imgs = np.random.randint(low=0,high=256,size=(1000,64,3,3))
print("forward:\n")
print("input_img_shape:",imgs.shape)
out = maxpool.forward(imgs)
print("out_shape:",out.shape)
print("backward:\n")
dout = np.random.randint(low=0,high=256,size=out.shape)
dout = maxpool.backward(dout)
print("dout_shape:",dout.shape)  
```

输出：

```
forward:

input_img_shape: (1000, 64, 3, 3)
out_shape: (1000, 64, 1, 1)
backward:

dout_shape: (1000, 64, 3, 3)
```

| 参数    | 说明                                 | 类型 |      |
| ------- | ------------------------------------ | ---- | ---- |
| pool_h  | 池化窗口高度，默认为2                | int  | 可选 |
| pool_w  | 池化窗口宽度，默认为2                | int  | 可选 |
| stride  | 滑动步长，默认为1                    | int  | 可选 |
| padding | 填充，默认为0                        | int  | 可选 |
| name    | 为该层取一个昵称，默认为“MaxPooling” | int  | 可选 |



### 激活函数

**Relu:**

```python
from layers.relu import Relu
import numpy as np
relu = Relu()
x = np.random.randint(low=-10, high=10,size=(10,))
print("x:",x)
out = relu.forward(x)
print("forward:",out)
dout = np.random.randint(low=-10, high=10,size=(10,))
print("dout:",dout)
dout = relu.backward(out)
print("backward:",dout)
```

输出：

```
x: [ 7  5  0  0  6  5  9 -3 -2  2]
forward: [7 5 0 0 6 5 9 0 0 2]
dout: [  2  -9  -4 -10   5   3   9  -6  -1  -3]
backward: [7 5 0 0 6 5 9 0 0 2]
```

**Sigmoid：**

```python
from layers.sigmoid import Sigmoid
import numpy as np
sigmoid = Sigmoid()
x = np.random.randint(low=-10, high=10,size=(5,))
print("x:",x)
out = sigmoid.forward(x)
print("forward:",out)
dout = np.random.randint(low=-10, high=10,size=(5,))
print("dout:",dout)
dx = sigmoid.backward(dout)
print("backward:",dx)
```

输出：

```
x: [-6  2 -7  8 -9]
forward: [2.47262316e-03 8.80797078e-01 9.11051194e-04 9.99664650e-01
 1.23394576e-04]
dout: [  9   9   3  -8 -10]
backward: [ 0.02219858  0.94494227  0.00273066 -0.0026819  -0.00123379]
```



### Batch Normalization

**BatchNormalization(gamma, beta, momentum=0.9, running_mean=None, running_var=None,name="Batch Normalization")**

示例：

```python
from layers.batchnormalization import BatchNormalization
import numpy as np
batch = BatchNormalization(gamma=0.5,beta=0.6)
imgs = np.random.randint(low=0,high=256,size=(100,3,28,28))
print("forward:\n")
print("input_img_shape:",imgs.shape)
out = batch.forward(imgs)
print("out_shape:",out.shape)
print("backward:\n")
dout = np.random.randint(low=0,high=256,size=out.shape)
dout = batch.backward(dout)
print("dout_shape:",dout.shape)
```

输出：

```
forward:

input_img_shape: (100, 3, 28, 28)
out_shape: (100, 3, 28, 28)
backward:

dout_shape: (100, 3, 28, 28)
```



| 参数         | 说明                        | 类型   |      |
| ------------ | --------------------------- | ------ | ---- |
| gamma        |                             | float  | 必选 |
| beta         |                             | float  | 必选 |
| momentum     | 默认为0.9                   | float  | 可选 |
| running_mean | 测试情况下的均值            | float  | 可选 |
| running_var  | 测试情况下的方差            | float  | 可选 |
| name         | 默认为“Batch Normalization” | string | 可选 |



### Dropout

**Dropout(drop_ratio = 0.5,name="Doupout")**

示例：

```python
from layers.dropout import Dropout
dropout = Dropout(drop_ratio = 0.5,name="Doupout_1")
```



## net

### Model

**Model(x_train, t_train, epochs=30, weight_decay_lambda=0, sample_batches=True,**

​         **batch_size=100, optimizer='SGD', optimizer_param={'lr': 0.01}, learning_rate_decay=1, verbose=True)**

属性：

| 属性         | 说明                 |
| ------------ | -------------------- |
| loss_history | 记录每次迭代的损失值 |
| acc_history  | 记录每个epoch的精度  |

初始化参数：

| 参数                | 说明                                                         | 类型    |      |
| ------------------- | ------------------------------------------------------------ | ------- | ---- |
| x_train             | 训练数据                                                     | 张量    | 必选 |
| t_train             | 训练数据标签                                                 | 张量    | 必选 |
| epochs              | 训练时期数，默认为30                                         | int     | 可选 |
| weight_decay_lambda | 权值衰减系数，默认为0                                        | float   | 可选 |
| sample_batches      | 是否按照mini-batcha训练，默认为True,如果为False，则将所有数据一次性计算; | boolean | 可选 |
| batch_size          | mini-batch大小，默认为100                                    | int     | 可选 |
| optimizer           | 梯度下降优化器,默认为sgd，可选的有：momentum,adagrad,adam;   | string  | 可选 |
| optimizer_param     | 优化器参数，例如学习率等等;默认为{'lr': 0.01}                | dict    | 可选 |
| learning_rate_decay | 学习率衰减系数,默认为1;                                      | float   | 可选 |
| verbose             | 是否打印训练进度,默认为True;                                 | boolean | 可选 |

方法：

**Model.add(layer,loss=False)**

将layer添加进model中，layer属于Layer类，loss=False表示添加的layer不是损失函数那一层。例如:

```python
from net.model import Model
model = Model(x_train,t_train)
model.add(Dense(hidden_size=100))
model.add(Relu())
```

**Model.init_weights()**

初始化Model中各层的参数，注意：在使用前向传播时必须先调用此方法

**Model.desc()**

输出各层的详细信息,注意：该方法需要在调用init_weight方法之后使用

**Model.predict(x,train_flg=False)**

对输入的x进行预测，train_flg=False表示此次预测表示训练，如果在训练过程中，train_flg需要置为True，返回预测值

**Model.accuracy(x,t)**

计算准确度，x是输入数据，t是正确解标签,返回一个浮点数

**Model.test(x_test,t_test)**

计算测试集的精度,返回一个浮点数

**Model.train()**

开始训练

### LeNet-5

ExpressDL实现了LeNet-5，导入即可直接使用

**LeNet5(x_train, t_train, x_test, t_test, epochs=30, weight_decay_lambda=0, sample_batches=True,** 

​        **batch_size=100, optimizer="SGD", optimizer_param={"lr": 0.01}, learning_rate_decay=1, verbose=True)**

参数和Model的一致

示例：

```python
from net.lenet5 import LeNet5
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_train = x_train[:10000]
t_train = t_train[:10000]
x_test = x_test[:1000]
t_test = t_test[:1000]

lenet5 = LeNet5(x_train,t_train,x_test,t_test,epochs=10,optimizer='adam',weight_decay_lambda=0.01,learning_rate_decay=0.95)
lenet5.train()#开始训练
lenet5.desc()#打印网络详细信息
```

**LeNet5.train()**

开始训练

**LeNet5.getModel()**

返回一个Model对象

**LeNet5.predict(x)**

对x预测

**LeNet5.desc()**

打印网络详细信息

## optimizers

SGD:随机梯度下降

```python
from optimizers.sgd import SGD
op = SGD(lr=0.001)
op.update(model) #对model进行参数更新
```

Momentum:

```python
from optimizers.momentum import Momentum
op = Momentum(lr=0.001,momentum=0.9)
op.update(model) #对model进行参数更新
```

AdaGrad：

```python
from optimizers.adagrad import AdaGrad
op = AdaGrad(lr=0.001)
op.update(model) #对model进行参数更新
```

Adam:

```python
from optimizers.adam import Adam
op = Adam(lr=0.001, beta1=0.9, beta2=0.999)
op.update(model) #对model进行参数更新
```

# 模型保存与加载

## 模型保存

```python
from layers.utils import save_model
save_model(model,"model.pkl")
```



## 模型加载

```python
from layers.utils import load_model
model = load_model("model.pkl")
```

