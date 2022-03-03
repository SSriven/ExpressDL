
from datetime import datetime
from optimizers.adam import Adam
from optimizers.adagrad import AdaGrad
from optimizers.momentum import Momentum
from optimizers.sgd import SGD
import numpy as np
import sys
import os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


class Model:

    """
    Parameters
    ----------
    x_train；训练数据;
    t_train: 训练数据标签;
    epochs: 训练时期数;
    weight_decay_lambda: 权值衰减系数;
    sample_batches:是否按照mini-batcha训练，默认为True,如果为False，则将所有数据一次性计算;
    batch_size:mini-batch大小;
    optimizer:梯度下降优化器,默认为sgd，可选的有：momentum,adagrad,adam;
    optimizer_param:优化器参数，例如学习率等等;
    learning_rate_decay:学习率衰减系数,默认为1;
    verbose:是否打印训练进度,默认为True;
    Returns
    -------
    Model对象
    """

    def __init__(self, x_train, t_train, epochs=30, weight_decay_lambda=0, sample_batches=True,
                 batch_size=100, optimizer='SGD', optimizer_param={'lr': 0.01}, learning_rate_decay=1, verbose=True):
        self.layers = []  # 存储各个层，例如卷积层，池化层等等
        self.criterion = None  # 存储最后一层
        self.weight_decay_lambda = weight_decay_lambda  # 权值衰减系数
        self.verbose = verbose  # 是否打印训练进度,默认为True
        self.x_train = x_train  # 训练数据
        self.t_train = t_train  # 训练数据标签

        # 是否按照mini-batcha训练，默认为True,如果为False，则将所有数据一次性计算
        self.sample_batches = sample_batches
        self.epochs = epochs
        self.batch_size = batch_size  # mini-batch大小
        self.learning_rate_decay = learning_rate_decay  # 学习率衰减系数

        # optimzer
        optimizer_class_dict = {
            'sgd': SGD, 'momentum': Momentum, 'adagrad': AdaGrad, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](
            **optimizer_param)

        self.current_iter = 1  # 记录当前迭代次数
        self.current_epoch = 1  # 记录当前epoch
        self.bar_i = 1  # 用于打印训练日志

        self.loss_history = []  # 存储损失值
        self.acc_history = [] # 存储精度

    """
    Parameters
    ----------
    
    layer:Layer对象

    Returns
    -------
    """

    def add(self, layer, loss=False):
        if loss:
            self.criterion = layer
        else:
            self.layers.append(layer)

    # 初始化各层的权重参数
    def init_weights(self):
        input_shape = self.x_train.shape
        for layer in self.layers:
            input_shape = layer.init_weights(input_shape)

    # 输出各层的详细信息,注意：该方法需要在调用init_weight方法之后使用
    def desc(self):
        print("=================================================================================")
        print("layer                         desc                                               ")
        print("=================================================================================")
        for layer in self.layers:
            print(layer.name.ljust(30) + "input：" +
                  str(layer.in_out_shape[0]) + "\t output：" + str(layer.in_out_shape[1]))
            print(
                "--------------------------------------------------------------------------------------")

    # 预测
    def predict(self, x,train_flg = False):
        out = x
        for layer in self.layers:
            if layer.op == "Dropout" or layer.op == "BatchNorm":
                out = layer.forward(out,train_flg)
            else:
                out = layer.forward(out)
        return out

    # 计算准确度，x是输入数据，t是正确解标签
    def accuracy(self, x, t):
        y = self.predict(x,train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = np.sum(y == t) / y.size
        return acc

    def test(self,x_test,t_test):
        test_acc = self.accuracy(x_test, t_test)
        print("=============== Final Test Accuracy ===============")
        print("test acc:" + str(test_acc))

    # 前向传播
    def forward(self, x, t):
        y = self.predict(x,train_flg=True)

        weight_decay = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                W = layer.W
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        loss = self.criterion.forward(y, t) + weight_decay
        return y, loss

    # 反向传播
    def backward(self):
        dout = self.criterion.backward()
        layers = list(self.layers)
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            if hasattr(layer, "W"):
                layer.dw += self.weight_decay_lambda * layer.W

    # 打印训练进度
    def train_progress(self):
        a = "▋" * int((self.bar_i / self.iter_per_epochs) * 50)
        # b = "-" * int(((self.iter_per_epochs - self.bar_i) / self.iter_per_epochs) * 50)
        c = (self.bar_i / self.iter_per_epochs) * 100
        print("\r epoch: {} / {} {:^3.0f}% {} ".format(
            self.current_epoch, self.epochs, c,a), end="")
        self.bar_i += 1

    # 训练步骤
    def train_step(self):

        # 打印训练进度
        if self.verbose:
            self.train_progress()

        if self.sample_batches:
            x_batch = self.x_batch[self.current_iter % self.iter_per_epochs -1]
            t_batch = self.t_batch[self.current_iter % self.iter_per_epochs -1]
        else:
            x_batch = self.x_train
            t_batch = self.t_train

        # 前向传播
        y, loss = self.forward(x_batch, t_batch)
        # 反向传播
        self.backward()
        # 优化参数
        self.optimizer.update(self)
        self.loss_history.append(loss)

        if self.current_iter % self.iter_per_epochs == 0:
            if self.verbose:
                train_acc = self.accuracy(self.x_train, self.t_train)
                self.acc_history.append(train_acc)
                print(' loss: %f, train acc: %f, lr: %e '
                      % (loss, train_acc, self.optimizer.lr))
            else:
                print()
            self.bar_i = 1
            self.current_epoch += 1

            if self.current_iter > 0:
                self.optimizer.lr *= self.learning_rate_decay

        self.current_iter += 1

    # 将训练数据分成多个批次，例如将1000笔训练数据分成10批，每一批100笔数据
    # 首先将1000笔训练数据随机打乱
    # 然后将打乱后的数据按照0-99，100-199，....，900-899的顺序分成10批
    def split_dataset(self):
        batch_mask = np.random.choice(
            self.x_train.shape[0], self.x_train.shape[0], replace=False)  # 随机打乱顺序
        x_random_train = self.x_train[batch_mask]
        t_random_train = self.t_train[batch_mask]
        # 拆分数据
        self.x_batch = np.array(np.array_split(
            x_random_train, self.iter_per_epochs))
        self.t_batch = np.array(np.array_split(
            t_random_train, self.iter_per_epochs))

    def train(self):
        if self.sample_batches:
            self.iter_per_epochs = max(
                self.x_train.shape[0] // self.batch_size, 1)  # 每个epoch迭代的次数
            # 分割数据集
            self.split_dataset()
        else:
            self.iter_per_epochs = 1

        max_iter = int(self.epochs * self.iter_per_epochs)   # 总的迭代次数

        start_time = datetime.now()  # 记录训练结束时间
        # 开始训练
        print("开始训练...")
        for i in range(max_iter):
            self.train_step()

        end_time = datetime.now()  # 记录训练结束时间
        train_time_seconds = (end_time - start_time).seconds
        m, s = divmod(train_time_seconds, 60)
        h, m = divmod(m, 60)
        print("训练结束\ntime：%d:%02d:%02d" % (h, m, s))
