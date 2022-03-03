

import numpy as np
from PIL import Image
from net.model import Model
from layers.utils import imgreshape
from layers.convolution import Conv2D
from layers.dense import Dense
from layers.maxpool import MaxPooling
from layers.relu import Relu
from layers.softmax_ce import SoftmaxWithLoss


class LeNet5():

    def __init__(self, x_train, t_train, x_test, t_test, epochs=30, weight_decay_lambda=0, sample_batches=True, 
                batch_size=100, optimizer="SGD", optimizer_param={"lr": 0.01}, learning_rate_decay=1, verbose=True):

        self.x_train = None
        self.t_train = t_train
        self.x_test = None
        self.t_test = t_test

        # 将图像同意转换成32*32
        self.x_train = imgreshape(x_train,size=(32,32))
        self.x_test = imgreshape(x_test,size=(32,32))


        self.model = Model(x_train=self.x_train, t_train=self.t_train, epochs=epochs,
                           weight_decay_lambda=weight_decay_lambda, sample_batches=sample_batches,
                            batch_size=batch_size, optimizer=optimizer, optimizer_param=optimizer_param, 
                            learning_rate_decay=learning_rate_decay, verbose=verbose)
        
        self.__init_layers()
        
    def __init_layers(self):

        self.model.add(Conv2D(kernels_num=6, kernel_h=5, kernel_w=5, stride=1,padding=0, name="Conv_1", weights_init_type="he")) #(32+0-5)/1+1=28
        self.model.add(Relu(name='Relu_1'))
        self.model.add(MaxPooling(name="MaxPooling_1", stride=2,pool_h=2,pool_w=2)) # (28-2)/2+1=14

        self.model.add(Conv2D(kernels_num=16, kernel_h=5, kernel_w=5, stride=1,padding=0, name="Conv_2", weights_init_type="he")) #(14+0-5)/1+1=10
        self.model.add(Relu(name='Relu_2'))
        self.model.add(MaxPooling(name="MaxPooling_1", stride=2,pool_h=2,pool_w=2)) # (10-2)/2+1=5

        self.model.add(Dense(hidden_size=120,name="Dense_1",weights_init_type="he")) #(,120)
        self.model.add(Relu(name='Relu_3'))
        self.model.add(Dense(hidden_size=84,name="Dense_2",weights_init_type="he")) #(,84)
        self.model.add(Relu(name='Relu_4'))
        self.model.add(Dense(hidden_size=10,name="Dense_3",weights_init_type="he")) #(,84)

        loss_layer = SoftmaxWithLoss()
        self.model.add(loss_layer, loss=True)

        self.model.init_weights()

    def train(self):
        self.model.train()
        self.model.test(self.x_test,self.t_test)

    def getModel(self):
        return self.model

    def predict(self,x):
         y = self.model.predict(x)
         y = np.argmax(y, axis=1)
         return y
    def desc(self):
        self.model.desc()