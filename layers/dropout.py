"""
Dropout
"""

import numpy as np
from layers.Layer import Layer

class Dropout(Layer):

    def __init__(self, drop_ratio = 0.5,name="Doupout"):
        super().__init__(name+"("+str(drop_ratio)+")",op="Dropout")
        self.drop_ratio = drop_ratio
        self.mask = None

    def forward(self,x,train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.drop_ratio)

    def backward(self,dout):
        return dout * self.mask