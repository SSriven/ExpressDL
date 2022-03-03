"""
激活层：sigmoid
"""
import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from layers.Layer import Layer

class Sigmoid(Layer):

    def __init__(self,name="Sigmoid"):
        super().__init__(name,op="Sigmoid")
        self.out = None

    def forward(self,x):
        out = None
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out

        return out

    def backward(self,dout):
        dx = dout * self.out * (1.0 - self.out)

        return dx

"""
Test
"""

if __name__ == "__main__":
    sigmoid = Sigmoid()
    x = np.random.randint(low=-10, high=10,size=(5,))
    print("x:",x)
    out = sigmoid.forward(x)
    print("forward:",out)
    dout = np.random.randint(low=-10, high=10,size=(5,))
    print("dout:",dout)
    dx = sigmoid.backward(dout)
    print("backward:",dx)