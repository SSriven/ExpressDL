"""
激活函数relu层
"""
import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from layers.Layer import Layer
class Relu(Layer):

    def __init__(self,name="Relu"):
        super().__init__(name,op="Relu")
        self.mask = None

    def forward(self,x):

        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self,dout):

        dout[self.mask] = 0
        dx = dout

        return dx

"""
Test
"""

if __name__ == "__main__":
    relu = Relu()
    x = np.random.randint(low=-10, high=10,size=(10,))
    print("x:",x)
    out = relu.forward(x)
    print("forward:",out)
    dout = np.random.randint(low=-10, high=10,size=(10,))
    print("dout:",dout)
    dout = relu.backward(out)
    print("backward:",dout)