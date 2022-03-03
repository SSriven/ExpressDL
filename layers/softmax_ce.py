"""
softmax交叉熵误差层
"""
import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from layers.loss_func import softmax
from layers.loss_func import cross_entropy_error
from layers.Layer import Layer

class SoftmaxWithLoss(Layer):

    def __init__(self,name="SoftmaxWithLoss"):
        super().__init__(name,op="SoftmaxWithLoss")
        self.loss = None # 损失
        self.y = None # softmax的输出
        self.t = None # 监督数据
        self.in_out_shape = None

    def forward(self,x,t):
        
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,t)

        self.in_out_shape = (x.shape,self.loss.shape)

        return self.loss

    def backward(self,dout = 1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

        
"""
Test
"""

if __name__ == "__main__":
    sml = SoftmaxWithLoss()
    x = np.around(np.random.randn(2,5),decimals=2)
    t = np.array([[0,0,1,0,0],[1,0,0,0,0]])
    print("x:",x.shape)
    print("t:",t.shape)
    out = sml.forward(x,t)
    print("forward:",out.shape,out)
    dx = sml.backward()
    print("backward:",dx.shape)