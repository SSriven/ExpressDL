"""
全连接层
"""
import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from layers.Layer import Layer
class Dense(Layer):
    """
    Parameters
    ----------
    hidden_size:隐藏层结点数（输出尺寸）
    weights_init_type:权重初始方式，可传入的值有浮点数值或者整型，字符型包括xavier，he；
    当weight_init_type为normal时，weights_init_std=0.01
    当weight_init_type为xavier时，weights_init_std= np.sqrt(1.0 / input_size) ,当激活函数为sigmoid建议使用xavier
    当weight_init_type为he时，weights_init_std= np.sqrt(2.0 / input_size)，当激活函数为relu建议使用he
    Returns
    -------
    """
    def __init__(self,hidden_size,weights_init_type="normal",name="Dense"):
        super().__init__(name + "(hiddien_nums:" + str(hidden_size) + ")",op="FC")
        self.hidden_size = hidden_size
        self.in_out_shape = None
        self.weights_init_type = weights_init_type

        # 缓存数据，用于backward
        self.x = None
        self.original_x_shape = None

        # 权重和偏置的梯度(导数)
        self.dw = None
        self.db = None


    # 初始化权重参数
    def init_weights(self,input_shape):

        if len(input_shape) == 1:
            input_shape = (1,input_shape[0])

        input_size = np.prod(input_shape) // input_shape[0]
        
        if self.weights_init_type == 'normal':
            weights_init_std = 0.01
        elif self.weights_init_type == 'xavier':
            weights_init_std = np.sqrt(1.0 / input_size)
        elif self.weights_init_type == 'he':
            weights_init_std = np.sqrt(2.0 / input_size)
        elif isinstance(self.weights_init_type,int) or isinstance(self.weights_init_type,float):
            weights_init_std = weights_init_type
        else:
            print("参数weights_init_type只能选择整型、浮点型或者xavier和he!")

        
        self.W = weights_init_std * np.random.randn(input_size,self.hidden_size)
        self.B = np.zeros(self.hidden_size)
        out_shape = (input_shape[0],self.hidden_size)
        self.in_out_shape = (input_shape,out_shape)
        return out_shape

    def forward(self,x):

        out = None
        self.original_x_shape = x.shape
        
        if x.ndim > 1:
            x = x.reshape(x.shape[0],-1)
        
        if x.ndim == 1:
            x = x.reshape(-1,x.shape[0])

        self.x = x
        out = np.dot(self.x , self.W) + self.B
        
        return out

    def backward(self,dout):

        dx = np.dot(dout , self.W.T)
        self.dw = np.dot(self.x.T , dout)
        self.db = np.sum(dout , axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


"""
Test
"""

if __name__ == "__main__":
    x = np.random.randn(100,256,6,6)
    dense = Dense(hidden_size=4096)
    dense.init_weights(x.shape)
    out = dense.forward(x)
    # print(dense.in_out_shape)
    print("forward:",out.shape)
    dout = np.random.randn(*out.shape)
    # print(dense.x.shape,dout.shape)
    dout = dense.backward(dout)
    print("backward:",dout.shape)
    