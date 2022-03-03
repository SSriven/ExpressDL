
import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from layers.utils import im2col,col2im
from layers.Layer import Layer

class MaxPooling(Layer):
    """
    Parameters
    ----------
    pool_h:池化窗口高
    pool_w:池化窗口宽
    stride:步长
    padding：填充
    name:名称
    Returns
    -------
    """
    def __init__(self,pool_h = 2,pool_w = 2,stride = 1,padding = 0,name="MaxPooling"):
        super().__init__(name + "(" + str(pool_h) + "*" + str(pool_w) + ")",op="MaxPooling")
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding
        self.in_out_shape = None

        self.x = None
        self.arg_max = None

    def init_weights(self,input_shape):
        N,C,H,W = input_shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        out_shape = (N,C,out_h,out_w)
        self.in_out_shape = (input_shape,out_shape)
        return out_shape

    def forward(self,x):
        out = None
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.padding)
        col = col.reshape(-1,self.pool_h * self.pool_w)

        arg_max = np.argmax(col,axis=1)
        out = np.max(col,axis=1)
        out = out.reshape(N,out_h,out_h,C).transpose(0,3,1,2)

        self.arg_max = arg_max
        self.x = x

        return out

    def backward(self,dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)
        
        return dx


"""
Test
"""

if __name__ == "__main__":
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