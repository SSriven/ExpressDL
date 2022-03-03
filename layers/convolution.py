import sys, os
sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
import numpy as np
from layers.utils import im2col,col2im
from layers.Layer import Layer

class Conv2D(Layer):
    """
    Parameters
    ----------
    in_dim:输入图像的形状,例如（3，28，28）表示输入通道数为3的28*28尺寸的图片
    kernels_num：卷积核数量，也即输出特征图的通道数
    kernel_h、kernel_w：卷积核的大小，默认为5*5
    stride：每次卷积滑动的步长，默认为1
    padding：空白填充，默认为0
    weights_init_type：权重参数初始方式，默认标准差是0.01
    Returns
    -------
    """
    def __init__(self,kernels_num ,kernel_h = 5,kernel_w = 5,stride = 1,padding = 0,weights_init_type="normal",name="Conv"):
        super().__init__(name + "(" + str(kernels_num) + "@" + str(kernel_h) + "*" + str(kernel_w) + ")",op="Conv2D")
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.padding = padding
        self.kernels_num = kernels_num
        self.in_out_shape = None
        self.weights_init_type = weights_init_type

        # 中间数据（backward时使用）s
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置的梯度
        self.dw = None
        self.db = None


    def init_weights(self,input_shape):

        N,C,H,W = input_shape
        input_size = C * H * W
        in_channnels = C

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
        
        self.W = weights_init_std * np.random.randn(self.kernels_num,in_channnels,self.kernel_h,self.kernel_w)
        self.B = np.zeros(self.kernels_num)

        FN,C,FH,FW = self.W.shape
        out_h = 1 + int((H + 2*self.padding - FH) / self.stride) # 输出特征图的高
        out_w = 1 + int((W + 2*self.padding - FW) / self.stride) # 输出特征图的宽
        out_shape = (N,self.kernels_num,out_h,out_w)
        self.in_out_shape = (input_shape,out_shape)
        return out_shape


    def forward(self,x):
        out = None
        N,C,H,W = x.shape
        
        FN,C,FH,FW = self.W.shape

        out_h = 1 + int((H + 2*self.padding - FH) / self.stride) # 输出特征图的高
        out_w = 1 + int((W + 2*self.padding - FW) / self.stride) # 输出特征图的宽

        col = im2col(x, FH, FW, self.stride, self.padding) # 将四维数据转换成二维数据
        col_W = self.W.reshape(FN, -1).T # 将四维卷积核转换成二维数据

        out = np.dot(col,col_W) + self.B
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # 恢复成四维数据

        self.x = x
        self.col = col
        self.col_W = col_W

        return out


    def backward(self,dout):
        dx = None
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dw = np.dot(self.col.T, dout)
        self.dw = self.dw.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.padding)

        return dx

"""
Test
"""

if __name__ == "__main__":
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
        