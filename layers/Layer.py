"""
    各个层的父类
    Parameters
    ----------
    
    name:layer名称

    Returns
    -------
"""

class Layer:

    def __init__(self,name,op):
        self.name = name
        self.op = op

    def init_weights(self,input_shape):
        self.in_out_shape = (input_shape,input_shape)
        return input_shape