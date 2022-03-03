
"""
随机梯度下降
"""
import numpy as np
class SGD:

    def __init__(self,lr = 0.01):
        self.lr = lr

    def update(self,model):
        
        for layer in model.layers:
            if hasattr(layer,"W"):
                layer.W -= self.lr * layer.dw
            if hasattr(layer,"B"):
                layer.B -= self.lr * layer.db