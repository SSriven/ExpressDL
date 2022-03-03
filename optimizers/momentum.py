
import numpy as np

class Momentum:

    def __init__(self,lr = 0.01,momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.vw = None
        self.vb = None

    def update(self,model):
        if self.vw is None or self.vb is None:
            self.vw = {}
            self.vb = {}
            i = 0
            for layer in model.layers:
                if hasattr(layer,"W"):
                    self.vw[i] = np.zeros_like(layer.dw)
                if hasattr(layer,"B"):
                    self.vb[i] = np.zeros_like(layer.db)
                i += 1
        j = 0        
        for layer in model.layers:
            if hasattr(layer,"W"):
                self.vw[j] = self.momentum * self.vw[j] - self.lr * layer.dw
                layer.W += self.vw[j] 
            if hasattr(layer,"B"):
                self.vb[j] = self.momentum * self.vb[j] - self.lr * layer.db
                layer.B += self.vb[j] 
            j +=1
        