
import numpy as np

class AdaGrad:

    def __init__(self,lr = 0.01):
        self.lr = lr
        self.hw = None
        self.hb = None

    def update(self,model):
        if self.hw is None or self.hb is None:
            self.hw = {}
            self.hb = {}
            i = 0
            for layer in model.layers:
                if hasattr(layer,"W"):
                    self.hw[i] = np.zeros_like(layer.dw)
                if hasattr(layer,"B"):
                    self.hb[i] = np.zeros_like(layer.db)
                i += 1
        j = 0        
        for layer in model.layers:
            if hasattr(layer,"W"):
                self.hw[j] += layer.dw * layer.dw
                layer.W -= self.lr * layer.dw / (np.sqrt(self.hw[j]) + 1e-7) 
            if hasattr(layer,"B"):
                self.hb[j] += layer.db * layer.db
                layer.B -= self.lr * layer.db / (np.sqrt(self.hb[j]) + 1e-7) 
            j +=1
        