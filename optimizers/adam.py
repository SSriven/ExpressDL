import numpy as np
class Adam:
    
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.mw = None
        self.mb = None
        self.vw = None
        self.vb = None
        
    def update(self, model):
        if self.mw is None:
            self.mw,self.mb,self.vw,self.vb = {}, {},{},{}
            i = 0
            for layer in model.layers:
                if hasattr(layer,"W"):
                    self.mw[i] = np.zeros_like(layer.dw)
                    self.vw[i] = np.zeros_like(layer.dw)
                if hasattr(layer,"B"):
                    self.mb[i] = np.zeros_like(layer.db)
                    self.vb[i] = np.zeros_like(layer.db)
                i += 1
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        j = 0        
        for layer in model.layers:
            if hasattr(layer,"W"):
                self.mw[j] += (1 - self.beta1) * (layer.dw - self.mw[j])
                self.vw[j] += (1 - self.beta1) * (layer.dw**2 - self.vw[j])
                layer.W -= lr_t * self.mw[j] / (np.sqrt(self.vw[j]) + 1e-7)
            if hasattr(layer,"B"):
                self.mb[j] += (1 - self.beta1) * (layer.db - self.mb[j])
                self.vb[j] += (1 - self.beta1) * (layer.db**2 - self.vb[j])
                layer.B -= lr_t * self.mb[j] / (np.sqrt(self.vb[j]) + 1e-7) 
            j +=1