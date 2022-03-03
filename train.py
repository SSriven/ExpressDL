# import sys, os
# sys.path.append(os.pardir) # 为了导入父目录的文件而进行的设定
from net.lenet5 import LeNet5
from dataset.mnist import load_mnist
from layers.utils import save_model,load_model,imgreshape
"""

"""

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_train = x_train[:10000]
t_train = t_train[:10000]
x_test = x_test[:1000]
t_test = t_test[:1000]

lenet5 = LeNet5(x_train,t_train,x_test,t_test,epochs=10,optimizer='adam',weight_decay_lambda=0.01,learning_rate_decay=0.95)
lenet5.train()

# save_model(lenet5,"lenet5.pkl")
lenet5 = load_model("lenet5.pkl")
test = imgreshape(x_test,(32,32))
y = lenet5.predict(test)
print(y[:100] == t_test[:100])
