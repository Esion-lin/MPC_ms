#该文件用于定义优化器，优化器在调用layer的backward时调用
from nn.pcell import PrivateCell
class optimizer(PrivateCell):
    def __init__(self, lr):
        self.lr = lr
    def construct(self, **kwargs):
        #输入梯度，根据优化器算法更新权重
        raise NotImplementedError("试图调用未定义的方法")
class GD(optimizer):
    def __init__(self,lr):
        self.lr = lr
        self.v = 0
    def construct(self, weight, delta):
        self.v = delta
        weight = weight - self.v * self.lr
        return weight
    
class Momentum(optimizer):
    def __init__(self,lr,momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = 0
    def construct(self, **kwargs):
        self.v = self.momentum*self.v + self.lr*kwargs["delta"]
        kwargs["weight"] -= self.v