from nn.pcell import PrivateCell
from protocol import get_protocol
from functools import partial
from common.placeholder import Placeholder
from crypto.factory import Counter  
class Conv(PrivateCell):
    def __init__(self, stride = 1, padding = 0, weight = None,**kwargs):
        PrivateCell.__init__(self,**kwargs)
        self.stride = stride
        self.padding = padding
        self.conv2d = get_protocol().Conv2d(self.stride, self.padding, name = "{}_conv2d".format(self.name))
        if weight is None:
            self.weight = self.set_weight()
        else:
            self.weight = weight
    def construct(self, input_var, weight = None, output_var = None,triple = None):
        if triple is not None:
            self.conv2d = partial(self.conv2d, triples = triple) #添加对空triple的判断
        if output_var: 
            self.conv2d(x = input_var, y = weight if weight is not None else self.weight, z = output_var)
            return output_var
        else:
            #创建临时变量
            tmp = Placeholder(name = "tmp_{}".format(Counter.get_counter()))
            self.conv2d(x = input_var, y = weight if weight is not None else self.weight, z = tmp)
            return tmp
    #deltaX 前向传播， deltaW本层更新
    # def backward(self, delta, input_var, learning_rate = 0.1):
    #     # 此处需要加上batch的方法
    #     deltaX,deltaW = _conv_dz_IntTensor(input_var,self.weight,self.stride,self.padding)
    #     #更新
    #     self.weight = self.weight - learning_rate * deltaW
    #     return deltaX

    def set_weight(self):
        pass
