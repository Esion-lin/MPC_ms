from nn.pcell import PrivateCell
from protocol import get_protocol
from functools import partial
import time
class Conv(PrivateCell):
    def __init__(self, stride = 1, padding = False, weight = None):
        self.stride = stride
        self.padding = padding
        self.conv2d = get_protocol().nn["conv"](self.stride, self.padding)
    def construct(self, input_var, weight = None, output_var = None,triple = None):
        if weight is None:
            weight = self.set_weight()
        if triple is not None:
            self.conv2d = partial(self.conv2d, triples = triple) #添加对空triple的判断
        if output_var: 
            self.conv2d(x = input_var, y = weight, z = output_var)
        else:
            #创建临时变量
            tmp = Placeholder(name = "tmp_{}".format(time.time()))
            self.conv2d(x = input_var, y = weight, z = tmp)
            return tmp
    def set_weight():
        pass
