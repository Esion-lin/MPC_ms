from nn.pcell import PrivateCell
from protocol import get_protocol
class Pooling(PrivateCell):
    def __init__(self, stride, padding):
        self.stride = stride
        self.padding = padding
    #TODO:实现一些pooling共有的方法
    pass

class avgPooling2D(Pooling):
    def construct(self, input_var):
        return get_protocol().avgpool2d(input_var, self.stride, self.padding)

class maxPooling2D(Pooling):
    def construct(self, input_var):
        return get_protocol().maxpool2d(input_var, self.stride, self.padding)
