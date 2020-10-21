from nn.pcell import PrivateCell
from protocol import get_protocol
class Pooling(PrivateCell):
    def __init__(self, kernel_size, stride, padding = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    #TODO:实现一些pooling共有的方法
    def set_weight(self):
        return 2

class avgPooling2D(Pooling):
    def construct(self, input_var):
        return get_protocol().avgpool2d(input_var, self.kernel_size, self.stride, self.padding)

class maxPooling2D(Pooling):
    def construct(self, input_var):
        return get_protocol().maxpool2d(input_var, self.kernel_size, self.stride, self.padding)
