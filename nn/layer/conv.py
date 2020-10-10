from nn.pcell import PrivateCell
from protocol import get_protocol
class Conv(PrivateCell):
    def __init__(self, stride = 1, padding = False, weight = None):
        self.pro = get_protocol()
        self.peremeter = weight?weight:self.set_weight()
        self.stride = stride
        self.padding = padding
    def construct(self, input_var):
        get_protocol().Conv2d(input_var, self.peremeter, self.stride, self.padding)

