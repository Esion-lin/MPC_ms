from nn.pcell import PrivateCell
from protocol import get_protocol
import time
class Conv(PrivateCell):
    def __init__(self, stride = 1, padding = False, weight = None):
        self.pro = get_protocol()
        self.peremeter = weight if weight != None else self.set_weight()
        self.stride = stride
        self.padding = padding
    def construct(self, input_var, output_var = None):
        if output_var: 
            get_protocol().Conv2d(input_var, self.peremeter, self.stride, self.padding, output_var)
        else:
            #创建临时变量
            tmp = Placeholder(name = "tmp_{}".format(time.time()))
            get_protocol().Conv2d(input_var, self.peremeter, self.stride, self.padding, tmp)
            return tmp

