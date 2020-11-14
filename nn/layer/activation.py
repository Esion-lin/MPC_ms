from nn.pcell import PrivateCell
from protocol import get_protocol
class Relu(PrivateCell):
    def construct(self, input_var):
        return get_protocol().relu(input_var)
    def backward(self, delta, opt)
        return delta
    def set_weight(self):
        pass
