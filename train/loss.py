from nn.pcell import PrivateCell
from protocol import get_protocol
class softmax(PrivateCell):
    def construct(self, *args, **kwargs):
        pass
    def backward(self,**kwargs):
        pass


class L2NormLoss(PrivateCell):
    def __init__(self):
        self.l2loss = get_protocol().L2NormLoss()
    def construct(self, *args, **kwargs):
        x = kwargs["x"]
        y = kwargs["y"]
        return self.l2loss(x = x, y = y)