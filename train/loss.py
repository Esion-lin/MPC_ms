from nn.pcell import PrivateCell
from protocol import get_protocol
class softmax(PrivateCell):
    def construct(self, *args, **kwargs):
        pass
    def backward(self,**kwargs):
        pass


class L2NormLoss(PrivateCell):
    def __init__(self):
        super(L2NormLoss, self).__init__()
        self.l2loss = get_protocol().L2NormLoss()
    def construct(self, *args, **kwargs):
        x = args[0]
        y = args[1]
        return self.l2loss(x = x, y = y)
    def backward(self, delta, opt):
        return self.l2loss.backward(delta)