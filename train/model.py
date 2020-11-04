from nn.pcell import PrivateCell

class Model(PrivateCell):
    def __init__(self, net, loss_func, net_opt):
        self.net = net
        self.loss_func = loss_func
        self.net_opt = net_opt
    def construct(self, **kwargs):
        self.train(**kwargs)
    def train(self, size_of_epoch, dataset):
        for i in range(dataset.shape[0]):
            tmp = self.net(dataset[i][0])
            err = self.loss_func(tmp,dataset[i][1])
            err = self.loss_func.backward(err, self.net_opt)
            self.net.backward(err,self.net_opt)
