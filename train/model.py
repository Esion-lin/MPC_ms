from nn.pcell import PrivateCell

class Model(PrivateCell):
    def __init__(self, net, loss_func, net_opt,**kwargs):
        PrivateCell.__init__(self,**kwargs)
        self.net = net
        self.loss_func = loss_func
        self.net_opt = net_opt

    def construct(self, *args, **kwargs):
        self.train_test(*args, **kwargs)

    def train(self, size_of_epoch, dataset, label):
        assert dataset.shape[0] == label.shape[0]
        for i in range(dataset.shape[0]):
            tmp = self.net(dataset[i])
            err = self.loss_func(tmp,label[i])
            err = self.loss_func.backward(err, self.net_opt)
            self.net.backward(err,self.net_opt)

    def train_test(self, dataset, label):
        tmp = self.net(dataset)
        err = self.loss_func(tmp,label)
        err = self.loss_func.backward(err, self.net_opt)
        self.net.backward(err,self.net_opt)