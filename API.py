#1>.init var pool
#2>.init connection pool
#3>.load configure
#4>.init local player

#5>.Define variable

import PlaceHolder

x = PlaceHolder("x") #Correspond with input function
y = PlaceHolder("label")
'''
@myDecorator.to_(player_name = "Bob", var_name = "x")
def input():
    from protocol.test_protocol import Protocol
    from common.tensor import PrivateTensor,IntTensor
    ptensor = PrivateTensor(shared = True, tensor = IntTensor([998,1234,9.88]))
    get_var_pool()["x"] = ptensor
    return ptensor.share()

''' 
#6>.Define Network
import MPC_ms.nn
class LeNet5(nn.PCell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
    def construct(self, x):
        self.load_weight("Bob")
        x = self.max_pool2d(self.relu(self.conv1(x)))
        ...
        x = self.fc3(x)
        return x
    '''@myDecorator.to_(player_name = "Bob", var_name = "[weight]")
    '''
    def load_weight(self, str):
        for ele in self.cells:
            ele.load_weight(str)

#6>.Prediction
net = LeNet5()
input()
y = net(x)
#7>.Define Loss function
net_loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

#8>.Train
model = Model(net, loss = None, opt = None)
model.train(x, epoch_size)