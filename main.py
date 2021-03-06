
from player import set_player
from config import *
from nn.layer.conv import Conv
from common.tensor import PrivateTensor,IntTensor
import sys
from common.placeholder import Placeholder
from train.model import Model
from train.loss import L2NormLoss
from train.opt import GD
import datetime
import numpy as np
from User import *
def main(argv):

	my_player = get_config().players[argv[1]]
	set_player(my_player)
	start_task()

	#test conv
	from nn.pcell import PrivateCell
	from nn import Conv
	from nn.layer.activation import Relu
	from nn.layer.pooling import avgPooling2D
	class testNet(PrivateCell):
		def __init__(self, weight):
			super(testNet, self).__init__()
			self.conv2d = Conv(stride=1,padding=0, weight=weight[0],name = "conv2d")
			self.conv2d2 = Conv(stride=1,padding=0, weight=weight[1],name = "conv2d2")
			self.relu = Relu(name = "relu")
			self.pool = avgPooling2D(kernel_size = 2, stride = 1)
			self.conv2d3 = Conv(stride=1,padding=0, weight=weight[2],name = "conv2d3")
		def construct(self, input_var):
			tmp = self.conv2d(input_var)
			
			# tmp = self.conv2d2(tmp)
			# tmp = self.relu(tmp)
			# #tmp = self.pool(tmp)
			# tmp = self.conv2d3(tmp)
			return tmp
		def set_weight(self):
			#实现默认的权重赋值
			pass
	x = input_data("x", IntTensor([1,2,3,4,-1,-2,-32,32,-128,128,-9,9,7,-7,1111]))
	from protocol import get_protocol
	ans = get_protocol().truncate(x, 2**14)
	res = open("Emme", ans)
	print("None" if res is None else "res is {}".format(res))
	# w = input("w", IntTensor(np.random.random((1,1,2,2))))
	# w2 = input("w2", IntTensor(np.random.random((3,3,2,2))))
	# w3 = input("w3", IntTensor(np.random.random((3,3,2,2))))
	w = input_data("w", IntTensor([[[[-1,-1],[-1,-1]],[[-1,-1],[-1,-1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]]]))
	w2 = input_data("w2", IntTensor([[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]]]))
	w3 = input_data("w3", IntTensor([[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]]]))
	image = input_data("image", IntTensor([[[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],
										[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],
										[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]]]))
	# print("image",image)
	# print("image shape",image.shape)
	#image = input("image", IntTensor([[[[1,1]]]]))
	# image = input("image", IntTensor(np.random.random((1,3,5,5))))
	# label = input_data("label", IntTensor(np.random.random((1,3,1,1))))
	# net = testNet(weight = [w,w2,w3])
	# #model = Model(net, loss_func = L2NormLoss(), net_opt = GD(1))
	
	# start = datetime.datetime.now()
	# print("start ")
	# #model(image, label)
	# y = net(image)
	# print("open ",y.name)
	# ans = open("Emme", y)
	# print("None" if ans is None else "res is {}".format(ans.to_native()))
	# end = datetime.datetime.now()
	# print("time ",end-start)
	# from nn import Conv
	# conv = Conv(1,0)
	# res = Placeholder("res")
	# conv(input_var = image, weight = w, output_var = res)
	# ans = open_with_player("Emme", "res")
	# print("None" if ans is None else "mul res is {}".format(ans.to_native()))
	
	#test triple
	#Protocol.make_triples("[tmp]","Emme", [3,3,3])
	#check2()

	'''
	from nn.pcell import PrivateCell
	from nn.layer.conv import Conv
	from nn.layer.activation import Relu
	
	class testNet(PrivateCell):
		def __init__(self):
			self.conv2d = Conv(stride=2,padding=True)
			self.relu = Relu()
		def construct(self, input_var):
			tmp = self.conv2d(input_var)
			tmp = self.relu(tmp)
			return tmp
	testNet(x)		
	'''

	#test triple
	# triples([1,1,3])
	# check()
	# check2()
	my_player.destroy()


if __name__ == '__main__':
	main(sys.argv)

