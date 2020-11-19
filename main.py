from communication import init_pool,get_pool
from player import set_player
from config import *
from communication import CallBack
from communication.callback import Dealer
from common.var_pool import get_pool as get_var_pool 
from nn.layer.conv import Conv
from common.tensor import PrivateTensor,IntTensor
import sys
from protocol.test_protocol import Protocol
from protocol.command_fun import *
from common.placeholder import Placeholder
from train.model import Model
from train.loss import L2NormLoss
from train.opt import GD
import datetime
from crypto.factory import encodeFP32
import numpy as np
def main(argv):
	config = Config(filename = "./config")
	set_config(config)
	my_player = get_config().players[argv[1]]
	net_cb = CallBack(Dealer())
	my_player.start_node(net_cb)

	set_player(my_player)
	init_pool()
	from common.wrap_function import PlayerDecorator, set_global_deco
	myDecorator = PlayerDecorator(my_player)
	set_global_deco(myDecorator)
	@myDecorator.from_(player_name = "esion")
	def mul():
		print("test")

	def input_data(name, jtensor):
		ptensor = PrivateTensor(shared = True, tensor = jtensor, name = name)
		return input_with_player("Bob", name, ptensor)

	@myDecorator.open_(player_name = "Emme", var_name = "x")
	def open():
		print(get_var_pool()["x"].open())
		return True

	@myDecorator.to_(player_name = "esion", var_name = '[x]')
	def triples(shape):
		from protocol.test_protocol import Protocol
		from common.tensor import PrivateTensor
		tmp = [PrivateTensor(tensor = i, shared = True) for i in Protocol.triple(shape)]
		get_var_pool()[var_name] = tmp
		return list(zip(*[ele.share() for ele in tmp]))

	def print_triple():
		print([ele.convert_public() for ele in get_var_pool()["[x]"]])

	@myDecorator.from_(player_name = "esion")
	def check():
		a,b,c = [ele.open() for ele in get_var_pool()["[x]"]]
		#print("get [{},{},{}]".format(a,b,c))
		if a * b == c:
			print("work well")
		else:
			print("{} != {}".format(a * b,c))
	
	@myDecorator.open_(player_name = "Emme", var_name = "[tmp]")
	def check2():
		a,b,c = [ele.open() for ele in get_var_pool()["[tmp]"]]
		#print("get [{},{},{}]".format(a,b,c))
		if a * b == c:
			print("triples work well")
		else:
			print("{} != {}".format(a * b,c))
	'''test input and open'''
	# x = input_data("x", IntTensor([1.1,0.2,1]))
	# print("x",x)
	# y = input_data("y", IntTensor([1.23,0.5,1]))
	# print("y",y)
	# res = Protocol.Mul(x,y)
	# print("res",res)
	# z = input_data("z", IntTensor([1.2,0.2,2]))
	# print("res",res)
	# res2 = Protocol.Mul(res,z)
	# res3 = Protocol.Mul(x,z)
	# ans = open_with_player("Emme", res)
	# ans2 = open_with_player("Emme", res2)
	# ans3 = open_with_player("Emme", res3)
	# print("None" if ans is None else ans.to_native())
	# print("None" if ans2 is None else ans2.to_native())
	# print("None" if ans3 is None else ans3.to_native())
	# return 0
	# res2 = Placeholder("res2")
	# Protocol.Mul(x,y,res2)
	# ans = open_with_player("Emme", "res2")
	# print("" if ans is None else "mul res is {}".format(ans.to_native()))
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
			
			tmp = self.conv2d2(tmp)
			#tmp = self.relu(tmp)
			#tmp = self.pool(tmp)
			#tmp = self.conv2d3(tmp)
			return tmp
		def set_weight(self):
			#实现默认的权重赋值
			pass

	# w = input("w", IntTensor(np.random.random((1,1,2,2))))
	# w2 = input("w2", IntTensor(np.random.random((3,3,2,2))))
	# w3 = input("w3", IntTensor(np.random.random((3,3,2,2))))
	w = input_data("w", IntTensor([[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]]]))
	w2 = input_data("w2", IntTensor([[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]]]))
	w3 = input_data("w3", IntTensor([[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]]]))
	image = input_data("image", IntTensor([[[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],
										[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],
										[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]]]))
	print("image",image)
	print("image shape",image.shape)
	#image = input("image", IntTensor([[[[1,1]]]]))
	# image = input("image", IntTensor(np.random.random((1,3,5,5))))
	label = input_data("label", IntTensor(np.random.random((1,3,1,1))))
	net = testNet(weight = [w,w2,w3])
	#model = Model(net, loss_func = L2NormLoss(), net_opt = GD(1))
	
	start = datetime.datetime.now()
	print("start ")
	#model(image, label)
	y = net(image)
	print("open ",y.name)
	ans = open_with_player("Emme", y)
	print("None" if ans is None else "res is {}".format(ans.to_native()))
	end = datetime.datetime.now()
	print("time ",end-start)
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

