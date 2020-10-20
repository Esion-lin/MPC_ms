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

	def input(name, jtensor):
		ptensor = PrivateTensor(shared = True, tensor = jtensor)
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
	# x = input("x", IntTensor([0.1,0.34,0.088]))
	# x.fill()
	# y = input("y", IntTensor([0.1,0.2,0.1]))
	# y.fill()
	# res = Placeholder("res")
	# Protocol.Add(x,y,res)
	# ans = Protocol.open_with_player("Emme", "res")
	# print("None" if ans is None else ans.to_native())
	# res2 = Placeholder("res2")
	# Protocol.Mul(x,y,res2)
	# ans = open_with_player("Emme", "res2")
	# print("" if ans is None else "mul res is {}".format(ans.to_native()))
	#test conv
	
	w = input("w", IntTensor([[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]]]))
	image = input("image", IntTensor([[[[1,1,1],[1,1,1],[1,1,1]],
										[[1,1,1],[1,1,1],[1,1,1]],
										[[1,1,1],[1,1,1],[1,1,1]]]]))
	from nn import Conv
	conv = Conv(1,0)
	res = Placeholder("res")
	conv(input_var = image, weight = w, output_var = res)
	ans = open_with_player("Emme", "res")
	print("None" if ans is None else "mul res is {}".format(ans.to_native()))
	
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

