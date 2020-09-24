from communication import init_pool,get_pool
from player import set_player
from config import *
from communication import CallBack
from communication.callback import Dealer
from common.var_pool import get_pool as get_var_pool 
import sys
def main(argv):
	config = Config(filename = "./config")
	set_config(config)
	my_player = get_config().players[argv[1]]
	net_cb = CallBack(Dealer())
	my_player.start_node(net_cb)

	set_player(my_player)
	init_pool()
	from common.wrap_function import PlayerDecorator
	myDecorator = PlayerDecorator(my_player)

	@myDecorator.from_(player_name = "esion")
	def mul():
		print("test")

	@myDecorator.to_(player_name = "Bob", var_name = "x")
	def input():
		from protocol.test_protocol import Protocol
		from common.tensor import PrivateTensor,IntTensor
		ptensor = PrivateTensor(shared = True, tensor = IntTensor([998,1234,9.88]))
		get_var_pool()["x"] = ptensor
		return ptensor.share()

	@myDecorator.open_(player_name = "Emme", var_name = "x")
	def open():
		print(get_var_pool()["x"].open())
		return True

	@myDecorator.to_(player_name = "esion", var_name = '[x]')
	def triples(shape):
		from protocol.test_protocol import Protocol
		from common.tensor import PrivateTensor
		tmp = [PrivateTensor(tensor = i, shared = True) for i in Protocol.triple(shape)]
		get_var_pool()["[x]"] = tmp
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
	
	@myDecorator.open_(player_name = "Emme", var_name = "[x]")
	def check2():
		a,b,c = [ele.open() for ele in get_var_pool()["[x]"]]
		#print("get [{},{},{}]".format(a,b,c))
		if a * b == c:
			print("work well")
		else:
			print("{} != {}".format(a * b,c))
	'''test input and open'''
	# input()
	# open()
	
	#test triple
	triples([1,1,3])
	check()
	check2()
	my_player.destroy()


if __name__ == '__main__':
	main(sys.argv)

