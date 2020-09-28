from crypto import Factory
from common.tensor import IntTensor
from common.placeholder import Placeholder
class Protocol:
	'''
	dispatch function:
	IntTensor -> (IntTensor, [IntTensor])
	'''
	@staticmethod
	def dispatch(value:IntTensor):
		value0 = IntTensor(Factory.get_uniform(value.shape), internal = True)
		value1 = IntTensor(Factory.get_uniform(value.shape), internal = True)
		#TODO module addition 
		value2 = value - value0 - value1
		return (value0, [value1, value2])

	@staticmethod
	def check_open(share0, share1):
		if not share0 is None and len(share1) == 2:
			return True
		else:
			return False

	@staticmethod
	def composite(share0, share1):
		value = share0 + share1[0] + share1[1]
		return value

	@staticmethod
	def triple(shape:list):
		a = IntTensor(Factory.get_uniform(shape), internal = True)
		b = IntTensor(Factory.get_uniform(shape), internal = True)
		c = a * b
		return [a,b,c]


	@staticmethod
	def open_with_player(player_name,var_name):
		from common.wrap_function import get_global_deco
		dec = get_global_deco()
		if isinstance(var_name, Placeholder):
			var_name = var_name.name
		@dec.open_(player_name = player_name, var_name = var_name)
		def open():
			return get_var_pool()[var_name].open()
		return open()

	@staticmethod
	def make_triples(triples_name = "", maked_player = "", triples_shape = [1,1,1]):
		@myDecorator.to_(player_name = maked_player, var_name = triples_name)
		def triples(shape):
			from protocol.test_protocol import Protocol
			from common.tensor import PrivateTensor
			tmp = [PrivateTensor(tensor = i, shared = True) for i in Protocol.triple()]
			get_var_pool()[var_name] = tmp
			return list(zip(*[ele.share() for ele in tmp]))
		return triples(triples_shape)

	@staticmethod
	def Mul(x:Placeholder,y:Placeholder, triple = None):
		if x.check() and y.check():
			if x.shape != y.shape:
				raise TypeError("except shape {}, but got {}!".format(x.shape,y.shape))
			if triple == None:
				triple = make_triples(triples_name = "[tmp]", maked_player = "triples_provider", triples_shape = x.shape)
				
				#需要生成triples
			else:
				if not triple.is_list:
					raise TypeError("triples need to be a tuple!!!")
				if triple[0].shape != x.shape:
					raise IndexError("triples shape invalid!!!")
				#使用现成的triples
			a = triple[0]
			b = triple[1]
			c = triple[2]
			x_0 = x.fill()
			y_0 = y.fill()
			alpha = x_0 - a
			beta = y_0 - b
			open_with_player(player_name = "", var_name = alpha)
			open_with_player(player_name = "", var_name = beta)
			z = alpha*y_0 + beta*x_0 + c
		else:
			raise NameError("Uninitialized placeholder!!")
		return None

