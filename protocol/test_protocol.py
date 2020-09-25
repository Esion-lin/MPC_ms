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
		@dec.open_(player_name = player_name, var_name = var_name)
		def open():
			return get_var_pool()["x"].open()
		return open()
	
	@staticmethod
	def Mul(x:Placeholder,y:Placeholder):
		pass

