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
	def Add(x:Placeholder,y:Placeholder,z:Placeholder):
		assert x.check() and y.check()
		if x.shape != y.shape:
			raise TypeError("except shape {}, but got {}!".format(x.shape,y.shape))
		x_0 = x.fill()
		y_0 = y.fill()
		z.set_value(x_0 + y_0)
		# fluent interface
		return z
	@staticmethod
	def Mul(x:Placeholder,y:Placeholder,z:Placeholder, triple = None):
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
			Placeholder.register(alpha,"alpha")
			Placeholder.register(beta,"beta")
			Alpha = open_with_player(player_name = "", var_name = "alpha")
			Beta = open_with_player(player_name = "", var_name = "beta")
			#Todo:实现PlaceHolder
			z.set_value(Alpha*Beta + b*Alpha + a*Beta - c)
		else:
			raise NameError("Uninitialized placeholder!!")
		# fluent interface
		return z
	@staticmethod
	def Conv2d(x:Placeholder, w:Placeholder, stride, padding, y:Placeholder):
		'''
		w*x ->  y
		使用tensor明文下的卷积操作构建协议的卷积
		'''
		# fluent interface
		return y
	@staticmethod
	def square(x:Placeholder, y:Placeholder, triple = None):
		if x.check():
			if triple == None:
				triple = make_triples(triples_name = "[tmp]", maked_player = "triples_provider", triples_shape = x.shape)
			#TODO 使用squre——triple实现该方法
			pass
		else:
			raise NameError("Uninitialized placeholder!!")
	@staticmethod
	def relu(x:Placeholder):
		w0 = 0.44015372000819103
		w1 = 0.500000000
		w2 = 0.11217537671414643
		w4 = -0.0013660836712429923
		w6 = 9.009136367360004e-06
		w8 = -2.1097433984e-08
		#TODO 计算多项式，高次项如何展开有待商榷
		pass
		return x

	@staticmethod
	def avgpool2d(x, pool_size, strides, padding):
		#TODO: 实现平均池化
		pass
		return x
	def maxpool2d(x, pool_size, strides, padding):
		#TODO: 实现z最大池化
		pass
		return x