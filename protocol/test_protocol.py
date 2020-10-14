from crypto import Factory
from crypto.factory import encodeFP32
from common.tensor import IntTensor, PrivateTensor
from common.placeholder import Placeholder
from common.var_pool import get_pool as get_var_pool 
from nn.pcell import PrivateCell
from common.wrap_function import get_global_deco
class Protocol:
	'''
	dispatch function:
	IntTensor -> (IntTensor, [IntTensor])
	'''
	dec = get_global_deco()
	@classmethod
	def dispatch(cls, value:IntTensor):
		value0 = IntTensor(Factory.get_uniform(value.shape), internal = True)
		value1 = IntTensor(Factory.get_uniform(value.shape), internal = True)
		#TODO module addition 
		value2 = value - value0 - value1
		return (value0, [value1, value2])

	@classmethod
	def check_open(cls, share0, share1):
		if not share0 is None and len(share1) >= 2:
			return True
		else:
			return False

	@classmethod
	def composite(cls, share0, share1):
		value = share0 + share1[0] + share1[1]
		return value

	@classmethod
	def triple(cls, shape:list):
		a = IntTensor(Factory.get_uniform(shape), internal = True)
		b = IntTensor(Factory.get_uniform(shape), internal = True)
		c = a * b
		return [a,b,c]

	@classmethod
	def mat_triple(cls, shapeX:list, shapeY:list):
		def check_mat(shapeA, shapeB):
			if len(shapeA) != len(shapeB):
				return False
			elif shapeA[-1] != shapeA[-2]:
				return False
			return True
		if check_mat(shapeX, shapeY):
			a = IntTensor(Factory.get_uniform(shapeX), internal = True)
			b = IntTensor(Factory.get_uniform(shapeY), internal = True)
			c = a.Matmul(b)
			return [a,b,c]
		else:
			raise TypeError("shapes do not match!")

	@classmethod
	def open_with_player(cls, player_name,var_name):
		if isinstance(var_name, Placeholder):
			var_name = var_name.name
		@cls.dec.open_(player_name = player_name, var_name = var_name)
		def open():
			return get_var_pool()[var_name].open()
		return open()

	@classmethod
	def input_with_player(cls, player_name,var_name, ptensor):
		@cls.dec.to_(player_name = player_name, var_name = var_name)
		def input():
			get_var_pool()[var_name] = ptensor
			return ptensor.share()
		return input()

	@classmethod
	def make_triples(cls, triples_name = "", maked_player = "", triples_shape = [1,1,1]):
		@cls.dec.to_(player_name = maked_player, var_name = triples_name)
		def triples(shape):
			from protocol.test_protocol import Protocol
			from common.tensor import PrivateTensor
			tmp = [PrivateTensor(tensor = i, shared = True) for i in cls.triple(shape)]
			get_var_pool()[var_name] = tmp
			return list(zip(*[ele.share() for ele in tmp]))
		return triples(triples_shape)
		
	@classmethod
	def make_mat_triples(cls, triples_name = "", maked_player = "", triples_shapeA, triples_shapeB):
		@cls.dec.to_(player_name = maked_player, var_name = triples_name)
		def triples(shape_a, shape_b):
			from protocol.test_protocol import Protocol
			from common.tensor import PrivateTensor
			tmp = [PrivateTensor(tensor = i, shared = True) for i in cls.mat_triple(shape_a, shape_b)]
			get_var_pool()[var_name] = tmp
			return list(zip(*[ele.share() for ele in tmp]))
		return triples(triples_shapeA, triples_shapeB)

	@classmethod
	def Add(cls, x:Placeholder,y:Placeholder,z:Placeholder = None):
		assert x.check() and y.check()
		if x.shape != y.shape:
			raise TypeError("except shape {}, but got {}!".format(x.shape,y.shape))
		x_0 = x.fill()
		y_0 = y.fill()
		if z == None:
			z = Placeholder("z")
		z.set_value(x_0 + y_0)
		z.inject()
		# fluent interface
		return z
	@classmethod
	def Add_cons(cls, x, y):
		@cls.dec.from_(player_name = "Emme")
		def add(input_x, input_y):
			if isinstance(input_x, Placeholder):
				input_x.value = input_x.value + input_y
				input_x.inject()
			elif isinstance(input_x, PrivateTensor):
				input_x = input_x + input_y
			return input_x
		ans = add(x, y)
		return x if ans is None else ans 

	@classmethod
	def Mul(cls, x:Placeholder,y:Placeholder,z:Placeholder, triple = None):
		if x.check() and y.check():
			print("starting mul, shape is {}".format(x.shape))
			if x.shape != y.shape:
				raise TypeError("except shape {}, but got {}!".format(x.shape,y.shape))
			if triple == None:
				#测试用例， 需要讨论是否指定生成的用户
				triple = cls.make_triples(triples_name = "[tmp]", maked_player = "Emme", triples_shape = x.shape)
				
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
			# from common.wrap_function import get_global_deco
			# dec = get_global_deco()
			# @dec.open_(player_name = "", var_name = "[tmp]")
			# def check():
			# 	aa,bb,cc = [ele.open() for ele in get_var_pool()["[tmp]"]]
			# 	#print("get [{},{},{}]".format(a,b,c))
			# 	if aa * bb == cc:
			# 		print("triples work well {} * {} == {}".format(aa, bb,cc))
			# 	else:
			# 		print("{} != {}".format(aa * bb,cc))
			# check()
			x_0 = x.fill()
			y_0 = y.fill()
			alpha = x_0 - a
			beta = y_0 - b
			Placeholder.register(alpha,"alpha")
			Placeholder.register(beta,"beta")
			Alpha = cls.open_with_player(player_name = "", var_name = "alpha")
			Beta = cls.open_with_player(player_name = "", var_name = "beta")
			# print("Alpha is {}".format(Alpha),"Beta is {}".format(Beta), "sadfa is {}".format(-(Alpha*Beta)))

			# xxxx = cls.open_with_player(player_name = "Emme", var_name = "x")
			# yyyy = cls.open_with_player(player_name = "Emme", var_name = "y")
			# Placeholder.register(y_0*Alpha + x_0*Beta,"kkk")
			# Placeholder.register(x_0*Beta,"ddd")
			# kkk = cls.open_with_player(player_name = "Emme", var_name = "kkk")
			# ddd = cls.open_with_player(player_name = "Emme", var_name = "ddd")
			# print("x is {}".format(xxxx),"y is {}".format(yyyy))
			# print("kkk is {}".format(kkk),"ddd is {}".format(ddd))
			#Todo:实现PlaceHolder
			z.set_value(cls.Add_cons(y_0*Alpha + x_0*Beta + c, -(Alpha*Beta)) / encodeFP32.scale_size())
			#																	^此处会导致结果出错, 需要使用截断协议
		else:
			raise NameError("Uninitialized placeholder!!")
		# fluent interface
		return z
	@classmethod
	def Conv2d(cls, x:Placeholder, w:Placeholder, stride, padding, y:Placeholder):
		'''
		w*x ->  y
		使用tensor明文下的卷积操作构建协议的卷积
		'''
		# fluent interface
		return y
	@classmethod
	def square(cls, x:Placeholder, y:Placeholder, triple = None):
		if x.check():
			if triple == None:
				triple = make_triples(triples_name = "[tmp]", maked_player = "triples_provider", triples_shape = x.shape)
			#TODO 使用squre——triple实现该方法
			pass
		else:
			raise NameError("Uninitialized placeholder!!")
	@classmethod
	def relu(cls, x:Placeholder):
		w0 = 0.44015372000819103
		w1 = 0.500000000
		w2 = 0.11217537671414643
		w4 = -0.0013660836712429923
		w6 = 9.009136367360004e-06
		w8 = -2.1097433984e-08
		#TODO 计算多项式，高次项如何展开有待商榷
		pass
		return x

	@classmethod
	def avgpool2d(cls, x, pool_size, strides, padding):
		#TODO: 实现平均池化
		pass
		return x
	@classmethod
	def maxpool2d(cls, x, pool_size, strides, padding):
		#TODO: 实现最大池化
		pass
		return x
class Conv2d(PrivateCell):
	'''
	w*x ->  y
	使用tensor明文下的卷积操作构建协议的卷积
	'''
	def __init__(self, in_channels, out_channels, stride, padding):
		pass
	def construct(self,**input_var):
		x = input_var["x"]
		y = input_var["y"]
		z = input_var["z"]
		if "triples" in input_var:
			triples = input_var["triples"]
		else:
			triples = cls.make_triples(triples_name = "[tmp]", maked_player = "Emme", triples_shapeA = x.shape, triples_shapeB = y.shape)
		a = triples[0]
		b = triples[1]
		c = triples[2]
		#获得privateTensor
		x_0 = x.fill()
		y_0 = y.fill()
		alpha = x_0 - a
		beta = y_0 - b
		Placeholder.register(alpha,"alpha")
		Placeholder.register(beta,"beta")
		Alpha = cls.open_with_player(player_name = "", var_name = "alpha")
		Beta = cls.open_with_player(player_name = "", var_name = "beta")
		b_cov = nn.Conv2d(in_channels, out_channels, Beta.shape,stride,padding = padding, weight_init=Beta.value)
		y_cov = nn.Conv2d(in_channels, out_channels, Beta.shape,stride,padding = padding, weight_init=y.value)
		z.set_value(cls.Add_cons(y_cov(Alpha) + b_cov(x_0) + c, -(b_cov(Alpha))) / encodeFP32.scale_size())