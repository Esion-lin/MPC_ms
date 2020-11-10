from crypto import Factory
from crypto.factory import encodeFP32
from common.tensor import IntTensor, PrivateTensor
from common.placeholder import Placeholder
from common.var_pool import get_pool as get_var_pool 
from common.VCM import vcm
from nn.pcell import PrivateCell
from mindspore import nn
from common.wrap_function import get_global_deco
from .command_fun import * 
from common.wrap_of_ms.extend_tensor import avgpool, _conv_dz_PrivateTensor
from typing import Union
import sys
class Protocol:
	'''
	dispatch function:
	IntTensor -> (IntTensor, [IntTensor])
	'''
	class Conv2d(PrivateCell):
		'''
		w*x ->  y
		使用tensor明文下的卷积操作构建协议的卷积
		TODO:添加channel检查的逻辑 
		'''
		def __init__(self, stride, padding,**kwargs):
			PrivateCell.__init__(self,**kwargs)
			self.stride = stride
			self.padding = padding
		def construct(self,**input_var):
			with vcm(self.name) as vcm_controller:
				x = input_var["x"]
				y = input_var["y"]
				z = input_var["z"]
				if "triples" in input_var:
					triples = input_var["triples"]
				else:
					triples = make_triples(triple_type = "conv_triple", triples_name = "[{}_tmp]".format(self.name), maked_player = "Emme", shapeX = x.shape, shapeY = y.shape, stride = self.stride, padding = self.padding)
				a = triples[0]
				b = triples[1]
				c = triples[2]
				#获得privateTensor
				alpha = x - a
				beta = y - b
				Alpha = open_with_player(player_name = "", var_name = alpha)
				Beta = open_with_player(player_name = "", var_name = beta)
				z.set_value(Protocol.Add_cons(Alpha.Conv(y,self.stride,self.padding) + x.Conv(Beta,self.stride,self.padding) + c, -(Alpha.Conv(Beta,self.stride,self.padding))) / encodeFP32.scale_size())
				#																			^此处会导致结果出错, 需要使用截断协议
				return z
		def backward(self,**kwargs):
			delta = kwargs["delta"]
			opt = kwargs["opt"]
			if "learning_rate" in kwargs:
				learning_rate = kwargs["learning_rate"]
			# 此处需要加上batch的方法
			x = kwargs["x"]
			y = kwargs["y"]
			x_0 = x.fill()
			y_0 = y.fill()
			deltaX,deltaW = _conv_dz_PrivateTensor(x_0,y_0,self.stride,self.padding)
			#更新
			y_0 = opt(y_0,deltaW)
			y_0 = y_0 - learning_rate * Protocol.Mul(deltaW,delta)#
			y.set_value(y_0, force_sys = True)
			z = Placeholder("z")
			return Protocol.Mul(deltaX,delta,z)

		def set_weight(self):
			raise NotImplementedError("试图调用未定义的方法")
	class L2NormLoss(PrivateCell):
		def construct(self, *args, **kwargs):
			x = kwargs["x"]
			y = kwargs["y"]
			if isinstance(x, Placeholder):
				x_0 = x.fill()
			else:
				x_0 = x
			if isinstance(y, Placeholder):
				y_0 = y.fill()
			else:
				y_0 = y
			z_0 = x_0-y_0
			Placeholder.register(z_0,"z_0")
			ans = Protocol.Square(z_0)
		def backward(self,**kwargs):
			delta = kwargs["delta"]
			return 2*delta
	@classmethod
	def dispatch(cls, value:IntTensor):
		'''
		该函数返回一个tuple其中第一位为自己持有的share， 第二位list为其他人的share -> （自己的share，[player1的share，player2的share...]）
		'''
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
	def add(x:Placeholder, y:Placeholder):
		assert x.check() and y.check()
		if x.shape != y.shape:
			raise TypeError("except shape {}, but got {}!".format(x.shape,y.shape))
		return x + y


	@classmethod
	def Add_cons(cls, x, y):
		dec = get_global_deco()
		@dec.from_(player_name = "Emme")
		def add(input_x, input_y):
			return input_x + input_y
		ans = add(x, y)
		return x if ans is None else ans 

	@classmethod
	def Square(cls, x, triples = None):
		if isinstance(x,Placeholder):
			assert x.check()
			x_0 = x.fill()
		else:
			x_0 = x
		if triples is None:
			triples = make_triples(triple_type = "square_triple", triples_name = "[tmp]", maked_player = "Emme", shape = x.shape)
		a = triple[0]
		b = triple[1]
		alpha = x_0 - a
		Placeholder.register(alpha,"alpha")
		Alpha = open_with_player(player_name = "", var_name = "alpha")
		if register and z is not None:
			z.set_value(cls.Add_cons(x_0*Alpha +b, -(Alpha*Alpha)) )
		else:
			z = cls.Add_cons(x_0*Alpha +b, -(Alpha*Alpha)) 
		if with_trunc:
			cls.truncate(x = z, d = encodeFP32.scale_size())
		return z
	@classmethod
	def Mul(cls, x:Union[Placeholder, PrivateTensor],y:Union[Placeholder, PrivateTensor],z, triple = None, with_trunc = True, register = True):
		#添加 private 类型的计算
		if isinstance(x,Placeholder):
			assert x.check()
		if isinstance(y,Placeholder):
			assert y.check()
		
		print("starting mul, shape is {}".format(x.shape))
		if x.shape != y.shape:
			raise TypeError("except shape {}, but got {}!".format(x.shape,y.shape))
		if triple == None:
			#测试用例， 需要讨论是否指定生成的用户
			triple = make_triples(triples_name = "[tmp]", maked_player = "Emme", shape = x.shape)
			
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
		if isinstance(x,Placeholder):
			x_0 = x.fill()
		else:
			x_0 = x
		if isinstance(y,Placeholder):
			y_0 = y.fill()
		else:
			y_0 = y
		alpha = x_0 - a
		beta = y_0 - b
		Placeholder.register(alpha,"alpha")
		Placeholder.register(beta,"beta")
		Alpha = open_with_player(player_name = "", var_name = "alpha")
		Beta = open_with_player(player_name = "", var_name = "beta")
		#Todo:实现PlaceHolder
		if register and z is not None:
			z.set_value(cls.Add_cons(y_0*Alpha + x_0*Beta + c, -(Alpha*Beta)) )
		else:
			z = cls.Add_cons(y_0*Alpha + x_0*Beta + c, -(Alpha*Beta))
		if with_trunc:
			cls.truncate(x = z, d = encodeFP32.scale_size())
		# fluent interface
		return z

	@classmethod
	def truncate(cls, x:Placeholder, d,y = None, triple = None):
		if triple is None:
			triple = make_triples(triple_type = "trunc_triple",triples_name = "[tmp2]", maked_player = "Emme", shape = x.shape, d = d)
			#																		^just for test
		a = triple[0]
		b = triple[1]
		x_0 = x.fill()
		alpha = x_0 - a
		Placeholder.register(alpha,"alpha2")
		Alpha = open_with_player(player_name = "", var_name = "alpha2")
		if y is None:
			x.set_value(cls.Add_cons(b, Alpha/d), force_sys = True)
			return x
		else:
			y.set_value(cls.Add_cons(b, Alpha/d))
			return y
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
	def avgpool2d(cls, x, pool_size, strides, padding, y = None):
		#TODO: 实现平均池化
		x_0 = x.fill()
		tmp = avgpool(x_0, kernel_size = pool_size, stride = strides)
		if y is None:
			x.set_value(tmp, force_sys = True)
			return x
		else:
			y.set_value(tmp)
			return y
	@classmethod
	def maxpool2d(cls, x, pool_size, strides, padding):
		#TODO: 实现最大池化
		pass
