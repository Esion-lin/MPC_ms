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
from common.wrap_of_ms.extend_tensor import avgpool, _conv_dz_PrivateTensor,_conv_dz_Placeholder
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
			with vcm() as vcm_controller:
				x = input_var["x"]
				y = input_var["y"]
				
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
				tmp = Alpha.Conv(y,self.stride,self.padding) + x.Conv(Beta,self.stride,self.padding) + c
				ans_tmp = open_with_player(player_name = "", var_name = tmp)
				ans_tmp2 = -(Alpha.Conv(Beta,self.stride,self.padding))
				# print("tmp data ",ans_tmp)
				# print("tmp2 data ",ans_tmp2)
				z = Protocol.Add_cons(tmp, ans_tmp2) 
				if "with_trunc" in input_var and input_var["with_trunc"] == False:
					return z	
				else:
					print("z_shape",z.shape)
					z = Protocol.truncate(x = z, d = encodeFP32.scale_size)
				return z
				#																			^此处会导致结果出错, 需要使用截断协议

		def backward(self,*args,**kwargs):
			with vcm() as vcm_controller:
				delta = kwargs["delta"]
				opt = kwargs["opt"]
				if "learning_rate" in kwargs:
					learning_rate = kwargs["learning_rate"]
				# 此处需要加上batch的方法
				x = kwargs["x"]
				y = kwargs["y"]
				(deltaX,deltaW) = _conv_dz_Placeholder(x,y,self.stride,self.padding, delta)
				#更新
				y = opt(y,deltaW)
				return deltaX

		def set_weight(self):
			raise NotImplementedError("试图调用未定义的方法")
	class L2NormLoss(PrivateCell):
		def construct(self, *args, **kwargs):
			with vcm() as vcm_controller:
				x = kwargs["x"]
				y = kwargs["y"]
				z = x - y
				ans = Protocol.Square(z)
				return ans
		def backward(self,delta):
			with vcm() as vcm_controller:
				return delta*2
	@classmethod
	def dispatch(cls, value:IntTensor):
		'''
		该函数返回一个tuple其中第一位为自己持有的share， 第二位list为其他人的share -> （自己的share，[player1的share，player2的share...]）
		'''
		value0 = IntTensor(Factory.get_uniform(value.shape), internal = True)
		value1 = IntTensor(Factory.get_uniform(value.shape), internal = True)
		#TODO module addition 
		value2 = value - value0 - value1
		# print("dispatch:",value0,value1,value2)
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
		with vcm() as vcm_controller:
			assert x.check() and y.check()
			if x.shape != y.shape:
				raise TypeError("except shape {}, but got {}!".format(x.shape,y.shape))
			return x + y


	@classmethod
	def Add_cons(cls, x, y, out_P = True):
		with vcm() as vcm_controller:
			dec = get_global_deco()
			@dec.from_(player_name = "Emme")
			def add(input_x, input_y):
				out = input_x + input_y
				return out
			ans = add(x, y)
			if out_P:
				if ans is None:
					if isinstance(x, Placeholder):
						x.rename("{}_add_cons_ans".format(Placeholder.tmp_name))
						return x 
					return Placeholder("{}_add_cons_ans".format(Placeholder.tmp_name),value = x)
				elif isinstance(ans, Placeholder):
					ans.rename("{}_add_cons_ans".format(Placeholder.tmp_name))
					return ans
				else:
					return Placeholder("{}_add_cons_ans".format(Placeholder.tmp_name),value = ans)
			return x if ans is None else ans 

	@classmethod
	def Square(cls, x, triples = None, out_P = True, with_trunc = True):
		if triples is None:
			triples = make_triples(triple_type = "square_triple", triples_name = "[tmp]", maked_player = "Emme", shape = x.shape)
		a = triples[0]
		b = triples[1]
		alpha = x - a
		Alpha = open_with_player(player_name = "", var_name = alpha)
		z = cls.Add_cons(x*Alpha +b, -(Alpha*Alpha), out_P = out_P)
		if with_trunc:
			z = cls.truncate(x = z, d = encodeFP32.scale_size, out_P = out_P)
		return z
	@classmethod
	def Mul(cls, x:Union[Placeholder, PrivateTensor],y:Union[Placeholder, PrivateTensor], triple = None, with_trunc = True, out_P = True):
		#添加 private 类型的计算
		with vcm() as vcm_controller:
			if isinstance(x,Placeholder):
				assert x.check()
			if isinstance(y,Placeholder):
				assert y.check()
			
			print("starting mul, shape is {}".format(x.shape))
			if x.shape != y.shape:
				raise TypeError("except shape {}, but got {}!".format(x.shape,y.shape))
			if triple == None:
				#测试用例， 需要讨论是否指定生成的用户
				triple = make_triples(triples_name = "[{}_tmp]".format(vcm.id()), maked_player = "Emme", shape = x.shape)
				
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
			alpha = x - a
			beta = y - b
			Alpha = open_with_player(player_name = "", var_name = alpha)
			Beta = open_with_player(player_name = "", var_name = beta)
			w = y*Alpha + x*Beta + c
			print("alpha",Alpha)
			print("beta",Beta)
			print("y*Alpha + x*Beta + c", w.fill())
			print("-(Alpha*Beta)", -(Alpha*Beta))
			z = cls.Add_cons(w, -(Alpha*Beta), out_P = out_P)
			print(z.name,z.fill())
			if with_trunc:
				z = cls.truncate(x = z, d = encodeFP32.scale_size, out_P = out_P)
			return z
	
	@classmethod
	def truncate(cls, x:Placeholder, d, y = None, triple = None, out_P = True):
		with vcm() as vcm_controller:
			if triple is None:
				triple = make_triples(triple_type = "trunc_triple",triples_name = "[{}_tmp2]".format(vcm.id()), maked_player = "Emme", shape = x.shape, d = d)
				#																		^just for test
			
			a = triple[0]
			b = triple[1]
			alpha = x + a
			Alpha = open_with_player(player_name = "", var_name = alpha)
			return cls.Add_cons(-b, Alpha/d, out_P = out_P)

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
		with vcm() as vcm_controller:
			return avgpool(x, kernel_size = pool_size, stride = strides)
	@classmethod
	def maxpool2d(cls, x, pool_size, strides, padding):
		#TODO: 实现最大池化
		pass
