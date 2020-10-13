from mindspore import Tensor
import mindspore
import numpy as np
from crypto.factory import encodeFP32
from mindspore.ops import operations as P

class IntTensor:
	'''

	'''
	#TODO: Access management
	def __init__(self, tensor, internal = False):
		if internal == False:
			if isinstance(tensor, Tensor):
				#transfer to int
				self.value = Tensor((tensor * encodeFP32.scale_size()).asnumpy(), dtype = mindspore.int32)
			elif isinstance(tensor, list):
				self.value = Tensor(np.asarray(tensor) * encodeFP32.scale_size(), dtype = mindspore.int32)
			else:
				self.value = Tensor(tensor * encodeFP32.scale_size(), dtype = mindspore.int32)
		else:
			if isinstance(tensor, Tensor):
				#transfer to int
				self.value = Tensor(tensor.asnumpy(), dtype = mindspore.int32)
			else:
				self.value = Tensor(tensor, dtype = mindspore.int32)
		self.shape = self.value.shape
		self.add = P.TensorAdd()
		self.inv = P.Invert()
		self.mul = P.Mul()
		self.div = P.FloorDiv()
		self.matmul = P.MatMul()
	#TODO: 需要添加对其他类型数据计算的重载

	
	def __add__(self, other):
		if isinstance(other, IntTensor):
			return IntTensor((self.add(self.value, other.value)).asnumpy() % encodeFP32.module(),internal = True)
		elif isinstance(other, PrivateTensor):
			# 数据与share相加得到share
			return PrivateTensor(tensor = self + other.convert_public())
		#self.value = Tensor(self.add(self.value, other.value).asnumpy() % encodeFP32.module())
		#return self

	def __sub__(self, other):
		return IntTensor((self.value.asnumpy() - other.value.asnumpy()) % encodeFP32.module(),internal = True)

	def __mul__(self, other):
		if isinstance(other, int):
			return IntTensor((self.value.asnumpy() * other) % encodeFP32.module(), internal = True)
		else:
			ans = IntTensor((self.mul(self.value, other.value)).asnumpy()  % encodeFP32.module(),internal = True) 
			return ans

	def __truediv__(self, other):
		if isinstance(other, IntTensor):
			return IntTensor((self.div(self.value, other.value)).asnumpy() % encodeFP32.module(),internal = True) 
		elif isinstance(other, int):
			return IntTensor(self.value.asnumpy() / other,internal = True)

	def __eq__(self, other):
		return (self.value.asnumpy() == other.value.asnumpy()).all()
	
	def __neg__(self):
		return IntTensor(-self.value.asnumpy() % encodeFP32.module(),internal = True)
	
	def deserialization(self):
		return self.value.asnumpy().tolist()
	
	def to_native(self):
		return Tensor(self.value, dtype = mindspore.float32) / encodeFP32.scale_size()

	def __repr__(self):
		return "IntTensor({})".format(self.value)
	def Matmul(self, other):
		'''
		fluent interface
		'''
		#self.value = self.matmul(self.value, other.value).asnumpy() / encodeFP32.scale_size() % encodeFP32.module()
		self.value = Tensor(np.dot(self.value.asnumpy(), other.value.asnumpy()) % encodeFP32.module(), dtype = mindspore.int32)
		return self
	def im2col(self, h, w, padding, stride):
		'''
		self -> col
		TODO:返回一个新的展开的IntTensor
		'''
		pass 


	def Conv(self, filters, stride, padding):
		'''
		TODO：使用matmul得到卷积结果的Tensor
		'''
		pass


class PrivateTensor:
	'''
	args:
		shared: if it need share
		dispatch: if it need share, how to share(supported by protocol)
		tensor: data

	method:
		open: (supported by protocol)
	object:
		__value: IntTensor
		__store_value: [IntTensor]
	'''
	def __init__(self, **kwargs):
		'''
		Check whether the parameter protocol is included, 
		if included, get the function from the protocol
		'''
		if "protocol" in kwargs:
			self.protocol = kwargs["protocol"]
		else:
			if __debug__:
				pass#print("use default protocol")
			from protocol.test_protocol import Protocol
			self.protocol = Protocol
		if "shared" in kwargs:
			#share a public tensor
			if "dispatch" not in kwargs and self.protocol == None:
				raise NameError("need keyword dispatch")
			if "tensor" not in kwargs:
				raise NameError("need keyword tensor!")

			self.tensor = self.check_tensor(kwargs)

			if self.protocol:
				self.__value, self.__store_value = self.protocol.dispatch(self.tensor)
			else:
				self.__value, self.__store_value = kwargs["dispatch"](self.tensor)

		else:
			if "tensor" not in kwargs:
				raise NameError("need keyword tensor!")
			self.__value = self.check_tensor(kwargs)
			self.__store_value = []
			#public -> private
		self.shape = self.__value.shape
	def check_tensor(self, dictory):
		tensor = dictory["tensor"]
		if isinstance(tensor, Tensor) or isinstance(tensor, list):
			#explicit convert
			if "internal" not in dictory:
				raise RuntimeError("Explicit declaration is required when converting from primitive type to private type!")
			return IntTensor(tensor,internal=dictory["internal"])
		elif isinstance(tensor, IntTensor):
			return tensor
		else:
			raise TypeError("tensor should be a serializable type")

	def deserialization(self):
		return self.__value.deserialization()
		
	def convert_public(self):
		return self.__value
	
	def share(self):
		if isinstance(self.__store_value[0], PrivateTensor):
			return self.__store_value
		elif isinstance(self.__store_value[0], IntTensor):
			return [PrivateTensor(tensor = value) for value in self.__store_value]


	def check_open(self):
		'''
		check if the shares meet the open requirements
		'''
		if self.protocol == None:
			raise RuntimeError("Initialize tensor without protocol")

		return self.protocol.check_open(self.__value, self.__store_value)
	
	'''
	args:
		reveal: s_0,s_1,s_2 -> s
	'''
	@staticmethod
	def open(*args, **kwargs):
		if "reveal" not in kwargs:
			raise NameError("need keyword reveal")
		for ptensor in args:
			return kwargs["reveal"](args)

	def add_value(self, value):
		self.__store_value.append(value)
		

	def open(self, composite = None):
		if isinstance(self.__store_value[0], PrivateTensor):
			__store_value = [tens.convert_public() for tens in self.__store_value]
		elif isinstance(self.__store_value[0], IntTensor):
			__store_value = self.__store_value
		if composite == None:
			return self.protocol.composite(self.__value, __store_value)
		elif not callable(composite):
			raise TypeError("composite should be function")
		elif not self.check_open():
			raise IndexError("value length error")
		else:
			return composite(self.__value, __store_value)


	'''
	TODO: operation overwriting()
	'''
	def __add__(self, other):
		if isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = self.__value + other.convert_public())
		elif isinstance(other, int):
			pass
		elif isinstance(other, IntTensor):
			return PrivateTensor(tensor = self.__value + other)


	def __sub__(self, other):
		if isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = self.__value - other.convert_public())
		elif isinstance(other, int):
			pass
		elif isinstance(other, IntTensor):
			return PrivateTensor(tensor = self.__value - other)


	def __mod__(self, other):
		pass

	def __mul__(self, other):
		if isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = self.__value * other.convert_public())
		elif isinstance(other, IntTensor) or isinstance(other, int):
			return PrivateTensor(tensor = self.__value * other)
		else:
			raise TypeError("Does not support multiplication of privateTensor and {}".format(type(other)))

	def __truediv__(self, other):
		if isinstance(other, int):
			return PrivateTensor(tensor = self.__value / other)
		elif isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = self.__value / other.convert_public())
	def __floordiv__(self, other):
		pass


	
