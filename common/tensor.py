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
	def __add__(self, other):
		return IntTensor((self.add(self.value, other.value)).asnumpy() % encodeFP32.module(),internal = True)

	def __sub__(self, other):
		return IntTensor((self.value.asnumpy() - other.value.asnumpy()) % encodeFP32.module(),internal = True)

	def __mul__(self, other):
		return IntTensor((self.mul(self.value, other.value) / encodeFP32.scale_size()).asnumpy() % encodeFP32.module(),internal = True) 

	def __div__(self, other):
		return IntTensor((self.div(self.value, other.value)).asnumpy() % encodeFP32.module(),internal = True) 
	def __eq__(self, other):
		return (self.value.asnumpy() == other.value.asnumpy()).all()

	def deserialization(self):
		return self.value.asnumpy().tolist()
	
	def to_native(self):
		return Tensor(self.value, dtype = mindspore.float32) / encodeFP32.scale_size()

	def __repr__(self):
		return "IntTensor({})".format(self.value)



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
				print("use default protocol in dispatch")
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
	TODO: operation overwriting(protocol dependence)
	
	def __and__(self, other):
		return Tensor(self.__value + other.convert_public())

	def __sub__(self, other):
		return Tensor(self.__value - other.convert_public())

	def __mod__(self, other):
		return

	def __mul__(self, other):
		return 

	def __floordiv__(self, other):
		return


	'''
