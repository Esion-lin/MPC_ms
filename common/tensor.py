from mindspore import Tensor
import mindspore
import numpy as np
from crypto.factory import encodeFP32
from mindspore.ops import operations as P
from mindspore import nn
from .event_queue import add_share_que
from common.fake.fakeconv import conv
class IntTensor:
	'''

	'''
	#TODO: Access management
	def __init__(self, tensor, internal = False,**kwargs):
		#TODO :添加module
		if isinstance(tensor, IntTensor):
			#copy
			if "name" in kwargs:
				self.name = kwargs["name"] 
			self.value = tensor.value
		elif isinstance(tensor, PrivateTensor):
			#explicit convert
			self.name = tensor.name
			self.value = tensor.convert_public().value
		#   ^ top down复制或类型转换的方法 
		elif internal == False:
			if isinstance(tensor, Tensor):
				#transfer to int
				self.value = Tensor((tensor * encodeFP32.scale_size).asnumpy() % encodeFP32.module, dtype = mindspore.int32)
			elif isinstance(tensor, list):
				self.value = Tensor(np.asarray(tensor) * encodeFP32.scale_size % encodeFP32.module, dtype = mindspore.int32)
			else:
				self.value = Tensor(tensor * encodeFP32.scale_size % encodeFP32.module , dtype = mindspore.int32)
		else:
			if isinstance(tensor, Tensor):
				#transfer to int
				self.value = Tensor(tensor.asnumpy(), dtype = mindspore.int32)
			else:
				self.value = Tensor(tensor, dtype = mindspore.int32)
		if "module" in kwargs:
			self.module =  kwargs["module"]
		# self.inv = P.Invert()
		
		# self.div = P.FloorDiv()
		# self.matmul = P.MatMul()

	#TODO: 需要添加对其他类型数据计算的重载
	def convert_float32(self):
		return Tensor(self.value.asnumpy().astype(np.float32),dtype = mindspore.float32)
	@property
	def shape(self):
		return self.value.shape

	def __add__(self, other):
		self.add = P.TensorAdd()
		if isinstance(other, IntTensor):
			return IntTensor((self.add(self.value, other.value)).asnumpy() % encodeFP32.module,internal = True)
		elif isinstance(other, PrivateTensor):
			# 数据与share相加得到share
			return PrivateTensor(tensor = self + other.convert_public())
		else:
			raise NotImplementedError("未实现的方法")
		#self.value = Tensor(self.add(self.value, other.value).asnumpy() % encodeFP32.module)
		#return self

	def __sub__(self, other):
		if not isinstance(other,IntTensor):
			raise NotImplementedError("未实现的方法")
		return IntTensor((self.value.asnumpy() - other.value.asnumpy()) % encodeFP32.module,internal = True)

	def __mul__(self, other):
		self.mul = P.Mul()
		if isinstance(other, int):
			return IntTensor((self.value.asnumpy() * other) % encodeFP32.module, internal = True)
		elif not isinstance(other, IntTensor):
			raise NotImplementedError("未实现的方法")
		else:
			ans = IntTensor((self.mul(self.value, other.value)).asnumpy()  % encodeFP32.module,internal = True) 
			return ans

	def __truediv__(self, other):
		self.div = P.FloorDiv()
		if isinstance(other, IntTensor):
			return IntTensor((self.div(self.value, other.value)).asnumpy() % encodeFP32.module,internal = True) 
		elif isinstance(other, int):
			return IntTensor(self.value.asnumpy() / other,internal = True)
		else:
			raise NotImplementedError("未实现的方法")

	def __eq__(self, other):
		if isinstance(other, IntTensor):
			return (self.value.asnumpy() == other.value.asnumpy()).all()
		else:
			raise NotImplementedError("未实现的方法")

	def __neg__(self):
		return IntTensor(-self.value.asnumpy() % encodeFP32.module,internal = True)
	def __mod__(self, other):
		if isinstance(other, IntTensor):
			return IntTensor(self.value.asnumpy() % other.asnumpy(),internal = True) 
		elif isinstance(other, int):
			return IntTensor(self.value.asnumpy() % other,internal = True)
		else:
			raise NotImplementedError("未实现的方法")
	def deserialization(self):
		return self.value.asnumpy().tolist()
	
	def to_native(self):
		tmp = self.value.asnumpy()
		ans = np.where(tmp < encodeFP32.threshold, tmp, tmp-encodeFP32.threshold*encodeFP32.base)
		return Tensor(ans, dtype = mindspore.float32) / encodeFP32.scale_size

	def __repr__(self):
		return "IntTensor({})".format(self.value)
	def Matmul(self, other):
		self.matmul = P.MatMul()
		#self.value = self.matmul(self.value, other.value).asnumpy() / encodeFP32.scale_size % encodeFP32.module
		return IntTensor(np.dot(self.value.asnumpy(), other.value.asnumpy()) % encodeFP32.module, internal = True)

	def Conv(self, filters, stride, padding):
		'''
		
		'''
		if not isinstance(filters, IntTensor):
			return filters.rConv(self, stride, padding)
		ans =  IntTensor(conv(self.value.asnumpy(), filters.value.asnumpy(), padding=padding, stride=stride), internal = True)
		#self.cov = nn.Conv2d(self.shape[1], filters.shape[-3], filters.shape[-2:], stride,pad_mode = "pad", padding = padding, weight_init=filters.to_native())
		# #																																^需要修改为int
		#ans =  IntTensor((self.cov(self.to_native())).asnumpy()*encodeFP32.scale_size, internal = False)% encodeFP32.module
		# #						^目前不支持整数，需要修改成整数
		print("check tensor cov", self,"\n", filters,"\n", ans,"\n", encodeFP32.module)
		ans = ans  % encodeFP32.module
		return ans
	def reshape(self, shape):
		reshape = P.Reshape()
		self.value = reshape(self.value, shape)
	
	def save(self, name):
		np.save(name,self.value.asnumpy())
	def load(self, name):
		self.value = Tensor(np.load(name), dtype = mindspore.int32)

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
		if "name" in kwargs:
			self.name = kwargs["name"]

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
			else:printhare_que.set_ele(self.name).unlock()
		else:
			if "tensor" not in kwargs:
				raise NameError("need keyword tensor!")
			self.__value = self.check_tensor(kwargs)
			self.__store_value = []
			#public -> private
		
	@property
	def shape(self):
		return self.__value.shape

	def __repr__(self):
		return "PrivateTensor({})".format(self.__value)

	def set_name(self, name):
		self.name = name
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
		if self.check_open():
			add_share_que.set_ele(self.name).unlock()

	
	def set_value(self, value):
		assert isinstance(value, IntTensor)
		self.__value = value

	def open(self, composite = None):
		if isinstance(self.__store_value[0], PrivateTensor):
			__store_value = [tens.convert_public() for tens in self.__store_value]
		elif isinstance(self.__store_value[0], IntTensor):
			__store_value = self.__store_value
		if composite == None:
			return IntTensor(self.protocol.composite(self.__value, __store_value),name = self.name)
		elif not callable(composite):
			raise TypeError("composite should be function")
		elif not self.check_open():
			raise IndexError("value length error")
		else:
			return IntTensor(composite(self.__value, __store_value),name = self.name)


	'''
	TODO: operation overwriting()
	'''
	def __add__(self, other):
		if isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = self.__value + other.convert_public())
		elif isinstance(other, int):
			# TODO 实现加上一个Int变量的操作
			pass
		elif isinstance(other, IntTensor):
			return PrivateTensor(tensor = self.__value + other)
		else:
			raise NotImplementedError("调用未实现的方法")
	
	def __radd__(self, other):
		return self.__add__(other)

	def __sub__(self, other):
		if isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = self.__value - other.convert_public())
		elif isinstance(other, int):
			pass
		elif isinstance(other, IntTensor):
			return PrivateTensor(tensor = self.__value - other)
		else:
			raise NotImplementedError("调用未实现的方法")

	def __rsub__(self, other):
		if isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = other.convert_public() - self.__value)
		elif isinstance(other, int):
			pass
		elif isinstance(other, IntTensor):
			return PrivateTensor(tensor = other - self.__value)
	
	def __mod__(self, other):
		pass

	def __mul__(self, other):
		if isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = self.__value * other.convert_public())
		elif isinstance(other, IntTensor) or isinstance(other, int):
			return PrivateTensor(tensor = self.__value * other)
		else:
			raise NotImplementedError("调用未实现的方法")
	
	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		if isinstance(other, int):
			return PrivateTensor(tensor = self.__value / other)
		elif isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = self.__value / other.convert_public())
		elif isinstance(other, IntTensor):
			return PrivateTensor(tensor = self.__value / other)
		else:
			raise NotImplementedError("调用未实现的方法")
	
	def __rtruediv__(self,other):
		if isinstance(other, PrivateTensor):
			return PrivateTensor(tensor = other.convert_public() / self.__value )
		elif isinstance(other, IntTensor):
			return PrivateTensor(tensor = other / self.__value)
		
		
	def __floordiv__(self, other):
		pass
	def rConv(self, img, stride, padding):
		return PrivateTensor(tensor = img.Conv(self.convert_public(), stride, padding))
	def Conv(self, filters, stride, padding):
		if isinstance(filters, PrivateTensor):
			filters = filters.fill()
		elif not isinstance(filters, IntTensor):
			return filters.rConv(self, stride, padding)
		return PrivateTensor(tensor = self.__value.Conv(filters, stride, padding))
	def reshape(self, shape):
		self.__value.rehape(shape)
		for ele in self.__store_value:
			ele.rehape(shape)
	
	def __neg__(self):
		return PrivateTensor(tensor = -self.__value)
	
	def save(self, name):
		self.__value.save(name)
	def load(self, name):
		self.__value.load(name)

# wrap with Tensor class
class Conv2d(nn.Conv2d):
	def __init__(self, *args, **kwargs):
		self.private = False
		if isinstance(kwargs["weight_init"], PrivateTensor):
			kwargs["weight_init"] = kwargs["weight_init"].convert_public().value
			self.private = True
		elif isinstance(kwargs["weight_init"], IntTensor):
			kwargs["weight_init"] = kwargs["weight_init"].value
		super(Conv2d,self).__init__(*args,**kwargs)
	def __call__(self, *args):
		for i in range(len(args)):
			if isinstance(args[i], PrivateTensor):
				self.private = True
				args[i] = args[i].convert_public().value
			elif isinstance(args[i], IntTensor):
				args[i] = args[i].value
		ans = super().__call__(*args)
		if self.private:
			return PrivateTensor(tensor = IntTensor(ans, internal = True))
		else:
			return IntTensor(ans, internal = True)
