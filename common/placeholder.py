from .var_pool import get_pool as get_var_pool
import time
import re
from common.tensor import PrivateTensor,IntTensor
import inspect
class Placeholder:
	'''
	#### 同步
	需要做到和变量池的状态同步，会出现3种情况
	1. 未初始化的Placeholder，而变量还未由用户输入
	2. 变量已经由用户输入，Placeholder未定义
	3. 初始化的Placeholder， 变量已经填入
	4. 未初始化的Placeholder，而变量已经由用户输入
	#### API
	fill(): 查询变量池，将对应的数据填入Placeholder
	inject(): 将Placeholder的内容填充到变量池
	set_value(ptensor): 使用ptensor设置Placeholder的内容
	
	'''
	def __init__(self, name = None, shape = None, value = None):
		self.name = name
		self.is_list = True if re.match("\[.*\]", name) else False 
		if value is not None:
			self.is_fill = True
			self.value = value
		else:
			self.is_fill = False
			# 如果该变量已经在变量池中，则直接填入。
			if self.check():
				self.fill()
		self.locked = False
	@property
	def value(self):
		return get_var_pool()[self.name]
	
	@value.setter
	def value(self, _value):
		self.is_fill = True
		get_var_pool()[self.name] = _value

	@property
	def shape(self):
		if isinstance(self.value, list):
			return [len(self.value)].extend(self.value[0].shape)
		return self.value.shape
	
	def reshape(self, shape):
		if isinstance(self.value, list):
			raise TypeError("Not support reshape for the type of list")
		self.value.reshape(shape)
		
	def fill(self):
		if self.is_fill:
			return self.value
		while not self.check():
			time.sleep(0.01)
		self.is_fill = True
		return self.value
		
	def check(self):
		return self.name in get_var_pool()

	def __getitem__(self, key):
		if not self.is_list:
			raise TypeError("PrivateTensor cannot be accessed by subscripts!")
		if not isinstance(key, int):
			raise TypeError("index should be type of int!!")
		return self.value[key]
	
	def __len__(self):
		if not self.is_list:
			return 1
		else:
			return len(self.value)

	def rename(self, name):
		if self.name in get_var_pool():
			get_var_pool()[name] = get_var_pool()[self.name]
			if self.locked == True:
				get_var_pool().lock(name)
				get_var_pool().unlock(self.name)
		self.name = name	

	def lock(self):
		#上锁，防止被清理
		get_var_pool().lock(self.name)
		self.locked = True

	def unlock(self):
		#上锁，防止被清理
		get_var_pool().unlock(self.name)
		self.locked = False

	def __iter__(self):
		if not self.is_list:
			raise TypeError("PrivateTensor cannot be accessed by subscripts!")
		return iter(self.value)

	# def __getattribute__(self, name):
	# 	value = object.__getattribute__(self,name)
	# 	che_list = ["check","fill","__init__","__dict__", "name", "set_value", "shape","inject"]
	# 	if name not in che_list and not self.check():
	# 		print("Init PlaceHolder first!(by running \"fill\" method)")
	# 	else:
	# 		return value
	
	@staticmethod
	def register(variable, name):
		get_var_pool()[name] = variable
		return Placeholder(name = name,shape = variable.shape)
	
	def inject(self):
		if self.name in get_var_pool():
			get_var_pool()[self.name].set_value(self.value.convert_public())
		else:
			get_var_pool()[self.name] = self.value

	def erase(self):
		if name in get_var_pool():
			del get_var_pool()[name]
		del self
		return None

	def __dispatch_form(self, a, b, opt, *args):
		def add(a, b):
			return a + b
		def sub(a,b):
			return a - b
		def mul(a,b):
			return a * b
		def truediv(a,b):
			return a / b
		def conv(a,b,*args):
			return a.Conv(b,*args)
		switch = {	"add":add,
					"sub":sub,
					"mul":mul,
					"truediv":truediv,
					"conv":conv}
		return switch[opt](a, b, *args)

	def dispatch(self, other, opt, *args, reverse = False):
		assert self.check()
		if reverse:
			newptensor = self.__dispatch_form(other, self.fill(), opt, *args)
		elif isinstance(other, Placeholder):
			assert other.check()
			newptensor = self.__dispatch_form(self.fill(), other.fill(), opt, *args)
		else:
			newptensor = self.__dispatch_form(self.fill(), other, opt, *args)
		if self.tmp_name is None:
			raise RuntimeWarning("在使用Placeholder进行直接计算时，请使用with语句，否则极有可能计算错误")
		if isinstance(other, int) or "name" not in other.__dict__:
			return Placeholder.register(newptensor,"{}_{}_{}_cons".format(self.tmp_name, self.name, opt))
		return Placeholder.register(newptensor,"{}_{}_{}_{}".format(self.tmp_name, self.name, opt, other.name))

	def __add__(self, other):
		return self.dispatch(other,"add")
	def __sub__(self, other):
		return self.dispatch(other,"sub")
	def __mul__(self, other):
		return self.dispatch(other,"mul")
	def __truediv__(self, other):
		return self.dispatch(other,"truediv")
	def Conv(self, filters, stride, padding):
		return self.dispatch(filters,"conv",stride, padding)

	def __radd__(self, other):
		return self.dispatch(other,"add", reverse = True)
	def __rsub__(self, other):
		return self.dispatch(other,"sub", reverse = True)
	def __rmul__(self, other):
		return self.dispatch(other,"mul", reverse = True)
	def __rtruediv__(self, other):
		return self.dispatch(other,"truediv", reverse = True)
	def rConv(self, filters, stride, padding):
		return self.dispatch(filters,"conv",stride, padding, reverse = True)
