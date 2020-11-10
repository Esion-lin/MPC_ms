from .var_pool import get_pool as get_var_pool
import time
import re
from common.tensor import PrivateTensor
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
	def __init__(self, name, shape = None):
		self.is_fill = False
		self.name = name
		self.shape = shape
		self.is_list = True if re.match("\[.*\]", name) else False 
		# 如果该变量已经在变量池中，则直接填入。
		if self.check():
			self.fill()
		

	def fill(self):
		if self.is_fill:
			return self.value
		while not self.check():
			time.sleep(0.01)
		self.value = get_var_pool()[self.name]
		self.is_fill = True
		if self.is_list ^ isinstance(self.value,list):
			raise TypeError("except type {}, but got {}!".format("list" if self.is_list else "PrivateTensor", self.value))
		if self.shape != None:
			if isinstance(self.value, list):
				if self.shape != self.value[0].shape:
					raise IndexError("except shape {}, but got {}!".format(self.shape,self.value.shape))
			elif self.shape != self.value.shape:
				raise IndexError("except shape {}, but got {}!".format(self.shape,self.value.shape))
		else:
			if isinstance(self.value, list):
				self.shape = self.value[0].shape
			else:
				self.shape = self.value.shapePrivateTensor
			raise TypeError("PrivateTensor cannot be accessed by subscripts!")
		if not isinstance(key, int):
			raise TypeError("index should be type of int!!")
		return self.value[key]
	
	def __len__(self):
		if not self.is_list:
			return 1
		else:
			return len(self.value)
	def set_value(self, ptensor, force_sys = False):
		self.value = ptensor
		self.shape = ptensor.shape
		self.is_fill = True
		if force_sys:
			self.inject()
		else:
			if self.check():
				raise RuntimeError("The variable exists in the variable pool and cannot be set!")
			else:
				self.inject()
			


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
	@staticmethod
	def __dispatch_form(a, b, opt, *args):
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
			newptensor = __dispatch_form(other, self.fill(), opt, *args)
		elif isinstance(other, Placeholder):
			assert other.check()
			newptensor = __dispatch_form(self.fill(), other.fill(), opt, *args)
		else:
			newptensor = __dispatch_form(self.fill(), other, opt, *args)
		if self.tmp_name is None:
			raise RuntimeWarning("在使用Placeholder进行直接计算时，请使用with语句，否则极有可能计算错误")
		if other.name is None:
			raise RuntimeError("尝试使用为注册的变量")
		return Placeholder.register(newptensor,"{}_{}_{}_{}".format(self.tmp_name, self.name, opt, other.name))

	def __add__(self, other):
		self.dispatch(other,"add")
	def __sub__(self, other):
		self.dispatch(other,"sub")
	def __mul__(self, other):
		self.dispatch(other,"mul")
	def __truediv__(self, other):
		self.dispatch(other,"truediv")
	def Conv(self, filters, stride, padding):
		self.dispatch(filters,"conv",stride, padding)

	def __radd__(self, other):
		self.dispatch(other,"add", reverse = True)
	def __rsub__(self, other):
		self.dispatch(other,"sub", reverse = True)
	def __rmul__(self, other):
		self.dispatch(other,"mul", reverse = True)
	def __rtruediv__(self, other):
		self.dispatch(other,"truediv", reverse = True)
	def rConv(self, filters, stride, padding, reverse = True):
		self.dispatch(filters,"conv",stride, padding)
