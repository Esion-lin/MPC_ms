from .var_pool import get_pool as get_var_pool
import time
import re

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
			time.sleep(1)
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
				self.shape = self.value.shape
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
		