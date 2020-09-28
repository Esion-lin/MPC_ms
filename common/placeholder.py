from .var_pool import get_pool as get_var_pool
import time
import re

class Placeholder:
	def __init__(self, name, shape = None):
		self.name = name
		self.shape = shape
		self.is_list = re.match("\[.*\]", key)? True:False 

	def fill(self):
		while not self.check():
			time.sleep(1)
		self.value = get_var_pool()[self.name]
		if self.is_list ^ isinstance(self.value,list):
			raise TypeError("except type {}, but got {}!".format(self.is_list?"list":"PrivateTensor", self.value))
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
	
	def __iter__(self):
		if not self.is_list:
			raise TypeError("PrivateTensor cannot be accessed by subscripts!")
		return iter(self.value)

	def __getattribute__(self, name):
		value = object.__getattribute__(self,name)
		if value == None:
			raise AttributeError("key Error {}".format(name))
		che_list = ["check","fill","__init__","__dict__", "name"]
		if name not in che_list and not self.check():
			print("Init PlaceHolder first!(by running \"fill\" method)")
		else
			return value