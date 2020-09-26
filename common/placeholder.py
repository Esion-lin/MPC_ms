from .var_pool import get_pool as get_var_pool
import time

class Placeholder:
	def __init__(self, name, shape = None):
		self.name = name
		self.shape = shape

	def fill(self):
		while not self.check():
			time.sleep(1)
		self.value = get_var_pool()[self.name]
		if self.shape != None:
			if self.shape != self.value.shape:
				raise IndexError("except shape {}, but got {}!".format(self.shape,self.value.shape))
		else:
			self.shape = value.shape
		return self.value
		
	def check(self):
		return self.name in get_var_pool()