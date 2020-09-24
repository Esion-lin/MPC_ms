
from .tensor import PrivateTensor

class VarPool:
	def __init__(self, ctype, **kwargs):
		self.__dict__ = kwargs
		self.ctype = ctype
	def check_list(self, arr:list):
		for i in arr:
			if not isinstance(i, self.ctype):
				return False
		return True

	def __len__(self):
		return len(self.__dict__)

	def __setitem__(self, key, value):
		if isinstance(value, self.ctype):
			self.__dict__[key] = value
		elif isinstance(value, list) and self.check_list(value):
			self.__dict__[key] = value
		else:
			raise TypeError("need {} type".format(self.ctype.__name__))

	def __getitem__(self, key):
		return self.__dict__[key]
		
	def __contains__(self, key):
		return key in self.__dict__


	

__tensor_pool__ = VarPool(PrivateTensor)

def get_pool():
	return __tensor_pool__

def set_pool(pool):
	global __tensor_pool__
	__tensor_pool__ = pool
	