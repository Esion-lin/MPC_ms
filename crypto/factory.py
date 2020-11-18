'''
Factory for random/decode/encode/...

'''
from mindspore import Tensor
import numpy as np


class Encode:
	def __init__(self, int_precision: int, frac_precision: int, module = 2**23, base = 2):
		self.int_precision = int_precision
		self.frac_precision = frac_precision
		self.base = base
		self.__module = module
	@property
	def scale_size(self):
		return 2**self.frac_precision
	
	@property
	def module(self):
		return self.__module

	@module.setter
	def module(self, value):
		self.__module = value
	def pos(self, num):
		if num < 2 ** 25:
			return True
		return False
encodeFP32 = Encode(10,8)


class Factory:
	@staticmethod
	def gen_uniform(module = encodeFP32.module):
		#TODO: replace with secure random!!!
		import random
		return random.randrange(0, module)
	@staticmethod
	def get_uniform(shape:list, minum = 0, module = encodeFP32.module):
		return np.random.randint(minum,high = module,size=shape)
	@staticmethod
	def get_sym_random(seed):
		return hash(seed)
		#简单的实现，后面需要修改

class Counter:
	counter = 0
	@classmethod
	def clear(cls):
		cls.counter = 0
	@classmethod
	def get_counter(cls):
		cls.counter += 1
		return cls.counter


