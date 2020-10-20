'''
Factory for random/decode/encode/...

'''
from mindspore import Tensor
import numpy as np


class Encode:
	def __init__(self, int_precision: int, frac_precision: int, base = 2):
		self.int_precision = int_precision
		self.frac_precision = frac_precision
		self.base = base
	def scale_size(self):
		return 2**self.frac_precision

	def module(self):
		return 2**(self.frac_precision + self.int_precision)
encodeFP32 = Encode(12,12)


class Factory:
	@staticmethod
	def gen_uniform(module = encodeFP32.module()):
		#TODO: replace with secure random!!!
		import random
		return random.randrange(0, module)
	@staticmethod
	def get_uniform(shape:list, module = encodeFP32.module()):
		return np.random.randint(module,size=shape)

