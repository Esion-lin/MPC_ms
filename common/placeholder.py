from .var_pool import get_pool as get_var_pool
import time

class Placeholder:
	def __init__(self, name):
		self.name = name

	def fill(self):
		while not self.check():
			time.sleep(1)
		return get_var_pool()[self.name]


	def check(self):
		return self.name in get_var_pool()