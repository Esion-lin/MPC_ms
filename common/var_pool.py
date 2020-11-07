# -*- coding: utf-8 -*-
from .tensor import PrivateTensor
import re
from .event_queue import ins_messs_que
'''
用于triples的管理
'''
class TupleManage:
	def __init__(self):
		self.t_dict = {}
	def unpack(self, key):
		if re.match("\[.*\]", key):
			return key[1:-1]
	def __setitem__(self, key, value):
		key = self.unpack(key)
		self.t_dict[key] = value
	def __getitem__(self, key):
		key = self.unpack(key)
		return self.t_dict[key]
	def __contains__(self, key):
		key = self.unpack(key)
		return key in self.t_dict
	def pop(self):
		key = min(self.t_dict.keys())
		value = self.t_dict[key]
		del self.t_dict[key]
		return value
	def __delitem__(self, key):
		key = self.unpack(key)
		del self.t_dict[key]

class VarPool:
		
	'''
	key: "[alphabet]": 普通变量； "[\[alphabet\]]"：数组变量
			^							^
			对应单个PrivateTensor		对应PrivateTensor数组
	'''

	def __init__(self, ctype, **kwargs):
		self.__dict__ = kwargs
		self.ctype = ctype
		self.tm = TupleManage()
	
	def check_key(self, key):
		if re.match("\[.*\]", key):
			return True
		return False

	def check_list(self, arr:list):
		for i in arr:
			if not isinstance(i, self.ctype):
				return False
		return True

	def __len__(self):
		return len(self.__dict__)

	def __setitem__(self, key, value):
		lockstitch = ins_messs_que.set_ele(key)
		lockstitch.unlock()
		if isinstance(value, self.ctype):
			value.set_name(key)
			self.__dict__[key] = value
		elif isinstance(value, list) and self.check_list(value):
			for ele in value:
				ele.set_name(key)
			self.tm[key] = value
		else:
			raise TypeError("need {} type".format(self.ctype.__name__))

	def __getitem__(self, key):
		if self.check_key(key):
			return self.tm[key]
		return self.__dict__[key]
		
	def __contains__(self, key):
		if self.check_key(key):
			return key in self.tm
		return key in self.__dict__
	def __delitem__(self, key):
		if self.check_key(key):
			del self.tm[key]
		else:
			del self.__dict__[key]





__tensor_pool__ = VarPool(PrivateTensor)

def get_pool():
	return __tensor_pool__

def set_pool(pool):
	global __tensor_pool__
	__tensor_pool__ = pool
	