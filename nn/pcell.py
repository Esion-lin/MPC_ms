from mindspore.nn import Cell
from .trait import Trait
import abc
from abc import abstractmethod
class PrivateCell(abc.ABC):
	'''
	input -> Shelling(privateTensor -> Tensor)
	output -> Packing(Tensor -> privateTensor)
	
	'''
	def __init__(self, **kwargs):
		if "name" in kwargs:
			self.name = kwargs["name"] 	# 用于进行范围管理
		self.train = False
		self.pcells_list = []
		#define env
	@abstractmethod
	def construct(self, *args, **kwargs):
		#Separate network and graph
		pass

	def __setattr__(self, name, value):
		if name == "weight":
			object.__setattr__(self, "need_weight", True)
		cells = self.__dict__.get('_cells')
		pcells = self.__dict__.get('_pcells')
		if isinstance(value, Cell):
			if cells is None:
				raise AttributeError("Can not assign cells before PrivateCell.__init__() call.")
			if name in self.__dict__:
				del self.__dict__[name]
			if traits and name in traits:
				raise TypeError("Expected type is Trait, but got Cell.")
			cells[name] = value
		elif isinstance(value, PrivateCell):
			self.pcells_list.append(value)
			object.__setattr__(self, name, value)
		else:
			object.__setattr__(self, name, value)

	def __getattr__(self, name):
		if '_cells' in self.__dict__:
			cells = self.__dict__['_cells']
			if name in cells:
				return cells[name]
		if '_traits' in self.__dict__:
			traits = self.__dict__['_traits']
			if name in traits:
				return traits[name]
	
	def __call__(self, *args,**kwargs):
		return self.construct(*args, **kwargs)

	def backward(self, err, opt):
		for layer in self.pcells_list[::-1]:
			err = layer.backward(err,opt)
	# @abstractmethod
	# def get_grad(self, input):
	# 	pass
	
	# @abstractmethod
	# def construct_extractor(self):
	# 	pass

	def set_weight(self):
		pass
	
	def inject(self, weight):
		#if self.__dict__.get('_pcells')
		if "need_weight" in self.__dict__:
			self.weight = weight[0]
			weight = weight[1:]
		pcells = self.__dict__.get('_pcells')
		for ele in pcells:
			ele.inject(weight[0])
			weight = weight[1:]
	
	def set_train(self, flag):
		self.train = flag