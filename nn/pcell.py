from mindspore.nn import Cell
from .trait import Trait
import abc
class PrivateCell(abc.ABC):
	'''
	input -> Shelling(privateTensor -> Tensor)
	output -> Packing(Tensor -> privateTensor)
	
	'''
	def __init__(self):
		pass
		#define env
	@abstractmethod
	def construct(self, input):
		#Separate network and graph
		pass

	def __setattr__(self, name, value):
		cells = self.__dict__.get('_cells')
		pcells = self.__dict__.get('_pcells')
		traits = self.__dict__.get("_traits")
		if isinstance(value, Cell):
			if cells is None:
				raise AttributeError("Can not assign cells before PrivateCell.__init__() call.")
			if name in self.__dict__:
				del self.__dict__[name]
			if traits and name in traits:
				raise TypeError("Expected type is Trait, but got Cell.")
			cells[name] = value
		elif isinstance(value, Trait):
			if traits is None:
				raise AttributeError("Can not assign traits before PrivateCell.__init__() call.")
			if name in self.__dict__:
				del self.__dict__[name]
			if cells and name in cells:
				raise TypeError("Expected type is Cell, but got Trait.")
			traits[name] = value
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
	
	def __call__(self, *args):
		self.construct(args)
	
	@abstractmethod
	def get_grad(self, input):
		pass
	
	@abstractmethod
	def construct_extractor(self):
		pass

	@abstractmethod
	def set_weight(self):
		pass