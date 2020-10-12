'''
callback class

'''
from common import get_pool
from common.constant import ACTION
from functools import partial
from mindspore import Tensor
from common.tensor import IntTensor,PrivateTensor
import re
#Process the received data
'''
{
	"action":xxxx (int)
	"key":xxxx (str)
	"value":xxxx (tensor)
}

'''
def reveal(data):
	if "key" not in data:
		raise KeyError("no key in json!")
	if "value" not in data:
		raise KeyError("no value in json!")
	# global pool open variable
	key = data["key"]
	value = data["value"]
	myPool = get_pool()
	
	# TODO : filter for selization tensor
	if re.match("\[.*\]", key) != None:
		if len(myPool[key]) != len(value):
			raise IndexError("The shape of the local variable conflicts with the shape of the data accepted by the network!!")
		for i in range(len(myPool[key])):
			myPool[key][i].add_value(PrivateTensor(tensor = value[i], internal = True))
	else:
		myPool[key].add_value(PrivateTensor(tensor = value, internal = True))
		
	
def private_input(data):
	if "key" not in data:
		raise KeyError("no key in json!")
	if "value" not in data:
		raise KeyError("no value in json!")
	key = data["key"]
	value = data["value"]
	myPool = get_pool()
	if re.match("\[.*\]", key) != None:
		myPool[key] = [PrivateTensor(tensor = i, internal = True) for i in value]
	else:
		myPool[key] = PrivateTensor(tensor = value, internal = True)
	
switch = {
	ACTION.OPEN:reveal,
	ACTION.SHARE:private_input
}

class Dealer:
	'''
	deal with data received
	'''
	def __init__(self):
		pass
	def __call__(self, data):
		if "action" not in data:
			raise KeyError("no action in json!")
		switch[data["action"]](data)



class CallBack:
	def __init__(self, dealer = None):
		if dealer == None:
			self.dealer = partial(print, "data is ")
		else:
			self.dealer = dealer
	def __call__(self, event, main_node, connected_node, data):
		if event != 'node_request_to_stop': 
			if __debug__:
				print('Event: {} from main node {}: connected node {}: {}'.format(event, main_node.id, connected_node.id, data))
		if data != None and data != {}:
			self.dealer(data)





