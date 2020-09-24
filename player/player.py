''' player'''

import re
from p2pnetwork.node import Node

class Player:
	'''
	'''
	def __init__(self, name, host, port = 8001):
		self.name = name
		self.port = port
		self.host = host
		self.on = False
	
	def __setattr__(self, name, value):
		if name == "host":	
			if len(value.split(':')) == 2:
				self.port = int(value.split(':')[1])
				value = value.split(':')[0]
			elif len(value.split(':')) != 1:
				raise TypeError("illegal host")

			#check ip
			p = re.compile('^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
			if not p.match(value):
				raise TypeError("illegal host")
		elif name == "port":
			if not isinstance(value, int):
				raise TypeError("The port type needs to be int!")
		object.__setattr__(self, name, value)
	
	#TODO callback logical
	def start_node(self, node_callback):
		self.node = Node(self.host, self.port, node_callback)
		self.node.start()
		self.on = True

	def destroy(self):
		if self.on:
			self.node.stop()

__player__ = Player("test", "127.0.0.1:8001")

def get_player():
	return __player__
def set_player(player):
	global __player__
	__player__ = player
