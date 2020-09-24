
'''
conection pool
'''
from player import get_player
from config import get_config
class ConnectionPool:
	'''
	use local player'node initiate pool
	'''
	def __init__(self, node, players = None):
		self.node = node
		self.players = players
	
	def init(self, players):
		if self.players == None:
			self.players = players
		myName = get_player().name
		for name in self.players:
			if name == myName:
				continue
			else:
				if __debug__:
					print("connect to {}".format(name))
				self.node.connect_with_node(self.players[name].host, self.players[name].port)


	def check_connection(self):
		if self.players == None:
			raise RuntimeError("pool is not initialized!")
		if len(self.node.nodes_outbound) != len(self.players) - 1:
			print("reconnecting...")
			myName = get_player().name
			for (name,player) in self.players.items():
				if name == myName:
					continue
				else:
					if __debug__:
						print("reconnect to {}".format(name))
					self.node.connect_with_node(player.host, player.port)
			if len(self.node.nodes_outbound) != len(self.players) - 1:
				print("some servers not ready")
				return False
		else:
			return True

	def check_node(self, player):
		#TODO check if one node connection
		pass

	def broadcast(self, data):
		if not isinstance(data, dict):
			raise TypeError("data need to be json!")
		self.node.send_to_nodes(data)

	def send(self, data, player):
		for node in self.node.nodes_outbound:
			if node.host == player.host and node.port == player.port:
				self.node.send_to_node(node, data)
				return
		print("no such player")
__pool = None
def init_pool():
	global __pool
	__pool = ConnectionPool(get_player().node, get_config().players)
	__pool.init(players = None)


def get_pool():
	return __pool
