import json
from player import Player
class Config:
	def __init__(self, player_list = None, filename = None):
		if filename is not None:
			self.load(filename)
		elif isinstance(player_list, dict):
			self._player_list = player_list
		else:
			self.players = {}
			return
		self.players = {name:Player(name, self._player_list[name]) for name in self._player_list}


	def load(self, filename):
		with open(filename, "r") as f:
			self._player_list = json.load(f)


	def __serializable(self):
		self._player_list = {name: self.players[name].host + ':' + self.players[name].port for name in self.players}

	def save(self, filename):
		self.__serializable()
		with open(filename, "w") as f:
			json.dump(self._player_list, f)

	def add_player(self, player):
		assert isinstance(player, Player)
		if not player.name in self.players:
			self.players[player.name] = player


__config__ = Config()
def get_config():
	return __config__
def set_config(config):
	global __config__
	__config__ = config

if __name__ == "__main__":
	config = Config(filename = "./config")
	if __debug__:
		print(config.players["esion"].host)
