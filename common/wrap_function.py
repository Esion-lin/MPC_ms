from functools import wraps
from functools import partial
from player import get_player
from config import get_config
from .var_pool import get_pool as get_var_pool
from communication.pool import get_pool as get_net_pool
from .constant import ACTION
from .placeholder import Placeholder
import time
import re
from .event_queue import ins_messs_que
from .event_queue import add_share_que
def wrap_json(action, key, value):
	if re.match("\[.*\]", key):
		# array type
		return {
			"action": action,
			"key": key,
			"value": [i.deserialization() for i in value]
			}
	else:	
		return {
			"action": action,
			"key": key,
			"value": value.deserialization()
			}
class PlayerDecorator:
	
	def __init__(self, player):
		self.player = player
	
	def __call__(self):
		raise NotImplementedError

	def _pass(self):
		if __debug__:
			print("pass operation!")


	def check_player(self, name):
		if self.player.name == name:
			return True
		return False
	'''
	replace the placeholder with privateTensor
	'''
	def replace_(self, func = None):
		@wraps(func)
		def wrapper(*args, **kwargs):
			new_args = []
			new_kwargs = {}
			for value in args:
				if isinstance(value, Placeholder):
					new_args.append(value.fill())
				else:
					new_args.append(value)
			for key,value in kwargs.items():
				if isinstance(value, Placeholder):
					new_kwargs[key] = value.fill()
				else:
					new_kwargs[key] = value
			return func(*new_args, **new_kwargs)
		return wrapper

	def fill_(self, func = None):
		@wraps(func)
		def wrapper(*args, **kwargs):
			for value in args:
				if isinstance(value, Placeholder):
					value.fill()
			for key,value in kwargs.items():
				if isinstance(value, Placeholder):
					value.fill()
			return func(*args, **kwargs)
		return wrapper

	'''

	'''
	def from_(self, func = None, player_name = ""):
		if func is None:
			return partial(self.from_, player_name = player_name)

		@wraps(func)
		def wrapper(*args, **kwargs):
			
			if player_name != "":
				if self.check_player(player_name):
					return func(*args, **kwargs)
				else:
					return self._pass()
			return func(*args, **kwargs)
		return wrapper

	'''
	Distribute the result of func to each player
	func: -> share
	player_name: who run the func
	target: who get the result
	var_name: the variable stored name
	'''
	def to_(self, func = None, player_name = "", target = "", var_name = "x"):
		if func is None:
			return partial(self.to_, player_name = player_name, target = target, var_name = var_name)

		@wraps(func)
		def wrapper(*args, **kwargs):
			'''
			add var_name to func
			'''
			g_dict = func.__globals__
			g_dict["var_name"] = var_name
			''''''
			if player_name != "":
				while not get_net_pool().check_connection():
					print("cannot connect all node! reconnecting...")
					time.sleep(2)
				if self.check_player(player_name):
					#check args
					dispatch_var = func(*args, **kwargs)
					i = 0
					for (name,player) in get_config().players.items():
						if name != player_name:
							data = wrap_json(ACTION.SHARE, var_name, dispatch_var[i])
							# if __debug__:
								# print(data,player.host)
							get_net_pool().send(data = data, player = player)
							i = i + 1
					#TODO: placeholder / True
					return Placeholder(var_name)
				else:
					#wait io
					ins_messs_que.set_ele(var_name).stand()
					return Placeholder(var_name)
			#TODO: Three type of combines of player_name and target {"","" : "xxx","xxx" : "","xxx"}
					
		return wrapper




	def open_(self, func = None, player_name = "", var_name = ""):
		if func is None:
			return partial(self.open_, player_name = player_name, var_name = var_name)

		@wraps(func)
		def wrapper(*args, **kwargs):
			'''
			add var_name to func
			'''
			g_dict = func.__globals__
			g_dict["var_name"] = var_name
			''''''
			#send share to palyer
			if player_name != "":
				if not self.check_player(player_name):
					if not player_name in get_config().players:
						raise KeyError("player_name is not defined in config!")
					#check var_pool, get label data
					data = wrap_json(ACTION.OPEN, var_name, get_var_pool()[var_name])
					get_net_pool().send(data = data, player = get_config().players[player_name])
					return self._pass()
				else:
					get_var_pool().lock(var_name)
					#wait io
					#TODO 需要重新设计更合理的检查方法
					add_share_que.set_ele(var_name).stand()
					# if re.match("\[.*\]", var_name):
						
					# 	while not get_var_pool()[var_name][0].check_open():
					# 		add_share_que
					# 		time.sleep(0.01)
					# else:
					# 	while not get_var_pool()[var_name].check_open():
					# 		time.sleep(0.01)
					# #TODO: timeout check
					# pass
					return func(*args, **kwargs)
					
			else:
				while not get_net_pool().check_connection():
					print("cannot connect all node! reconnecting...")
					time.sleep(2)
				#broadcast
				data = wrap_json(ACTION.OPEN, var_name, get_var_pool()[var_name])
				get_net_pool().broadcast(data)
				get_var_pool().lock(var_name)
				add_share_que.set_ele(var_name).stand()
				#player run open
				# if re.match("\[.*\]", var_name):
				# 	while not get_var_pool()[var_name][0].check_open():
				# 		time.sleep(0.01)
				# else:
				# 	while not get_var_pool()[var_name].check_open():
				# 		time.sleep(0.01)
				# #TODO: timeout check
				# pass
				return func(*args, **kwargs)
		return wrapper
__deco__ = None
def set_global_deco(deco):
	global __deco__
	__deco__ = deco
	print("[wrap_function]:", "init deco success")

def get_global_deco():
	return __deco__