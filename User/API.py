from common.placeholder import Placeholder
from protocol.command_fun import *
from communication import CallBack,init_pool
from communication.callback import Dealer
from player import get_player
def input_data(var_name, jtensor = None, player_name = "Bob"):
		if jtensor is None:
			return Placeholder(var_name)
		ptensor = PrivateTensor(shared = True, tensor = jtensor, name = var_name)
		return input_with_player(player_name, var_name, ptensor)

def input_from_Record(file_path, player_name):
    # return placeholder
    pass

def input_with_interact(player_name):
    pass

def open(with_player, var):
    return open_with_player(with_player, var)

def start_task():
    if get_player() is None:
        raise RuntimeError("请先指定用户")
    init_pool()