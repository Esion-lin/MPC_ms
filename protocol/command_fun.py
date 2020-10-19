from common.var_pool import get_pool as get_var_pool 
from common.wrap_function import get_global_deco
from .triples_gen import Triple_generator
from common.tensor import IntTensor, PrivateTensor
from common.placeholder import Placeholder
def open_with_player(player_name,var_name):
	if isinstance(var_name, Placeholder):
		var_name = var_name.name
	dec = get_global_deco()
	@dec.open_(player_name = player_name, var_name = var_name)
	def open():
		return get_var_pool()[var_name].open()
	return open()

def input_with_player(player_name,var_name, ptensor):
	dec = get_global_deco()
	@dec.to_(player_name = player_name, var_name = var_name)
	def input():
		get_var_pool()[var_name] = ptensor
		return ptensor.share()
	return input()
triple_select = {
    "triple": Triple_generator.triple,
    "mat_triple": Triple_generator.mat_triple,
    "conv_triple": Triple_generator.conv_triple,
    "square_triple": Triple_generator.square_triple,
}
def make_triples(triple_type = "triple", triples_name = "", maked_player = "", **kwargs):
	dec = get_global_deco()
	@dec.to_(player_name = maked_player, var_name = triples_name)
	def triples(**kwargs):
		from protocol.test_protocol import Protocol
		from common.tensor import PrivateTensor
		tmp = [PrivateTensor(tensor = i, shared = True) for i in triple_select[triple_type](**kwargs)]
		get_var_pool()[var_name] = tmp
		return list(zip(*[ele.share() for ele in tmp]))
	return triples(**kwargs)