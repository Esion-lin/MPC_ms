from .player import player
from protocol.test_protocol import Protocol
__all__ = [
	"player",
	"p2pnetwork",
]


def set_protocol(protocol):
	global __protocol__
	__protocol__ = protocol

def get_protocol():
	return __protocol__

set_protocol(Protocol)