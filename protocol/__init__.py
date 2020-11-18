from .test_protocol import Protocol


__protocol__ = Protocol

def get_protocol():
	return __protocol__
def set_protocol(pro):
	global __protocol__
	__protocol__ = pro

__all__ = [
	"get_protocol",
	"set_protocol",
]