from mindspore import nn
from common.tensor import *
#使用setattr扩展tensor，或直接调用
def _avgpool_IntTensor(self, kernel_size, stride):
    pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    return IntTensor(pool(self.to_native()), internal = False)

def _avgpool_PrivateTensor(self, kernel_size, stride):
    return PrivateTensor(tensor = _avgpool_IntTensor(self.convert_public(), kernel_size, stride))

'''
'''
avgpool = _avgpool_PrivateTensor

__all__ = [
    "avgpool",
]