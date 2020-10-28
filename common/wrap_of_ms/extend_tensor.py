from mindspore import nn
from mindspore.ops import composite as C
from common.tensor import *
import time
from mindspore import ParameterTuple
#使用setattr扩展tensor，或直接调用
def _avgpool_IntTensor(self, kernel_size, stride):
    pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    return IntTensor(pool(self.to_native()), internal = False)

def _avgpool_PrivateTensor(self, kernel_size, stride):
    return PrivateTensor(tensor = _avgpool_IntTensor(self.convert_public(), kernel_size, stride))

def _dz(network,input_value,weight):
    if network not None:
        params = ParameterTuple(network.trainable_params())
        grad = C.GradOperation(name = "{}".format(time.time()),get_by_list=True)
        gradient_function = grad(conv, params)
        dz = gradient_function(input_value)
        #^ (dx, dw)
        return dz
    else:
        raise RuntimeError("Uninitialized network!!!")
def _conv_dz_IntTensor(self,weight,stride,padding):
    input_value = self.to_native()
    conv = nn.Conv2d(input_value.shape[1], weight.shape[-3], weight.shape[-2:], stride,pad_mode = "pad", padding = padding, weight_init=weight.to_native())
    return (IntTensor(ele, internal = False) for ele in _dz(conv, self.value))
    
def _conv_dz_PrivateTensor(self,weight,stride,padding):
    return ( PrivateTensor(tensor = ele) for ele in _dz_IntTensor(self.convert_public(),weight,stride,padding) )
'''
'''
avgpool = _avgpool_PrivateTensor

__all__ = [
    "avgpool",
]