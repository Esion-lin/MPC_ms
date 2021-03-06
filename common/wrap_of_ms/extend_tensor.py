from mindspore import nn
from mindspore.ops import composite as C
from common.tensor import *
from common.placeholder import Placeholder
import time
from mindspore import ParameterTuple
#使用setattr扩展tensor，或直接调用
def _avgpool_IntTensor(self, kernel_size, stride):
    pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    return IntTensor(pool(self.to_native()), internal = False)

def _avgpool_PrivateTensor(self, kernel_size, stride):
    return PrivateTensor(tensor = _avgpool_IntTensor(self.convert_public(), kernel_size, stride))

def _avgpool_Placeholder(self, kernel_size, stride):
    if Placeholder.tmp_name is None:
        raise RuntimeError("在使用Placeholder进行直接计算时，请使用with语句")
    if isinstance(self, Placeholder):
        return Placeholder(name = "avgpool_ans_{}".format(Placeholder.tmp_name), value = _avgpool_PrivateTensor(self.fill(), kernel_size, stride))
    else:
        return Placeholder(name = "avgpool_ans_{}".format(Placeholder.tmp_name), value = _avgpool_PrivateTensor(self, kernel_size, stride))
    
def _dz(network,input_value, delta):
    if network is not None:
        params = ParameterTuple(network.trainable_params())
        grad = C.GradOperation(name = "{}".format(time.time()),get_by_list=True, get_all=True, sens_param=True)
        gradient_function = grad(network, params)
        dz = gradient_function(input_value, delta)
        print(dz)
        #^ (dx, dw)
        return dz
    else:
        raise RuntimeError("Uninitialized network!!!")
def _conv_dz_IntTensor(self,weight,stride,padding, delta):
    input_value = self.to_native()
    conv = nn.Conv2d(input_value.shape[1], weight.shape[-3], weight.shape[-2:], stride,pad_mode = "pad", padding = padding, weight_init=weight.to_native())
    dx, dw = _dz(conv, input_value, delta.to_native())
    return [IntTensor(dx[0], internal = False), IntTensor(dw[0], internal = False)]
    
def _conv_dz_PrivateTensor(self, weight, stride, padding, delta):
    dx, dw = _conv_dz_IntTensor(self.convert_public(),weight.convert_public(),stride,padding, delta.convert_public())
    return [ PrivateTensor(tensor = dx), PrivateTensor(tensor = dw)]

def _conv_dz_Placeholder(self, weight, stride, padding, delta):
    dx, dw = _conv_dz_PrivateTensor(self.fill(),weight.fill(),stride,padding, delta.fill())
    return [Placeholder(name = "{}_conv_dx_ans".format(Placeholder.tmp_name),value = dx), Placeholder(name = "{}_conv_dw_ans".format(Placeholder.tmp_name),value = dw)]


'''
'''
avgpool = _avgpool_Placeholder

__all__ = [
    "avgpool",
]