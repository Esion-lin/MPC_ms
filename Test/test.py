import sys
import time

#test common
from common.tensor import IntTensor

x = IntTensor([[10.11,123.12],[1.23,2.11]])
y = IntTensor([[10.11],[1.23]])

x.Matmul(y)
print(x)