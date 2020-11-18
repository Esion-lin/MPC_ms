
class vcm:
    __counter = 0
    @classmethod
    def id(cls):
        return cls.__counter
    def __init__(self):
        vcm.__counter += 1
        self.prv_name = None
    def __enter__(self):
        from common.tensor import PrivateTensor,IntTensor
        from common.placeholder import Placeholder
        if "tmp_name" in Placeholder.__dict__:
            self.prv_name = Placeholder.tmp_name
        PrivateTensor.tmp_name = vcm.__counter
        Placeholder.tmp_name = vcm.__counter
        IntTensor.tmp_name = vcm.__counter
    def __exit__(self, exc_type,exc_value, traceback):
        from common.tensor import PrivateTensor,IntTensor
        from common.placeholder import Placeholder
        PrivateTensor.tmp_name = self.prv_name
        Placeholder.tmp_name = self.prv_name
        IntTensor.tmp_name = self.prv_name
        if exc_type is not None:
            print(exc_type,exc_value)
