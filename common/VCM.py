class vcm:
    def __init__(self,name):
        self.name = name
    def __enter__(self):
        from common.tensor import PrivateTensor
        from common.placeholder import Placeholder
        PrivateTensor.tmp_name = self.name
        Placeholder.tmp_name = self.name
    def __exit__(self, exc_type,exc_value, traceback):
        from common.tensor import PrivateTensor
        from common.placeholder import Placeholder
        PrivateTensor.tmp_name = None
        Placeholder.tmp_name = None
        if exc_type not None:
            print(exc_type,exc_value)
