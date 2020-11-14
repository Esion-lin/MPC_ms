import mindspore.ops as P
class ops:
    def __init__(self):
        pass
    def construct(self,  *args, **kwargs):
        pass
    def __call__(self, *args,**kwargs):
        return self.construct(*args, **kwargs)


class Reshape:
    def __init__(self):
        self.reshape = P.Reshape()
    def construct(self,  *args, **kwargs):
        shape = kwargs["shape"]
        data = kwargs["data"]
        return self.reshape(data, shape)