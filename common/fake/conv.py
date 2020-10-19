import math
def shape_infer(im_shape, filter_shape, stride, padding):
    size_end = math.ceil(((im_shape[-1] + 2*padding - filter_shape[-1]) / stride)) + 1
    return [im_shape[-4],filter_shape[-4],size_end,size_end]