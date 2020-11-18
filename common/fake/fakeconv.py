from crypto.factory import encodeFP32
import numpy as np
def conv(image, filters, padding=0, stride=1):
    '''
    image: 卷积层输入张量，形状（Batch_size，Input_channels,H,W）
    filters: 卷积核，形状（Input_channels，Onput_channels,H_f,W_f）
    padding: (Union[int, tuple[int]])
    stride: 步长
    '''

    if isinstance(padding, int):
        padding=(padding, padding)
    padding_image = np.lib.pad(image, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)
    batch_size, input_channels, height, width = padding_image.shape
    input_channels_f, output_channels, H_f, W_f = filters.shape
    if isinstance(stride,int):
        stride= (stride,stride)
    assert (height - H_f) % stride[0] == 0, '步长必须能够被整除'
    assert (width - W_f) % stride[1] == 0, '步长必须能够被整除'
    conv_z = np.zeros((batch_size, output_channels, 1 + (height - H_f) // stride[0], 1 + (width - W_f) // stride[1]))
    for n in np.arange(batch_size):
        for d in np.arange(output_channels):
            for h in np.arange(height - H_f + 1)[::stride[0]]:
                for w in np.arange(width - W_f + 1)[::stride[1]]:
                    conv_z[n, d, h // stride[0], w // stride[1]] = np.sum(padding_image[n, :, h:h + H_f, w:w + W_f] * filters[:, d] % encodeFP32.module)
    return conv_z
