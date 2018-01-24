"""Functional interface"""

import warnings
import math
from operator import mul
from functools import reduce

import torch
from torch._C import _infer_size, _add_docstr
from . import _functions
from .modules import utils
from ._functions.linear import Bilinear
from ._functions.padding import ConstantPadNd
from ._functions.vision import GridSampler, AffineGridGenerator
from torch.autograd import Variable
from .modules.utils import _single, _pair, _triple

# Convolutions
_ConvNd = torch._C._functions.ConvNd


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    r"""对几个输入平面组成的输入信号应用一个1D卷积.

    关于细节和输出形状大小, 请参见 :class:`~torch.nn.Conv1d`. 

    参数:
        input: 形状为 (minibatch x in_channels x iW) 的输入张量
        weight: 形状为 (out_channels x in_channels x kW) 的滤波器
        bias: 可选的偏置,形状为 (out_channels). 默认值: None
        stride: 卷积核的步长. 可以是单个数字, 也可以是一个元组 (sW, ). 默认值: 1
        padding: 输入两端隐式零填充的个数. 可以是单个数字, 也可以是一个元组 (padW, ). 默认值: 0
        dilation: 卷积核中元素之间的空洞大小. 可以是单个数字, 也可以是一个元组 (dW, ). 默认值: 1
        groups: 将输入分成的组的个数. in_channels 的值要求能够被 groups 的值整除. 默认值: 1

    例子::

        >>> filters = autograd.Variable(torch.randn(33, 16, 3))
        >>> inputs = autograd.Variable(torch.randn(20, 16, 50))
        >>> F.conv1d(inputs, filters)
    """
    if input is not None and input.dim() != 3:
        raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = _ConvNd(_single(stride), _single(padding), _single(dilation), False,
                _single(0), groups, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    r"""对几个输入平面组成的输入信号应用一个2D卷积.

    关于细节和输出形状大小, 请参见 :class:`~torch.nn.Conv2d`. 

    参数:
        input: 形状为 (minibatch x in_channels x iH x iW) 的输入张量
        weight: 形状为 (out_channels x in_channels/groups x kH x kW) 的滤波器
        bias: 可选的偏置,形状为 (out_channels). 默认值: None
        stride: 卷积核的步长. 可以是单个数字, 也可以是一个元组 (sH, sW). 默认值: 1
        padding: 输入两端隐式零填充的个数. 可以是单个数字, 也可以是一个元组 (padH, padW). 默认值: 0
        dilation: 卷积核中元素之间的空洞大小. 可以是单个数字, 也可以是一个元组 (dH, dW). 默认值: 1
        groups: 将输入分成的组的个数. in_channels 的值要求能够被 groups 的值整除. 默认值: 1

    例子::
    
        >>> # With square kernels and equal stride
        >>> filters = autograd.Variable(torch.randn(8,4,3,3))
        >>> inputs = autograd.Variable(torch.randn(1,4,5,5))
        >>> F.conv2d(inputs, filters, padding=1)
    """
    if input is not None and input.dim() != 4:
        raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = _ConvNd(_pair(stride), _pair(padding), _pair(dilation), False,
                _pair(0), groups, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    r"""对几个输入平面组成的输入信号应用一个3D卷积.

    关于细节和输出形状大小, 请参见 :class:`~torch.nn.Conv3d`. 

    参数:
        input: 形状为 (minibatch x in_channels x iT x iH x iW) 的输入张量
        weight: 形状为 (out_channels x in_channels/groups x kT x kH x kW) 的滤波器
        bias: 可选的偏置,形状为 (out_channels). 默认值: None
        stride: 卷积核的步长. 可以是单个数字, 也可以是一个元组 (sT, sH, sW). 默认值: 1
        padding: 输入两端隐式零填充的个数. 可以是单个数字, 也可以是一个元组 (padT, padH, padW). 默认值: 0
        dilation: 卷积核中元素之间的空洞大小. 可以是单个数字, 也可以是一个元组 (dT, dH, dW). 默认值: 1
        groups: 将输入分成的组的个数. in_channels 的值要求能够被 groups 的值整除. 默认值: 1

    例子::

        >>> filters = autograd.Variable(torch.randn(33, 16, 3, 3, 3))
        >>> inputs = autograd.Variable(torch.randn(20, 16, 50, 10, 20))
        >>> F.conv3d(inputs, filters)
    """

    if input is not None and input.dim() != 5:
        raise ValueError("Expected 5D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = _ConvNd(_triple(stride), _triple(padding), _triple(dilation), False,
                _triple(0), groups, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    r"""对几个输入平面组成的输入信号应用一个1D转置卷积,该操作有的时候也被称为“反卷积”.

    关于细节和输出形状大小, 请参见 :class:`~torch.nn.ConvTranspose1d`. 

    参数:
        input: 形状为 (minibatch x in_channels x iW) 的输入张量
        weight: 形状为 (out_channels x in_channels x kW) 的滤波器
        bias: 可选的偏置,形状为 (out_channels). 默认值: None
        stride: 卷积核的步长. 可以是单个数字, 也可以是一个元组 (sW, ). 默认值: 1
        padding: 输入两端隐式零填充的个数. 可以是单个数字, 也可以是一个元组 (padW, ). 默认值: 0
        output_padding: 输出两端隐式零填充的个数,范围为 0 <= padding < stride.
        可以是单个数字, 也可以是一个元组 (out_padW, ). 默认值: 0
        groups: 将输入分成的组的个数. in_channels 的值要求能够被 groups 的值整除. 默认值: 1
        dilation: 卷积核中元素之间的空洞大小. 可以是单个数字, 也可以是一个元组 (dW, ). 默认值: 1
    """
    if input is not None and input.dim() != 3:
        raise ValueError("Expected 3D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = _ConvNd(_single(stride), _single(padding), _single(dilation), True,
                _single(output_padding),
                groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.enabled)
    return f(input, weight, bias)


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    r"""对几个输入平面组成的输入信号应用一个2D转置卷积,该操作有的时候也被称为 "反卷积".

    关于细节和输出形状大小, 请参见 :class:`~torch.nn.ConvTranspose2d`. 

    参数:
        input: 形状为 (minibatch x in_channels x iH x iW) 的输入张量
        weight: 形状为 (out_channels x in_channels x kH x kW) 的滤波器
        bias: 可选的偏置,形状为 (out_channels). 默认值: None
        stride: 卷积核的步长. 可以是单个数字, 也可以是一个元组 (sH, sW). 默认值: 1
        padding: 输入两端隐式零填充的个数. 可以是单个数字, 也可以是一个元组 (padH, padW). 默认值: 0
        output_padding: 输出两端隐式零填充的个数,范围为 0 <= padding < stride.
        可以是单个数字, 也可以是一个元组 (out_padH, out_padW). 默认值: 0
        groups: 将输入分成的组的个数. in_channels 的值要求能够被 groups 的值整除. 默认值: 1
        dilation: 卷积核中元素之间的空洞大小. 可以是单个数字, 也可以是一个元组 (dH, dW). 默认值: 1
    """
    if input is not None and input.dim() != 4:
        raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = _ConvNd(_pair(stride), _pair(padding), _pair(dilation), True,
                _pair(output_padding), groups, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    r"""对几个输入平面组成的输入信号应用一个3D转置卷积, 该操作有的时候也被称为 "反卷积".

    关于细节和输出形状大小, 请参见 :class:`~torch.nn.ConvTranspose3d`. 

    参数:
        input: 形状为 (minibatch x in_channels x iT x iH x iW) 的输入张量
        weight: 形状为 (out_channels x in_channels x kH x kW) 的滤波器
        bias: 可选的偏置,形状为 (out_channels). 默认值: None
        stride: 卷积核的步长. 可以是单个数字, 也可以是一个元组 (sT, sH, sW). 默认值: 1
        padding: 输入两端隐式零填充的个数. 可以是单个数字, 也可以是一个元组 (padT, padH, padW). 默认值: 0
        output_padding: 输出两端隐式零填充的个数,范围为 0 <= padding < stride.
        可以是单个数字, 也可以是一个元组 (out_padT, out_padH, out_padW). 默认值: 0
        groups: 将输入分成的组的个数. in_channels 的值要求能够被 groups 的值整除. 默认值: 1
        dilation: 卷积核中元素之间的空洞大小. 可以是单个数字, 也可以是一个元组 (dT, dH, dW). 默认值: 1
    """
    if input is not None and input.dim() != 5:
        raise ValueError("Expected 5D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = _ConvNd(_triple(stride), _triple(padding), _triple(dilation), True,
                _triple(output_padding), groups, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)


# Pooling
def avg_pool1d(input, kernel_size, stride=None, padding=0,
               ceil_mode=False, count_include_pad=True):
    r"""对由几个输入通道组成的输入信号进行一维平均池化。

    有关详细信息和输出形状，请参阅 :class:`~torch.nn.AvgPool1d` 。

    Args:
        input: 输入张量 (minibatch x in_channels x iW)
        kernel_size: 窗口的大小。可以是单个数字或者 tuple (kW,)
        stride: 窗口的步长。可以是单个数字或者 tuple (sW,)。 默认值: :attr:`kernel_size`
        padding: 在输入周围隐式零填充。可以是单个数字或者 tuple (padW,)。默认值: 0
        ceil_mode: 当为 True 时, 将使用 `ceil` 代替 `floor` 来计算输出的 shape。默认值: ``False``
        count_include_pad: 当为 True 时, 在平均计算时将包括零填充。默认值: ``True``

    Example:
        >>> # pool of square window of size=3, stride=2
        >>> input = Variable(torch.Tensor([[[1,2,3,4,5,6,7]]]))
        >>> F.avg_pool1d(input, kernel_size=3, stride=2)
        Variable containing:
        (0 ,.,.) =
          2  4  6
        [torch.FloatTensor of size 1x1x3]
    """
    if input.dim() != 3:
        raise ValueError('expected 3D input (got {} dimensions)'
                         .format(input.dim()))
    kernel_size = _single(kernel_size) + (1,)
    stride = _single(stride) + (1,) if stride is not None else kernel_size
    padding = _single(padding) + (0,)
    return avg_pool2d(input.unsqueeze(3), kernel_size, stride, padding,
                      ceil_mode, count_include_pad).squeeze(3)


avg_pool2d = _add_docstr(torch._C._nn.avg_pool2d, r"""
avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Variable

在 kh x kw 区域中应用步长为 dh x dw 的二维平均池化操作。输出特征的数量等于输入通道的数量。

有关详细信息和输出形状，请参阅 :class:`~torch.nn.AvgPool2d` 。

Args:
    input: 输入张量 (minibatch x in_channels x iH x iW)
    kernel_size: 池化区域的大小。可以是单个数字或者 tuple (kH x kW)
    stride: 池化操作的步长。 可以是单个数字或者 tuple (sH, sW)。默认等于 kernel 的大小
    padding: 在输入周围隐式零填充。可以是单个数字或者 tuple (padH, padW)。默认值: 0
    ceil_mode: 当为 True 时, 将使用公式中的 `ceil` 代替 `floor` 来计算输出的 shape。默认值: ``False``
    count_include_pad: 当为 True 时, 在平均计算时将包括零填充。默认值: ``True``
""")

avg_pool3d = _add_docstr(torch._C._nn.avg_pool3d, r"""
avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Variable

在 kt x kh x kw 区域中应用步长为 dt x dh x dw 的三维平均池化操作。
输出特征的数量等于输入通道的数量/dt。

有关详细信息和输出形状，请参阅 :class:`~torch.nn.AvgPool3d` 。

Args:
    input: 输入张量 (minibatch x in_channels x iT x iH x iW)
    kernel_size: 池化区域的大小。可以是单个数字或者 tuple (kT x kH x kW)
    stride: 池化操作的步长。 可以是单个数字或者 tuple (sT, sH, sW). 默认等于 kernel 的大小
    padding: 在输入周围隐式零填充。可以是单个数字或者 tuple (padT, padH, padW), 默认值: 0
    ceil_mode: 当为 True 时, 将使用公式中的 `ceil` 代替 `floor` 来计算输出的 shape
    count_include_pad: 当为 True 时, 在平均计算时将包括零填充
""")


# share the same interface
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    """对由几个输入通道组成的输入信号进行一维最大池化。

    有关详细信息，请参阅 :class:`~torch.nn.MaxPool1d` 。
    """
    ret = _functions.thnn.MaxPool1d.apply(input, kernel_size, stride, padding, dilation,
                                          ceil_mode)
    return ret if return_indices else ret[0]


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    """对由几个输入通道组成的输入信号进行二维最大池化。

    有关详细信息，请参阅 :class:`~torch.nn.MaxPool2d` 。
    """
    ret = torch._C._nn.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    return ret if return_indices else ret[0]


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    """对由几个输入通道组成的输入信号进行三维最大池化。

    有关详细信息，请参阅 :class:`~torch.nn.MaxPool2d` 。
    """
    ret = _functions.thnn.MaxPool3d.apply(input, kernel_size, stride, padding, dilation,
                                          ceil_mode)
    return ret if return_indices else ret[0]


def _unpool_output_size(input, kernel_size, stride, padding, output_size):
    input_size = input.size()
    default_size = []
    for d in range(len(kernel_size)):
        default_size.append((input_size[d + 2] - 1) * stride[d] +
                            kernel_size[d] - 2 * padding[d])
    if output_size is None:
        return default_size

    output_size = list(output_size)
    if len(output_size) == len(kernel_size) + 2:
        output_size = output_size[2:]
    if len(output_size) != len(kernel_size):
        raise ValueError("output_size should be a sequence containing "
                         "{} or {} elements, but it has a length of '{}'"
                         .format(len(kernel_size), len(kernel_size) + 2,
                                 len(output_size)))
    for d in range(len(kernel_size)):
        min_size = default_size[d] - stride[d]
        max_size = default_size[d] + stride[d]
        if not (min_size < output_size[d] < max_size):
            raise ValueError(
                'invalid output_size "{}" (dim {} must be between {} and {})'
                .format(output_size, d, min_size, max_size))

    return output_size


def max_unpool1d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    """计算 :class:`MaxPool1d` 的部分逆

    有关详细信息，请参阅 :class:`~torch.nn.MaxUnpool1d` 。
    """
    kernel_size = _single(kernel_size)
    stride = _single(stride)
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool2d(input.unsqueeze(3), indices.unsqueeze(3), output_size + [1]).squeeze(3)


def max_unpool2d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    """计算 :class:`MaxPool2d` 的部分逆。

    有关详细信息，请参阅 :class:`~torch.nn.MaxUnpool2d` 。
    """
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool2d(input, indices, output_size)


def max_unpool3d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    """计算 :class:`MaxPool3d` 的部分逆。

    有关详细信息，请参阅 :class:`~torch.nn.MaxUnpool3d` 。
    """
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool3d(input, indices, output_size, stride, padding)


def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    """对由几个输入通道组成的输入信号进行二维幂平均池化。

    有关详细信息，请参阅 :class:`~torch.nn.LPPool2d` 。
    """
    kw, kh = utils._pair(kernel_size)
    out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return out.mul(kw * kh).pow(1. / norm_type)


def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    """对由几个输入通道组成的输入信号进行一维幂平均池化。

    有关详细信息，请参阅 :class:`~torch.nn.LPPool1d` 。
    """
    out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return out.mul(kernel_size).pow(1. / norm_type)


def adaptive_max_pool1d(input, output_size, return_indices=False):
    r"""对由几个输入通道组成的输入信号进行一维自适应最大池化。

    有关详细信息和输出形状，请参阅 :class:`~torch.nn.AdaptiveMaxPool1d` 。

    Args:
        output_size: 目标输出大小（单个整数）
        return_indices: 是否返回池化索引。 默认值: ``False``
    """
    ret = _functions.thnn.AdaptiveMaxPool1d.apply(input, output_size)
    return ret if return_indices else ret[0]


def adaptive_max_pool2d(input, output_size, return_indices=False):
    r"""对由几个输入通道组成的输入信号进行二维自适应最大池化。

    有关详细信息和输出形状，请参阅 :class:`~torch.nn.AdaptiveMaxPool2d` 。

    Args:
        output_size: 目标输出大小（单个整数或者两个整数的 tuple ）
        return_indices: 是否返回池化索引。 默认值: ``False``
    """
    ret = _functions.thnn.AdaptiveMaxPool2d.apply(input, output_size)
    return ret if return_indices else ret[0]


def adaptive_max_pool3d(input, output_size, return_indices=False):
    r"""对由几个输入通道组成的输入信号进行三维自适应最大池化。

    有关详细信息和输出形状，请参阅 :class:`~torch.nn.AdaptiveMaxPool3d` 。

    Args:
        output_size: 目标输出大小（单个整数或者三个整数的 tuple ）
        return_indices: 是否返回池化索引。 默认值: ``False``
    """
    ret = _functions.thnn.AdaptiveMaxPool3d.apply(input, output_size)
    return ret if return_indices else ret[0]


def adaptive_avg_pool1d(input, output_size):
    r"""对由几个输入通道组成的输入信号进行一维自适应平均池化。

    有关详细信息和输出形状，请参阅 :class:`~torch.nn.AdaptiveAvgPool1d` 。

    Args:
        output_size: 目标输出大小（单个整数）
    """
    return _functions.thnn.AdaptiveAvgPool1d.apply(input, output_size)


def adaptive_avg_pool2d(input, output_size):
    r"""对由几个输入通道组成的输入信号进行二维自适应平均池化。

    有关详细信息和输出形状，请参阅 :class:`~torch.nn.AdaptiveAvgPool2d` 。

    Args:
        output_size: 目标输出大小（单个整数或者两个整数的 tuple ）
    """
    return _functions.thnn.AdaptiveAvgPool2d.apply(input, output_size)


def adaptive_avg_pool3d(input, output_size):
    r"""对由几个输入通道组成的输入信号进行三维自适应平均池化。

    有关详细信息和输出形状，请参阅 :class:`~torch.nn.AdaptiveAvgPool3d` 。

    Args:
        output_size: 目标输出大小（单个整数或者三个整数的 tuple ）
    """
    return _functions.thnn.AdaptiveAvgPool3d.apply(input, output_size)


# Activation functions

def dropout(input, p=0.5, training=False, inplace=False):
    return _functions.dropout.Dropout.apply(input, p, training, inplace)


def alpha_dropout(input, p=0.5, training=False):
    r"""将 dropout 应用于输入数据( dropou 是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃,防止过拟合)。

    有关详细信息，请参阅 :class:`~torch.nn.AlphaDropout`

    Args:
        p (float, optional): 丢弃的概率。默认值: 0.5
        training (bool, optional): 决定是否在训练和测试模式之间的切换. 默认值: ``False``
    """
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))

    if p == 0 or not training:
        return input

    alpha = -1.7580993408473766
    keep_prob = 1 - p
    # TODO avoid casting to byte after resize
    noise = input.data.new().resize_(input.size())
    noise.bernoulli_(p)
    noise = Variable(noise.byte())

    output = input.masked_fill(noise, alpha)

    a = (keep_prob + alpha ** 2 * keep_prob * (1 - keep_prob)) ** (-0.5)
    b = -a * alpha * (1 - keep_prob)

    return output.mul_(a).add_(b)


def dropout2d(input, p=0.5, training=False, inplace=False):
    return _functions.dropout.FeatureDropout.apply(input, p, training, inplace)


def dropout3d(input, p=0.5, training=False, inplace=False):
    return _functions.dropout.FeatureDropout.apply(input, p, training, inplace)


threshold = _add_docstr(torch._C._nn.threshold, r"""
threshold(input, threshold, value, inplace=False) -> Variable

Thresholds each element of the input Tensor.

See :class:`~torch.nn.Threshold` for more details.
""")


def relu(input, inplace=False):
    """relu(input, threshold, value, inplace=False) -> Variable

    以元素的方式应用修正线性单元函数. 请参阅
    :class:`~torch.nn.ReLU` 可以获取更多细节.
    """
    return threshold(input, 0, 0, inplace)


glu = _add_docstr(torch._C._nn.glu, r"""
glu(input, dim=-1) -> Variable

门控线性单元. 计算方式如下:

.. math ::

    H = A \times \sigma(B)

其中输入沿着轴拆分为A和B两部分.

请参阅 使用门控卷积网络进行语言建模 <https://arxiv.org/abs/1612.08083>.

Args:
    input (Variable): 输入变量
    dim (int): 指定分裂的轴
""")

hardtanh = _add_docstr(torch._C._nn.hardtanh, r"""
hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Variable

Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
details.
""")


def relu6(input, inplace=False):
    r"""relu6(input, inplace=False) -> Variable

    以元素方式应用HardTanh函数 :math:`{ReLU6}(x) = min(max(0,x), 6)`.

    请参阅 :class:`~torch.nn.ReLU6` 可以获取更多细节.
    """
    return hardtanh(input, 0, 6, inplace)


elu = _add_docstr(torch._C._nn.elu, r"""
elu(input, alpha=1., inplace=False) -> Variable

以元素方式使用,
:math:`f(x) = max(0,x) + min(0, alpha * (exp(x) - 1))`.

请参阅 :class:`~torch.nn.ELU` 可以获取更多细节.
""")


def selu(input, inplace=False):
    r"""selu(input, inplace=False) -> Variable

    以元素方式使用,
    :math:`f(x) = scale * (\max(0,x) + \min(0, alpha * (\exp(x) - 1)))`,
    with ``alpha=1.6732632423543772848170429916717`` and
    ``scale=1.0507009873554804934193349852946``.

    请参阅 :class:`~torch.nn.SELU` 可以获取更多细节.
    """
    return _functions.thnn.SELU.apply(input, inplace)


leaky_relu = _add_docstr(torch._C._nn.leaky_relu, r"""
leaky_relu(input, negative_slope=0.01, inplace=False) -> Variable

以元素方式使用,
:math:`f(x) = max(0, x) + {negative\_slope} * min(0, x)`

请参阅 :class:`~torch.nn.LeakyReLU` 可以获取更多细节.
""")


# prelu = _add_docstr(torch._C._nn.prelu, r"""
# prelu(input, weight) -> Variable
# """)
def prelu(input, weight):
    r"""prelu(input, weight) -> Variable

    以元素方式使用方法
    :math:`PReLU(x) = max(0,x) + weight * min(0,x)` where weight is a
    learnable parameter.

    请参阅 :class:`~torch.nn.PReLU` 可以获取更多细节.
    """
    return _functions.thnn.PReLU.apply(input, weight)


rrelu = _add_docstr(torch._C._nn.rrelu, r"""
rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Variable
""")

logsigmoid = _add_docstr(torch._C._nn.log_sigmoid, r"""
logsigmoid(input) -> Variable

以元素方式使用 :math:`LogSigmoid(x) = log( 1 / (1 + exp(-x_i)))`

请参阅 :class:`~torch.nn.LogSigmoid` 可以获取更多细节.
""")

hardshrink = _add_docstr(torch._C._nn.hardshrink, r"""
hardshrink(input, lambd=0.5) -> Variable

以元素方式使用 hard shrinkage 方法

请参阅 :class:`~torch.nn.Hardshrink` 可以获取更多细节.


""")


def tanhshrink(input):
    r"""tanhshrink(input) -> Variable

    以元素方式使用, :math:`Tanhshrink(x) = x - Tanh(x)`

    请参阅 :class:`~torch.nn.Tanhshrink` 可以获取更多细节.
    """
    return input - input.tanh()


def softsign(input):
    r"""softsign(input) -> Variable

    以元素方式使用方法 :math:`f(x) = x / (1 + |x|)`

    请参阅 :class:`~torch.nn.Softsign` 可以获取更多细节.
    """
    return input / (input.abs() + 1)


softplus = _add_docstr(torch._C._nn.softplus, r"""
softplus(input, beta=1, threshold=20) -> Variable
""")


def _get_softmax_dim(name, ndim, stacklevel):
    warnings.warn("Implicit dimension choice for " + name + " has been deprecated. "
                  "Change the call to include dim=X as an argument.", stacklevel=stacklevel)
    if ndim == 0 or ndim == 1 or ndim == 3:
        return 0
    else:
        return 1


def softmin(input, dim=None, _stacklevel=3):
    r"""使用一个 softmin 函数.

    注意 softmin(x) = softmax(-x). 请参阅 softmax 数学公式的定义.

    请参阅 :class:`~torch.nn.Softmin` 可以获取更多细节.

    Arguments:
        input (Variable): 输入
        dim (int): softmin 将沿着指定轴 dim 计算(所以沿着轴的切片累加和为 1).
    """
    if dim is None:
        dim = _get_softmax_dim('softmin', input.dim(), _stacklevel)
    return torch._C._nn.softmax(-input, dim)


def softmax(input, dim=None, _stacklevel=3):
    r"""使用一个 softmax 函数.

    Softmax被定义为:

    :math:`softmax(x) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

    函数会应用于沿着指定轴的所有切片，并且会标准化结果让每个切片的计算结果映射到（0,1）范围内，让总和为 1.

    请参阅 :class:`~torch.nn.Softmax` 可以获取更多细节.

    Arguments:
        input (Variable): 输入
        dim (int): softmax 将沿着指定轴 dim 计算.

    .. note::
        该函数不直接与 NLLLoss 一起工作，
        NLLLoss 期望在 Softmax 和它自身之间计算对数.
        使用 log_softmax 代替（log_softmax 更快并且对数值型支持度更好）.

    """
    if dim is None:
        dim = _get_softmax_dim('softmax', input.dim(), _stacklevel)
    return torch._C._nn.softmax(input, dim)


def log_softmax(input, dim=None, _stacklevel=3):
    r"""使用对数形式的 softmax 函数.

    虽然在数学上等同于 log（softmax（x）），但单独执行这两个
    操作的速度较慢，而且数值不稳定。
    这个功能使用另一个公式来正确计算输出和梯度。

    请参阅 :class:`~torch.nn.LogSoftmax` 可以获取更多细节.

    Arguments:
        input (Variable): 输入
        dim (int): log_softmax 将沿着指定轴dim计算.
    """
    if dim is None:
        dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)
    return torch._C._nn.log_softmax(input, dim)


def softshrink(input, lambd=0.5):
    r"""softshrink(input, lambd=0.5) -> Variable

    以元素的方式使用 soft shrinkage 函数

    请参阅 :class:`~torch.nn.Softshrink` 可以获取更多细节.
    """
    return _functions.thnn.auto.Softshrink.apply(input, lambd)


def tanh(input):
    r"""tanh(input) -> Variable

    以元素的方式使用,
    :math:`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

    请参阅 :class:`~torch.nn.Tanh` 可以获取更多细节.
    """
    return input.tanh()


def sigmoid(input):
    r"""sigmoid(input) -> Variable

    以元素的方式使用函数 :math:`f(x) = 1 / ( 1 + exp(-x))`

    请参阅 :class:`~torch.nn.Sigmoid` 可以获取更多细节.
    """
    return input.sigmoid()


# etc.

def linear(input, weight, bias=None):
    """
    对输入的数据应用线性转换: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` 其中 * 表示任意数量的附加维度
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        return torch.addmm(bias, input, weight.t())

    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    return output


def bilinear(input1, input2, weight, bias=None):
    if bias is None:
        return Bilinear.apply(input1, input2, weight)
    else:
        return Bilinear.apply(input1, input2, weight, bias)


def embedding(input, embedding_matrix,
              max_norm=None, norm_type=2, scale_grad_by_freq=False,
              sparse=False):
    r"""A simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    Args:
        input: tensor, containing indices into the embedding matrix
        embedding_matrix:
                Number of rows should correspond to the maximum possible index + 1,
                number of columns is the embedding size
        max_norm (float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the frequency of
                                                the words in the mini-batch.
        sparse (boolean, optional): if ``True``, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for
                                    more details regarding sparse gradients.

    Shape:
        - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Embedding_matrix: FloatTensor `(V, embedding_dim)`, V = maximum index + 1, embedding_dim = embedding size
        - Output: `(N, W, embedding_dim)`

    Notes:
        It is advised to only use `sparse=True` if `embedding_matrix` is a leaf Variable,
        since some autograd functions may not propagate sparse gradients correctly.
        Additionally, keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's `optim.SGD` (`cuda` and `cpu`), and `optim.Adagrad` (`cpu`)

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = Variable(torch.rand(10, 3))
        >>> torch.nn.functional.embedding(input, embedding_matrix)

        Variable containing:
        (0 ,.,.) =
         -1.0822  1.2522  0.2434
          0.8393 -0.6062 -0.3348
          0.6597  0.0350  0.0837
          0.5521  0.9447  0.0498

        (1 ,.,.) =
          0.6597  0.0350  0.0837
         -0.1527  0.0877  0.4260
          0.8393 -0.6062 -0.3348
         -0.8738 -0.9054  0.4281
        [torch.FloatTensor of size 2x4x3]

        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = Variable(weights)
        >>> input = Variable(torch.LongTensor([[0,2,0,5]]))
        >>> torch.nn.functional.embedding(input, embedding_matrix)

        Variable containing:
        (0 ,.,.) =
          0.0000  0.0000  0.0000
          0.3452  0.4937 -0.9361
          0.0000  0.0000  0.0000
          0.0706 -2.1962 -0.6276
        [torch.FloatTensor of size 1x4x3]

    """
    return _functions.thnn.Embedding.apply(
        input, embedding_matrix,
        -1, max_norm, norm_type,
        scale_grad_by_freq, sparse
    )


def embedding_bag(embedding_matrix, indices, offsets=None,
                  max_norm=None, norm_type=2, scale_grad_by_freq=False, mode='mean'):
    r"""Computes sums or means of 'bags' of embeddings, without instantiating the
        intermediate embeddings.

        For bags of constant length,
            * embedding_bag with `mode=sum` is equivalent to nn.functional.embedding followed by `torch.sum(dim=1)`
            * with `mode=mean` is equivalent to nn.functional.embedding followed by `torch.mean(dim=1)`

        However, embedding_bag is much more time and memory efficient than using a chain of these
        operations.

        Args:
            embedding_matrix: FloatTensor, where number of rows should correspond to the maximum possible index + 1,
                              number of columns is the embedding size
            indices (N or BxN): LongTensor containing the indices of the embeddings to extract.
                                When `input` is 1D Tensor of shape `N`, an `offsets` Tensor is given, that contains the
                                starting position of each new sequence in the mini-batch.
            offsets (B or None): LongTensor containing the starting positions of each sample in a mini-batch of variable
                                 length sequences. If `input` is 2D (BxN), then offsets does not need to be given,
                                 as the `input` is treated as a mini-batch of fixed length sequences of length `N` each.
            max_norm (float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
            norm_type (float, optional): The p of the p-norm to compute for the max_norm option
            scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the frequency of
                                                    the words in the dictionary.
            mode (string, optional): 'sum' | 'mean'. Specifies the way to reduce the bag. Default: 'mean'

        Shape:
            - Embedding_matrix: FloatTensor `(V, embedding_dim)`,
                                V = number of embeddings, embedding_dim = embedding size
            - Input: LongTensor `N`, N = number of embeddings to extract
                     (or) LongTensor `BxN`, B = number of sequences in mini-batch,
                                            N = number of embeddings per sequence
            - Offsets: LongTensor `B`, B = number of bags. The values are the
                       offsets in `input` for each bag, i.e. the cumsum of lengths.
                       Offsets is not given if Input is 2D `BxN` Tensor,
                       the input is considered to be of fixed-length sequences
            - Output: `(B, embedding_dim)`

        Examples::

            >>> # an Embedding module containing 10 tensors of size 3
            >>> embedding_matrix = Variable(torch.rand(10, 3))
            >>> # a batch of 2 samples of 4 indices each
            >>> input = Variable(torch.LongTensor([1,2,4,5,4,3,2,9]))
            >>> offsets = Variable(torch.LongTensor([0,4]))
            >>> embedding_bag(embedding_matrix, input, offsets)

            Variable containing:
            -1.1840 -0.2547 -0.5860
            -0.7126  0.0002 -0.3411
            [torch.FloatTensor of size 2x3]

        """
    if indices.dim() == 2:
        if offsets is not None:
            raise ValueError("if input is 2D, then offsets has to be None"
                             ", as input is treated is a mini-batch of"
                             " fixed length sequences. However, found "
                             "offsets of type {}".format(type(offsets)))
        else:
            offsets = Variable(torch.arange(0, indices.numel(), indices.size(1),
                               out=indices.data.new().long()))
            indices = indices.view(-1)

    elif indices.dim() != 1:
        raise ValueError("input has to be 1D or 2D Tensor,"
                         " but got Tensor of dimension {}".format(indices.dim()))

    if offsets is None:
        raise ValueError("offsets has to be a 1D Tensor but got None")

    return _functions.thnn.EmbeddingBag.apply(
        embedding_matrix, indices, offsets,
        max_norm, norm_type,
        scale_grad_by_freq, mode
    )


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    if training:
        size = list(input.size())
        if reduce(mul, size[2:], size[0]) == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
    f = torch._C._functions.BatchNorm(running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled)
    return f(input, weight, bias)


# loss

def nll_loss(input, target, weight=None, size_average=True, ignore_index=-100, reduce=True):
    r"""The negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or `(N, C, H, W)`
            in case of 2D - Loss
        target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. If size_average
            is False, the losses are summed for each minibatch. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            True, the loss is averaged over non-ignored targets. Default: -100

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = autograd.Variable(torch.randn(3, 5))
        >>> # each element in target has to have 0 <= value < C
        >>> target = autograd.Variable(torch.LongTensor([1, 0, 4]))
        >>> output = F.nll_loss(F.log_softmax(input), target)
        >>> output.backward()
    """
    dim = input.dim()
    if torch.is_tensor(weight):
        weight = Variable(weight)
    if dim == 2:
        return torch._C._nn.nll_loss(input, target, weight, size_average, ignore_index, reduce)
    elif dim == 4:
        return torch._C._nn.nll_loss2d(input, target, weight, size_average, ignore_index, reduce)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def poisson_nll_loss(input, target, log_input=True, full=False, size_average=True, eps=1e-8):
    r"""泊松分布的负对数似然损失 (Negative log likelihood loss).

    详见 :class:`~torch.nn.PoissonNLLLoss`.

    参数:
        input: 泊松分布的期望值.
        target: 随机样本 :math:`target \sim Pois(input)`.
        log_input:  如果设置为 ``True`` , 损失将会按照公式 `exp(input) - target * input` 
            来计算, 如果设置为 ``False`` , 损失将会按照 `input - target * log(input+eps)` 计算.
            默认: ``True``
        full:  是否计算全部的损失, i. e. 加上 Stirling 近似项 
            `target * log(target) - target + 0.5 * log(2 * pi * target)`. 默认: ``False``
        size_average:  默认情况下, 该损失函数的值会在每个 mini-batch（小批量） 上取平均值. 
            如果字段 size_average 被设置为  ``False``, 损失函数的值会在每 个 mini-batch（小批量）上求和.
            默认: ``True``.
        eps (float, optional): 当 ··log_input==False`` 时, 取一个很小的值用来
            避免计算 log(0) . 默认: 1e-8
    """
    if log_input:
        loss = torch.exp(input) - target * input
    else:
        loss = input - target * torch.log(input + eps)
    if full:
        mask = target > 1
        loss[mask] += (target * torch.log(target) - target + 0.5 * torch.log(2 * math.pi * target))[mask]
    if size_average:
        return torch.mean(loss)
    else:
        return torch.sum(loss)


kl_div = _add_docstr(torch._C._nn.kl_div, r"""
`Kullback-Leibler divergence` 损失.

详见ee :class:`~torch.nn.KLDivLoss`.

参数:
    input: 任意形状变量
    target: 与输入形状相同的变量
    size_average: 如果是 ``True`` 输出值会除以输入 tensor 的元素总数. 默认: ``True``
    reduce (bool, optional): 默认情况下, 该损失函数的值会根据 size_average 在每个 
            mini-batch（小批量）上求平均值或者求和. 当 reduce 是 ``False`` 时, 损
            失函数会对每个 batch 元素都返回一个损失值并忽略 size_average. 默认: ``True``
""")


def cross_entropy(input, target, weight=None, size_average=True, ignore_index=-100, reduce=True):
    r"""这个标准把 `log_softmax` 和 `nll_loss` 结合到了一个方程中.

    详见 :class:`~torch.nn.CrossEntropyLoss`.

    参数:
        input: 变量 :math:`(N, C)` 其中 `C` 为分类的数量
        target: 变脸 :math:`(N)` 其中每个值 `0 <= targets[i] <= C-1`
        weight (Tensor, optional): 自定义的每个类别的权重. 必须是一个大小为 `C` 的 Tensor
        size_average (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量） 
                上取平均值. 如果字段 size_average 被设置为 ``False``, 损失函数的值会在每个 
                mini-batch（小批量）上求和. 当 reduce 的值为 False 时会被忽略. 默认: ``True``
        ignore_index (int, optional): 设置一个目标值, 该目标值会被忽略, 从而不会影响到输入的梯度. 
                当 size_average 为 True 时, 损失函数的值将会在没有被忽略的元素上取平均. 
                默认: -100
        reduce (bool, optional):  默认情况下, 该损失函数的值会根据 size_average 的取值, 在每个 
                mini-batch（小批量）上求平均值或者求和. 当 reduce 是 False 时, 损失函数会对每个 
                batch 元素都返回一个损失值并忽略 size_average. 默认值: ``True``

    实例::

        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.LongTensor(3).random_(5))
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
    """
    return nll_loss(log_softmax(input, 1), target, weight, size_average, ignore_index, reduce)


def binary_cross_entropy(input, target, weight=None, size_average=True):
    r"""计算目标 `target` 与输出 `output` 之间的二进制交叉熵 (Binary Cross Entropy).

    详见 :class:`~torch.nn.BCELoss`.

    参数:
        input: 任意形状的变量
        target: 与输入形状相同的变量
        weight (Variable, optional): 一个可手动指定每个类别的权重.如果给定的话, 会反复与
                输入 tensor 形状相匹配.
        size_average (bool, optional): 默认情况下, 是mini-batchloss的平均值. 然而, 
                如果size_average=False, 则是 `mini-batch` loss的总和. 默认: ``True``

    实例::

        >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
        >>> target = autograd.Variable(torch.LongTensor(3).random_(2))
        >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
        >>> loss.backward()
    """
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if input.nelement() != target.nelement():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.nelement(), input.nelement()))

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)
        if torch.is_tensor(weight):
            weight = Variable(weight)

    return torch._C._nn.binary_cross_entropy(input, target, weight, size_average)


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True):
    r"""Function that measures Binary Cross Entropy between target and output
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Default: ``True``

    Examples::

         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


smooth_l1_loss = _add_docstr(torch._C._nn.smooth_l1_loss, r"""

Function that uses a squared term if the absolute
element-wise error falls below 1 and an L1 term otherwise.

See :class:`~torch.nn.SmoothL1Loss` for details.
""")

l1_loss = _add_docstr(torch._C._nn.l1_loss, r"""
这个方程计算逐个变量的差的绝对值的平均数.

详见 :class:`~torch.nn.L1Loss`.
""")

mse_loss = _add_docstr(torch._C._nn.mse_loss, r"""
计算变量之间的均方差.

详见 :class:`~torch.nn.MSELoss`.
""")


def margin_ranking_loss(input1, input2, target, margin=0, size_average=True):
    """margin_ranking_loss(input1, input2, target, margin=0, size_average=True) -> Variable

    See :class:`~torch.nn.MarginRankingLoss` for details.
    """
    return _functions.loss.MarginRankingLoss.apply(input1, input2, target, margin, size_average)


def hinge_embedding_loss(input, target, margin=1.0, size_average=True):
    """
    详见 :class:`~torch.nn.HingeEmbeddingLoss`.
    """
    return _functions.loss.HingeEmbeddingLoss.apply(input, target, margin, size_average)


multilabel_margin_loss = _add_docstr(torch._C._nn.multilabel_margin_loss, r"""
multilabel_margin_loss(input, target, size_average=True) -> Variable

See :class:`~torch.nn.MultiLabelMarginLoss` for details.
""")

soft_margin_loss = _add_docstr(torch._C._nn.soft_margin_loss, r"""
soft_margin_loss(input, target, size_average=True) -> Variable

See :class:`~torch.nn.SoftMarginLoss` for details.
""")


def multilabel_soft_margin_loss(input, target, weight=None, size_average=True):
    """multilabel_soft_margin_loss(input, target, weight=None, size_average=True) -> Variable

    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    input = torch.sigmoid(input)
    return binary_cross_entropy(input, target, weight, size_average)


def cosine_embedding_loss(input1, input2, target, margin=0, size_average=True):
    """
    详见 :class:`~torch.nn.CosineEmbeddingLoss`.
    """
    return _functions.loss.CosineEmbeddingLoss.apply(input1, input2, target, margin, size_average)


def multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=True):
    """multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=True) -> Variable

    See :class:`~torch.nn.MultiMarginLoss` for details.
    """
    if p != 1 and p != 2:
        raise ValueError('only p == 1 and p == 2 supported')
    if weight is not None and weight.dim() != 1:
        raise ValueError('weight must be one-dimensional')

    return torch._C._nn.multi_margin_loss(input, target, p, margin, weight, size_average)


def pixel_shuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C*r^2, H, W]`` to a
    tensor of shape ``[C, H*r, W*r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples::

        >>> ps = nn.PixelShuffle(3)
        >>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])
    """
    batch_size, channels, in_height, in_width = input.size()
    channels //= upscale_factor ** 2

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, upscale_factor, upscale_factor,
        in_height, in_width)

    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


def upsample(input, size=None, scale_factor=None, mode='nearest'):
    r"""Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for upsampling is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric upsampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [depth] x [height] x width`

    The modes available for upsampling are: `nearest`, `linear` (3D-only),
    `bilinear` (4D-only), `trilinear` (5D-only)

    Args:
        input (Variable): input
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
        mode (string): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear'. Default: 'nearest'
    """
    if input.dim() == 3 and mode == 'nearest':
        return _functions.thnn.UpsamplingNearest1d.apply(input, _single(size), scale_factor)
    elif input.dim() == 4 and mode == 'nearest':
        return _functions.thnn.UpsamplingNearest2d.apply(input, _pair(size), scale_factor)
    elif input.dim() == 5 and mode == 'nearest':
        return _functions.thnn.UpsamplingNearest3d.apply(input, _triple(size), scale_factor)
    elif input.dim() == 3 and mode == 'linear':
        return _functions.thnn.UpsamplingLinear1d.apply(input, _single(size), scale_factor)
    elif input.dim() == 3 and mode == 'bilinear':
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    elif input.dim() == 3 and mode == 'trilinear':
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    elif input.dim() == 4 and mode == 'linear':
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    elif input.dim() == 4 and mode == 'bilinear':
        return _functions.thnn.UpsamplingBilinear2d.apply(input, _pair(size), scale_factor)
    elif input.dim() == 4 and mode == 'trilinear':
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    elif input.dim() == 5 and mode == 'linear':
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    elif input.dim() == 5 and mode == 'bilinear':
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")
    elif input.dim() == 5 and mode == 'trilinear':
        return _functions.thnn.UpsamplingTrilinear3d.apply(input, _triple(size), scale_factor)
    else:
        raise NotImplementedError("Input Error: Only 3D, 4D and 5D input Tensors supported"
                                  " (got {}D) for the modes: nearest | linear | bilinear | trilinear"
                                  " (got {})".format(input.dim(), mode))


def upsample_nearest(input, size=None, scale_factor=None):
    r"""Upsamples the input, using nearest neighbours' pixel values.

    **Note:: This function is deprecated. Use nn.functional.upsample instead**

    Currently spatial and volumetric upsampling are supported (i.e. expected
    inputs are 4 or 5 dimensional).

    Args:
        input (Variable): input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatia
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.upsample_nearest is deprecated. Use nn.functional.upsample instead.")
    return upsample(input, size, scale_factor, mode='nearest')


def upsample_bilinear(input, size=None, scale_factor=None):
    r"""Upscales the input, using bilinear upsampling.

    **Note:: This function is deprecated. Use nn.functional.upsample instead**

    Expected inputs are spatial (4 dimensional). Use upsample_trilinear fo
    volumetric (5 dimensional) inputs.

    Args:
        input (Variable): input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.upsample_bilinear is deprecated. Use nn.functional.upsample instead.")
    return upsample(input, size, scale_factor, mode='bilinear')


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros'):
    r"""Given an :attr:`input` and a flow-field :attr:`grid`, computes the
    `output` using input pixel locations from the grid.

    Uses bilinear interpolation to sample the input pixels.
    Currently, only spatial (4 dimensional) inputs are supported.

    For each output location, :attr:`grid` has `x` and `y`
    input pixel locations which are used to compute output.

    :attr:`grid` has values in the range of `[-1, 1]`. This is because the
    pixel locations are normalized by the input height and width.

    For example, values: x: -1, y: -1 is the left-top pixel of the input
                 values: x: 1, y: 1 is the right-bottom pixel of the input

    If :attr:`grid` has values outside the range of `[-1, 1]`, those locations
    are handled as defined by `padding_mode`. Options are `zeros` or `border`,
    defining those locations to use 0 or image border values as contribution
    to the bilinear interpolation.

    .. Note:: This function is used in building Spatial Transformer Networks

    Args:
        input (Variable): input batch of images (N x C x IH x IW)
        grid (Variable): flow-field of size (N x OH x OW x 2)
        padding_mode (str): padding mode for outside grid values
            'zeros' | 'border'. Default: 'zeros'

    Returns:
        output (Variable): output Tensor

    """
    batch_size, channels, in_height, in_width = input.size()
    return GridSampler.apply(input, grid, padding_mode)


def affine_grid(theta, size):
    r"""Generates a 2d flow field, given a batch of affine matrices :attr:`theta`
    Generally used in conjunction with :func:`grid_sample` to
    implement Spatial Transformer Networks.

    Args:
        theta (Variable): input batch of affine matrices (N x 2 x 3)
        size (torch.Size): the target output image size (N x C x H x W)
                           Example: torch.Size((32, 3, 24, 24))

    Returns:
        output (Variable): output Tensor of size (N x H x W x 2)
    """
    return AffineGridGenerator.apply(theta, size)


def pad(input, pad, mode='constant', value=0):
    r"""Pads tensor.

    Nd constant padding:  The number of dimensions to pad is
        len(padding) // 2 and the dimensions that gets padded begins with the
        last dimension and moves forward.  See below for examples.

    1D, 2D and 3D "reflect"/"replicate" padding:
        1D: 3D input with padding in form (pad_l, pad_r)
        2D: 4D input tensor pad should be in form
        (pad_l, pad_r, pad_t, pad_b ).
        3D: 5D pad (pleft, pright, ptop, pbottom, pfront, pback). No "reflect"
        implementation

    Args:
        input (Variable): Nd tensor
        pad (tuple): m-elem tuple, where m // 2 <= input dimensions and m % 2 == 0
        mode: 'constant', 'reflect' or 'replicate'. Default: 'constant'
        value: fill value for 'constant' padding. Default: 0

    Examples::

        >>> t4d = torch.Tensor(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)
        >>> print(out.data.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.data.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.Tensor(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.data.size())
        torch.Size([3, 9, 7, 3])
    """
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= input.dim(), 'Padding length too large'
    if mode == 'constant':
        return ConstantPadNd.apply(input, pad, value)
    elif input.dim() == 3:
        assert len(pad) == 2, '3D tensors expect 2 values for padding'
        if mode == 'reflect':
            return _functions.thnn.ReflectionPad1d.apply(input, *pad)
        elif mode == 'replicate':
            return _functions.thnn.ReplicationPad1d.apply(input, *pad)
    elif input.dim() == 4:
        assert len(pad) == 4, '4D tensors expect 4 values for padding'
        if mode == 'reflect':
            return _functions.thnn.ReflectionPad2d.apply(input, *pad)
        elif mode == 'replicate':
            return _functions.thnn.ReplicationPad2d.apply(input, *pad)
    elif input.dim() == 5:
        assert len(pad) == 6, '5D tensors expect 6 values for padding'
        if mode == 'reflect':
            raise NotImplementedError
        elif mode == 'replicate':
            return _functions.thnn.ReplicationPad3d.apply(input, *pad)
    else:
        raise NotImplementedError("Only 3D, 4D, 5D padding with non-constant padding are supported for now")


# distance

def pairwise_distance(x1, x2, p=2, eps=1e-6):
    r"""
    计算向量 v1,v2 之间的分批成对距离(意思是可以计算多个，可以参看后面的参数):

    .. math ::
        \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}

    Args:
        x1: 第一个输入张量
        x2: 第二个输入张量
        p: 矩阵范数的维度。默认值是2，即二范数
        eps (float, optional): 指定一个很小的值以避免被零除. 默认值: 1e-6

    Shape:
        - Input: :math:`(N, D)` 其中 `D = vector dimension (矢量维数)`
        - Output: :math:`(N, 1)`

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.pairwise_distance(input1, input2, p=2)
        >>> output.backward()
    """
    assert x1.size() == x2.size(), "Input sizes must be equal."
    assert x1.dim() == 2, "Input must be a 2D matrix."
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff + eps, p).sum(dim=1, keepdim=True)
    return torch.pow(out, 1. / p)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    r"""返回沿着 dim(矢量的维度) 计算的 x1 和 x2 之间的余弦相似度。

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        x1 (Variable): 第一个输入。
        x2 (Variable): 第二个输入。 (大小和 x1 匹配).
        dim (int, optional): 矢量的维度。 默认: 1
        eps (float, optional): 指定一个很小的值以避免被零除. 默认值: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` 其中 D 位于 `dim` 位置.
        - Output: :math:`(\ast_1, \ast_2)` 其中 1 位于`dim`位置.

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.

    The distance swap is described in detail in the paper `Learning shallow
    convolutional feature descriptors with triplet losses`_ by
    V. Balntas, E. Riba et al.

    .. math::
        L(a, p, n) = \frac{1}{N} \left( \sum_{i=1}^N \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\} \right)

    where :math:`d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p`.

    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
        margin: the margin value. Default: 1
        p: the norm degree. Default: 2
        eps: small epsilon value to avoid numerical issues. Default: 1e-6
        swap: compute distance swap. Default: ``False``

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    Example::

        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> input3 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.triplet_margin_loss(input1, input2, input3, p=2)
        >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Input must be a 2D matrix."
    assert margin > 0.0, 'Margin should be positive value.'
    d_p = pairwise_distance(anchor, positive, p, eps)
    d_n = pairwise_distance(anchor, negative, p, eps)
    if swap:
        d_s = pairwise_distance(positive, negative, p, eps)
        d_n = torch.min(d_n, d_s)

    dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)
    loss = torch.mean(dist_hinge)
    return loss


def normalize(input, p=2, dim=1, eps=1e-12):
    r"""  对指定维度的输入执行 :math:`L_p` 规则化。
    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    对于输入的维度的每个 subtensor(子张量) V 扩展。每个子张量展开成一个向量, i.e. :math:`\lVert v \rVert_p` 不是一个规则的矩阵。

    使用默认参数在第二个维度上用欧几里得范数规则化。

    Args:
        input: 输入任何 shape(形状) 的张量
        p (float): 规范化公式中的指数值。默认值: 2
        dim (int): 要减少的维度。默认值: 1
        eps (float): 指定一个很小的值，避免被零除。默认值: 1e-12
    """
    return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
