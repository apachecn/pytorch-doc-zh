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
    r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    See :class:`~torch.nn.Conv1d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight: filters of shape (out_channels x in_channels x kW)
        bias: optional bias of shape (out_channels). Default: None
        stride: the stride of the convolving kernel. Can be a single number or
          a tuple (sW,). Default: 1
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple (padW,). Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple (dW,). Default: 1
        groups: split input into groups, in_channels should be divisible by
          the number of groups. Default: 1

    Examples::

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
    r"""Applies a 2D convolution over an input image composed of several input
    planes.

    See :class:`~torch.nn.Conv2d` for details and output shape.

    Args:
        input: input tensor (minibatch x in_channels x iH x iW)
        weight: filters tensor (out_channels x in_channels/groups x kH x kW)
        bias: optional bias tensor (out_channels). Default: None
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple (sH, sW). Default: 1
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple (padH, padW). Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple (dH, dW). Default: 1
        groups: split input into groups, in_channels should be divisible by the
          number of groups. Default: 1

    Examples::

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
    r"""Applies a 3D convolution over an input image composed of several input
    planes.

    See :class:`~torch.nn.Conv3d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight: filters tensor of shape (out_channels x in_channels x kT x kH x kW)
        bias: optional bias tensor of shape (out_channels). Default: None
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple (sT, sH, sW). Default: 1
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple (padT, padH, padW). Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple (dT, dH, dW). Default: 1
        groups: split input into groups, in_channels should be divisible by
          the number of groups. Default: 1

    Examples::

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
    r"""Applies a 1D transposed convolution operator over an input signal
    composed of several input planes, sometimes also called "deconvolution".

    See :class:`~torch.nn.ConvTranspose1d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight: filters of shape (in_channels x out_channels x kW)
        bias: optional bias of shape (out_channels). Default: None
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple (sW,). Default: 1
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple (padW,). Default: 0
        output_padding: implicit zero-paddings of 0 <= padding < stride on both
          sides of the output. Can be a single number or a tuple (out_padW,).
          Default: 0
        groups: split input into groups, in_channels should be divisible by the
          number of groups. Default: 1
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple (dW,). Default: 1
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
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes, sometimes also called "deconvolution".

    See :class:`~torch.nn.ConvTranspose2d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight: filters of shape (in_channels x out_channels x kH x kW)
        bias: optional bias of shape (out_channels). Default: None
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple (sH, sW). Default: 1
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple (padH, padW). Default: 0
        output_padding: implicit zero-paddings of 0 <= padding < stride on both
          sides of the output. Can be a single number or a tuple
          (out_padH, out_padW). Default: 0
        groups: split input into groups, in_channels should be divisible by the
          number of groups. Default: 1
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple (dH, dW). Default: 1
    """

    if input is not None and input.dim() != 4:
        raise ValueError("Expected 4D tensor as input, got {}D tensor instead.".format(input.dim()))

    f = _ConvNd(_pair(stride), _pair(padding), _pair(dilation), True,
                _pair(output_padding), groups, torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cudnn.enabled)
    return f(input, weight, bias)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    r"""Applies a 3D transposed convolution operator over an input image
    composed of several input planes, sometimes also called "deconvolution"

    See :class:`~torch.nn.ConvTranspose3d` for details and output shape.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight: filters of shape (in_channels x out_channels x kH x kW)
        bias: optional bias of shape (out_channels). Default: None
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple (sT, sH, sW). Default: 1
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple (padT, padH, padW). Default: 0
        output_padding: implicit zero-paddings of 0 <= padding < stride on both
          sides of the output. Can be a single number or a tuple
          (out_padT, out_padH, out_padW). Default: 0
        groups: split input into groups, in_channels should be divisible by the
          number of groups. Default: 1
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple (dT, dH, dW). Default: 1
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
    r"""Applies a 1D average pooling over an input signal composed of several
    input planes.

    See :class:`~torch.nn.AvgPool1d` for details and output shape.

    Args:
        input: input tensor (minibatch x in_channels x iW)
        kernel_size: the size of the window. Can be a single number or a
          tuple (kW,)
        stride: the stride of the window. Can be a single number or a tuple
          (sW,). Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple (padW,). Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the
            output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``

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

Applies 2D average-pooling operation in kh x kw regions by step size
dh x dw steps. The number of output features is equal to the number of
input planes.

See :class:`~torch.nn.AvgPool2d` for details and output shape.

Args:
    input: input tensor (minibatch x in_channels x iH x iW)
    kernel_size: size of the pooling region. Can be a single number or a
      tuple (kH x kW)
    stride: stride of the pooling operation. Can be a single number or a
      tuple (sH, sW). Default is equal to kernel size
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple (padH, padW). Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` in the formula
        to compute the output shape. Default: ``False``
    count_include_pad: when True, will include the zero-padding in th
        averaging calculation. Default: ``True``
""")

avg_pool3d = _add_docstr(torch._C._nn.avg_pool3d, r"""
avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Variable

Applies 3D average-pooling operation in kt x kh x kw regions by step
size dt x dh x dw steps. The number of output features is equal to the
number of input planes / dt.

See :class:`~torch.nn.AvgPool3d` for details and output shape.

Args:
    input: input tensor (minibatch x in_channels x iT x iH x iW)
    kernel_size: size of the pooling region. Can be a single number or a
      tuple (kT x kH x kW)
    stride: stride of the pooling operation. Can be a single number or a
      tuple (sT, sH, sW). Default is equal to kernel size
    padding: implicit zero paddings on both sides of the input. Can be a
      single number or a tuple (padT, padH, padW), Default: 0
    ceil_mode: when True, will use `ceil` instead of `floor` in the formula
        to compute the output shape
    count_include_pad: when True, will include the zero-padding in th
        averaging calculation
""")


# share the same interface
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    """Applies a 1D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool1d` for details.
    """
    ret = _functions.thnn.MaxPool1d.apply(input, kernel_size, stride, padding, dilation,
                                          ceil_mode)
    return ret if return_indices else ret[0]


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    """Applies a 2D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool2d` for details.
    """
    ret = torch._C._nn.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    return ret if return_indices else ret[0]


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    """Applies a 3D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool2d` for details.
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
    """Computes a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
    kernel_size = _single(kernel_size)
    stride = _single(stride)
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool2d(input.unsqueeze(3), indices.unsqueeze(3), output_size + [1]).squeeze(3)


def max_unpool2d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    """Computes a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool2d(input, indices, output_size)


def max_unpool3d(input, indices, kernel_size, stride=None, padding=0,
                 output_size=None):
    """Computes a partial inverse of :class:`MaxPool3d`.

    See :class:`~torch.nn.MaxUnpool3d` for details.
    """
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
    output_size = _unpool_output_size(input, kernel_size, stride, padding,
                                      output_size)
    return torch._C._nn.max_unpool3d(input, indices, output_size, stride, padding)


def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    """Applies a 2D power-average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.LPPool2d` for details.
    """
    kw, kh = utils._pair(kernel_size)
    out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return out.mul(kw * kh).pow(1. / norm_type)


def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    """Applies a 1D power-average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.LPPool1d` for details.
    """
    out = avg_pool1d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return out.mul(kernel_size).pow(1. / norm_type)


def adaptive_max_pool1d(input, output_size, return_indices=False):
    r"""Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    ret = _functions.thnn.AdaptiveMaxPool1d.apply(input, output_size)
    return ret if return_indices else ret[0]


def adaptive_max_pool2d(input, output_size, return_indices=False):
    r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    ret = _functions.thnn.AdaptiveMaxPool2d.apply(input, output_size)
    return ret if return_indices else ret[0]


def adaptive_max_pool3d(input, output_size, return_indices=False):
    r"""Applies a 3D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    ret = _functions.thnn.AdaptiveMaxPool3d.apply(input, output_size)
    return ret if return_indices else ret[0]


def adaptive_avg_pool1d(input, output_size):
    r"""Applies a 1D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
    """
    return _functions.thnn.AdaptiveAvgPool1d.apply(input, output_size)


def adaptive_avg_pool2d(input, output_size):
    r"""Applies a 2D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
    """
    return _functions.thnn.AdaptiveAvgPool2d.apply(input, output_size)


def adaptive_avg_pool3d(input, output_size):
    r"""Applies a 3D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
    """
    return _functions.thnn.AdaptiveAvgPool3d.apply(input, output_size)


# Activation functions

def dropout(input, p=0.5, training=False, inplace=False):
    return _functions.dropout.Dropout.apply(input, p, training, inplace)


def alpha_dropout(input, p=0.5, training=False):
    r"""Applies alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.

    Args:
        p (float, optional): the drop probability. Default: 0.5
        training (bool, optional): switch between training and evaluation mode. Default: ``False``
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

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    return threshold(input, 0, 0, inplace)


glu = _add_docstr(torch._C._nn.glu, r"""
glu(input, dim=-1) -> Variable

The gated linear unit. Computes:

.. math ::

    H = A \times \sigma(B)

where `input` is split in half along `dim` to form `A` and `B`.

See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

Args:
    input (Variable): input variable
    dim (int): dimension on which to split the input
""")

hardtanh = _add_docstr(torch._C._nn.hardtanh, r"""
hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Variable

Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
details.
""")


def relu6(input, inplace=False):
    r"""relu6(input, inplace=False) -> Variable

    Applies the element-wise function :math:`{ReLU6}(x) = min(max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """
    return hardtanh(input, 0, 6, inplace)


elu = _add_docstr(torch._C._nn.elu, r"""
elu(input, alpha=1., inplace=False) -> Variable

Applies element-wise,
:math:`f(x) = max(0,x) + min(0, alpha * (exp(x) - 1))`.

See :class:`~torch.nn.ELU` for more details.
""")


def selu(input, inplace=False):
    r"""selu(input, inplace=False) -> Variable

    Applies element-wise,
    :math:`f(x) = scale * (\max(0,x) + \min(0, alpha * (\exp(x) - 1)))`,
    with ``alpha=1.6732632423543772848170429916717`` and
    ``scale=1.0507009873554804934193349852946``.

    See :class:`~torch.nn.SELU` for more details.
    """
    return _functions.thnn.SELU.apply(input, inplace)


leaky_relu = _add_docstr(torch._C._nn.leaky_relu, r"""
leaky_relu(input, negative_slope=0.01, inplace=False) -> Variable

Applies element-wise,
:math:`f(x) = max(0, x) + {negative\_slope} * min(0, x)`

See :class:`~torch.nn.LeakyReLU` for more details.
""")


# prelu = _add_docstr(torch._C._nn.prelu, r"""
# prelu(input, weight) -> Variable
# """)
def prelu(input, weight):
    r"""prelu(input, weight) -> Variable

    Applies element-wise the function
    :math:`PReLU(x) = max(0,x) + weight * min(0,x)` where weight is a
    learnable parameter.

    See :class:`~torch.nn.PReLU` for more details.
    """
    return _functions.thnn.PReLU.apply(input, weight)


rrelu = _add_docstr(torch._C._nn.rrelu, r"""
rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Variable
""")

logsigmoid = _add_docstr(torch._C._nn.log_sigmoid, r"""
logsigmoid(input) -> Variable

Applies element-wise :math:`LogSigmoid(x) = log( 1 / (1 + exp(-x_i)))`

See :class:`~torch.nn.LogSigmoid` for more details.
""")

hardshrink = _add_docstr(torch._C._nn.hardshrink, r"""
hardshrink(input, lambd=0.5) -> Variable

Applies the hard shrinkage function element-wise

See :class:`~torch.nn.Hardshrink` for more details.


""")


def tanhshrink(input):
    r"""tanhshrink(input) -> Variable

    Applies element-wise, :math:`Tanhshrink(x) = x - Tanh(x)`

    See :class:`~torch.nn.Tanhshrink` for more details.
    """
    return input - input.tanh()


def softsign(input):
    r"""softsign(input) -> Variable

    Applies element-wise, the function :math:`f(x) = x / (1 + |x|)`

    See :class:`~torch.nn.Softsign` for more details.
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
    r"""Applies a softmin function.

    Note that softmin(x) = softmax(-x). See softmax definition for mathematical formula.

    See :class:`~torch.nn.Softmin` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which softmin will be computed (so every slice
            along dim will sum to 1).
    """
    if dim is None:
        dim = _get_softmax_dim('softmin', input.dim(), _stacklevel)
    return torch._C._nn.softmax(-input, dim)


def softmax(input, dim=None, _stacklevel=3):
    r"""Applies a softmax function.

    Softmax is defined as:

    :math:`softmax(x) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which softmax will be computed.

    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).

    """
    if dim is None:
        dim = _get_softmax_dim('softmax', input.dim(), _stacklevel)
    return torch._C._nn.softmax(input, dim)


def log_softmax(input, dim=None, _stacklevel=3):
    r"""Applies a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    See :class:`~torch.nn.LogSoftmax` for more details.

    Arguments:
        input (Variable): input
        dim (int): A dimension along which log_softmax will be computed.
    """
    if dim is None:
        dim = _get_softmax_dim('log_softmax', input.dim(), _stacklevel)
    return torch._C._nn.log_softmax(input, dim)


def softshrink(input, lambd=0.5):
    r"""softshrink(input, lambd=0.5) -> Variable

    Applies the soft shrinkage function elementwise

    See :class:`~torch.nn.Softshrink` for more details.
    """
    return _functions.thnn.auto.Softshrink.apply(input, lambd)


def tanh(input):
    r"""tanh(input) -> Variable

    Applies element-wise,
    :math:`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

    See :class:`~torch.nn.Tanh` for more details.
    """
    return input.tanh()


def sigmoid(input):
    r"""sigmoid(input) -> Variable

    Applies the element-wise function :math:`f(x) = 1 / ( 1 + exp(-x))`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    return input.sigmoid()


# etc.

def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
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
    r"""Poisson negative log likelihood loss.

    See :class:`~torch.nn.PoissonNLLLoss` for details.

    Args:
        input: expectation of underlying Poisson distribution.
        target: random sample :math:`target \sim Pois(input)`.
        log_input: if ``True`` the loss is computed as
            `exp(input) - target * input`, if ``False`` then loss is
            `input - target * log(input+eps)`. Default: ``True``
        full: whether to compute full loss, i. e. to add the Stirling
            approximation term. Default: ``False``
            `target * log(target) - target + 0.5 * log(2 * pi * target)`.
        size_average: By default, the losses are averaged over observations for
            each minibatch. However, if the field sizeAverage is set to False,
            the losses are instead summed for each minibatch. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of log(0) when
            log_input=False. Default: 1e-8
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
kl_div(input, target, size_average=True) -> Variable

The `Kullback-Leibler divergence`_ Loss.

See :class:`~torch.nn.KLDivLoss` for details.

Args:
    input: Variable of arbitrary shape
    target: Variable of the same shape as input
    size_average: if ``True`` the output is divided by the number of elements
        in input tensor. Default: ``True``
    reduce (bool, optional): By default, the losses are averaged
        over observations for each minibatch, or summed, depending on
        size_average. When reduce is False, returns a loss per batch
        element instead and ignores size_average. Default: ``True``

""")


def cross_entropy(input, target, weight=None, size_average=True, ignore_index=-100, reduce=True):
    r"""This criterion combines `log_softmax` and `nll_loss` in a single
    function.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input: Variable :math:`(N, C)` where `C = number of classes`
        target: Variable :math:`(N)` where each value is
            `0 <= targets[i] <= C-1`
        weight (Tensor, optional): a manual rescaling weight given to each
                class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Ignored if reduce is False. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
                and does not contribute to the input gradient. When size_average is
                True, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per batch element instead and ignores
                size_average. Default: ``True``

    Examples::

        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.LongTensor(3).random_(5))
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
    """
    return nll_loss(log_softmax(input, 1), target, weight, size_average, ignore_index, reduce)


def binary_cross_entropy(input, target, weight=None, size_average=True):
    r"""Function that measures the Binary Cross Entropy
    between the target and the output.

    See :class:`~torch.nn.BCELoss` for details.

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
smooth_l1_loss(input, target, size_average=True) -> Variable

Function that uses a squared term if the absolute
element-wise error falls below 1 and an L1 term otherwise.

See :class:`~torch.nn.SmoothL1Loss` for details.
""")

l1_loss = _add_docstr(torch._C._nn.l1_loss, r"""
l1_loss(input, target, size_average=True, reduce=True) -> Variable

Function that takes the mean element-wise absolute value difference.

See :class:`~torch.nn.L1Loss` for details.
""")

mse_loss = _add_docstr(torch._C._nn.mse_loss, r"""
mse_loss(input, target, size_average=True, reduce=True) -> Variable

Measures the element-wise mean squared error.

See :class:`~torch.nn.MSELoss` for details.
""")


def margin_ranking_loss(input1, input2, target, margin=0, size_average=True):
    """margin_ranking_loss(input1, input2, target, margin=0, size_average=True) -> Variable

    See :class:`~torch.nn.MarginRankingLoss` for details.
    """
    return _functions.loss.MarginRankingLoss.apply(input1, input2, target, margin, size_average)


def hinge_embedding_loss(input, target, margin=1.0, size_average=True):
    """hinge_embedding_loss(input, target, margin=1.0, size_average=True) -> Variable

    See :class:`~torch.nn.HingeEmbeddingLoss` for details.
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
    """cosine_embedding_loss(input1, input2, target, margin=0, size_average=True) -> Variable

    See :class:`~torch.nn.CosineEmbeddingLoss` for details.
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
    Computes the batchwise pairwise distance between vectors v1,v2:

    .. math ::
        \Vert x \Vert _p := \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}

    Args:
        x1: first input tensor
        x2: second input tensor
        p: the norm degree. Default: 2
        eps (float, optional): Small value to avoid division by zero. Default: 1e-6

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
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
    r"""Returns cosine similarity between x1 and x2, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.

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
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """
    return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
