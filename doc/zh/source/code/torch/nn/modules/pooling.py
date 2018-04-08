import torch
from torch.autograd import Variable

from .module import Module
from .utils import _single, _pair, _triple
from .. import functional as F


class MaxPool1d(Module):
    r"""对于多个输入通道组成的输入信号,应用一维的最大池化 ``max pooling`` 操作

    最简单的例子, 如果输入大小为 :math:`(N, C, L)`, 输出大小为 :math:`(N, C, L_{out})`,
    该层输出值可以用下式精确计算:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, k)  = \max_{{m}=0}^{{kernel\_size}-1} input(N_i, C_j, stride * k + m)
        \end{array}

    | 如果 :attr:`padding` 不是0,那么在输入数据的每条边上会隐式填补对应 :attr:`padding` 数量的0值点
    | :attr:`dilation` 用于控制内核点之间的间隔, `link`_ 很好地可视化展示了 :attr:`dilation` 的功能

    Args:
        kernel_size: 最大池化操作时的窗口大小
        stride: 最大池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding: 输入的每条边隐式补0的数量
        dilation: 用于控制窗口中元素的步长的参数
        return_indices: 如果等于 ``True``, 在返回 max pooling 结果的同时返回最大值的索引.
                        这在之后的 Unpooling 时很有用
        ceil_mode: 如果等于 ``True``, 在计算输出大小时,将采用向上取整来代替默认的向下取整的方式

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})` 遵从如下关系
          :math:`L_{out} = floor((L_{in}  + 2 * padding - dilation * (kernel\_size - 1) - 1) / stride + 1)`

    Examples::

        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50))
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool1d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', dilation=' + str(self.dilation) \
            + ', ceil_mode=' + str(self.ceil_mode) + ')'


class MaxPool2d(Module):
    r"""对于多个输入通道组成的输入信号,应用二维的最大池化 ``max pooling`` 操作

    最简单的例子, 如果输入大小为 :math:`(N, C, H, W)`, 输出大小为 :math:`(N, C, H_{out}, W_{out})`,
    池化窗口大小 :attr:`kernel_size` 为 :math:`(kH, kW)`
    该层输出值可以用下式精确计算:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, h, w)  = \max_{{m}=0}^{kH-1} \max_{{n}=0}^{kW-1}
                               input(N_i, C_j, stride[0] * h + m, stride[1] * w + n)
        \end{array}

    | 如果 :attr:`padding` 不是0, 那么在输入数据的每条边上会隐式填补对应 :attr:`padding` 数量的0值点
    | :attr:`dilation` 用于控制内核点之间的间隔, `link`_ 很好地可视化展示了 :attr:`dilation` 的功能

    参数 :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` 
    可以是以下任意一种数据类型:

        - 单个 ``int`` 类型数据 -- 此时在 height 和 width 维度上将使用相同的值
        - 包含两个 int 类型数据的 ``tuple`` 元组 -- 此时第一个 `int` 数据表示 height 维度上的数值,
          第二个 `int` 数据表示 width 维度上的数值

    Args:
        kernel_size: 最大池化操作时的窗口大小
        stride: 最大池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding: 输入的每条边隐式补0的数量
        dilation: 用于控制窗口中元素的步长的参数
        return_indices: 如果等于 ``True``, 在返回 max pooling 结果的同时返回最大值的索引
                        这在之后的 Unpooling 时很有用
        ceil_mode: 如果等于 ``True``, 在计算输出大小时,将采用向上取整来代替默认的向下取整的方式

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` 遵从如下关系
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 and padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ')'


class MaxUnpool1d(Module):
    r""" :class:`MaxPool1d` 的逆过程

    要注意的是 :class:`MaxPool1d` 并不是完全可逆的, 因为在max pooling过程中非最大值已经丢失

    :class:`MaxUnpool1d` 以 :class:`MaxPool1d` 的输出, 包含最大值的索引作为输入
    计算max poooling的部分逆过程(对于那些最大值区域), 对于那些非最大值区域将设置为0值

    .. note:: `MaxPool1d` 可以将多个输入大小映射到相同的输出大小, 因此反演过程可能会模棱两可
              为适应这一点, 在调用forward函数时可以将需要的输出大小作为额外的参数 `output_size` 传入.
              具体用法,请参阅下面的输入和示例

    Args:
        kernel_size (int or tuple): 最大池化操作时的窗口大小
        stride (int or tuple): 最大池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding (int or tuple): 输入的每条边填充0值的个数

    Inputs:
        - `input`: 需要转化的输入的 Tensor
        - `indices`: `MaxPool1d` 提供的最大值索引
        - `output_size` (可选) : `torch.Size` 类型的数据指定输出的大小

    Shape:
        - Input: :math:`(N, C, H_{in})`
        - Output: :math:`(N, C, H_{out})` 遵从如下关系
          :math:`H_{out} = (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0]`
          或者在调用时指定输出大小 :attr:`output_size`

    Example::

        >>> pool = nn.MaxPool1d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool1d(2, stride=2)
        >>> input = Variable(torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]]))
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        Variable containing:
        (0 ,.,.) =
           0   2   0   4   0   6   0   8
        [torch.FloatTensor of size 1x1x8]

        >>> # Example showcasing the use of output_size
        >>> input = Variable(torch.Tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9]]]))
        >>> output, indices = pool(input)
        >>> unpool(output, indices, output_size=input.size())
        Variable containing:
        (0 ,.,.) =
           0   2   0   4   0   6   0   8   0
        [torch.FloatTensor of size 1x1x9]

        >>> unpool(output, indices)
        Variable containing:
        (0 ,.,.) =
           0   2   0   4   0   6   0   8
        [torch.FloatTensor of size 1x1x8]

    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool1d, self).__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)

    def forward(self, input, indices, output_size=None):
        return F.max_unpool1d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) + ')'


class MaxUnpool2d(Module):
    r""" :class:`MaxPool2d` 的逆过程

    要注意的是 :class:`MaxPool2d` 并不是完全可逆的, 因为在max pooling过程中非最大值已经丢失

    :class:`MaxUnpool2d` 以 :class:`MaxPool2d` 的输出, 包含最大值的索引作为输入
    计算max poooling的部分逆过程(对于那些最大值区域), 对于那些非最大值区域将设置为0值

    .. note:: `MaxPool2d` 可以将多个输入大小映射到相同的输出大小, 因此反演过程可能会模棱两可.
              为适应这一点, 在调用forward函数时可以将需要的输出大小作为额外的参数 `output_size` 传入.
              具体用法,请参阅下面的输入和示例

    Args:
        kernel_size (int or tuple): 最大池化操作时的窗口大小
        stride (int or tuple): 最大池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding (int or tuple): 输入的每条边填充0值的个数

    Inputs:
        - `input`: 需要转化的输入的 Tensor
        - `indices`: `MaxPool2d` 提供的最大值索引
        - `output_size` (可选) : `torch.Size` 类型的数据指定输出的大小

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` 遵从如下关系
          :math:`H_{out} = (H_{in} - 1) * stride[0] -2 * padding[0] + kernel\_size[0]`
          :math:`W_{out} = (W_{in} - 1) * stride[1] -2 * padding[1] + kernel\_size[1]`
          或者在调用时指定输出大小 :attr:`output_size`

    Example::

        >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool2d(2, stride=2)
        >>> input = Variable(torch.Tensor([[[[ 1,  2,  3,  4],
        ...                                  [ 5,  6,  7,  8],
        ...                                  [ 9, 10, 11, 12],
        ...                                  [13, 14, 15, 16]]]]))
        >>> output, indices = pool(input)
        >>> unpool(output, indices)
        Variable containing:
        (0 ,0 ,.,.) =
           0   0   0   0
           0   6   0   8
           0   0   0   0
           0  14   0  16
        [torch.FloatTensor of size 1x1x4x4]

        >>> # specify a different output size than input size
        >>> unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
        Variable containing:
        (0 ,0 ,.,.) =
           0   0   0   0   0
           6   0   8   0   0
           0   0   0  14   0
          16   0   0   0   0
           0   0   0   0   0
        [torch.FloatTensor of size 1x1x5x5]

    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)

    def forward(self, input, indices, output_size=None):
        return F.max_unpool2d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) + ')'


class MaxUnpool3d(Module):
    r""" :class:`MaxPool3d` 的逆过程

    要注意的是 :class:`MaxPool3d` 并不是完全可逆的, 因为在max pooling过程中非最大值已经丢失
    :class:`MaxUnpool3d` 以 :class:`MaxPool3d` 的输出, 包含最大值的索引作为输入
    计算max poooling的部分逆过程(对于那些最大值区域), 对于那些非最大值区域将设置为0值

    .. note:: `MaxPool3d` 可以将多个输入大小映射到相同的输出大小, 因此反演过程可能会模棱两可.
              为适应这一点, 在调用forward函数时可以将需要的输出大小作为额外的参数 `output_size` 传入.
              具体用法,请参阅下面的输入和示例

    Args:
        kernel_size (int or tuple): 最大池化操作时的窗口大小
        stride (int or tuple): 最大池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding (int or tuple): 输入的每条边填充0值的个数

    Inputs:
        - `input`: 需要转化的输入的 Tensor
        - `indices`: `MaxPool3d` 提供的最大值索引
        - `output_size` (可选) : `torch.Size` 类型的数据指定输出的大小

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` 遵从如下关系
          :math:`D_{out} = (D_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0]`
          :math:`H_{out} = (H_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1]`
          :math:`W_{out} = (W_{in} - 1) * stride[2] - 2 * padding[2] + kernel\_size[2]`
          或者在调用时指定输出大小 :attr:`output_size`

    Example::

        >>> # pool of square window of size=3, stride=2
        >>> pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        >>> unpool = nn.MaxUnpool3d(3, stride=2)
        >>> output, indices = pool(Variable(torch.randn(20, 16, 51, 33, 15)))
        >>> unpooled_output = unpool(output, indices)
        >>> unpooled_output.size()
        torch.Size([20, 16, 51, 33, 15])
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool3d, self).__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride if stride is not None else kernel_size)
        self.padding = _triple(padding)

    def forward(self, input, indices, output_size=None):
        return F.max_unpool3d(input, indices, self.kernel_size, self.stride,
                              self.padding, output_size)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) + ')'


class AvgPool1d(Module):
    r"""对于多个输入通道组成的输入信号,应用一维的平均池化 ``average pooling`` 操作

    最简单的例子, 如果输入大小为 :math:`(N, C, L)`, 输出大小为 :math:`(N, C, L_{out})`,
    池化窗口大小 :attr:`kernel_size` 为 :math:`k`
    该层输出值可以用下式精确计算:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, l)  = 1 / k * \sum_{{m}=0}^{k}
                               input(N_i, C_j, stride * l + m)
        \end{array}

    | 如果 :attr:`padding` 不是0, 那么在输入数据的每条边上会隐式填补对应 :attr:`padding` 数量的0值点

    参数 :attr:`kernel_size`, :attr:`stride`, :attr:`padding` 可以为单个 ``int`` 类型的数据
    或者是一个单元素的tuple元组

    Args:
        kernel_size: 平均池化操作时取平均值的窗口的大小
        stride: 平均池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding: 输入的每条边隐式补0的数量
        ceil_mode: 如果等于 ``True``, 在计算输出大小时,将采用向上取整来代替默认的向下取整的方式
        count_include_pad: 如果等于 ``True``, 在计算平均池化的值时,将考虑 ``padding`` 填充的0

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})` 遵从如下关系
          :math:`L_{out} = floor((L_{in}  + 2 * padding - kernel\_size) / stride + 1)`

    Examples::

        >>> # pool with window of size=3, stride=2
        >>> m = nn.AvgPool1d(3, stride=2)
        >>> m(Variable(torch.Tensor([[[1,2,3,4,5,6,7]]])))
        Variable containing:
        (0 ,.,.) =
          2  4  6
        [torch.FloatTensor of size 1x1x3]
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool1d, self).__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return F.avg_pool1d(
            input, self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'


class AvgPool2d(Module):
    r"""对于多个输入通道组成的输入信号,应用二维的平均池化 ``average pooling`` 操作

    最简单的例子,如果输入大小为 :math:`(N, C, H, W)`,输出大小为 :math:`(N, C, H_{out}, W_{out})`,
    池化窗口大小 :attr:`kernel_size` 为 :math:`(kH, kW)`
    该层输出值可以用下式精确计算:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, h, w)  = 1 / (kH * kW) * \sum_{{m}=0}^{kH-1} \sum_{{n}=0}^{kW-1}
                               input(N_i, C_j, stride[0] * h + m, stride[1] * w + n)
        \end{array}

    | 如果 :attr:`padding` 不是0, 那么在输入数据的每条边上会隐式填补对应 :attr:`padding` 数量的0值点

    参数 :attr:`kernel_size`, :attr:`stride`, :attr:`padding`
    可以是以下任意一种数据类型:

        - 单个 ``int`` 类型数据 -- 此时在 height 和 width 维度上将使用相同的值
        - 包含两个 int 类型数据的 ``tuple`` 元组 -- 此时第一个 `int` 数据表示 height 维度上的数值, 
          第二个 `int` 数据表示 width 维度上的数值

    Args:
        kernel_size: 平均池化操作时取平均值的窗口的大小
        stride: 平均池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding: 输入的每条边隐式补0的数量
        ceil_mode: 如果等于 ``True``, 在计算输出大小时,将采用向上取整来代替默认的向下取整的方式
        count_include_pad: 如果等于 ``True``, 在计算平均池化的值时,将考虑 ``padding`` 填充的0

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` 遵从如下关系
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - kernel\_size[0]) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - kernel\_size[1]) / stride[1] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'


class MaxPool3d(Module):
    r"""对于多个输入通道组成的输入信号,应用三维的最大池化 ``max pooling`` 操作

    最简单的例子, 如果输入大小为 :math:`(N, C, D, H, W)`,输出大小为 :math:`(N, C, D_{out}, H_{out}, W_{out})`
    池化窗口大小 :attr:`kernel_size` 为 :math:`(kD, kH, kW)`
    该层输出值可以用下式精确计算:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, d, h, w)  = \max_{{k}=0}^{kD-1} \max_{{m}=0}^{kH-1} \max_{{n}=0}^{kW-1}
                         input(N_i, C_j, stride[0] * k + d, stride[1] * h + m, stride[2] * w + n)
        \end{array}

    | 如果 :attr:`padding` 不是0, 那么在输入数据的每条边上会隐式填补对应 :attr:`padding` 数量的0值点
    | :attr:`dilation` 用于控制内核点之间的间隔, `link`_ 很好地可视化展示了 :attr:`dilation` 的功能

    参数 :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation`
    可以是以下任意一种数据类型:

        - 单个 ``int`` 类型数据 -- 此时在 depth, height 和 width 维度上将使用相同的值
        - 包含三个 int 类型数据的 ``tuple`` 元组 -- 此时第一个 `int` 数据表示 depth 维度上的数值, 
          第二个 `int` 数据表示 height 维度上的数值,第三个 `int` 数据表示 width 维度上的数值

    Args:
        kernel_size: 最大池化操作时的窗口大小
        stride: 最大池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding: 输入所有三条边上隐式补0的数量
        dilation: 用于控制窗口中元素的步长的参数
        return_indices: 如果等于 ``True``, 在返回 max pooling 结果的同时返回最大值的索引
                        这在之后的 Unpooling 时很有用
        ceil_mode: 如果等于 ``True``, 在计算输出大小时,将采用向上取整来代替默认的向下取整的方式

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` 遵从如下关系
          :math:`D_{out} = floor((D_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`H_{out} = floor((H_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[2] - dilation[2] * (kernel\_size[2] - 1) - 1) / stride[2] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool3d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', dilation=' + str(self.dilation) \
            + ', ceil_mode=' + str(self.ceil_mode) + ')'


class AvgPool3d(Module):
    r"""对于多个输入通道组成的输入信号,应用三维的平均池化 ``average pooling`` 操作

    最简单的例子, 如果输入大小为 :math:`(N, C, D, H, W)`,输出大小为 :math:`(N, C, D_{out}, H_{out}, W_{out})`
    池化窗口大小 :attr:`kernel_size` 为 :math:`(kD, kH, kW)`
    该层输出值可以用下式精确计算:

    .. math::

        \begin{array}{ll}
        out(N_i, C_j, d, h, w)  = 1 / (kD * kH * kW) * \sum_{{k}=0}^{kD-1} \sum_{{m}=0}^{kH-1} \sum_{{n}=0}^{kW-1}
                               input(N_i, C_j, stride[0] * d + k, stride[1] * h + m, stride[2] * w + n)
        \end{array}

    | 如果 :attr:`padding` 不是0, 那么在输入数据的每条边上会隐式填补对应 :attr:`padding` 数量的0值点

    参数 :attr:`kernel_size`, :attr:`stride` 可以是以下任意一种数据类型:

        - 单个 ``int`` 类型数据 -- 此时在 depth, height 和 width 维度上将使用相同的值
        - 包含三个 int 类型数据的 ``tuple`` 元组 -- 此时第一个 `int` 数据表示 depth 维度上的数值, 
          第二个 `int` 数据表示 height 维度上的数值,第三个 `int` 数据表示 width 维度上的数值

    Args:
        kernel_size: 平均池化操作时取平均值的窗口的大小
        stride: 平均池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        padding: 输入的每条边隐式补0的数量
        ceil_mode: 如果等于 ``True``, 在计算输出大小时,将采用向上取整来代替默认的向下取整的方式
        count_include_pad: 如果等于 ``True``, 在计算平均池化的值时,将考虑 ``padding`` 填充的0

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` 遵从如下关系
          :math:`D_{out} = floor((D_{in} + 2 * padding[0] - kernel\_size[0]) / stride[0] + 1)`
          :math:`H_{out} = floor((H_{in} + 2 * padding[1] - kernel\_size[1]) / stride[1] + 1)`
          :math:`W_{out} = floor((W_{in} + 2 * padding[2] - kernel\_size[2]) / stride[2] + 1)`

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
        >>> output = m(input)
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(AvgPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return F.avg_pool3d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

    def __setstate__(self, d):
        super(AvgPool3d, self).__setstate__(d)
        self.__dict__.setdefault('padding', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'


class FractionalMaxPool2d(Module):
    r"""对于多个输入通道组成的输入信号,应用二维的分数最大池化 ``fractional max pooling`` 操作

    分数最大池化 ``Fractiona MaxPooling`` 的具体细节描述,详见Ben Graham论文 `Fractional MaxPooling`_

    由目标输出大小确定随机步长,在 kH x kW 区域内进行最大池化的操作
    输出特征的数量与输入通道的数量相同

    Args:
        kernel_size: 最大池化操作时窗口的大小.
                     可以是单个数字 k (等价于 k x k 的正方形窗口) 或者是 一个元组 tuple (kh x kw)
        output_size: oH x oW 形式的输出图像的尺寸.
                     可以用 一个 tuple 元组 (oH, oW) 表示 oH x oW 的输出尺寸, 
                     或者是单个的数字 oH 表示 oH x oH 的输出尺寸
        output_ratio: 如果想用输入图像的百分比来指定输出图像的大小,可选用该选项.
                      使用范围在 (0,1) 之间的一个值来指定.
        return_indices: 如果等于 ``True``,在返回输出结果的同时返回最大值的索引,该索引对 nn.MaxUnpool2d 有用.
                        默认情况下该值等于 ``False``

    Examples:
        >>> # pool of square window of size=3, and target output size 13x12
        >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """

    def __init__(self, kernel_size, output_size=None, output_ratio=None,
                 return_indices=False, _random_samples=None):
        super(FractionalMaxPool2d, self).__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.return_indices = return_indices
        self.register_buffer('_random_samples', _random_samples)
        if output_size is not None:
            self.outh, self.outw = _pair(output_size)
            self.rh, self.rw = None, None
            assert output_ratio is None
        elif output_ratio is not None:
            self.outh, self.outw = None, None
            self.rh, self.rw = _pair(output_ratio)
            assert output_size is None
            assert 0 < self.rh < 1
            assert 0 < self.rw < 1
        else:
            raise ValueError("FractionalMaxPool2d requires specifying either "
                             "an output size, or a pooling ratio")

    def forward(self, input):
        output_size, output_ratio = None, None
        if self.outh is not None:
            output_size = self.outh, self.outw
        else:
            output_ratio = self.rh, self.rw
        ret = self._backend.FractionalMaxPool2d.apply(input, self.kw, self.kh, output_size, output_ratio,
                                                      self._random_samples)
        return ret if self.return_indices else ret[0]


class LPPool2d(Module):
    r"""对于多个输入通道组成的输入信号,应用二维的幂平均池化 ``power-average pooling`` 操作

    在每个窗口内, 输出的计算方式: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`

        - 当 p 无穷大时,等价于最大池化 ``Max Pooling`` 操作
        - 当 ``p=1`` 时, 等价于平均池化 ``Average Pooling`` 操作

    参数 :attr:`kernel_size`, :attr:`stride` 可以是以下任意一种数据类型:

        - 单个 ``int`` 类型数据 -- 此时在height和width维度上将使用相同的值
        - 包含两个 int 类型数据的 ``tuple`` 元组 -- 此时第一个 `int` 数据表示 height 维度上的数值, 
          第二个 `int` 数据表示 width 维度上的数值

    Args:
        kernel_size: 幂平均池化时窗口的大小
        stride: 幂平均池化操作时窗口移动的步长, 默认值是 :attr:`kernel_size`
        ceil_mode: 如果等于 ``True``, 在计算输出大小时,将采用向上取整来代替默认的向下取整的方式

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` 遵从如下关系
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`

    Examples::

        >>> # power-2 pool of square window of size=3, stride=2
        >>> m = nn.LPPool2d(2, 3, stride=2)
        >>> # pool of non-square window of power 1.2
        >>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m(input)

    """

    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False):
        super(LPPool2d, self).__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.lp_pool2d(input, self.norm_type, self.kernel_size,
                           self.stride, self.ceil_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.norm_type) + ', ' \
            + str(self.kernel_size) + ', ' \
            + 'stride=' + str(self.stride) + ', ' \
            + 'ceil_mode=' + str(self.ceil_mode) + ')'


class LPPool1d(Module):
    r"""对于多个输入通道组成的输入信号,应用一维的幂平均池化 ``power-average pooling`` 操作

    在每个窗口内, 输出的计算方式: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`

        - 当 p 无穷大时,等价于最大池化 ``Max Pooling`` 操作
        - 当 ``p=1`` 时, 等价于平均池化 ``Average Pooling`` 操作

    Args:
        kernel_size: 单个 ``int`` 类型的数据,池化窗口的大小
        stride: 单个 ``int`` 类型的数据, 池化操作时窗口移动的步长. 默认值是 :attr:`kernel_size`
        ceil_mode: 如果等于 ``True``, 在计算输出大小时,将采用向上取整来代替默认的向下取整的方式

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})` 遵从如下关系
          :math:`L_{out} = floor((L_{in} + 2 * padding - kernel\_size) / stride + 1)`

    Examples::
        >>> # power-2 pool of window of length 3, with stride 2.
        >>> m = nn.LPPool1d(2, 3, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50))
        >>> output = m(input)
    """

    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False):
        super(LPPool1d, self).__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.lp_pool1d(input, self.norm_type, self.kernel_size,
                           self.stride, self.ceil_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.norm_type) + ', ' \
            + str(self.kernel_size) + ', ' \
            + 'stride=' + str(self.stride) + ', ' \
            + 'ceil_mode=' + str(self.ceil_mode) + ')'


class AdaptiveMaxPool1d(Module):
    r"""对于多个输入通道组成的输入信号,应用一维的自适应最大池化 ``adaptive max pooling`` 操作

    对于任意大小的输入,可以指定输出的尺寸为 H
    输出特征的数量与输入通道的数量相同.

    Args:
        output_size: 目标输出的尺寸 H
        return_indices: 如果等于 ``True``,在返回输出结果的同时返回最大值的索引,该索引对 nn.MaxUnpool1d 有用.
                        默认情况下该值等于 ``False``

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveMaxPool1d(5)
        >>> input = autograd.Variable(torch.randn(1, 64, 8))
        >>> output = m(input)

    """

    def __init__(self, output_size, return_indices=False):
        super(AdaptiveMaxPool1d, self).__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input):
        return F.adaptive_max_pool1d(input, self.output_size, self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'output_size=' + str(self.output_size) + ')'


class AdaptiveMaxPool2d(Module):
    r"""对于多个输入通道组成的输入信号,应用二维的自适应最大池化 ``adaptive max pooling`` 操作

    对于任意大小的输入,可以指定输出的尺寸为 H x W
    输出特征的数量与输入通道的数量相同.

    Args:
        output_size: H x W 形式的输出图像的尺寸.
                     可以用 一个 tuple 元组 (H, W) 表示 H x W 的输出尺寸, 
                     或者是单个的数字 H 表示 H x H 的输出尺寸
        return_indices: 如果等于 ``True``,在返回输出结果的同时返回最大值的索引,该索引对 nn.MaxUnpool2d 有用.
                        默认情况下该值等于 ``False``

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveMaxPool2d((5,7))
        >>> input = autograd.Variable(torch.randn(1, 64, 8, 9))
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveMaxPool2d(7)
        >>> input = autograd.Variable(torch.randn(1, 64, 10, 9))
        >>> output = m(input)

    """

    def __init__(self, output_size, return_indices=False):
        super(AdaptiveMaxPool2d, self).__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input):
        return F.adaptive_max_pool2d(input, self.output_size, self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'output_size=' + str(self.output_size) + ')'


class AdaptiveMaxPool3d(Module):
    r"""对于多个输入通道组成的输入信号,应用三维的自适应最大池化 ``adaptive max pooling`` 操作

    对于任意大小的输入,可以指定输出的尺寸为 D x H x W
    输出特征的数量与输入通道的数量相同.

    Args:
        output_size: D x H x W 形式的输出图像的尺寸.
                     可以用 一个 tuple 元组 (D, H, W) 表示 D x H x W 的输出尺寸, 
                     或者是单个的数字 D 表示 D x D x D 的输出尺寸
        return_indices: 如果等于 ``True``,在返回输出结果的同时返回最大值的索引,该索引对 nn.MaxUnpool3d 有用.
                        默认情况下该值等于 ``False``

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveMaxPool3d((5,7,9))
        >>> input = autograd.Variable(torch.randn(1, 64, 8, 9, 10))
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveMaxPool3d(7)
        >>> input = autograd.Variable(torch.randn(1, 64, 10, 9, 8))
        >>> output = m(input)

    """

    def __init__(self, output_size, return_indices=False):
        super(AdaptiveMaxPool3d, self).__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input):
        return F.adaptive_max_pool3d(input, self.output_size, self.return_indices)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'output_size=' + str(self.output_size) + ')'


class AdaptiveAvgPool1d(Module):
    r"""对于多个输入通道组成的输入信号,应用一维的自适应平均池化 ``adaptive average pooling`` 操作

    对于任意大小的输入,可以指定输出的尺寸为 H
    输出特征的数量与输入通道的数量相同.

    Args:
        output_size: 目标输出的尺寸 H

    Examples:
        >>> # target output size of 5
        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = autograd.Variable(torch.randn(1, 64, 8))
        >>> output = m(input)

    """

    def __init__(self, output_size):
        super(AdaptiveAvgPool1d, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool1d(input, self.output_size)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'output_size=' + str(self.output_size) + ')'


class AdaptiveAvgPool2d(Module):
    r"""对于多个输入通道组成的输入信号,应用二维的自适应平均池化 ``adaptive average pooling`` 操作

    对于任意大小的输入,可以指定输出的尺寸为 H x W
    输出特征的数量与输入通道的数量相同.

    Args:
        output_size: H x W 形式的输出图像的尺寸.
                     可以用 一个 tuple 元组 (H, W) 表示 H x W 的输出尺寸, 
                     或者是单个的数字 H 表示 H x H 的输出尺寸

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = autograd.Variable(torch.randn(1, 64, 8, 9))
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = autograd.Variable(torch.randn(1, 64, 10, 9))
        >>> output = m(input)

    """

    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool2d(input, self.output_size)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'output_size=' + str(self.output_size) + ')'


class AdaptiveAvgPool3d(Module):
    r"""对于多个输入通道组成的输入信号,应用三维的自适应平均池化 ``adaptive average pooling`` 操作

    对于任意大小的输入,可以指定输出的尺寸为 D x H x W
    输出特征的数量与输入通道的数量相同.

    Args:
        output_size: D x H x W 形式的输出图像的尺寸.
                     可以用 一个 tuple 元组 (D, H, W) 表示 D x H x W 的输出尺寸, 
                     或者是单个的数字 D 表示 D x D x D 的输出尺寸

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveAvgPool3d((5,7,9))
        >>> input = autograd.Variable(torch.randn(1, 64, 8, 9, 10))
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = autograd.Variable(torch.randn(1, 64, 10, 9, 8))
        >>> output = m(input)

    """

    def __init__(self, output_size):
        super(AdaptiveAvgPool3d, self).__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool3d(input, self.output_size)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'output_size=' + str(self.output_size) + ')'
