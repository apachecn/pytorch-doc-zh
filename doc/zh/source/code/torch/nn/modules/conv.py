# coding=utf-8
import math
import torch
from torch.nn.parameter import Parameter
from .. import functional as F
from .module import Module
from .utils import _single, _pair, _triple


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv1d(_ConvNd):
    r"""一维卷积层
    输入矩阵的维度为 :math:`(N, C_{in}, L)`, 输出矩阵维度为 :math:`(N, C_{out}, L_{out})`.
    其中N为输入数量, C为每个输入样本的通道数量,  L为样本中一个通道下的数据的长度.
    算法如下:

    .. math::

        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}

    :math:`\star` 是互相关运算符, 上式带 :math:`\star` 项为卷积项.

    | :attr:`stride` 计算相关系数的步长, 可以为 tuple .
    | :attr:`padding` 处理边界时在两侧补0数量  
    | :attr:`dilation` 采样间隔数量. 大于1时为非致密采样, 如对(a,b,c,d,e)采样时, 若池化规模为2, 
    dilation 为1时, 使用 (a,b);(b,c)... 进行池化, dilation 为1时, 使用 (a,c);(b,d)... 进行池化.
    | :attr:`groups` 控制输入和输出之间的连接, group=1, 输出是所有输入的卷积；group=2, 此时相当于
    有并排的两个卷基层, 每个卷积层只在对应的输入通道和输出通道之间计算, 并且输出时会将所有
    输出通道简单的首尾相接作为结果输出.
     `in_channels` 和 `out_channels` 都要可以被 groups 整除.

    .. note::
        数据的最后一列可能会因为 kernal 大小设定不当而被丢弃（大部分发生在 kernal 大小不能被输入
        整除的时候, 适当的 padding 可以避免这个问题）. 

    Args:
        - in_channels (int):  输入信号的通道数.
        - out_channels (int): 卷积后输出结果的通道数.
        - kernel_size (int or tuple): 卷积核的形状.
        - stride (int or tuple, optional): 卷积每次移动的步长, 默认为1.
        - padding (int or tuple, optional): 处理边界时填充0的数量, 默认为0(不填充).
        - dilation (int or tuple, optional): 采样间隔数量, 默认为1, 无间隔采样.
        - groups (int, optional): 输入与输出通道的分组数量. 当不为1时, 默认为1(全连接).
        - bias (bool, optional): 为 ``True`` 时,  添加偏置.

    Shape:
        - 输入 Input: :math:`(N, C_{in}, L_{in})`
        - 输出 Output: :math:`(N, C_{out}, L_{out})` 其中 
          :math:`L_{out} = floor((L_{in}  + 2 * padding - dilation * (kernel\_size - 1) - 1) / stride + 1)`

    Attributes:
        weight (Tensor): 卷积网络层间连接的权重, 是模型需要学习的变量, 形状为
            (out_channels, in_channels, kernel_size)
        bias (Tensor): 偏置, 是模型需要学习的变量, 形状为
            (out_channels)

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50))
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv2d(_ConvNd):
    r"""二维卷积层
    输入矩阵的维度为 :math:`(N, C_{in}, H, W)` , 输出矩阵维度为 :math:`(N, C_{out}, H_{out}, W_{out})` .
    其中N为输入数量, C为每个输入样本的通道数量, H, W 分别为样本中一个通道下的数据的形状. 
    算法如下: 

    .. math::

        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}

    :math:`\star` 是互相关运算符, 上式带 :math:`\star` 项为卷积项.

    | :attr:`stride` 计算相关系数的步长, 可以为 tuple . 
    | :attr:`padding` 处理边界时在每个维度首尾补0数量. 
    | :attr:`dilation` 采样间隔数量. 大于1时为非致密采样. 
    | :attr:`groups` 控制输入和输出之间的连接,  group=1, 输出是所有输入的卷积； group=2, 此时
    相当于有并排的两个卷基层, 每个卷积层只在对应的输入通道和输出通道之间计算, 并且输出时会将所有
    输出通道简单的首尾相接作为结果输出. 
            `in_channels` 和 `out_channels` 都要可以被 groups 整除. 
    
    :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` 可以为:

        -  单个 ``int`` 值  -- 宽和高均被设定为此值. 
        -  由两个 ``int`` 组成的 ``tuple``  -- 第一个 ``int`` 为高,  第二个 ``int`` 为宽. 

    .. note::
        数据的最后一列可能会因为 kernal 大小设定不当而被丢弃（大部分发生在 kernal 大小不能被输入
        整除的时候, 适当的 padding 可以避免这个问题）. 
        
    Args:
        - in_channels (int): 输入信号的通道数. 
        - out_channels (int): 卷积后输出结果的通道数. 
        - kernel_size (int or tuple): 卷积核的形状. 
        - stride (int or tuple, optional): 卷积每次移动的步长, 默认为1. 
        - padding (int or tuple, optional): 处理边界时填充0的数量, 默认为0(不填充). 
        - dilation (int or tuple, optional): 采样间隔数量, 默认为1, 无间隔采样. 
        - groups (int, optional): 输入与输出通道的分组数量. 当不为1时, 默认为1(全连接). 
        - bias (bool, optional): 为 ``True`` 时,  添加偏置. 

    Shape:
        - 输入 Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - 输出 Output: :math:`(N, C_{out}, H_{out}, W_{out})` 其中
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`

    Attributes:
        weight (Tensor): 卷积网络层间连接的权重, 是模型需要学习的变量, 形状为
            (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   偏置, 是模型需要学习的变量, 形状为 (out_channels)
       

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv3d(_ConvNd):
    r"""三维卷基层
    输入矩阵的维度为 :math:`(N, C_{in}, D, H, W)`, 输出矩阵维度为::math:`(N, C_{out}, D_{out}, H_{out}, W_{out})`. 
    其中N为输入数量, C为每个输入样本的通道数量,  D,  H,  W 分别为样本中一个通道下的数据的形状. 
    算法如下: 

    .. math::

        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}

    :math:`\star` 是互相关运算符, 上式带 :math:`\star` 项为卷积项.

    | :attr:`stride` 计算相关系数的步长, 可以为 tuple . 
    | :attr:`padding` 处理边界时在每个维度首尾补0数量. 
    | :attr:`dilation` 采样间隔数量. 大于1时为非致密采样. 
    | :attr:`groups` 控制输入和输出之间的连接,  group=1, 输出是所有输入的卷积； group=2, 此时
    相当于有并排的两个卷基层, 每个卷积层只在对应的输入通道和输出通道之间计算, 并且输出时会将所有
    输出通道简单的首尾相接作为结果输出. 
            `in_channels` 和 `out_channels` 都要可以被 groups 整除. 

    :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` 可以为:

        -  单个 ``int`` 值  -- 宽和高和深度均被设定为此值. 
        -  由三个 ``int`` 组成的 ``tuple``  -- 第一个 ``int`` 为深度,  第二个 ``int`` 为高度, 第三个 ``int`` 为宽度. 

    .. note::
        数据的最后一列可能会因为 kernal 大小设定不当而被丢弃（大部分发生在 kernal 大小不能被输入
        整除的时候, 适当的 padding 可以避免这个问题）. 
           
    Args:
        - in_channels (int): 输入信号的通道数. 
        - out_channels (int): 卷积后输出结果的通道数. 
        - kernel_size (int or tuple): 卷积核的形状. 
        - stride (int or tuple, optional): 卷积每次移动的步长, 默认为1. 
        - padding (int or tuple, optional): 处理边界时填充0的数量, 默认为0(不填充). 
        - dilation (int or tuple, optional): 采样间隔数量, 默认为1, 无间隔采样. 
        - groups (int, optional): 输入与输出通道的分组数量. 当不为1时, 默认为1(全连接). 
        - bias (bool, optional): 为 ``True`` 时,  添加偏置. 
        
    Shape:
        - 输入 Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - 输出 Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 其中
          :math:`D_{out} = floor((D_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`H_{out} = floor((H_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[2] - dilation[2] * (kernel\_size[2] - 1) - 1) / stride[2] + 1)`

    Attributes:
        weight (Tensor): 卷积网络层间连接的权重, 是模型需要学习的变量, 形状为
            (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor): 偏置, 是模型需要学习的变量, 形状为 (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias)

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class _ConvTransposeMixin(object):

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])


class ConvTranspose1d(_ConvTransposeMixin, _ConvNd):
    r"""一维反卷积层
    反卷积层可以理解为输入的数据和卷积核的位置反转的卷积操作. 
    反卷积有时候也会被翻译成解卷积. 

    | :attr:`stride` 计算相关系数的步长. 
    | :attr:`padding` 处理边界时在每个维度首尾补0数量. 
    | :attr:`output_padding` 输出时候在首尾补0的数量. （卷积时, 形状不同的输入数据
    对相同的核函数可以产生形状相同的结果；反卷积时, 同一个输入对相同的核函数可以产生多
    个形状不同的输出, 而输出结果只能有一个, 因此必须对输出形状进行约束）.
    | :attr:`dilation` 采样间隔数量. 大于1时为非致密采样. 
    | :attr:`groups` 控制输入和输出之间的连接,  group=1, 输出是所有输入的卷积； group=2, 此时
    相当于有并排的两个卷基层, 每个卷积层只在对应的输入通道和输出通道之间计算, 并且输出时会将所有
    输出通道简单的首尾相接作为结果输出. 
            `in_channels` 和 `out_channels` 都要可以被 groups 整除. 
    
    .. note::
        数据的最后一列可能会因为 kernal 大小设定不当而被丢弃（大部分发生在 kernal 大小不能被输入
        整除的时候, 适当的 padding 可以避免这个问题）. 

    Args:
        - in_channels (int): 输入信号的通道数. 
        - out_channels (int): 卷积后输出结果的通道数. 
        - kernel_size (int or tuple): 卷积核的形状. 
        - stride (int or tuple, optional): 卷积每次移动的步长, 默认为1. 
        - padding (int or tuple, optional): 处理边界时填充0的数量, 默认为0(不填充). 
        - output_padding (int or tuple, optional): 输出时候在首尾补值的数量, 默认为0. （卷积时, 形状不同的输入数据
        对相同的核函数可以产生形状相同的结果；反卷积时, 同一个输入对相同的核函数可以产生多
        个形状不同的输出, 而输出结果只能有一个, 因此必须对输出形状进行约束）
        - groups (int, optional): 输入与输出通道的分组数量. 当不为1时, 默认为1(全连接). 
        - bias (bool, optional): 为 ``True`` 时,  添加偏置. 
        - dilation (int or tuple, optional): 采样间隔数量, 默认为1, 无间隔采样. 

    Shape:
        - 输入 Input: :math:`(N, C_{in}, L_{in})`
        - 输出 Output: :math:`(N, C_{out}, L_{out})` 其中
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + kernel\_size + output\_padding`

    Attributes:
        weight (Tensor): 卷积网络层间连接的权重, 是模型需要学习的变量, 形状为weight (Tensor): 卷积网络层间连接的权重, 是模型需要学习的变量, 形状为
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   偏置, 是模型需要学习的变量, 形状为 (out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super(ConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose1d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    r"""二维反卷积层
    反卷积层可以理解为输入的数据和卷积核的位置反转的卷积操作. 
    反卷积有时候也会被翻译成解卷积. 
    
    | :attr:`stride` 计算相关系数的步长. 
    | :attr:`padding` 处理边界时在每个维度首尾补0数量. 
    | :attr:`output_padding` 输出时候在每一个维度首尾补0的数量. （卷积时, 形状不同的输入数据
    对相同的核函数可以产生形状相同的结果；反卷积时, 同一个输入对相同的核函数可以产生多
    个形状不同的输出, 而输出结果只能有一个, 因此必须对输出形状进行约束）.
    | :attr:`dilation` 采样间隔数量. 大于1时为非致密采样. 
    | :attr:`groups` 控制输入和输出之间的连接,  group=1, 输出是所有输入的卷积； group=2, 此时
    相当于有并排的两个卷基层, 每个卷积层只在对应的输入通道和输出通道之间计算, 并且输出时会将所有
    输出通道简单的首尾相接作为结果输出. 
            `in_channels` 和 `out_channels` 都应当可以被 groups 整除. 
   
    :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding` 可以为:

        - 单个 ``int`` 值  -- 宽和高均被设定为此值. 
        - 由两个 ``int`` 组成的 ``tuple``  -- 第一个 ``int`` 为高度,  第二个 ``int`` 为宽度. 

    .. note::
        数据的最后一列可能会因为 kernal 大小设定不当而被丢弃（大部分发生在 kernal 大小不能被输入
        整除的时候, 适当的 padding 可以避免这个问题）.  

    Args:
        - in_channels (int): 输入信号的通道数. 
        - out_channels (int): 卷积后输出结果的通道数. 
        - kernel_size (int or tuple): 卷积核的形状. 
        - stride (int or tuple, optional): 卷积每次移动的步长, 默认为1. 
        - padding (int or tuple, optional): 处理边界时填充0的数量, 默认为0(不填充). 
        - output_padding (int or tuple, optional): 输出时候在首尾补值的数量, 默认为0. （卷积时, 形状不同的输入数据
        对相同的核函数可以产生形状相同的结果；反卷积时, 同一个输入对相同的核函数可以产生多
        个形状不同的输出, 而输出结果只能有一个, 因此必须对输出形状进行约束）
        - groups (int, optional): 输入与输出通道的分组数量. 当不为1时, 默认为1(全连接). 
        - bias (bool, optional): 为 ``True`` 时,  添加偏置. 
        - dilation (int or tuple, optional): 采样间隔数量, 默认为1, 无间隔采样. 
    
    Shape:
        - 输入 Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - 输出 Output: :math:`(N, C_{out}, H_{out}, W_{out})` 其中
          :math:`H_{out} = (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
          :math:`W_{out} = (W_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`

    Attributes:
        weight (Tensor): 卷积网络层间连接的权重, 是模型需要学习的变量, 形状为weight (Tensor): 卷积网络层间连接的权重, 是模型需要学习的变量, 形状为
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   偏置, 是模型需要学习的变量, 形状为 (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class ConvTranspose3d(_ConvTransposeMixin, _ConvNd):
    r"""三维反卷积层
    反卷积层可以理解为输入的数据和卷积核的位置反转的卷积操作. 
    反卷积有时候也会被翻译成解卷积. 

    | :attr:`stride` 计算相关系数的步长. 
    | :attr:`padding` 处理边界时在每个维度首尾补0数量. 
    | :attr:`output_padding` 输出时候在每一个维度首尾补0的数量. （卷积时, 形状不同的输入数据
    对相同的核函数可以产生形状相同的结果；反卷积时, 同一个输入对相同的核函数可以产生多
    个形状不同的输出, 而输出结果只能有一个, 因此必须对输出形状进行约束）
    | :attr:`dilation` 采样间隔数量. 大于1时为非致密采样. 
    | :attr:`groups` 控制输入和输出之间的连接,  group=1, 输出是所有输入的卷积； group=2, 此时
    相当于有并排的两个卷基层, 每个卷积层只在对应的输入通道和输出通道之间计算, 并且输出时会将所有
    输出通道简单的首尾相接作为结果输出. 
            `in_channels` 和 `out_channels` 都应当可以被 groups 整除. 

    :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding` 可以为:

        - 单个 ``int`` 值  -- 深和宽和高均被设定为此值. 
        - 由三个 ``int`` 组成的 ``tuple``  -- 第一个 ``int`` 为深度,  第二个 ``int`` 为高度,第三个 ``int`` 为宽度. 

    .. note::
        数据的最后一列可能会因为 kernal 大小设定不当而被丢弃（大部分发生在 kernal 大小不能被输入
        整除的时候, 适当的 padding 可以避免这个问题）. 

    Args:
        - in_channels (int): 输入信号的通道数. 
        - out_channels (int): 卷积后输出结果的通道数. 
        - kernel_size (int or tuple): 卷积核的形状. 
        - stride (int or tuple, optional): 卷积每次移动的步长, 默认为1. 
        - padding (int or tuple, optional): 处理边界时填充0的数量, 默认为0(不填充). 
        - output_padding (int or tuple, optional): 输出时候在首尾补值的数量, 默认为0. （卷积时, 形状不同的输入数据
        对相同的核函数可以产生形状相同的结果；反卷积时, 同一个输入对相同的核函数可以产生多
        个形状不同的输出, 而输出结果只能有一个, 因此必须对输出形状进行约束）
        - groups (int, optional): 输入与输出通道的分组数量. 当不为1时, 默认为1(全连接). 
        - bias (bool, optional): 为 ``True`` 时,  添加偏置. 
        - dilation (int or tuple, optional): 采样间隔数量, 默认为1, 无间隔采样. 

    Shape:
        - 输入 Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - 输出 Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` 其中
          :math:`D_{out} = (D_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
          :math:`H_{out} = (H_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`
          :math:`W_{out} = (W_{in} - 1) * stride[2] - 2 * padding[2] + kernel\_size[2] + output\_padding[2]`

    Attributes:
        卷积网络层间连接的权重, 是模型需要学习的变量, 形状为weight (Tensor): 卷积网络层间连接的权重, 是模型需要学习的变量, 形状为
                         (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   偏置, 是模型需要学习的变量, 形状为 (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super(ConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose3d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


# TODO: Conv2dLocal
# TODO: Conv2dMap
# TODO: ConvTranspose2dMap
