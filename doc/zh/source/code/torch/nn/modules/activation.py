import warnings
import torch
from torch.nn.parameter import Parameter

from .module import Module
from .. import functional as F


class Threshold(Module):
    r"""基于 Tensor 中的每个元素创造阈值函数

    Threshold 被定义为 ::

         y =  x        if x >  threshold
              value    if x <= threshold

    Args:
        threshold: 阈值
        value: 输入值小于阈值则会被 value 代替
        inplace: 选择是否进行覆盖运算. 默认值: ``False``
        
    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(N, *)`, 和输入的格式 shape 一致


    例::

        >>> m = nn.Threshold(0.1, 20)
        >>> input = Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, threshold, value, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    def forward(self, input):
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + str(self.threshold) \
            + ', ' + str(self.value) \
            + inplace_str + ')'


class ReLU(Threshold):
    r"""对输入运用修正线性单元函数
    :math:`{ReLU}(x)= max(0, x)`

    Args:
        inplace: 选择是否进行覆盖运算 Default: ``False``

    Shape:
        - Input: :math:`(N, *)` `*` 代表任意数目附加维度
        - Output: :math:`(N, *)`, 与输入拥有同样的 shape 属性

    Examples::

        >>> m = nn.ReLU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + inplace_str + ')'


class RReLU(Module):

    def __init__(self, lower=1. / 8, upper=1. / 3, inplace=False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + str(self.lower) \
            + ', ' + str(self.upper) \
            + inplace_str + ')'


class Hardtanh(Module):
    r"""对输入的每一个元素运用 HardTanh

    HardTanh 被定义为::

       f(x) = +1, if x  >  1
       f(x) = -1, if x  < -1
       f(x) =  x,  otherwise

    线性区域的范围 :math:`[-1, 1]` 可以被调整

    Args:
        min_val: 线性区域范围最小值. 默认值: -1
        max_val: 线性区域范围最大值.  默认值: 1
        inplace: 选择是否进行覆盖运算. 默认值: ``False``

    关键字参数 :attr:`min_value` 以及 :attr:`max_value` 已被弃用. 
    更改为 :attr:`min_val` 和 :attr:`max_val`

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意维度组合
        - Output: :math:`(N, *)`, 与输入有相同的 shape 属性

    例 ::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, min_val=-1, max_val=1, inplace=False, min_value=None, max_value=None):
        super(Hardtanh, self).__init__()
        if min_value is not None:
            warnings.warn("keyword argument min_value is deprecated and renamed to min_val")
            min_val = min_value
        if max_value is not None:
            warnings.warn("keyword argument max_value is deprecated and renamed to max_val")
            max_val = max_value

        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        assert self.max_val > self.min_val

    def forward(self, input):
        return F.hardtanh(input, self.min_val, self.max_val, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'min_val=' + str(self.min_val) \
            + ', max_val=' + str(self.max_val) \
            + inplace_str + ')'


class ReLU6(Hardtanh):
    r"""对输入的每一个元素运用函数 :math:`{ReLU6}(x) = min(max(0,x), 6)`

    Args:
        inplace: 选择是否进行覆盖运算 默认值: ``False``

    Shape:
        - Input: :math:`(N, *)`, `*` 代表任意数目附加维度
        - Output: :math:`(N, *)`, 与输入拥有同样的 shape 属性

    Examples::

        >>> m = nn.ReLU6()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(0, 6, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + inplace_str + ')'


class Sigmoid(Module):
    r"""对每个元素运用 Sigmoid 函数. Sigmoid 定义如下 :math:`f(x) = 1 / ( 1 + exp(-x))`

    Shape:
        - Input: :math:`(N, *)` `*` 表示任意维度组合
        - Output: :math:`(N, *)`, 与输入有相同的 shape 属性

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return torch.sigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Tanh(Module):
    r"""对输入的每个元素, 
    :math:`f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

    Shape:
        - Input: :math:`(N, *)` `*` 表示任意维度组合
        - Output: :math:`(N, *)`, 与输入有相同的 shape 属性

    Examples::

        >>> m = nn.Tanh()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return torch.tanh(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ELU(Module):
    r"""对输入的每一个元素运用函数,
    :math:`f(x) = max(0,x) + min(0, alpha * (exp(x) - 1))`

    Args:
        alpha: ELU 定义公式中的 alpha 值. 默认值: 1.0
        inplace: 选择是否进行覆盖运算 默认值: ``False``

    Shape:
        - Input: :math:`(N, *)` `*` 代表任意数目附加维度
        - Output: :math:`(N, *)`, 与输入拥有同样的 shape 属性

    Examples::

        >>> m = nn.ELU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, alpha=1., inplace=False):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + 'alpha=' + str(self.alpha) \
            + inplace_str + ')'


class SELU(Module):
    r"""对输入的每一个元素运用函数,
    :math:`f(x) = scale * (\max(0,x) + \min(0, alpha * (\exp(x) - 1)))`,
    ``alpha=1.6732632423543772848170429916717``, 
    ``scale=1.0507009873554804934193349852946``.

    更多地细节可以参阅论文 `Self-Normalizing Neural Networks`_ .

    Args:
        inplace (bool, optional): 选择是否进行覆盖运算. 默认值: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.SELU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))

    .. _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
    """

    def __init__(self, inplace=False):
        super(SELU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.selu(input, self.inplace)

    def __repr__(self):
        inplace_str = '(inplace)' if self.inplace else ''
        return self.__class__.__name__ + inplace_str


class GLU(Module):
    r"""Applies the gated linear unit function
    :math:`{GLU}(a, b)= a \otimes \sigma(b)` where `a` is the first half of
    the input vector and `b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(*, N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(*, N / 2, *)`

    Examples::

        >>> m = nn.GLU()
        >>> input = autograd.Variable(torch.randn(4, 2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, input):
        return F.glu(input, self.dim)

    def __repr__(self):
        return '{}(dim={})'.format(self.__class__.__name__, self.dim)


class Hardshrink(Module):
    r"""对每个元素运用 hard shrinkages 函数, hard shrinkage 定义如下::
        f(x) = x, if x >  lambda
        f(x) = x, if x < -lambda
        f(x) = 0, otherwise

    Args:
        lambd: Hardshrink 公式中的 beta 值. 默认值: 0.5

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(N, *)`, 和输入的格式 shape 一致

    例::

        >>> m = nn.Hardshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, lambd=0.5):
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input):
        return F.hardshrink(input, self.lambd)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.lambd) + ')'


class LeakyReLU(Module):
    r"""对输入的每一个元素运用,
    :math:`f(x) = max(0, x) + {negative\_slope} * min(0, x)`

    Args:
        negative_slope: 控制负斜率的角度, 默认值: 1e-2
        inplace: 选择是否进行覆盖运算 默认值: ``False``

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(N, *)`, 和输入的格式shape一致

    例::

        >>> m = nn.LeakyReLU(0.1)
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, negative_slope=1e-2, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + '(' \
            + str(self.negative_slope) \
            + inplace_str + ')'


class LogSigmoid(Module):
    r"""对输入的每一个元素运用函数
    :math:`LogSigmoid(x) = log( 1 / (1 + exp(-x_i)))`

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(N, *)`, 和输入的格式shape一致

    例::

        >>> m = nn.LogSigmoid()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return F.logsigmoid(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Softplus(Module):
    r"""对每个元素运用Softplus函数, Softplus 定义如下 ::
    :math:`f(x) = 1/beta * log(1 + exp(beta * x_i))`

    Softplus 函数是ReLU函数的平滑逼近. Softplus 函数可以使得输出值限定为正数.

    为了保证数值稳定性. 线性函数的转换可以使输出大于某个值.

    Args:
        beta: Softplus 公式中的 beta 值. 默认值: 1
        threshold: 阈值. 当输入到该值以上时我们的SoftPlus实现将还原为线性函数. 默认值: 20

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
          dimensions
        - Output: :math:`(N, *)`, 和输入的格式shape一致

    例::

        >>> m = nn.Softplus()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, beta=1, threshold=20):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'beta=' + str(self.beta) \
            + ', threshold=' + str(self.threshold) + ')'


class Softshrink(Module):
    r"""对输入的每一个元素运用 soft shrinkage 函数

    SoftShrinkage 运算符定义为::

        f(x) = x-lambda, if x > lambda >  f(x) = x+lambda, if x < -lambda
        f(x) = 0, otherwise

    Args:
        lambd: Softshrink 公式中的 lambda 值. 默认值: 0.5

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(N, *)`, 和输入的格式 shape 一致

    例::

        >>> m = nn.Softshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, lambd=0.5):
        super(Softshrink, self).__init__()
        self.lambd = lambd

    def forward(self, input):
        return F.softshrink(input, self.lambd)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.lambd) + ')'


class PReLU(Module):
    r"""对输入的每一个元素运用函数
    :math:`PReLU(x) = max(0,x) + a * min(0,x)` 这里的 "a" 是自学习的参数.
    当不带参数地调用时, nn.PReLU() 在所有输入通道中使用单个参数 "a" . 
    而如果用 nn.PReLU(nChannels) 调用, "a" 将应用到每个输入.


    .. note::
        当为了表现更佳的模型而学习参数 "a" 时不要使用权重衰减 (weight decay)

    Args:
        num_parameters: 需要学习的 "a" 的个数. 默认等于1
        init: "a" 的初始值. 默认等于0.25

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(N, *)`, 和输入的格式 shape 一致

    例::

        >>> m = nn.PReLU()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(PReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, input):
        return F.prelu(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'num_parameters=' + str(self.num_parameters) + ')'


class Softsign(Module):
    r"""对输入的每一个元素运用函数 :math:`f(x) = x / (1 + |x|)`

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(N, *)`, 和输入的格式 shape 一致

    例::

        >>> m = nn.Softsign()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return F.softsign(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Tanhshrink(Module):
    r"""对输入的每一个元素运用函数, :math:`Tanhshrink(x) = x - Tanh(x)`

    Shape:
        - Input: :math:`(N, *)` 其中 `*` 代表任意数目的附加维度
        - Output: :math:`(N, *)`, 和输入的格式shape一致

    例::

        >>> m = nn.Tanhshrink()
        >>> input = autograd.Variable(torch.randn(2))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        return F.tanhshrink(input)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Softmin(Module):
    r"""对n维输入张量运用 Softmin 函数, 将张量的每个元素缩放到
    (0,1) 区间且和为 1.

    :math:`f(x) = \frac{\exp(-x_i)}{\sum_j \exp(-x_j)}`

    Shape:
        - Input: 任意shape
        - Output: 和输入相同

    Arguments:
        dim (int): 这是将计算 Softmax 的维度 (所以每个沿着 dim 的切片和为 1).

    Returns:
        返回结果是一个与输入维度相同的张量, 每个元素的取值范围在 [0, 1] 区间.

    例::

        >>> m = nn.Softmin()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """
    def __init__(self, dim=None):
        super(Softmin, self).__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmin(input, self.dim, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Softmax(Module):
    r"""对n维输入张量运用 Softmax 函数, 将张量的每个元素缩放到
    (0,1) 区间且和为 1. Softmax 函数定义如下
    :math:`f_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    Shape:
        - Input: 任意shape
        - Output: 和输入相同

    Returns:
        返回结果是一个与输入维度相同的张量, 每个元素的取值范围在 [0, 1] 区间. 

    Arguments:
        dim (int): 这是将计算 Softmax 的那个维度 (所以每个沿着 dim 的切片和为 1).

    .. note::

        如果你想对原始 Softmax 数据计算 Log 进行收缩, 并不能使该模块直接使用 NLLLoss 负对数似然损失函数.
        取而代之, 应该使用 Logsoftmax (它有更快的运算速度和更好的数值性质).

    例::

        >>> m = nn.Softmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return F.softmax(input, self.dim, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Softmax2d(Module):
    r"""把 SoftMax 应用于每个空间位置的特征. 

    给定图片的 通道数 Channels x 高 Height x 宽 Width, 它将对图片的每一个位置
    使用 Softmax :math:`(Channels, h_i, w_j)`

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (格式 shape 与输入相同)

    Returns:
        一个维度及格式 shape 都和输入相同的 Tensor, 取值范围在[0, 1]

    例::

        >>> m = nn.Softmax2d()
        >>> # you softmax over the 2nd dimension
        >>> input = autograd.Variable(torch.randn(2, 3, 12, 13))
        >>> print(input)
        >>> print(m(input))
    """

    def forward(self, input):
        assert input.dim() == 4, 'Softmax2d 需要的输入是 4D tensor'
        return F.softmax(input, 1, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LogSoftmax(Module):
    r"""对每个输入的 n 维 Tensor 使用 Log(Softmax(x)). 
    LogSoftmax 公式可简化为

    :math:`f_i(x) = log(exp(x_i) / sum_j exp(x_j) )`

    Shape:
        - Input: 任意格式 shape
        - Output: 和输入的格式 shape 一致

    Arguments:
        dim (int): 这是将计算 Softmax 的那个维度 (所以每个沿着 dim 的切片和为1).

    Returns:
        一个维度及格式 shape 都和输入相同的 Tensor, 取值范围在 [-inf, 0)

    例::

        >>> m = nn.LogSoftmax()
        >>> input = autograd.Variable(torch.randn(2, 3))
        >>> print(input)
        >>> print(m(input))
    """

    def __init__(self, dim=None):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        return F.log_softmax(input, self.dim, _stacklevel=5)

    def __repr__(self):
        return self.__class__.__name__ + '()'
