from torch.autograd import Variable
import torch
from .module import Module
from .container import Sequential
from .activation import LogSoftmax
from .. import functional as F


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


class L1Loss(_Loss):
    r"""创建一个标准用来衡量输入的 `x` 与目标 `y` 之间的区别，该函数会逐元素地求出 
    `x` 和 `y` 之间的差值，最后返回这些差值的绝对值的平均值.

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` 和 `y` 可以是任意维度的数组，但需要有相同数量的n个元素.

    求和操作会对n个元素求和，最后除以 `n`.

    在构造函数的参数中设置 `size_average=False` 可以避免最后除以 `n` 的操作.

    Args:
        size_average (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False``, 损失函数的值会在每个 mini-batch（小批量）
            上求和. 当 reduce 的值为 ``False`` 时会被忽略. 默认值: ``True``
        reduce (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）上求平均值或者
            求和. 当 reduce 是 ``False`` 时, 损失函数会对每个 batch 元素都返回一个损失值并忽略
            size_average. 默认值: ``True``
    

    Shape:
        - 输入: :math:`(N, *)`, `*` 表示任意数量的额外维度
        - 目标: :math:`(N, *)`, 和输入的shape相同
        - 输出: 标量. 如果 reduce 是 ``False`` , 则输出为 :math:`(N, *)`, 
          shape与输出相同

    Examples::

        >>> loss = nn.L1Loss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(3, 5))
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, size_average=True, reduce=True):
        super(L1Loss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.l1_loss(input, target, size_average=self.size_average,
                         reduce=self.reduce)


class NLLLoss(_WeightedLoss):
    r"""负对数似然损失. 用 `C` 个类别来训练一个分类问题是很有效的. 可选参数 `weight` 是
    一个1维的 Tensor, 用来设置每个类别的权重. 当训练集不平衡时该参数十分有用.

    由前向传播得到的输入应该含有每个类别的对数概率: 输入必须是形如 `(minibatch, C)` 的
    2维 Tensor.

    在一个神经网络的最后一层添加 `LogSoftmax` 层可以得到对数概率. 如果你不希望在神经网络中
    加入额外的一层, 也可以使用 `CrossEntropyLoss` 函数.

    该损失函数需要的目标值是一个类别索引 `(0 to C-1, where C = number of classes)`

    该损失值可以描述为::

        loss(x, class) = -x[class]

    或者当 weight 参数存在时可以描述为::

        loss(x, class) = -weight[class] * x[class]

    又或者当 ignore_index 参数存在时可以描述为::

        loss(x, class) = class != ignoreIndex ? -weight[class] * x[class] : 0

    Args:
        weight (Tensor, optional): 自定义的每个类别的权重. 必须是一个长度为 `C` 的
            Tensor
        size_average (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为``False``, 损失函数的值会在每
            个 mini-batch（小批量）上求和. 当 reduce 的值为 ``False`` 时会被忽略. 默认值: ``True``
        ignore_index (int, optional): 设置一个目标值, 该目标值会被忽略, 从而不会影响到
            输入的梯度. 当 size_average 为 ``True`` 时, 损失函数的值将会在没有被忽略的元素上
            取平均.
        reduce (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）上求平均值或者
            求和. 当 reduce 是 ``False`` 时, 损失函数会对每个 batch 元素都返回一个损失值并忽略
            size_average. 默认值: ``True``

    Shape:
        - 输入: :math:`(N, C)`, 其中 `C = number of classes`
        - 目标: :math:`(N)`, 其中的每个元素都满足 `0 <= targets[i] <= C-1`
        - 输出: 标量. 如果 reduce 是 ``False``, 则输出为 :math:`(N)`.

    Examples::

        >>> m = nn.LogSoftmax()
        >>> loss = nn.NLLLoss()
        >>> # input is of size N x C = 3 x 5
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = autograd.Variable(torch.LongTensor([1, 0, 4]))
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(NLLLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.nll_loss(input, target, self.weight, self.size_average,
                          self.ignore_index, self.reduce)


class NLLLoss2d(NLLLoss):
    r"""对于图片输入的负对数似然损失. 它计算每个像素的负对数似然损失.

    Args:
        weight (Tensor, optional): 自定义的每个类别的权重. 必须是一个长度为 `C` 的
            Tensor
        size_average: 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为``False``, 损失函数的值会在每
            个 mini-batch（小批量）上求和. 当 reduce 的值为 ``False`` 时会被忽略. 默认值: ``True``
        reduce (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）上求平均值或者
            求和. 当 reduce 是 ``False`` 时, 损失函数会对每个 batch 元素都返回一个损失值并忽略
            size_average. 默认值: ``True``


    Shape:
        - Input: :math:`(N, C, H, W)` where `C = number of classes`
        - Target: :math:`(N, H, W)` where each value is `0 <= targets[i] <= C-1`
        - Output: scalar. If reduce is ``False``, then :math:`(N, H, W)` instead.

    Examples::

        >>> m = nn.Conv2d(16, 32, (3, 3)).float()
        >>> loss = nn.NLLLoss2d()
        >>> # input is of size N x C x height x width
        >>> input = autograd.Variable(torch.randn(3, 16, 10, 10))
        >>> # each element in target has to have 0 <= value < C
        >>> target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    pass


class PoissonNLLLoss(_Loss):
    r"""目标值为泊松分布的负对数似然损失.

    该损失可以描述为:

        target ~ Pois(input)
        loss(input, target) = input - target * log(input) + log(target!)

    最后一项可以被省略或者用 Stirling 公式来近似. 该近似用于大于1的目标值. 当目标值
    小于或等于1时, 则将0加到损失值中.

    Args:
        log_input (bool, optional): 如果设置为 ``True`` , 损失将会按照公
            式 `exp(input) - target * input` 来计算, 如果设置为 ``False`` , 损失
            将会按照 `input - target * log(input+eps)` 计算.
        full (bool, optional): 是否计算全部的损失, i. e. 加上 Stirling 近似项
            `target * log(target) - target + 0.5 * log(2 * pi * target)`.
        size_average (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为``False``, 损失函数的值会在每
            个 mini-batch（小批量）上求和.
        eps (float, optional): 当 log_input==``False`` 时, 取一个很小的值用来避免计算 log(0) .
            默认值: 1e-8

    Examples::

        >>> loss = nn.PoissonNLLLoss()
        >>> log_input = autograd.Variable(torch.randn(5, 2), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(5, 2))
        >>> output = loss(log_input, target)
        >>> output.backward()

    """

    def __init__(self, log_input=True, full=False, size_average=True, eps=1e-8):
        super(PoissonNLLLoss, self).__init__()
        self.log_input = log_input
        self.full = full
        self.size_average = size_average
        self.eps = eps

    def forward(self, log_input, target):
        _assert_no_grad(target)
        return F.poisson_nll_loss(log_input, target, self.log_input, self.full, self.size_average, self.eps)


class KLDivLoss(_Loss):
    r""" `Kullback-Leibler divergence`_ 损失

    KL 散度可用于衡量不同的连续分布之间的距离, 在连续的输出分布的空间上(离散采样)上进行直接回归时
    很有效.

    跟 `NLLLoss` 一样, `input` 需要含有*对数概率*, 不同于  `ClassNLLLoss`, `input` 可
    以不是2维的 Tensor, 因为该函数会逐元素地求值.

    该方法需要一个shape跟 `input` `Tensor` 一样的 `target` `Tensor`.

    损失可以描述为:

    .. math:: loss(x, target) = 1/n \sum(target_i * (log(target_i) - x_i))

    默认情况下, 损失会在每个 mini-batch（小批量）上和 **维度** 上取平均值. 如果字段
    `size_average` 设置为 ``False``, 则损失会不会取平均值.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
        size_average (bool, optional): 默认情况下, 损失会在每个 mini-batch（小批量）上
            和 **维度** 上取平均值. 如果设置为 ``False``, 则损失会不会取平均值.
        reduce (bool, optional): 默认情况下, 该损失函数的值会根据 size_average 在每
            个 mini-batch（小批量）上求平均值或者求和. 当 reduce 是 ``False`` 时, 损失函数会对每
            个 batch 元素都返回一个损失值并忽略 size_average. 默认值: ``True``
    
    Shape:
        - 输入: :math:`(N, *)`, 其中 `*` 表示任意数量的额外维度.
        - 目标: :math:`(N, *)`, shape 跟输入相同
        - 输出: 标量. 如果 `reduce` 是 ``True``, 则输出为 :math:`(N, *)`,
            shape 跟输入相同.
    """

    def __init__(self, size_average=True, reduce=True):
        super(KLDivLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.kl_div(input, target, size_average=self.size_average, reduce=self.reduce)


class MSELoss(_Loss):
    r"""输入 `x` 和 目标 `y` 之间的均方差

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|^2`

    `x` 和 `y` 可以是任意维度的数组，但需要有相同数量的n个元素.

    求和操作会对n个元素求和，最后除以 `n`.

    在构造函数的参数中设置 `size_average=False` 可以避免最后除以 `n` 的操作.

    要得到单个 batch 元素的损失, 设置 `reduce` 为 ``False``. 返回的损失将不会
    被平均, 也不会被 `size_average` 影响.

    Args:
        size_average (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为``False``, 损失函数的值会在每
            个 mini-batch（小批量）上求和. 只有当 reduce 的值为 ``True`` 才会生效. 默认值: ``True``
        reduce (bool, optional): 默认情况下, 该损失函数的值会根据 size_average 在每
            个 mini-batch（小批量）上求平均值或者求和. 当 reduce 是 ``False`` 时, 损失函数会对每
            个 batch 元素都返回一个损失值并忽略 size_average. 默认值: ``True``

    Shape:
        - 输入: :math:`(N, *)`, 其中 `*` 表示任意数量的额外维度.
        - 目标: :math:`(N, *)`, shape 跟输入相同

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(3, 5))
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, size_average=True, reduce=True):
        super(MSELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.mse_loss(input, target, size_average=self.size_average, reduce=self.reduce)


class BCELoss(_WeightedLoss):
    r"""创建一个标准用于衡量目标和输出之间的二值交叉熵:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    当定义了 weight 参数时:

    .. math:: loss(o, t) = - 1/n \sum_i weight[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    这可用于测量重构的误差, 例如自动编码机. 注意目标的值 `t[i]` 的范围为0到1之间.

    Args:
        weight (Tensor, optional): 自定义的每个 batch 元素的损失的的权重. 必须是一个长度为 "nbatch" 的
            的 Tensor
        size_average (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为``False``, 损失函数的值会在每
            个 mini-batch（小批量）上求和. 默认值: ``True``
    
    Shape:
        - 输入: :math:`(N, *)`, 其中 `*` 表示任意数量的额外维度.
        - 目标: :math:`(N, *)`, shape 跟输入相同
    
    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
        >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.binary_cross_entropy(input, target, weight=self.weight,
                                      size_average=self.size_average)


class BCEWithLogitsLoss(Module):
    r"""该损失函数把 `Sigmoid` 层集成到了 `BCELoss` 类中. 该版比用一个简单的 `Sigmoid`
    层和 `BCELoss` 在数值上更稳定, 因为把这两个操作合并为一个层之后, 可以利用 log-sum-exp 的
    技巧来实现数值稳定.

    目标和输出之间的二值交叉熵(不含sigmoid函数)是:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

    当定义了 weight 参数之后可描述为:

    .. math:: loss(o, t) = - 1/n \sum_i weight[i] * (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

    这可用于测量重构的误差, 例如自动编码机. 注意目标的值 `t[i]` 的范围为0到1之间.
    
    Args:
        weight (Tensor, optional): 自定义的每个 batch 元素的损失的权重. 必须是一个长度
            为 "nbatch" 的 Tensor
        size_average (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为``False``, 损失函数的值会在每
            个 mini-batch（小批量）上求和. 默认值: ``True``
    
    Shape:
        - 输入: :math:`(N, *)`, 其中 `*` 表示任意数量的额外维度.
        - 目标: :math:`(N, *)`, shape 跟输入相同


    Examples::

         >>> loss = nn.BCEWithLogitsLoss()
         >>> input = autograd.Variable(torch.randn(3), requires_grad=True)
         >>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
         >>> output = loss(input, target)
         >>> output.backward()
    """
    def __init__(self, weight=None, size_average=True):
        super(BCEWithLogitsLoss, self).__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)

    def forward(self, input, target):
        if self.weight is not None:
            return F.binary_cross_entropy_with_logits(input, target, Variable(self.weight), self.size_average)
        else:
            return F.binary_cross_entropy_with_logits(input, target, size_average=self.size_average)


class HingeEmbeddingLoss(_Loss):
    r"""Measures the loss given an input tensor `x` and a labels tensor `y`
    containing values (`1` or `-1`).
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as `x`, and is typically
    used for learning nonlinear embeddings or semi-supervised learning::

                         { x_i,                  if y_i ==  1
        loss(x, y) = 1/n {
                         { max(0, margin - x_i), if y_i == -1

    `x` and `y` can be of arbitrary shapes with a total of `n` elements each.
    The sum operation operates over all the elements.

    The division by `n` can be avoided if one sets the internal
    variable `size_average=False`.

    The `margin` has a default value of `1`, or can be set in the constructor.
    """

    def __init__(self, margin=1.0, size_average=True):
        super(HingeEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, self.margin, self.size_average)


class MultiLabelMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a multi-class multi-classification
    hinge loss (margin-based loss) between input `x`  (a 2D mini-batch `Tensor`)
    and output `y` (which is a 2D `Tensor` of target class indices).
    For each sample in the mini-batch::

        loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x.size(0)

    where `i == 0` to `x.size(0)`, `j == 0` to `y.size(0)`,
    `y[j] >= 0`, and `i != y[j]` for all `i` and `j`.

    `y` and `x` must have the same size.

    The criterion only considers the first non zero `y[j]` targets.

    This allows for different samples to have variable amounts of target classes
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.multilabel_margin_loss(input, target, size_average=self.size_average)


class SmoothL1Loss(_Loss):
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).
    Also known as the Huber loss::

                              { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
        loss(x, y) = 1/n \sum {
                              { |x_i - y_i| - 0.5,   otherwise

    `x` and `y` arbitrary shapes with a total of `n` elements each
    the sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal variable
    `size_average` to ``False``

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over all elements. However, if the field size_average is set to ``False``,
           the losses are instead summed. Ignored when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed
           over elements. When reduce is ``False``, the loss function returns
           a loss per element instead and ignores size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input

    """
    def __init__(self, size_average=True, reduce=True):
        super(SmoothL1Loss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.smooth_l1_loss(input, target, size_average=self.size_average,
                                reduce=self.reduce)


class SoftMarginLoss(_Loss):
    r"""Creates a criterion that optimizes a two-class classification
    logistic loss between input `x` (a 2D mini-batch Tensor) and
    target `y` (which is a tensor containing either `1` or `-1`).

    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()

    The normalization by the number of elements in the input can be disabled by
    setting `self.size_average` to ``False``.
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.soft_margin_loss(input, target, size_average=self.size_average)


class CrossEntropyLoss(_WeightedLoss):
    r"""这个标准把 `LogSoftMax` 和 `NLLLoss` 结合到了一个类中

    当训练有  `C` 个类别的分类问题时很有效. 可选参数 `weight` 必须是一个1维 Tensor, 
    权重将被分配给各个类别.
    对于不平衡的训练集非常有效.

    `input` 含有每个类别的分数

    `input` 必须是一个2维的形如 `(minibatch, C)` 的 `Tensor`.

    `target` 是一个类别索引 (0 to C-1), 对应于 `minibatch` 中的每个元素

    损失可以描述为::

        loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                       = -x[class] + log(\sum_j exp(x[j]))
    
    当 `weight` 参数存在时::

        loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))
    
    损失会在每个 mini-batch（小批量）上求平均.

    Args:
        weight (Tensor, optional): 自定义的每个类别的权重. 必须是一个长度为 `C` 的
            Tensor
        size_average (bool, optional): 默认情况下, 该损失函数的值会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为``False``, 损失函数的值会在每
            个 mini-batch（小批量）上求和. 当 reduce 的值为 ``False`` 时会被忽略.
        ignore_index (int, optional): 设置一个目标值, 该目标值会被忽略, 从而不会影响到
            输入的梯度. 当 size_average 为 ``True`` 时, 损失函数的值将会在没有被忽略的元素上
            取平均.
        reduce (bool, optional): 默认情况下, 该损失函数的值会根据 size_average 在每
            个 mini-batch（小批量）上求平均值或者求和. 当 reduce 是 ``False`` 时, 损失函数会对
            每个 batch 元素都返回一个损失值并忽略 size_average. 默认值: ``True``

    Shape:
        - 输入: :math:`(N, C)`, 其中 `C = number of classes`
        - 目标: :math:`(N)`, 其中的每个元素都满足 `0 <= targets[i] <= C-1`
        - 输出: 标量. 如果 reduce 是 ``False``, 则输出为 :math:`(N)`.


    Examples::

        >>> loss = nn.CrossEntropyLoss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.LongTensor(3).random_(5))
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True):
        super(CrossEntropyLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.cross_entropy(input, target, self.weight, self.size_average,
                               self.ignore_index, self.reduce)


class MultiLabelSoftMarginLoss(_WeightedLoss):
    r"""Creates a criterion that optimizes a multi-label one-versus-all
    loss based on max-entropy, between input `x`  (a 2D mini-batch `Tensor`) and
    target `y` (a binary 2D `Tensor`). For each sample in the minibatch::

       loss(x, y) = - sum_i (y[i] * log( 1 / (1 + exp(-x[i])) )
                         + ( (1-y[i]) * log(exp(-x[i]) / (1 + exp(-x[i])) ) )

    where `i == 0` to `x.nElement()-1`, `y[i]  in {0,1}`.
    `y` and `x` must have the same size.
    """

    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target, self.weight, self.size_average)


class CosineEmbeddingLoss(Module):
    r"""Creates a criterion that measures the loss given  an input tensors
    x1, x2 and a `Tensor` label `y` with values 1 or -1.
    This is used for measuring whether two inputs are similar or dissimilar,
    using the cosine distance, and is typically used for learning nonlinear
    embeddings or semi-supervised learning.

    `margin` should be a number from `-1` to `1`, `0` to `0.5` is suggested.
    If `margin` is missing, the default value is `0`.

    The loss function for each sample is::

                     { 1 - cos(x1, x2),              if y ==  1
        loss(x, y) = {
                     { max(0, cos(x1, x2) - margin), if y == -1

    If the internal variable `size_average` is equal to ``True``,
    the loss function averages the loss over the batch samples;
    if `size_average` is ``False``, then the loss function sums over the
    batch samples. By default, `size_average = True`.
    """

    def __init__(self, margin=0, size_average=True):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, self.margin, self.size_average)


class MarginRankingLoss(Module):
    r"""Creates a criterion that measures the loss given
    inputs `x1`, `x2`, two 1D mini-batch `Tensor`s,
    and a label 1D mini-batch tensor `y` with values (`1` or `-1`).

    If `y == 1` then it assumed the first input should be ranked higher
    (have a larger value) than the second input, and vice-versa for `y == -1`.

    The loss function for each sample in the mini-batch is::

        loss(x, y) = max(0, -y * (x1 - x2) + margin)

    if the internal variable `size_average = True`,
    the loss function averages the loss over the batch samples;
    if `size_average = False`, then the loss function sums over the batch
    samples.
    By default, `size_average` equals to ``True``.
    """

    def __init__(self, margin=0, size_average=True):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, self.margin, self.size_average)


class MultiMarginLoss(Module):
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input `x` (a 2D mini-batch `Tensor`) and
    output `y` (which is a 1D tensor of target class indices,
    `0` <= `y` <= `x.size(1)`):

    For each mini-batch sample::

        loss(x, y) = sum_i(max(0, (margin - x[y] + x[i]))^p) / x.size(0)
                     where `i == 0` to `x.size(0)` and `i != y`.

    Optionally, you can give non-equal weighting on the classes by passing
    a 1D `weight` tensor into the constructor.

    The loss function then becomes:

        loss(x, y) = sum_i(max(0, w[y] * (margin - x[y] - x[i]))^p) / x.size(0)

    By default, the losses are averaged over observations for each minibatch.
    However, if the field `size_average` is set to ``False``,
    the losses are instead summed.
    """

    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(MultiMarginLoss, self).__init__()
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        assert weight is None or weight.dim() == 1
        self.p = p
        self.margin = margin
        self.size_average = size_average
        self.weight = weight

    def forward(self, input, target):
        return F.multi_margin_loss(input, target, self.p, self.margin,
                                   self.weight, self.size_average)


class TripletMarginLoss(Module):
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
        p: the norm degree. Default: 2

    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`

    >>> triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> input3 = autograd.Variable(torch.randn(100, 128))
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()

    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, margin=1.0, p=2, eps=1e-6, swap=False):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, self.margin,
                                     self.p, self.eps, self.swap)

# TODO: L1HingeEmbeddingCriterion
# TODO: MSECriterion weight
# TODO: ClassSimplexCriterion
