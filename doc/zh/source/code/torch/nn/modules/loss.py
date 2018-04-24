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
    r"""创建一个衡量输入 `x` 与目标 `y` 之间差的绝对值的平均值的标准, 该
    函数会逐元素地求出 `x` 和 `y` 之间差的绝对值, 最后返回绝对值的平均值.

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` 和 `y` 可以是任意维度的数组, 但需要有相同数量的n个元素.

    求和操作会对n个元素求和, 最后除以 `n` .

    在构造函数的参数中传入 `size_average=False`, 最后求出来的绝对值将不会除以 `n`.

    Args:
        size_average (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False``, loss 将会在每个 mini-batch（小批量）
            上累加, 而不会取平均值. 当 reduce 的值为 ``False`` 时该字段会被忽略. 默认值: ``True``
        reduce (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）上求平均值或者
            求和. 当 reduce 是 ``False`` 时, 损失函数会对每个 batch 元素都返回一个 loss 并忽
            略 size_average 字段. 默认值: ``True``
    

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
    r"""负对数似然损失. 用于训练 `C` 个类别的分类问题. 可选参数 `weight` 是
    一个1维的 Tensor, 用来设置每个类别的权重. 当训练集不平衡时该参数十分有用.

    由前向传播得到的输入应该含有每个类别的对数概率: 输入必须是形如 `(minibatch, C)` 的
    2维 Tensor.

    在一个神经网络的最后一层添加 `LogSoftmax` 层可以得到对数概率. 如果你不希望在神经网络中
    加入额外的一层, 也可以使用 `CrossEntropyLoss` 函数.

    该损失函数需要的目标值是一个类别索引 `(0 到 C-1, 其中 C 是类别数量)`

    该 loss 可以描述为::

        loss(x, class) = -x[class]

    或者当 weight 参数存在时可以描述为::

        loss(x, class) = -weight[class] * x[class]

    又或者当 ignore_index 参数存在时可以描述为::

        loss(x, class) = class != ignoreIndex ? -weight[class] * x[class] : 0

    Args:
        weight (Tensor, optional): 自定义的每个类别的权重. 必须是一个长度为 `C` 的
            Tensor
        size_average (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False``, loss 将会在每个 mini-batch（小批量）
            上累加, 而不会取平均值. 当 reduce 的值为 ``False`` 时该字段会被忽略. 默认值: ``True``
        ignore_index (int, optional): 设置一个目标值, 该目标值会被忽略, 从而不会影响到
            输入的梯度. 当 size_average 为 ``True`` 时, loss 将会在没有被忽略的元素上
            取平均值.
        reduce (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）上求平均值或者
            求和. 当 reduce 是 ``False`` 时, 损失函数会对每个 batch 元素都返回一个 loss 并忽
            略 size_average 字段. 默认值: ``True``


    Shape:
        - 输入: :math:`(N, C)`, 其中 `C` 是类别的数量
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
        size_average: 默认情况下, loss 会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False``, loss 将会在每个 mini-batch（小批量）
            上累加, 而不会取平均值. 当 reduce 的值为 ``False`` 时该字段会被忽略. 默认值: ``True``
        reduce (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）上求平均值或者
            求和. 当 reduce 是 ``False`` 时, 损失函数会对每个 batch 元素都返回一个 loss 并忽
            略 size_average 字段. 默认值: ``True``


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
    小于或等于1时, 则将0加到 loss 中.

    Args:
        log_input (bool, optional): 如果设置为 ``True`` , loss 将会按照公
            式 `exp(input) - target * input` 来计算, 如果设置为 ``False`` , loss
            将会按照 `input - target * log(input+eps)` 计算.
        full (bool, optional): 是否计算全部的 loss, i. e. 加上 Stirling 近似项
            `target * log(target) - target + 0.5 * log(2 * pi * target)`.
        size_average (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False``, loss 将会在每个 mini-batch（小批量）
            上累加, 而不会取平均值.
        eps (float, optional): 当 log_input==``False`` 时, 取一个很小的值用来避免计算 log(0).
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

    跟 `NLLLoss` 一样, `input` 需要含有 *对数概率* , 不同于  `ClassNLLLoss`, `input` 可
    以不是2维的 Tensor, 因为该函数会逐元素地求值.

    该方法需要一个shape跟 `input` `Tensor` 一样的 `target` `Tensor`.

    损失可以描述为:

    .. math:: loss(x, target) = 1/n \sum(target_i * (log(target_i) - x_i))

    默认情况下, loss 会在每个 mini-batch（小批量）上和 **维度** 上取平均值. 如果字段
    `size_average` 设置为 ``False``, 则 loss 不会取平均值.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
        size_average (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）上
            和 **维度** 上取平均值. 如果设置为 ``False``, 则 loss 会累加, 而不是取平均值.
        reduce (bool, optional): 默认情况下, loss 会根据 size_average 在每
            个 mini-batch（小批量）上求平均值或者求和. 当 reduce 是 ``False`` 时, 损失函数会对每
            个 batch 元素都返回一个 loss 并忽略 size_average 字段. 默认值: ``True``
    
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

    `x` 和 `y` 可以是任意维度的数组, 但需要有相同数量的n个元素.

    求和操作会对n个元素求和, 最后除以 `n`.

    在构造函数的参数中传入 `size_average=False` , 最后求出来的绝对值将不会除以 `n`.

    要得到每个 batch 中每个元素的 loss, 设置 `reduce` 为 ``False``. 返回的 loss 将不会
    取平均值, 也不会被 `size_average` 影响.

    Args:
        size_average (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False`` , loss 会在每
            个 mini-batch（小批量）上求和. 只有当 reduce 的值为 ``True`` 才会生效. 默认值: ``True``
        reduce (bool, optional): 默认情况下, loss 会根据 size_average 的值在每
            个 mini-batch（小批量）上求平均值或者求和. 当 reduce 是 ``False`` 时, 损失函数会对每
            个 batch 元素都返回一个 loss 并忽略 size_average字段. 默认值: ``True``

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
    r"""计算目标和输出之间的二进制交叉熵:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    当定义了 weight 参数时:

    .. math:: loss(o, t) = - 1/n \sum_i weight[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    这可用于测量重构的误差, 例如自动编码机. 注意目标的值 `t[i]` 的范围为0到1之间.

    Args:
        weight (Tensor, optional): 自定义的每个 batch 元素的 loss 的权重. 必须是一个长度为 "nbatch" 的
            的 Tensor
        size_average (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False`` , loss 会在每
            个 mini-batch（小批量）上累加, 而不是取平均值. 默认值: ``True``
    
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
        weight (Tensor, optional): 自定义的每个 batch 元素的 loss 的权重. 必须是一个长度
            为 "nbatch" 的 Tensor
        size_average (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False`` , loss 会在每
            个 mini-batch（小批量）上累加, 而不是取平均值. 默认值: ``True``
    
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
    r"""衡量输入 Tensor(张量) `x` 和 目标 Tensor(张量) `y` (取值为 `1` 和 `-1`) 之间的损失值.
    此方法通常用来衡量两个输入值是否相似, 例如使用L1成对距离作为 `x`, 并且通常用来进行非线性嵌入学习或者
    半监督学习::

                         { x_i,                  if y_i ==  1
        loss(x, y) = 1/n {
                         { max(0, margin - x_i), if y_i == -1

    `x` 和 `y` 分别可以是具有 `n` 个元素的任意形状. 合计操作对所有元素进行计算.

    如果 `size_average=False`, 则计算时不会除以 `n` 取平均值.

    `margin` 的默认值是 `1`, 或者可以通过构造函数来设置.
    """

    def __init__(self, margin=1.0, size_average=True):
        super(HingeEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, self.margin, self.size_average)


class MultiLabelMarginLoss(_Loss):
    r"""创建一个标准, 用以优化多元分类问题的合页损失函数 (基于空白的损失), 计算损失值时
    需要2个参数分别为输入, `x` (一个2维小批量 `Tensor`) 和输出 `y` 
    (一个2维 `Tensor`, 其值为 `x` 的索引值). 
    对于mini-batch(小批量) 中的每个样本按如下公式计算损失::

        loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x.size(0)

    其中 `i` 的取值范围是 `0` 到 `x.size(0)`, `j` 的取值范围是 `0` 到 `y.size(0)`,
    `y[j] >= 0`, 并且对于所有 `i` 和 `j` 有 `i != y[j]`.

    `y` 和 `x` 必须有相同的元素数量.

    此标准仅考虑 `y[j]` 中最先出现的非零值.

    如此可以允许每个样本可以有数量不同的目标类别.
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.multilabel_margin_loss(input, target, size_average=self.size_average)


class SmoothL1Loss(_Loss):
    r"""创建一个标准, 当某个元素的错误值的绝对值小于1时使用平方项计算, 其他情况则使用L1范式计算.
    此方法创建的标准对于异常值不如 `MSELoss` 敏感, 但是同时在某些情况下可以防止梯度爆炸 (比如
    参见论文 "Fast R-CNN" 作者 Ross Girshick).
    也被称为 Huber 损失函数::

                              { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
        loss(x, y) = 1/n \sum {
                              { |x_i - y_i| - 0.5,   otherwise

    `x` 和 `y` 可以是任意形状只要都具备总计 `n` 个元素
    合计仍然针对所有元素进行计算, 并且最后除以 `n`.

    如果把内部变量 `size_average` 设置为 ``False``, 则不会被除以 `n`.

    Args:
        size_average (bool, optional): 损失值默认会按照所有元素取平均值. 但是, 如果 size_average 被
           设置为 ``False``, 则损失值为所有元素的合计. 如果 reduce 参数设为 ``False``, 则忽略此参数的值.
           默认: ``True`` 
        reduce (bool, optional): 损失值默认会按照所有元素取平均值或者取合计值. 当 reduce 设置为 ``False``
           时, 损失函数对于每个元素都返回损失值并且忽略 size_average 参数. 默认: ``True``

    Shape:
        - 输入: :math:`(N, *)` `*` 代表任意个其他维度
        - 目标: :math:`(N, *)`, 同输入
        - 输出: 标量. 如果 reduce 设为 ``False`` 则为
          :math:`(N, *)`, 同输入

    """
    def __init__(self, size_average=True, reduce=True):
        super(SmoothL1Loss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.smooth_l1_loss(input, target, size_average=self.size_average,
                                reduce=self.reduce)


class SoftMarginLoss(_Loss):
    r"""创建一个标准, 用以优化两分类的 logistic loss. 输入为 `x` (一个2维 mini-batch Tensor)和
    目标 `y` (一个包含 `1` 或者 `-1` 的 Tensor).

    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()

    可以通过设置 `self.size_average` 为 ``False`` 来禁用按照元素数量取平均的正则化操作.
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.soft_margin_loss(input, target, size_average=self.size_average)


class CrossEntropyLoss(_WeightedLoss):
    r"""该类把 `LogSoftMax` 和 `NLLLoss` 结合到了一个类中

    当训练有 `C` 个类别的分类问题时很有效. 可选参数 `weight` 必须是一个1维 Tensor, 
    权重将被分配给各个类别. 对于不平衡的训练集非常有效.

    `input` 含有每个类别的分数

    `input` 必须是一个2维的形如 `(minibatch, C)` 的 `Tensor`.

    `target` 是一个类别索引 (0 to C-1), 对应于 `minibatch` 中的每个元素

    loss 可以描述为::

        loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                       = -x[class] + log(\sum_j exp(x[j]))
    
    当 `weight` 参数存在时::

        loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))
    
    loss 在每个 mini-batch（小批量）上取平均值.

    Args:
        weight (Tensor, optional): 自定义的每个类别的权重. 必须是一个长度为 `C` 的
            Tensor
        size_average (bool, optional): 默认情况下, loss 会在每个 mini-batch（小批量）
            上取平均值. 如果字段 size_average 被设置为 ``False``, loss 将会在每个 mini-batch（小批量）
            上累加, 而不会取平均值. 当 reduce 的值为 ``False`` 时该字段会被忽略.
        ignore_index (int, optional): 设置一个目标值, 该目标值会被忽略, 从而不会影响到
            输入的梯度. 当 size_average 字段为 ``True`` 时, loss 将会在没有被忽略的元素上
            取平均.
        reduce (bool, optional): 默认情况下, loss 会根据 size_average 的值在每
            个 mini-batch（小批量）上求平均值或者求和. 当 reduce 是 ``False`` 时, 损失函数会对
            每个 batch 元素都返回一个 loss 并忽略 size_average 字段. 默认值: ``True``

    Shape:
        - 输入: :math:`(N, C)`, 其中 `C` 是类别的数量
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
    r"""创建一个标准, 基于输入 `x` 和目标 `y` 的 max-entropy(最大熵), 优化多标签 one-versus-all 损失.
    输入 `x` 为一个2维 mini-batch `Tensor`, 目标 `y` 为2进制2维 `Tensor`.
    对每个 mini-batch 中的样本, 对应的 loss 为::

       loss(x, y) = - sum_i (y[i] * log( 1 / (1 + exp(-x[i])) )
                         + ( (1-y[i]) * log(exp(-x[i]) / (1 + exp(-x[i])) ) )

    其中 `i == 0` 至 `x.nElement()-1`, `y[i]  in {0,1}`.
    `y` 和 `x` 必须具有相同的维度.
    """

    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target, self.weight, self.size_average)


class CosineEmbeddingLoss(Module):
    r"""新建一个标准, 用以衡量输入 `Tensor` x1, x2 和取值为 1 或者 -1 的标签 `Tensor` `y` 之间的
    损失值.
    此标准用 cosine 距离来衡量2个输入参数之间是否相似, 并且一般用来学习非线性 embedding 或者半监督
    学习.

    `margin` 应该取 `-1` 到 `1` 之间的值, 建议取值范围是 `0` 到 `0.5`.
    如果没有设置 `margin` 参数, 则默认值取 `0`.

    每个样本的损失函数如下::

                     { 1 - cos(x1, x2),              if y ==  1
        loss(x, y) = {
                     { max(0, cos(x1, x2) - margin), if y == -1

    如果内部变量 `size_average` 设置为 ``True``, 则损失函数以 batch 中所有的样本数取平均值;
    如果 `size_average` 设置为 ``False``, 则损失函数对 batch 中所有的样本求和. 默认情况下, 
    `size_average = True`.
    """

    def __init__(self, margin=0, size_average=True):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, self.margin, self.size_average)


class MarginRankingLoss(Module):
    r"""创建一个衡量 mini-batch(小批量) 中的2个1维 `Tensor` 的输入 `x1` 和 `x2`,
    和1个1维 `Tensor` 的目标 `y` ( `y` 的取值是 `1` 或者 `-1`) 之间损失的标准.

    如果 `y == 1` 则认为第一个输入值应该排列在第二个输入值之上(即值更大), `y == -1` 时则相反.

    对于 mini-batch(小批量) 中每个实例的损失函数如下::


        loss(x, y) = max(0, -y * (x1 - x2) + margin)

    如果内部变量 `size_average = True`, 则损失函数计算批次中所有实例的损失值的平均值;
    如果 `size_average = False`, 则损失函数计算批次中所有实例的损失至的合计.
    `size_average` 默认值为 ``True``.
    """

    def __init__(self, margin=0, size_average=True):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, self.margin, self.size_average)


class MultiMarginLoss(Module):
    r"""创建一个标准, 用以优化多元分类问题的合页损失函数 (基于空白的损失), 计算损失值时
    需要2个参数分别为输入, `x` (一个2维小批量 `Tensor`) 和输出 `y` 
    (一个1维 `Tensor`, 其值为 `x` 的索引值, `0` <= `y` <= `x.size(1)`):

    对于每个 mini-batch(小批量) 样本::

        loss(x, y) = sum_i(max(0, (margin - x[y] + x[i]))^p) / x.size(0)
                     其中 `i == 0` 至 `x.size(0)` 并且 `i != y`.

    可选择的, 如果您不想所有的类拥有同样的权重的话, 您可以通过在构造函数中传入 `weight` 参数来
    解决这个问题, `weight` 是一个1维 Tensor.

    传入 `weight` 后, 损失函数变为:

        loss(x, y) = sum_i(max(0, w[y] * (margin - x[y] - x[i]))^p) / x.size(0)

    默认情况下, 求出的损失值会对每个 minibatch 样本的结果取平均. 可以通过设置 `size_average`
    为 ``False`` 来用合计操作取代取平均操作.
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
    r"""创建一个标准, 用以衡量三元组合的损失值, 计算损失值时需要3个输入张量 `x1`, `x2`, `x3` 和
    一个大于零的 `margin` 值.
    此标准可以用来衡量输入样本间的相对相似性. 一个三元输入组合由 `a`, `p` 和 `n`: anchor,
    positive 样本 和 negative 样本组成. 所有输入变量的形式必须为 :math:`(N, D)`.

    距离交换的详细说明请参考论文 `Learning shallow convolutional feature descriptors with
    triplet losses`_ by V. Balntas, E. Riba et al.

    .. math::
        L(a, p, n) = \frac{1}{N} \left( \sum_{i=1}^N \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\} \right)

    其中 :math:`d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p`.

    Args:
        anchor: anchor 输入 tensor
        positive: positive 输入 tensor
        negative: negative 输入 tensor
        p: 正则化率. Default: 2

    Shape:
        - Input: :math:`(N, D)` 其中 `D = vector dimension`
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
