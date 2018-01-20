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
    r"""创建一个标准. 此标准被用作衡量输入 `x` 和目标 `y` 之间对应元素差的绝对值的
    平均数:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` 和 `y` 可以是拥有 n 个元素的任意形状.

    求和运算 (即上述公式中的 :math:`\sum`) 会遍历所有元素, 然后除以 n.

    如果构造函数中的参数被设置为 `size_average=False`, 那么除以 n 的操作将不会被执行.

    参数:
        size_average (bool, optional): 在默认情况下, loss 是通过对各组 (即上述公
           式中的 :math:`|x_i - y_i|`)求平均. 然而, 如果 size_average 被设置
           为 ``False``, 那么 loss 就变成了对各组求和 (即除以 n 的操作将不会执行). 
           当 reduce 是 ``False`` 时， 此参数取值会被忽略. 默认: ``True``
        reduce (bool, optional): 在默认情况下, loss 是对各组求平均或求和. 当 reduce
           是 ``False`` 时, loss 函数返回各组自己的损失同时忽略 size_average 的取值.
           默认: ``True``

    形状:
        - 输入: :math:`(N, *)`. `*` 的意思是任意多的维度数
        - 目标: :math:`(N, *)`, 和输入形状相同
        - 输出: 标量. 如果 reduce 是 ``False``, 那么
          :math:`(N, *)`, 和输入形状相同

    实例::

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
    r"""负对数似然损失 (negative log likelihood loss). 当训练一个类数为 `C` 的分类器
    时很有用.

    如果已经提供了的话, 参数 `weight` 应该是一个一维的 `Tensor`, 对每个类赋值权重. 当训练
    一个不平衡的训练集时,这个参数十分有用.

    输入需要包含每个类别的 `log-probabilities`: 输入必须是一个二维 `Tensor`, 形状是
     `(minibatch, C)`

    在一个神经网络中的 `log-probabilities`, 是可以通过在网络中最后一层后面添加一个 
    `LogSoftmax` 层获得的. 或者如果您不想额外添加一层的话, 也可以使用 
    `CrossEntropyLoss` 代替.
  
    此 loss 期望的 target 是一个类的索引 `(0 to C-1)`, C 是类的数量. 

    Loss 可以被表述为::

        loss(x, class) = -x[class]

    或者如果 weight 参数被指定的话::

        loss(x, class) = -weight[class] * x[class]

    或 ignore_index::

        loss(x, class) = class != ignoreIndex ? -weight[class] * x[class] : 0

    参数:
        weight (Tensor, optional): 手工调节的各个类的权重. 如果有的话, 必须是大小为
           `C` 的 `Tensor`.
        size_average (bool, optional): Loss 默认为对每个 `minibatch` 求平均.
           然而， 如果 size_average 被设置为 ``False``, loss 会变为对每个 `minibatch`
           求和. 如果 reduce 是 ``False``. 此参数会被忽略. 默认: ``True``
        ignore_index (int, optional): 指定一个被忽略的目标值并且不影响输入梯度. 当
           size_average 是 ``True`` 时, loss 通过对没有被忽略的目标求平均.
        reduce (bool, optional):  Loss 默认为取决于 size_average 的取值, 对每个 
           `minibatch` 求平均或求和. 当 reduce 是 ``False`` 时, 返回每个 batch 元素
           的 loss 并忽略 size_average 参数. 默认: ``True``

    形状:
        - 输入: :math:`(N, C)`. `C` 是类的数量
        - 目标: :math:`(N)` 每个值必须是 `0 <= targets[i] <= C-1`
        - 输出: 标量. 如果 reduce 是 ``False``, 那么 :math:`(N)`.

    实例::

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
    r"""这是负对数似然损失 (negative log likehood loss). 但是, 对于图像输入, 计算
    每个像素的 NLL 损失.

    参数:
        weight (Tensor, optional): 手动对各个类权重的调节. 如果给予, weight 必须是
            一个具有和类相同数量的元素的 1D Tensor.
        size_average (bool, optional): Loss 默认为对每个 `minibatch` 求平均.
           然而， 如果 size_average 被设置为 ``False``, loss 会变为对每个 `minibatch`
           求和. 如果 reduce 是 ``False``. 此参数会被忽略. 默认: ``True``
        reduce (bool, optional):  Loss 默认为取决于 size_average 的取值, 对每个 
           `minibatch` 求平均或求和. 当 reduce 是 ``False`` 时, 返回每个 batch 元素
           的 loss 并忽略 size_average 参数. 默认: ``True``


    形状:
        - 输入: :math:`(N, C, H, W)`. `C` 是类的数量
        - 目标: :math:`(N, H, W)` 每个值必须是 `0 <= targets[i] <= C-1`
        - 输出: 标量. 如果 reduce 是 ``False``, 那么改为 :math:`(N, H, W)`.

    实例::

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
    r"""泊松分布的负对数似然损失 (Negative log likelihood loss)

    Loss 可以被表述为::

        target ~ Pois(input)
        loss(input, target) = input - target * log(input) + log(target!)

    最后一项可以被忽略或用 Stirling formula 估算. 估算只对多余一的目标取值使用. 
    少于或等于一个零的目标会被加到loss中.

    参数:
        log_input (bool, optional): 如果是 ``True``, loss 用表达式
            `exp(input) - target * input` 计算, 如果是 ``False``, loss 用表达式
            `input - target * log(input+eps)` 计算.
        full (bool, optional): 是否计算full loss, 例如, 加入Stirling formula项
            `target * log(target) - target + 0.5 * log(2 * pi * target)`.
        size_average (bool, optional): loss 默认是对所有 minibatch 求平均. 然而, 
            如果 size_average 设置为 ``False``, loss 会变为对 minibatch 求和.
        eps (float, optional): 当 log_input==``False`` 时, 为避免 log(0) 运算而
            加入的很小值. 默认: 1e-8

    实例::

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
    r"""`Kullback-Leibler` 散度损失

    KL散度常用来描述两个分布的距离，并在输出分布的空间上执行直接回归时是有用的.

    与 `NLLLoss` 一样, `input` 需要包含 `log-probabilities`. 然而， 与 
    `ClassNLLLoss` 不同的是, `input` 不仅限于一个 2D Tensor, 因为这个标准是基于元素的.

    `target` 应该和 `input` 形状相同.

    Loss 可以表述为:

    .. math:: loss(x, target) = 1/n \sum(target_i * (log(target_i) - x_i))

    默认情况下, loss会基于元素求平均. 如果 `size_average=False`, loss会被累加.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    参数:
        size_average (bool, optional: loss 默认是对所有 minibatch 求平均. 然而, 
            如果 size_average 设置为 ``False``, loss 会变为对 minibatch 求和.
        reduce (bool, optional): Loss 默认为取决于 size_average 的取值, 对每个 
           `minibatch` 求平均或求和. 当 reduce 是 ``False`` 时, 返回每个 batch 元素
           的 loss 并忽略 size_average 参数. 默认: ``True``

    形状:
        - 输入: :math:`(N, *)`. `*` 的意思是任意多的维度数
        - 目标: :math:`(N, *)`, 和输入相同
        - 输出: 标量. 如果 `reduce` 是 ``True``, 那么 :math:`(N, *)`,
            和输入相同

    """
    def __init__(self, size_average=True, reduce=True):
        super(KLDivLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.kl_div(input, target, size_average=self.size_average, reduce=self.reduce)


class MSELoss(_Loss):
    r"""创建一个标准. 此标准被用作衡量输入 `x` 和目标 `y` 之间的均方误差:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|^2`

    `x` 和 `y` 可以是拥有 n 个元素的任意形状.

    求和运算 (即上述公式中的 :math:`\sum`) 会遍历所有元素, 然后除以 n.

    如果构造函数中的参数被设置为 `size_average=False`, 那么除以 n 的操作将不会被执行.

    设置 `reduce' 为 ``False`` 可以得到一组损失, 即每组各自的损失. 这些损失不会被
    参数 `size_average` 的值影响.

    参数:
        size_average (bool, optional): 在默认情况下, loss 是通过对各组 (即上述公
           式中的 :math:`|x_i - y_i|^2`)求平均. 然而, 如果 size_average 被设置
           为 ``False``, 那么 loss 就变成了对各组求和 (即除以 n 的操作将不会执行). 
           只有当 reduce 是 ``True`` 时， 此函数才会被执行. 默认: ``True``
        reduce (bool, optional): 在默认情况下, loss 是取决于 size_average, 对各
           组求平均或求和. 当 reduce 是 ``False`` 时, loss 函数返回各组自己的损失
           同时忽略 size_average 的取值. 默认: ``True``

    形状:
        - 输入: :math:`(N, *)`. `*` 的意思是任意多的维度数
        - 目标: :math:`(N, *)`, 和输入形状相同

    实例::

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
    r"""计算 `target` 与 `output` 之间的二进制交叉熵. 
    输出为:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    如果 `weight` 被指定:

    .. math:: loss(o, t) = - 1/n \sum_i weight[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    这个用于计算 `auto-encoder` 的 `reconstruction error`. 注意 0<=target[i]<=1.

    参数:
        weight (Tensor, optional): 手动对各个类权重的调节. 如果给予, weight 必须是
            一个具有和类相同数量的元素的 Tensor. 如果指定, 必须是一个有 "nbatch" 大小的
            Tensor. 
        size_average (bool, optional): 默认是对所有 minibatch 求平均. 然而, 
            如果 size_average 设置为 ``False``, loss 会变为对 minibatch 求和. 默认
            为 `True`.
    形状:
        - 输入: :math:`(N, *)`. `*` 的意思是任意多的维度数
        - 目标: :math:`(N, *)`, 和输入相同

    实例::

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
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    This Binary Cross Entropy between the target and the output logits
    (no sigmoid applied) is:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

    or in the case of the weight argument being specified:

    .. math:: loss(o, t) = - 1/n \sum_i weight[i] * (t[i] * log(sigmoid(o[i])) + (1 - t[i]) * log(1 - sigmoid(o[i])))

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. However, if the field
            size_average is set to ``False``, the losses are instead summed for
            each minibatch. Default: ``True``

     Shape:
         - Input: :math:`(N, *)` where `*` means, any number of additional
           dimensions
         - Target: :math:`(N, *)`, same shape as the input

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
    r"""给定一个输入 `x` 和对应的标签 `y`, `y` 的值只能是 1 或 -1. 此函数用来计算
    之间的损失值.

    这个 `loss` 通常用来测量两个输入是否相似. 例如, 使用 L1 成对距离为 `x`, 
    This is usually used for measuring whether two inputs are similar or
    dissimilar, e.g. using the L1 pairwise distance as `x`, 主要是用在学习非线
    性 `embedding` 或者半监督学习中::

                         { x_i,                  if y_i ==  1
        loss(x, y) = 1/n {
                         { max(0, margin - x_i), if y_i == -1

    `x` 和 `y` 可以是任意形状, 且都有n的元素. loss的求和操作作用在所有的元素上, 然后除以n.

    如果 `size_average=False`, 除以 n 的操作将不会执行.

    `margin` 的默认值为1,可以通过构造函数来设置.
    """

    def __init__(self, margin=1.0, size_average=True):
        super(HingeEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input, target):
        return F.hinge_embedding_loss(input, target, self.margin, self.size_average)


class MultiLabelMarginLoss(_Loss):
    r"""计算多标签分类的 `hinge loss` (margin-based loss), 计算loss时需要: 输入 `x` 
    (2-D mini-batch Tensor), 和输出 `y` (2-D Tensor 表示 `mini-batch` 中样本类别的索引).
    `mini-batch` 中每个样本::

        loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x.size(0)

    其中 `i == 0` 到 `x.size(0)`, `j == 0` 到 `y.size(0)`.

    `y[j] >= 0`, 且 `i != y[j]` 对于所有 `i` and `j`.

    `y` 和 `x` 必须大小相同.

    这个标准仅考虑了第一个非零目标 `y[j]`.

    此标准允许每个样本可以有不同类别.
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.multilabel_margin_loss(input, target, size_average=self.size_average)


class SmoothL1Loss(_Loss):
    r"""创建一个标准. 如果基于元素的 error 降到小于1和 L1 项的话, 则使用一个平方项. 
    
    相比于 `MSELoss`, `SmoothL1Loss` 对于 outliers 并不敏感且有时会防止梯度爆炸 (参看
    Ross Girshick 的论文 "Fast R_CNN").
    也被称为 Huber loss::

                              { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
        loss(x, y) = 1/n \sum {
                              { |x_i - y_i| - 0.5,   otherwise

    `x` 和 `y` 可以是任意形状, 且都有n的元素. 
    loss的求和操作作用在所有的元素上, 然后除以n.

    如果 `size_average=False`, 除以 n 的操作将不会执行.

    参数:
        size_average (bool, optional): 默认是对所有 minibatch 求平均. 然而, 
            如果 size_average 设置为 ``False``, loss 会变为对 minibatch 求和. 默认
            为 `True`.
        reduce (bool, optional): 在默认情况下, loss 是取决于 size_average, 对各
           组求平均或求和. 当 reduce 是 ``False`` 时, loss 函数返回各组自己的损失
           同时忽略 size_average 的取值. 默认: ``True``

    形状:
        - 输入: :math:`(N, *)` `*` 的意思是任意多的维度数
        - 目标: :math:`(N, *)`, 和输入相同
        - 输出: scalar. 如果 reduce 是 ``False``, 那么
          :math:`(N, *)`, 和输入相同

    """
    def __init__(self, size_average=True, reduce=True):
        super(SmoothL1Loss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.smooth_l1_loss(input, target, size_average=self.size_average,
                                reduce=self.reduce)


class SoftMarginLoss(_Loss):
    r"""创建一个标准, 用来优化2分类的 logistic loss. 输入为 `x`（一个 2-D mini-batch 
    Tensor）和目标 `y`（一个取值为1或-1的 Tensor）
    ::

        loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x.nelement()

    如果求出的 loss 不想被平均可以通过设置 `size_average=False` 取消.
    """
    def forward(self, input, target):
        _assert_no_grad(target)
        return F.soft_margin_loss(input, target, size_average=self.size_average)


class CrossEntropyLoss(_WeightedLoss):
    r"""此标准将 `LogSoftMax` 和 `NLLLoss` 集成到一个类中.

    当训练一个多类分类器的时候, 这个方法是十分有用的.

    如果预先提供了的话, 可选的 `weight` 参数应该是一个一维的 `Tensor`, 将权重赋值到每一个类中
    如果你的训练样本很不平均的话, 是极其有用的.

    `input` 包含每个类的得分.

    `input` 必须是一个二维的 `Tensor`, 大小为 `(minibatch, C)`.

    此标准将类的索引 (0 到 C-1) 视为 `target`, 对每一个大小为 `minibatch` 的一维 `Tensor`
    的值.

    Loss 可以表述为::

        loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                       = -x[class] + log(\sum_j exp(x[j]))

    或当 `weight` 参数被指定时::

        loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))

    Loss 是通过对每个 minibatch 求平均而得.

    参数:
        weight (Tensor, optional): 手工调节的各个类的权重. 如果有的话, 必须是大小为
           `C` 的 `Tensor`.
        size_average (bool, optional): Loss 默认为对每个 `minibatch` 求平均.
           然而， 如果 size_average 被设置为 ``False``, loss 会变为对每个 `minibatch`
           求和. 如果 reduce 是 ``False``. 此参数会被忽略.
        ignore_index (int, optional): 指定一个被忽略的目标值并且不影响输入梯度. 当
           size_average 是 ``True`` 时, loss 通过对没有被忽略的目标求平均.
        reduce (bool, optional):  Loss 默认为取决于 size_average 的取值, 对每个 
           `minibatch` 求平均或求和. 当 reduce 是 ``False`` 时, 返回每个 batch 元素
           的 loss 并忽略 size_average 参数. 默认: ``True``

    形状:
        - 输入: :math:`(N, C)` 其中 `C` 是类的数量
        - 目标: :math:`(N)` 其每个取值都大于等于 0 并 小于等于 C - 1
        - 输出: 标量. 如果 reduce 是 ``False``, 那么是 :math:`(N)`.

    实例::

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
    r"""创建一个标准，基于输入 `x` (一个 2D mini-batch `Tensor`) 和目标 `y` (一个二元
    2D `Tensor`) 之间的最大熵 (max-entropy) 和优化多标签一对全部 (one-versus-all) 的损失.
    对每个mini-batch中的样本, 对应的loss为::

       loss(x, y) = - sum_i (y[i] * log( 1 / (1 + exp(-x[i])) )
                         + ( (1-y[i]) * log(exp(-x[i]) / (1 + exp(-x[i])) ) )

    其中 `i == 0` 到 `x.nElement()-1`, `y[i]` 取值 0 或 1.
    `y` 和 `x` 必须有同样大小.
    """

    def forward(self, input, target):
        return F.multilabel_soft_margin_loss(input, target, self.weight, self.size_average)


class CosineEmbeddingLoss(Module):
    r"""给定输入 `Tensors` `x1`, `x2` 和一个标签 `Tensor` `y` (取值1或-1).
    此标准采用cosine距离, 判断两个输入是否相似, 一般用作学习非线性 embedding 或者半监
    督学习。

    `margin` 应该是-1到1之间的值，建议使用0到0.5。
    如果没有传入 `margin` 实参，默认值为0。

    每个样本的loss是::

                     { 1 - cos(x1, x2),              if y ==  1
        loss(x, y) = {
                     { max(0, cos(x1, x2) - margin), if y == -1

    如果 `size_average=True` 求出的loss会对batch求均值, 如果 `size_average=False` 的话,
    则会累加loss, 默认情况 `size_average=True`.
    """

    def __init__(self, margin=0, size_average=True):
        super(CosineEmbeddingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return F.cosine_embedding_loss(input1, input2, target, self.margin, self.size_average)


class MarginRankingLoss(Module):
    r"""创建一个标准, 给定输入 `x1`, `x2`, 两个 1-D mini-batch `Tensor`, 和一个
    1-D mini-batch `Tensor` `y`, `y` 里面的值只能是-1或1.

    如果 `y=1`, 代表第一个输入的值应该大于第二个输入的值, 如果 `y=-1` 的话, 则相反.

    `mini-batch` 中每个样本的loss的计算公式如下::

        loss(x, y) = max(0, -y * (x1 - x2) + margin)

    如果 `size_average=True`, 那么求出的loss将会对 `mini-batch` 求平均. 反之, 求出
    的loss会累加. 默认情况下, size_average=True. 
    """

    def __init__(self, margin=0, size_average=True):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.size_average = size_average

    def forward(self, input1, input2, target):
        return F.margin_ranking_loss(input1, input2, target, self.margin, self.size_average)


class MultiMarginLoss(Module):
    r"""创建一个标准, 用来计算输入 `x` (2D mini-batch `Tensor`) 和输出 `y` (包含目标
    参数索引的 1D `Tensor`, `0` <= `y` <= `x.size(1)`) 之间, 多类分类器的 hinge 
    loss（magin-based loss).

    对于每个 mini-batch 样本::

        loss(x, y) = sum_i(max(0, (margin - x[y] + x[i]))^p) / x.size(0)
                     where `i == 0` to `x.size(0)` and `i != y`.

    如果您不想所有的类拥有同样的权重的话, 您可以通过在构造函数中传入 1D `weights` 参数来
    随意对其赋值.

    loss 会变为:

        loss(x, y) = sum_i(max(0, w[y] * (margin - x[y] - x[i]))^p) / x.size(0)

    默认情况下, 求出的 loss 会对 mini-batch 取平均, 可以通过设置 `size_average=False` 
    来取消取平均操作.
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
