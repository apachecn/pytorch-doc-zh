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
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Ignored when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed
           for each minibatch. When reduce is ``False``, the loss function returns
           a loss per batch element instead and ignores size_average.
           Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input

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
    r"""The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    If provided, the optional argument `weight` should be a 1D Tensor assigning
    weight to each of the classes. This is particularly useful when you have an
    unbalanced training set.

    The input given through a forward call is expected to contain
    log-probabilities of each class: input has to be a 2D Tensor of size
    `(minibatch, C)`

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The target that this loss expects is a class index
    `(0 to C-1, where C = number of classes)`

    The loss can be described as::

        loss(x, class) = -x[class]

    or in the case of the weight argument it is specified as follows::

        loss(x, class) = -weight[class] * x[class]

    or in the case of ignore_index::

        loss(x, class) = class != ignoreIndex ? -weight[class] * x[class] : 0

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
           class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Ignored when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average
            is ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): By default, the losses are averaged or summed
            for each minibatch. When reduce is ``False``, the loss function returns
            a loss per batch element instead and ignores size_average.
            Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`
        - Target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        - Output: scalar. If reduce is ``False``, then :math:`(N)` instead.

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
    r"""This is negative log likehood loss, but for image inputs. It computes
    NLL loss per-pixel.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a 1D Tensor having as many elements,
            as there are classes.
        size_average: By default, the losses are averaged over observations
            for each minibatch. However, if the field size_average is set to
            ``False``, the losses are instead summed for each minibatch.
            Ignored when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed
            for each minibatch depending on size_average. When reduce is ``False``,
            the loss function returns a loss per batch element instead and
            ignores size_average. Default: ``True``


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
    r"""Negative log likelihood loss with Poisson distribution of target.

    The loss can be described as::

        target ~ Pois(input)
        loss(input, target) = input - target * log(input) + log(target!)

    The last term can be omitted or approximised with Stirling formula. The
    approximation is used for target values more than 1. For targets less or
    equal to 1 zeros are added to the loss.

    Args:
        log_input (bool, optional): if ``True`` the loss is computed as
            `exp(input) - target * input`, if ``False`` the loss is
            `input - target * log(input+eps)`.
        full (bool, optional): whether to compute full loss, i. e. to add the
            Stirling approximation term
            `target * log(target) - target + 0.5 * log(2 * pi * target)`.
        size_average (bool, optional): By default, the losses are averaged over
            observations for each minibatch. However, if the field size_average
            is set to ``False``, the losses are instead summed for each minibatch.
        eps (float, optional): Small value to avoid evaluation of log(0) when
            log_input==``False``. Default: 1e-8

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
    r"""The `Kullback-Leibler divergence`_ Loss

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    As with `NLLLoss`, the `input` given is expected to contain
    *log-probabilities*, however unlike `ClassNLLLoss`, `input` is not
    restricted to a 2D Tensor, because the criterion is applied element-wise.

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.

    The loss can be described as:

    .. math:: loss(x, target) = 1/n \sum(target_i * (log(target_i) - x_i))

    By default, the losses are averaged for each minibatch over observations
    **as well as** over dimensions. However, if the field
    `size_average` is set to ``False``, the losses are instead summed.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
        size_average (bool, optional: By default, the losses are averaged
            for each minibatch over observations **as well as** over
            dimensions. However, if ``False`` the losses are instead summed.
        reduce (bool, optional): By default, the losses are averaged
            over observations for each minibatch, or summed, depending on
            size_average. When reduce is ``False``, returns a loss per batch
            element instead and ignores size_average. Default: ``True``

    Shape:
        - input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - target: :math:`(N, *)`, same shape as the input
        - output: scalar. If `reduce` is ``True``, then :math:`(N, *)`,
            same shape as the input

    """
    def __init__(self, size_average=True, reduce=True):
        super(KLDivLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.kl_div(input, target, size_average=self.size_average, reduce=self.reduce)


class MSELoss(_Loss):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|^2`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal variable
    `size_average` to ``False``.

    To get a batch of losses, a loss per batch element, set `reduce` to
    ``False``. These losses are not averaged and are not affected by
    `size_average`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Only applies when reduce is ``True``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged
           over observations for each minibatch, or summed, depending on
           size_average. When reduce is ``False``, returns a loss per batch
           element instead and ignores size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

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
    r"""Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    .. math:: loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    or in the case of the weight argument being specified:

    .. math:: loss(o, t) = - 1/n \sum_i weight[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

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
    需要2个参数分别为输入，`x` (一个2维小批量 `Tensor`) 和输出 `y` 
    (一个2维 `Tensor`, 其值为 `x` 的索引值)。
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
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    创建一个标准，当某个元素的错误值的绝对值小于1时使用平方项计算, 其他情况则使用L1范式计算.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).
    此方法创建的标准对于异常值不如 `MSELoss`敏感, 但是同时在某些情况下可以防止梯度爆炸 (比如
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
    r"""This criterion combines `LogSoftMax` and `NLLLoss` in one single class.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument `weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain scores for each class.

    `input` has to be a 2D `Tensor` of size `(minibatch, C)`.

    This criterion expects a class index (0 to C-1) as the
    `target` for each value of a 1D tensor of size `minibatch`

    The loss can be described as::

        loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
                       = -x[class] + log(\sum_j exp(x[j]))

    or in the case of the `weight` argument being specified::

        loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))

    The losses are averaged across observations for each minibatch.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
           If given, has to be a Tensor of size "C"
        size_average (bool, optional): By default, the losses are averaged over observations for each minibatch.
           However, if the field size_average is set to ``False``, the losses are
           instead summed for each minibatch. Ignored if reduce is ``False``.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            ``True``, the loss is averaged over non-ignored targets.
        reduce (bool, optional): By default, the losses are averaged or summed over
            observations for each minibatch depending on size_average. When reduce
            is ``False``, returns a loss per batch element instead and ignores
            size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`
        - Target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
        - Output: scalar. If reduce is ``False``, then :math:`(N)` instead.

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
    创建一个标准, 基于输入 `x` 和目标 `y`的 max-entropy(最大熵), 优化多标签 one-versus-all 损失.
    输入 `x` 为一个2维 mini-batch `Tensor`, 目标 `y` 为2进制2维 `Tensor`.
    对每个 mini-batch 中的样本，对应的 loss 为::

       loss(x, y) = - sum_i (y[i] * log( 1 / (1 + exp(-x[i])) )
                         + ( (1-y[i]) * log(exp(-x[i]) / (1 + exp(-x[i])) ) )

    其中 `i == 0` 至 `x.nElement()-1`, `y[i]  in {0,1}`.
    `y` 和 `x` 必须具有相同的维度.
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
    r"""创建一个衡量 mini-batch(小批量) 中的2个1维 `Tensor` 的输入 `x1` 和 `x2`,
    和1个1维 `Tensor` 的目标 `y`(`y` 的取值是 `1` 或者 `-1`) 之间损失的标准.

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
