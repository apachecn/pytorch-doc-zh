import math
import random

import torch
from torch.autograd import Variable


def calculate_gain(nonlinearity, param=None):
    """返回给定非线性函数的推荐增益值.
    它们的值如下:

    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    tanh         :math:`5 / 3`
    relu         :math:`\sqrt{2}`
    leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    ============ ==========================================

    Args:
        nonlinearity: 非线性函数 (`nn.functional` name)
        param: 非线性函数的可选参数

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def uniform(tensor, a=0, b=1):
    """使用均匀分布 :math:`U(a, b)` 中的值填充输入的 Tensor（张量）或者 Variable（变量）.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable
        a: 均匀分布的下界
        b: 均匀分布的上界

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.uniform(w)
    """
    if isinstance(tensor, Variable):
        uniform(tensor.data, a=a, b=b)
        return tensor

    return tensor.uniform_(a, b)


def normal(tensor, mean=0, std=1):
    """使用从正态分布 :math:`N(mean, std)` 绘制的值填充输入的 Tensor（张量）或者 Variable（变量）.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable
        mean: 正态分布的均值
        std: 正态分布的标准差

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.normal(w)
    """
    if isinstance(tensor, Variable):
        normal(tensor.data, mean=mean, std=std)
        return tensor

    return tensor.normal_(mean, std)


def constant(tensor, val):
    """使用值 `val` 填充输入的 Tensor（张量）或者 Variable（变量）.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable
        val: 填充 tensor（张量）的值

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.constant(w)
    """
    if isinstance(tensor, Variable):
        constant(tensor.data, val)
        return tensor

    return tensor.fill_(val)


def eye(tensor):
    """用单位矩阵填充 2 维输入的 Tensor（张量）或 Variable（变量）.
    保留线性层中输入的标记, 尽可能多地保留输入.

    Args:
        tensor: 一个 2 维的 torch.Tensor 或 autograd.Variable

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.eye(w)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    if isinstance(tensor, Variable):
        eye(tensor.data)
        return tensor

    return tensor.copy_(torch.eye(tensor.size(0), tensor.size(1)))


def dirac(tensor):
    """用狄拉克三角函数填充 {3,4,5} 维输入的张量或变量.
    保留卷积层中输入的标记, 尽可能多地保存输入通道.

    Args:
        tensor: 一个 {3, 4, 5} 维的 torch.Tensor 或 autograd.Variable

    Examples:
        >>> w = torch.Tensor(3, 16, 5, 5)
        >>> nn.init.dirac(w)
    """
    dimensions = tensor.ndimension()
    if dimensions not in [3, 4, 5]:
        raise ValueError("Only tensors with 3, 4, or 5 dimensions are supported")

    if isinstance(tensor, Variable):
        dirac(tensor.data)
        return tensor

    sizes = tensor.size()
    min_dim = min(sizes[0], sizes[1])
    tensor.zero_()

    for d in range(min_dim):
        if dimensions == 3:  # Temporal convolution
            tensor[d, d, tensor.size(2) // 2] = 1
        elif dimensions == 4:  # Spatial convolution
            tensor[d, d, tensor.size(2) // 2, tensor.size(3) // 2] = 1
        else:  # Volumetric convolution
            tensor[d, d, tensor.size(2) // 2, tensor.size(3) // 2, tensor.size(4) // 2] = 1
    return tensor


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, gain=1):
    """根据 "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010)
    中描述的方法, 使用均匀分布填充输入张量或变量.
    作为结果的张量将具有从
    :math:`U(-a, a)` 中取样的值, 其中
    :math:`a = gain \\times \sqrt{2 / (fan\_in + fan\_out)} \\times \sqrt{3}`.
    也被称为 Glorot 初始化.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable
        gain: 一个可选的比例因子

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))
    """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """
    根据 "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010) 中描述的方法,
    使用正态分布填充输入张量或变量.
    作为结果的张量将具有从
    :math:`N(0, std)` 中取样的值, 其中
    :math:`std = gain \\times \sqrt{2 / (fan\_in + fan\_out)}`.
    也被称为 Glorot 初始化.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable
        gain: 一个可选的比例因子

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_normal(w)
    """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return tensor.normal_(0, std)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform(tensor, a=0, mode='fan_in'):
    """根据 "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015) 中所描述的方法, 填充输入的张量或变量
    作为结果的张量将具有从
    :math:`U(-bound, bound)` 中取样的值, 其中
    :math:`bound = \sqrt{2 / ((1 + a^2) \\times fan\_in)} \\times \sqrt{3}`.
    也被称为 Glorot 初始化.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable
        a: 在该层之后使用的 rectifier（整流器）的负斜率（默认情况下, ReLU 为 0）
        mode: 'fan_in' (default) 或 'fan_out' 其中的一个.
            选择 `fan_in` 保留 forward pass（前向传递）中权重方差的量级.
            选择 `fan_out` 来保存 backwards pass（反向传递）的量级.

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.kaiming_uniform(w, mode='fan_in')
    """
    if isinstance(tensor, Variable):
        kaiming_uniform(tensor.data, a=a, mode=mode)
        return tensor

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain('leaky_relu', a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return tensor.uniform_(-bound, bound)


def kaiming_normal(tensor, a=0, mode='fan_in'):
    """根据 "Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification" - He, K. et al. (2015) 中所描述的方法,
    使用正态分布填充输入张量或变量值. 
    作为结果的张量将具有从
    :math:`N(0, std)` 中取样的值, 其中
    :math:`std = \sqrt{2 / ((1 + a^2) \\times fan\_in)}`.
    也被称为 Glorot 初始化.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable
        a: 在该层之后使用的 rectifier（整流器）的负斜率（默认情况下, ReLU 为 0）
        mode: 'fan_in' (default) 或 'fan_out' 其中的一个.
            选择 `fan_in` 保留 forward pass（前向传递）中权重方差的量级.
            选择 `fan_out` 来保存 backwards pass（反向传递）的量级.

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.kaiming_normal(w, mode='fan_out')
    """
    if isinstance(tensor, Variable):
        kaiming_normal(tensor.data, a=a, mode=mode)
        return tensor

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain('leaky_relu', a)
    std = gain / math.sqrt(fan)
    return tensor.normal_(0, std)


def orthogonal(tensor, gain=1):
    """根据 "Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks" - Saxe, A. et al. (2013) 中描述的那样,
    使用（半）正交矩阵填充输入张量或变量.
    输入张量必须至少有 2 个维度, 对于 2 维以上的张量, 后面的维度须是平坦的.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable, 其中 n >= 2
        gain: 可选比例因子

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.orthogonal(w)
    """
    if isinstance(tensor, Variable):
        orthogonal(tensor.data, gain=gain)
        return tensor

    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor


def sparse(tensor, sparsity, std=0.01):
    """
    根据 "Deep learning via Hessian-free optimization" - Martens, J. (2010) 中描述的那样,
    将 2D 输入张量或变量填充为稀疏矩阵, 其中非零元素将从正态分布 :math:`N(0, 0.01)` 中绘制.

    Args:
        tensor: 一个 n 维的 torch.Tensor 或 autograd.Variable
        sparsity: 每列中元素的 fraction（部分）被设置为零
        std: 用于生成非零值的正态分布的标准差

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.sparse(w, sparsity=0.1)
    """
    if isinstance(tensor, Variable):
        sparse(tensor.data, sparsity, std=std)
        return tensor

    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    tensor.normal_(0, std)
    rows, cols = tensor.size(0), tensor.size(1)
    num_zeros = int(math.ceil(rows * sparsity))

    for col_idx in range(tensor.size(1)):
        row_indices = list(range(rows))
        random.shuffle(row_indices)
        zero_indices = row_indices[:num_zeros]
        for row_idx in zero_indices:
            tensor[row_idx, col_idx] = 0

    return tensor
