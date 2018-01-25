import torch
from .module import Module
from torch.nn.parameter import Parameter
from .. import functional as F


# TODO: check contiguous in THNN
# TODO: use separate backend functions?
class _BatchNorm(Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input):
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class (_BatchNorm):
    r""" 对 2d 或者 3d 的小批量 (mini-batch) 数据进行批标准化 (Batch Normalization) 操作.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    每个小批量数据中,计算各个维度的均值和标准差,并且 gamma 和 beta 是大小为 C 的可学习,
    可改变的仿射参数向量( C 为输入大小).

    在训练过程中,该层计算均值和方差,并进行平均移动,默认的平均移动动量值为 0.1.

    在验证时,训练得到的均值/方差,用于标准化验证数据.

    BatchNorm 在 'C' 维上处理,即 '(N,L)' 部分运行,被称作 'Temporal BatchNorm'

    Args:
        num_features: 预期输入的特征数,大小为 'batch_size x num_features [x width]'
        eps: 给分母加上的值,保证数值稳定(分母不能趋近0或取0),默认为 1e-5
        momentum: 动态均值和动态方差使用的移动动量值,默认为 0.1
        affine: 布尔值,设为 True 时,表示该层添加可学习,可改变的仿射参数,即 gamma 和 beta,默认为 True

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = autograd.Variable(torch.randn(20, 100))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm1d, self)._check_input_dim(input)


class BatchNorm2d(_BatchNorm):
    r""" 对小批量 (mini-batch) 3d 数据组成的 4d 输入进行标准化 (Batch Normalization) 操作.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    每个小批量数据中,计算各个维度的均值和标准差,
    并且 gamma 和 beta 是大小为 C 的可学习,可改变的仿射参数向量 (C 为输入大小).

    在训练过程中,该层计算均值和方差,并进行平均移动.默认的平均移动动量值为 0.1.

    在验证时,训练得到的均值/方差,用于标准化验证数据.

    BatchNorm 在 'C' 维上处理,即 '(N, H, W)' 部分运行,被称作 'Spatial BatchNorm'.

    Args:
        num_features: 预期输入的特征数,大小为 'batch_size x num_features x height x width'
        eps: 给分母加上的值,保证数值稳定(分母不能趋近0或取0),默认为 1e-5
        momentum: 动态均值和动态方差使用的移动动量值,默认为 0.1
        affine: 布尔值,设为 True 时,表示该层添加可学习,可改变的仿射参数,即 gamma 和 beta,默认为 True

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm2d, self)._check_input_dim(input)


class BatchNorm3d(_BatchNorm):
    r""" 对小批量 (mini-batch) 4d 数据组成的 5d 输入进行标准化 (Batch Normalization) 操作.

    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta

    每个小批量数据中,计算各个维度的均值和标准差,
    并且 gamma 和 beta 是大小为 C 的可学习,可改变的仿射参数向量 (C 为输入大小).

    在训练过程中,该层计算均值和方差,并进行平均移动.默认的平均移动动量值为 0.1.

    在验证时,训练得到的均值/方差,用于标准化验证数据.

    BatchNorm 在 'C' 维上处理,即 '(N, D, H, W)' 部分运行,被称作 'Volumetric BatchNorm' 或者 'Spatio-temporal BatchNorm'

    Args:
        num_features: 预期输入的特征数,大小为 'batch_size x num_features x depth x height x width'
        eps: 给分母加上的值,保证数值稳定(分母不能趋近0或取0),默认为 1e-5
        momentum: 动态均值和动态方差使用的移动动量值,默认为 0.1
        affine: 布尔值,设为 True 时,表示该层添加可学习,可改变的仿射参数,即 gamma 和 beta,默认为 True

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm3d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(BatchNorm3d, self)._check_input_dim(input)
