from .batchnorm import _BatchNorm
from .. import functional as F


class _InstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_InstanceNorm, self).__init__(
            num_features, eps, momentum, affine)

    def forward(self, input):
        b, c = input.size(0), input.size(1)

        # Repeat stored stats and affine transform params
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        weight, bias = None, None
        if self.affine:
            weight = self.weight.repeat(b)
            bias = self.bias.repeat(b)

        # Apply instance norm
        input_reshaped = input.contiguous().view(1, b * c, *input.size()[2:])

        out = F.batch_norm(
            input_reshaped, running_mean, running_var, weight, bias,
            True, self.momentum, self.eps)

        # Reshape back
        self.running_mean.copy_(running_mean.view(b, c).mean(0, keepdim=False))
        self.running_var.copy_(running_var.view(b, c).mean(0, keepdim=False))

        return out.view(b, c, *input.size()[2:])

    def eval(self):
        return self


class InstanceNorm1d(_InstanceNorm):
    r""" 对 2d 或者 3d 的小批量 (mini-batch) 数据进行实例标准化 (Instance Normalization) 操作.
    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * gamma + beta

    对小批量数据中的每一个对象,计算其各个维度的均值和标准差,并且 gamma 和 beta 是大小为 C 的可学习,
    可改变的仿射参数向量( C 为输入大小).

    在训练过程中,该层计算均值和方差,并进行平均移动,默认的平均移动动量值为 0.1.

    在验证时 ('.eval()'),InstanceNorm 模型默认保持不变,即求得的均值/方差不用于标准化验证数据,
    但可以用 '.train(False)' 方法强制使用存储的均值和方差.

    Args:
        num_features: 预期输入的特征数,大小为 'batch_size x num_features x width'
        eps: 给分母加上的值,保证数值稳定(分母不能趋近0或取0),默认为 1e-5
        momentum: 动态均值和动态方差使用的移动动量值,默认为 0.1
        affine: 布尔值,设为 True 时,表示该层添加可学习,可改变的仿射参数,即 gamma 和 beta,默认为 False

    Shape:
        - Input: :math:`(N, C, L)`
        - Output: :math:`(N, C, L)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm1d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm1d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 40))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))
        super(InstanceNorm1d, self)._check_input_dim(input)


class InstanceNorm2d(_InstanceNorm):
    r""" 对小批量 (mini-batch) 3d 数据组成的 4d 输入进行实例标准化 (Batch Normalization) 操作.
    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * gamma + beta

    对小批量数据中的每一个对象,计算其各个维度的均值和标准差,并且 gamma 和 beta 是大小为 C 的可学习,
    可改变的仿射参数向量( C 为输入大小).

    在训练过程中,该层计算均值和方差,并进行平均移动,默认的平均移动动量值为 0.1.

    在验证时 ('.eval()'),InstanceNorm 模型默认保持不变,即求得的均值/方差不用于标准化验证数据,
    但可以用 '.train(False)' 方法强制使用存储的均值和方差.

    Args:
        num_features: 预期输入的特征数,大小为 'batch_size x num_features x height x width'
        eps: 给分母加上的值,保证数值稳定(分母不能趋近0或取0),默认为 1e-5
        momentum: 动态均值和动态方差使用的移动动量值,默认为 0.1
        affine: 布尔值,设为 True 时,表示该层添加可学习,可改变的仿射参数,即 gamma 和 beta,默认为 False

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm2d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm2d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(InstanceNorm2d, self)._check_input_dim(input)


class InstanceNorm3d(_InstanceNorm):
    r""" 对小批量 (mini-batch) 4d 数据组成的 5d 输入进行实例标准化 (Batch Normalization) 操作.
    .. math::

        y = \frac{x - mean[x]}{ \sqrt{Var[x]} + \epsilon} * gamma + beta

    对小批量数据中的每一个对象,计算其各个维度的均值和标准差,并且 gamma 和 beta 是大小为 C 的可学习,
    可改变的仿射参数向量( C 为输入大小).

    在训练过程中,该层计算均值和方差,并进行平均移动,默认的平均移动动量值为 0.1.

    在验证时 ('.eval()'),InstanceNorm 模型默认保持不变,即求得的均值/方差不用于标准化验证数据,
    但可以用 '.train(False)' 方法强制使用存储的均值和方差.

    Args:
        num_features: 预期输入的特征数,大小为 'batch_size x num_features x depth x height x width'
        eps: 给分母加上的值,保证数值稳定(分母不能趋近0或取0),默认为 1e-5
        momentum: 动态均值和动态方差使用的移动动量值,默认为 0.1
        affine: 布尔值,设为 True 时,表示该层添加可学习,可改变的仿射参数,即 gamma 和 beta,默认为 False

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)

    Examples:
        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm3d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm3d(100, affine=True)
        >>> input = autograd.Variable(torch.randn(20, 100, 35, 45, 10))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        super(InstanceNorm3d, self)._check_input_dim(input)
