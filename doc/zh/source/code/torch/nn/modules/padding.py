from .module import Module
from .utils import _pair, _quadruple, _ntuple
from .. import functional as F


# TODO: grad_output size asserts in THNN


class ConstantPad1d(Module):
    r"""Pads the input tensor boundaries with a constant value.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in both boundaries. If a 2-tuple, uses (paddingLeft, paddingRight)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ConstantPad1d(3, 3.5)
        >>> input = autograd.Variable(torch.randn(16, 2, 480))
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ConstantPad1d((3, 5), 3.5)
        >>> output = m(input)

    """

    def __init__(self, padding, value):
        super(ConstantPad1d, self).__init__()
        self.padding = _pair(padding)
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ConstantPad2d(Module):
    r"""用一个常数值填充输入张量边界.

    对于 Nd-padding, 使用 nn.functional.pad().

    Args:
        padding (int, tuple):填充的大小.  如果是int, 则在所有边界使用相同的填充. 
        如果是4个元组, 使用 (paddingLeft, paddingRight, paddingTop, paddingBottom)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Example::

        >>> m = nn.ConstantPad2d(3, 3.5)
        >>> input = autograd.Variable(torch.randn(16, 3, 320, 480))
        >>> output = m(input)
        >>> # 使用不同的填充
        >>> m = nn.ConstantPad2d((3, 3, 6, 6), 3.5)
        >>> output = m(input)

    """

    def __init__(self, padding, value):
        super(ConstantPad2d, self).__init__()
        self.padding = _quadruple(padding)
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ConstantPad3d(Module):
    r"""Pads the input tensor boundaries with a constant value.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 6-tuple, uses
            (paddingLeft, paddingRight, paddingTop, paddingBottom, paddingFront, paddingBack)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = D_{in} + paddingFront + paddingBack`
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ConstantPad3d(3, 3.5)
        >>> input = autograd.Variable(torch.randn(16, 3, 10, 20, 30))
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
        >>> output = m(input)

    """

    def __init__(self, padding, value):
        super(ConstantPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReflectionPad1d(Module):
    r"""Pads the input tensor using the reflection of the input boundary.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 2-tuple, uses (paddingLeft, paddingRight)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ReflectionPad1d(3)
        >>> input = autograd.Variable(torch.randn(16, 3, 480))
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ReflectionPad1d((3, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReflectionPad1d, self).__init__()
        self.padding = _pair(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'reflect')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReflectionPad2d(Module):
    r"""使用输入边界的反射填充输入张量.

    Args:
        padding (int, tuple): 填充的大小.  如果是int, 则在所有边界填充使用相同的.
        如果是4个元组, 则使用 (paddingLeft, paddingRight, paddingTop, paddingBottom)
        
    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Example::

        >>> m = nn.ReflectionPad2d(3)
        >>> input = autograd.Variable(torch.randn(16, 3, 320, 480))
        >>> output = m(input)
        >>> # 使用不同的填充
        >>> m = nn.ReflectionPad2d((3, 3, 6, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'reflect')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReplicationPad1d(Module):
    r"""使用输入边界的复制填充输入张量.

    Args:
    padding (int, tuple): 填充的大小.  如果是int, 则在所有边界使用相同的填充. 
             如果一个2元组, 使用 (paddingLeft, paddingRight)
        
    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Example::

        >>> m = nn.ReplicationPad1d(3)
        >>> input = autograd.Variable(torch.randn(16, 3, 480))
        >>> output = m(input)
        >>> # 使用不同的填充
        >>> m = nn.ReplicationPad1d((3, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReplicationPad1d, self).__init__()
        self.padding = _pair(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'replicate')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReplicationPad2d(Module):
    r"""使用输入边界的复制填充输入张量.

    Args:
        padding (int, tuple): 填充的大小. 如果是int, 则在所有边界使用相同的填充.
         如果是4个元组, 则使用(paddingLeft, paddingRight, paddingTop, paddingBottom)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Example::

        >>> m = nn.ReplicationPad2d(3)
        >>> input = autograd.Variable(torch.randn(16, 3, 320, 480))
        >>> output = m(input)
        >>> # 使用不同的填充
        >>> m = nn.ReplicationPad2d((3, 3, 6, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReplicationPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'replicate')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReplicationPad3d(Module):
    r"""使用输入边界的复制填充输入张量.

    Args:
        padding (int, tuple): 填充的大小. 如果是int, 则在所有边界使用相同的填充.
        如果是四个元组, 则使用 (paddingLeft, paddingRight,
        paddingTop, paddingBottom, paddingFront, paddingBack)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = D_{in} + paddingFront + paddingBack`
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Example::

        >>> m = nn.ReplicationPad3d(3)
        >>> input = autograd.Variable(torch.randn(16, 3, 8, 320, 480))
        >>> output = m(input)
        >>> # 使用不同的填充
        >>> m = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReplicationPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'replicate')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ZeroPad2d(ConstantPad2d):
    r"""用零填充输入张量边界.

    Args:
        padding (int, tuple): 填充的大小. 如果是int, 则在所有边界使用相同的填充.
        . 如果是四个元组, 则使用 (paddingLeft, paddingRight, paddingTop, paddingBottom)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Example::

        >>> m = nn.ZeroPad2d(3)
        >>> input = autograd.Variable(torch.randn(16, 3, 320, 480))
        >>> output = m(input)
        >>> # 使用不同的填充
        >>> m = nn.ZeroPad2d((3, 3, 6, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ZeroPad2d, self).__init__(padding, 0)
