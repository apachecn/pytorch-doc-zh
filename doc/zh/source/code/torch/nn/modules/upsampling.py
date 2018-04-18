from numbers import Integral
import warnings

from .module import Module
from .. import functional as F


class Upsample(Module):
    r"""对给定的多通道一维时序数据, 二维空间数据, 或三维容积数据进行上采样.

    输入数据的格式为 `minibatch x channels x [depth] x [height] x width`. 
    因此, 对于2-D空间数据的输入, 期望得到一个4-D张量；对于3-D立体数据输入, 期望得到一个5-D张量.

    对3D, 4D, 5D的输入张量进行最近邻、线性、双线性和三线性采样, 可用于该上采样方法. 

    可以提供 :attr:`scale_factor` 或目标输出的 :attr:`size` 来计算输出的大小. （不能同时都给, 因为这样做是含糊不清的. ）

    Args:
        size (tuple, optional): 整型数的元组 ([D_out], [H_out], W_out) 输出大小
        scale_factor (int / tuple of ints, optional): 图像高度/宽度/深度的乘数
        mode (string, optional): 上采样算法: nearest | linear | bilinear | trilinear. 默认为: nearest

    Shape:
        - 输入: :math:`(N, C, W_{in})`, :math:`(N, C, H_{in}, W_{in})` 或 :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - 输出: :math:`(N, C, W_{out})`, :math:`(N, C, H_{out}, W_{out})`
          或 :math:`(N, C, D_{out}, H_{out}, W_{out})` 其中: 
          :math:`D_{out} = floor(D_{in} * scale\_factor)` 或 `size[-3]`
          :math:`H_{out} = floor(H_{in} * scale\_factor)` 或 `size[-2]`
          :math:`W_{out} = floor(W_{in}  * scale\_factor)` 或 `size[-1]`

    示例::

        >>> inp
        Variable containing:
        (0 ,0 ,.,.) =
          1  2
          3  4
        [torch.FloatTensor of size 1x1x2x2]

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear')
        >>> m(inp)
        Variable containing:
        (0 ,0 ,.,.) =
          1.0000  1.3333  1.6667  2.0000
          1.6667  2.0000  2.3333  2.6667
          2.3333  2.6667  3.0000  3.3333
          3.0000  3.3333  3.6667  4.0000
        [torch.FloatTensor of size 1x1x4x4]

        >>> inp
        Variable containing:
        (0 ,0 ,.,.) =
          1  2
          3  4
        [torch.FloatTensor of size 1x1x2x2]

        >>> m = nn.Upsample(scale_factor=2, mode='nearest')
        >>> m(inp)
        Variable containing:
        (0 ,0 ,.,.) =
          1  1  2  2
          1  1  2  2
          3  3  4  4
          3  3  4  4
        [torch.FloatTensor of size 1x1x4x4]


    """

    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        return F.upsample(input, self.size, self.scale_factor, self.mode)

    def __repr__(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return self.__class__.__name__ + '(' + info + ')'


class UpsamplingNearest2d(Upsample):
    r"""对多个输入通道组成的输入信号进行2维最近邻上采样. 

    为了指定采样范围, 提供了 :attr:`size` 或 :attr:`scale_factor` 作为构造参数. 

    当给定 `size`, 输出图像的大小为 (h, w). 

    Args:
        size (tuple, optional): 输出图片大小的整型元组(H_out, W_out)
        scale_factor (int, optional): 图像的 长和宽的乘子. 

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` 其中
          :math:`H_{out} = floor(H_{in} * scale\_factor)`
          :math:`W_{out} = floor(W_{in}  * scale\_factor)`

    示例::

        >>> inp
        Variable containing:
        (0 ,0 ,.,.) =
          1  2
          3  4
        [torch.FloatTensor of size 1x1x2x2]

        >>> m = nn.UpsamplingNearest2d(scale_factor=2)
        >>> m(inp)
        Variable containing:
        (0 ,0 ,.,.) =
          1  1  2  2
          1  1  2  2
          3  3  4  4
          3  3  4  4
        [torch.FloatTensor of size 1x1x4x4]

    """
    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingNearest2d, self).__init__(size, scale_factor, mode='nearest')

    def forward(self, input):
        warnings.warn("nn.UpsamplingNearest2d is deprecated. Use nn.Upsample instead.")
        return super(UpsamplingNearest2d, self).forward(input)


class UpsamplingBilinear2d(Upsample):
    r"""对多个输入通道组成的输入信号进行2维双线性上采样. 

    为了指定采样范围, 提供了 :attr:`size` 或 :attr:`scale_factor` 作为构造参数. 

    当给定 `size`, 输出图像的大小为 (h, w). 

    Args:
        size (tuple, optional): 输出图片大小的整型元组(H_out, W_out)
        scale_factor (int, optional): 图像的 长和宽的乘子. 

    shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` 其中
          :math:`H_{out} = floor(H_{in} * scale\_factor)`
          :math:`W_{out} = floor(W_{in}  * scale\_factor)`

    示例::::

        >>> inp
        Variable containing:
        (0 ,0 ,.,.) =
          1  2
          3  4
        [torch.FloatTensor of size 1x1x2x2]

        >>> m = nn.UpsamplingBilinear2d(scale_factor=2)
        >>> m(inp)
        Variable containing:
        (0 ,0 ,.,.) =
          1.0000  1.3333  1.6667  2.0000
          1.6667  2.0000  2.3333  2.6667
          2.3333  2.6667  3.0000  3.3333
          3.0000  3.3333  3.6667  4.0000
        [torch.FloatTensor of size 1x1x4x4]

    """
    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingBilinear2d, self).__init__(size, scale_factor, mode='bilinear')

    def forward(self, input):
        warnings.warn("nn.UpsamplingBilinear2d is deprecated. Use nn.Upsample instead.")
        return super(UpsamplingBilinear2d, self).forward(input)
