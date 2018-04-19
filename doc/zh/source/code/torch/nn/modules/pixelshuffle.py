from .module import Module
from .. import functional as F


class PixelShuffle(Module):
    r"""
    对张量中形如 :math:`(*, C * r^2, H, W]` 的元素, 重新排列成 :math:`(C, H * r, W * r)`.

    当使用 stride = :math:`1/r` 的高效子像素卷积很有用.

    参考如下论文获得更多信息:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    Shi et. al (2016) . 

    Args:
        upscale_factor (int): 增加空间分辨率的因子

    Shape:
        - 输入: :math:`(N, C * {upscale\_factor}^2, H, W)`
        - 输出: :math:`(N, C, H * {upscale\_factor}, W * {upscale\_factor})`

    Examples::

        >>> ps = nn.PixelShuffle(3)
        >>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return F.pixel_shuffle(input, self.upscale_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(upscale_factor=' + str(self.upscale_factor) + ')'
