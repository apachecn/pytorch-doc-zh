from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from . import functional as F

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "Scale", "CenterCrop", "Pad",
           "Lambda", "RandomCrop", "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop",
           "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation", "ColorJitter", "RandomRotation",
           "Grayscale", "RandomGrayscale"]


class Compose(object):
    """将多个变换组合到一起.

    Args:
        transforms (list of ``Transform`` objects): 要组合的变换列表.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """转换一个 ``PIL Image`` 或 ``numpy.ndarray`` 为 tensor（张量）.

    将范围 [0, 255] 中的 PIL Image 或 numpy.ndarray (H x W x C) 转换形状为
    (C x H x W) , 值范围为 [0.0, 1.0] 的 torch.FloatTensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): 将要被转换为 tensor 的 Image.

        Returns:
            Tensor: 转换后的 image.
        """
        return F.to_tensor(pic)


class ToPILImage(object):
    """转换一个 tensor 或 ndarray 为 PIL Image.

    转换一个形状为(C x H x W) 的 torch.*Tensor 或一个形状为(H x W x C )的numpy ndarray 至一个 PIL Image ,同时保留值范围.

    Args:
        mode (`PIL.Image mode`_): 输入数据的色域和像素深度 (可选).
            如果 ``mode`` 为 ``None`` (默认) ,这里对输入数据有一些假设:
            1. 如果输入有3个通道,  ``mode`` 假设为 ``RGB``.
            2. 如果输入有4个通道,  ``mode`` 假设为 ``RGBA``.
            3. 如果输入有1个通道,  ``mode`` 根据数据类型确定 (i,e,
            ``int``, ``float``, ``short``).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray):要转换为PIL Image的图像.

        Returns:
            PIL Image: 转换为PIL Image的图像.

        """
        return F.to_pil_image(pic, self.mode)


class Normalize(object):
    """用均值和标准偏差对张量图像进行归一化.
    给定均值: ``(M1,...,Mn)`` 和标准差: ``(S1,..,Sn)`` 用于 ``n`` 个通道,
    该变换将标准化输入 ``torch.*Tensor`` 的每一个通道.
    例如: ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): 每一个通道的均值序列.
        std (sequence): 每一个通道的标准差序列.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): 需要被归一化的大小为 (C, H, W)Tensor image.

        Returns:
            Tensor: 归一化后的 Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std)


class Resize(object):
    """调整输入的 PIL Image 尺寸为给定的 size（尺寸）.

    Args:
        size (sequence or int): 期望输出的尺寸. 如果 size（尺寸）是一个像
            (h, w) 这样的序列, 则 output size（输出尺寸）将于此匹配.
            如果 size（尺寸）是一个 int 类型的数字,
            图像较小的边缘将被匹配到该数字.
            例如, 如果 height > width, 那么图像将会被重新缩放到
            (size * height / width, size). 即按照size/width的比值缩放
        interpolation (int, optional): 期望的插值. 默认是
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)


class Scale(Resize):
    """
    Note: 为了支持 Resize, 该变换已经过时了.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                      "please use transforms.Resize instead.")
        super(Scale, self).__init__(*args, **kwargs)


class CenterCrop(object):
    """在中心裁剪指定的 PIL Image.

    Args:
        size (sequence or int): 期望裁剪的输出尺寸. 如果 size（尺寸）是 ``int`` 类型的整数, 而不是像 (h, w) 这样类型的序列, 裁剪出来的图像是 (size, size) 这样的正方形的.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return F.center_crop(img, self.size)


class Pad(object):
    """用指定的 "pad" 值填充指定的 PIL image.

    Args:
        padding (int or tuple): 填充每个边框. 如果提供了一个 int 型的整数, 则用于填充所有边界.
            如果提供长度为 2 的元组, 则这是分别在 左/右 和 上/下 的填充.
            如果提供长度为 4 的元组, 则这是分别用于 左, 上, 右 和 下 部边界的填充.
        fill: 像素填充.  默认值为 0. 如果长度为 3 的元组, 分别用于填充 R, G, B 通道.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, self.padding, self.fill)


class Lambda(object):
    """应用一个用户定义的 Lambda 作为变换.

    Args:
        lambd (function): Lambda/function 以用于 transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class RandomCrop(object):
    """在一个随机位置裁剪指定的 PIL Image.

    Args:
        size (sequence or int): 期望输出的裁剪尺寸. 如果 size（尺寸）是 ``int`` 类型的整数, 而不是像 (h, w) 这样类型的序列, 裁剪出来的图像是 (size, size) 这样的正方形的.
        padding (int or sequence, optional): 图像的每个边框上的可选填充. 缺省值是 0, 即没有填充. 如果提供长度为 4 的序列, 则分别用于填充左侧, 顶部, 右侧, 底部边界.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)


class RandomHorizontalFlip(object):
    """以概率0.5随机水平翻转图像"""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return F.hflip(img)
        return img


class RandomVerticalFlip(object):
    """以概率0.5随机垂直翻转图像."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return F.vflip(img)
        return img


class RandomResizedCrop(object):
    """将给定的 PIL 图像裁剪为随机大小和纵横比例.

    原始高宽比的随机大小（默认: 0.08 到 1.0）和随机宽高比（默认: 3/4 到 4/3）的裁剪.
    该裁剪最终会被调整为指定的尺寸.

    该操作普遍用于训练 Inception networks.



    Args:
        size: 每条边的期望的输出尺寸
        scale: 原始剪裁尺寸大小的范围
        ratio: 原始裁剪纵横比的范围
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly cropped and resize image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RandomSizedCrop(RandomResizedCrop):
    """
    Note: 为了支持 RandomResizedCrop, 该变换已经被弃用.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn("The use of the transforms.RandomSizedCrop transform is deprecated, " +
                      "please use transforms.RandomResizedCrop instead.")
        super(RandomSizedCrop, self).__init__(*args, **kwargs)


class FiveCrop(object):
    """将给定的 PIL Image 裁剪成四个角落和中心裁剪

    .. Note::
         该变换返回一个图像元组, 并且数据集返回的输入和目标的数量可能不匹配.
         请参阅下面的例子来处理这个问题.

    Args:
         size (sequence or int): 期望输出的裁剪尺寸. 如果 size（尺寸）是 `int`` 类型的整数, 而不是像 (h, w) 这样类型的序列, 裁剪出来的图像是 (size, size) 这样的正方形的..

    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # 一个 PIL Images 的列表
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # 返回一个4D Tensor
         >>> ])
         >>> #在你的测试循环可以如下操作:
         >>> input, target = batch # 输入是5DTensor,输出是2D
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return F.five_crop(img, self.size)


class TenCrop(object):
    """将给定的 PIL Image 裁剪成四个角, 中心裁剪, 并加上这些的翻转版本（默认使用水平翻转）

    .. Note::
         该变换返回一个图像元组, 并且数据集返回的输入和目标的数量可能不匹配.
         请参阅下面的例子来处理这个问题.

    Args:
        size (sequence or int): 期望输出的裁剪尺寸. 如果 size（尺寸）是 `int` 类型的整数, 而不是像 (h, w) 这样类型的序列, 裁剪出来的图像是 (size, size) 这样的正方形的.
        vertical_flip(bool): 使用垂直翻转而不是水平的方式

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return F.ten_crop(img, self.size, self.vertical_flip)


class LinearTransformation(object):
    """Transform a tensor image with a square transformation matrix computed
    offline.

    Given transformation_matrix, will flatten the torch.*Tensor, compute the dot
    product with the transformation matrix and reshape the tensor to its
    original shape.

    Applications:
    - whitening: zero-center the data, compute the data covariance matrix
                 [D x D] with np.dot(X.T, X), perform SVD on this matrix and
                 pass it as transformation_matrix.

    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
    """

    def __init__(self, transformation_matrix):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor.size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        flat_tensor = tensor.view(1, -1)
        transformed_tensor = torch.mm(flat_tensor, self.transformation_matrix)
        tensor = transformed_tensor.view(tensor.size())
        return tensor


class ColorJitter(object):
    """随机更改图像的亮度, 对比度和饱和度.

    Args:
        brightness (float): 亮度改变的范围. brightness_factor
            从 [max(0, 1 - brightness), 1 + brightness]的范围中一致选择.
        contrast (float): 对比度改变的范围. contrast_factor
            从 [max(0, 1 - contrast), 1 + contrast]的范围中一致选择.
        saturation (float): 饱和度改变的范围. saturation_factor
            从[max(0, 1 - saturation), 1 + saturation]的范围中一致选择.
        hue(float): 色调改变的范围. hue_factor 从
            [-hue, hue]的范围中一致选择. 应该 >=0 且 <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center)


class Grayscale(object):
    """将图像转换为灰度图像.

    Args:
        num_output_channels (int): (1 or 3) 输出图像所期望的通道数量

    Returns:
        PIL Image: 灰度版本的输入.
        - 如果 num_output_channels == 1 : 返回的图像是 1 通道
        - 如果 num_output_channels == 3 : 返回的图像是 3 通道, 并且 r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)


class RandomGrayscale(object):
    """随机将图像转换为灰度图像, 概率为 p (default 0.1).

    Args:
        p (float): 图像应该被转换成灰度的概率.

    Returns:
        PIL Image: 灰度版本的输入图像的概率为 p, 不变的概率为（1-p）
        - 如果输入图像为1个通道: 则灰度版本是 1 通道
        - 如果输入图像为3个通道: 则灰度版本是 3 通道, 并且 r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img
