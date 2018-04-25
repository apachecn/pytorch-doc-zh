from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torchvision import utils

__version__ = '0.2.0'

_image_backend = 'PIL'


def set_image_backend(backend):
    """
    指定用于加载图像的包. 

    Args:
        backend (string): 图像处理后端的名称. {'PIL', 'accimage'} 之一.
         :mod:`accimage` 使用 Intel IPP library（高性能图像加载和增强程序模拟的程序）.通常比PIL库要快, 但是不支持许多操作.
    """
    global _image_backend
    if backend not in ['PIL', 'accimage']:
        raise ValueError("Invalid backend '{}'. Options are 'PIL' and 'accimage'"
                         .format(backend))
    _image_backend = backend


def get_image_backend():
    """
    获取用于加载图像的包的名称
    """
    return _image_backend
