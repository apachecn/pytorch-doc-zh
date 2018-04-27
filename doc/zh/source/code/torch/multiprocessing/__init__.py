"""
torch.multiprocessing 是本地 :mod:`multiprocessing` 多进程处理模块的一个 wrapper（包装器）. 
它通过注册自定义的 reducers（缩减器）, 使用共享内存来提供不同进程中相同数据的共享视图.
一旦 tensor/storage（张量/存储）移动到共享内存 (请参阅 :func:`~torch.Tensor.share_memory_`),
就可以将其发送到其他进程而不做任何复制.

该 API 与原始模块 100% 兼容 - 只需将 ``import multiprocessing`` 更改为 ``import torch.multiprocessing`` 就
可以将所有张量通过队列发送, 或通过其它机制共享, 移动到共享内存.

由于 API 的相似性, 我们没有记录大部分这个包的内容, 我们参考引用原始模块中非常优秀的文档.
"""
import sys
from .reductions import init_reductions
import multiprocessing

__all__ = ['set_sharing_strategy', 'get_sharing_strategy',
           'get_all_sharing_strategies']


from multiprocessing import *


__all__ += multiprocessing.__all__


if sys.version_info < (3, 3):
    """Override basic classes in Python 2.7 and Python 3.3 to use ForkingPickler
    for serialization. Later versions of Python already use ForkingPickler."""
    from .queue import Queue, SimpleQueue
    from .pool import Pool


if sys.platform == 'darwin':
    _sharing_strategy = 'file_system'
    _all_sharing_strategies = {'file_system'}
else:
    _sharing_strategy = 'file_descriptor'
    _all_sharing_strategies = {'file_descriptor', 'file_system'}


def set_sharing_strategy(new_strategy):
    """为共享的 CPU 张量来设置策略.

    Arguments:
        new_strategy (str): 所选策略的名称. 必须是函数 :func:`get_all_sharing_strategies()` 所返回的值之一.
    """
    global _sharing_strategy
    assert new_strategy in _all_sharing_strategies
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    """返回用于共享 CPU 张量的当前策略."""
    return _sharing_strategy


def get_all_sharing_strategies():
    """返回当前系统支持的一组共享策略."""
    return _all_sharing_strategies


init_reductions()
