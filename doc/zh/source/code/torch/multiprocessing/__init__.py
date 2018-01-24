"""
torch.multiprocessing is a wrapper around the native :mod:`multiprocessing`
module. It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor/storage is moved
to shared_memory (see :func:`~torch.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import torch.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to very good docs of the original module.
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
    """Sets the strategy for sharing CPU tensors.

    Arguments:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    assert new_strategy in _all_sharing_strategies
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    """Returns the current strategy for sharing CPU tensors."""
    return _sharing_strategy


def get_all_sharing_strategies():
    """Returns a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies


init_reductions()
