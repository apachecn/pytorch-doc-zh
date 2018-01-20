"""
The torch package contains data structures for multi-dimensional
tensors and mathematical operations over these are defined.
Additionally, it provides many utilities for efficient serializing of
Tensors and arbitrary types, and other useful utilities.

It has a CUDA counterpart, that enables you to run your tensor computations
on an NVIDIA GPU with compute capability >= 3.0.
"""

import sys
from ._utils import _import_dotted_name
from .version import __version__

__all__ = [
    'typename', 'is_tensor', 'is_storage', 'set_default_tensor_type',
    'set_rng_state', 'get_rng_state', 'manual_seed', 'initial_seed',
    'save', 'load', 'set_printoptions', 'chunk', 'split', 'stack', 'matmul',
    'DoubleStorage', 'FloatStorage', 'LongStorage', 'IntStorage',
    'ShortStorage', 'CharStorage', 'ByteStorage',
    'DoubleTensor', 'FloatTensor', 'LongTensor', 'IntTensor',
    'ShortTensor', 'CharTensor', 'ByteTensor',
]

################################################################################
# Load the extension module
################################################################################

# Loading the extension with RTLD_GLOBAL option allows to not link extension
# modules against the _C shared object. Their missing THP symbols will be
# automatically filled by the dynamic loader.
import os as _dl_flags

# if we have numpy, it *must* be imported before the call to setdlopenflags()
# or there is risk that later c modules will segfault when importing numpy
try:
    import numpy as np
except ImportError:
    pass

# first check if the os package has the required flags
if not hasattr(_dl_flags, 'RTLD_GLOBAL') or not hasattr(_dl_flags, 'RTLD_LAZY'):
    try:
        # next try if DLFCN exists
        import DLFCN as _dl_flags
    except ImportError:
        # as a last attempt, use compile-time constants
        import torch._dl as _dl_flags

old_flags = sys.getdlopenflags()
sys.setdlopenflags(_dl_flags.RTLD_GLOBAL | _dl_flags.RTLD_LAZY)
try:
    import torch._nvrtc
except ImportError:
    pass
from torch._C import *

__all__ += [name for name in dir(_C)
            if name[0] != '_' and
            not name.endswith('Base')]

sys.setdlopenflags(old_flags)
del _dl_flags
del old_flags

################################################################################
# Define basic utilities
################################################################################


def typename(o):
    module = ''
    class_name = ''
    if hasattr(o, '__module__') and o.__module__ != 'builtins' \
            and o.__module__ != '__builtin__' and o.__module__ is not None:
        module = o.__module__ + '.'

    if hasattr(o, '__qualname__'):
        class_name = o.__qualname__
    elif hasattr(o, '__name__'):
        class_name = o.__name__
    else:
        class_name = o.__class__.__name__

    return module + class_name


def is_tensor(obj):
    r"""Returns True if `obj` is a pytorch tensor.

    Args:
        obj (Object): Object to test
    """
    return type(obj) in _tensor_classes


def is_storage(obj):
    r"""Returns True if `obj` is a pytorch storage object.

    Args:
        obj (Object): Object to test
    """
    return type(obj) in _storage_classes


def set_default_tensor_type(t):
    global Tensor
    global Storage
    Tensor = _import_dotted_name(t)
    Storage = _import_dotted_name(t.replace('Tensor', 'Storage'))
    _C._set_default_tensor_type(Tensor)


from .random import set_rng_state, get_rng_state, manual_seed, initial_seed
from .serialization import save, load
from ._tensor_str import set_printoptions

################################################################################
# Define Storage and Tensor classes
################################################################################

from .storage import _StorageBase
from .tensor import _TensorBase


class DoubleStorage(_C.DoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_C.FloatStorageBase, _StorageBase):
    pass


class HalfStorage(_C.HalfStorageBase, _StorageBase):
    pass


class LongStorage(_C.LongStorageBase, _StorageBase):
    pass


class IntStorage(_C.IntStorageBase, _StorageBase):
    pass


class ShortStorage(_C.ShortStorageBase, _StorageBase):
    pass


class CharStorage(_C.CharStorageBase, _StorageBase):
    pass


class ByteStorage(_C.ByteStorageBase, _StorageBase):
    pass


class DoubleTensor(_C.DoubleTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return DoubleStorage


class FloatTensor(_C.FloatTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return FloatStorage


class HalfTensor(_C.HalfTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return HalfStorage


class LongTensor(_C.LongTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return LongStorage


class IntTensor(_C.IntTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return IntStorage


class ShortTensor(_C.ShortTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return ShortStorage


class CharTensor(_C.CharTensorBase, _TensorBase):

    def is_signed(self):
        # TODO
        return False

    @classmethod
    def storage_type(cls):
        return CharStorage


class ByteTensor(_C.ByteTensorBase, _TensorBase):

    def is_signed(self):
        return False

    @classmethod
    def storage_type(cls):
        return ByteStorage


_storage_classes = {
    DoubleStorage, FloatStorage, LongStorage, IntStorage, ShortStorage,
    CharStorage, ByteStorage, HalfStorage
}

_tensor_classes = {
    DoubleTensor, FloatTensor, LongTensor, IntTensor, ShortTensor,
    CharTensor, ByteTensor, HalfTensor
}


set_default_tensor_type('torch.FloatTensor')

################################################################################
# Import interface functions defined in Python
################################################################################

from .functional import *


################################################################################
# Initialize extension
################################################################################

def manager_path():
    import os
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'lib', 'torch_shm_manager')
    if not os.path.exists(path):
        raise RuntimeError("Unable to find torch_shm_manager at " + path)
    return path.encode('utf-8')


# Shared memory manager needs to know the exact location of manager executable
_C._initExtension(manager_path())
del manager_path

################################################################################
# Remove unnecessary members
################################################################################

del DoubleStorageBase
del FloatStorageBase
del LongStorageBase
del IntStorageBase
del ShortStorageBase
del CharStorageBase
del ByteStorageBase
del DoubleTensorBase
del FloatTensorBase
del LongTensorBase
del IntTensorBase
del ShortTensorBase
del CharTensorBase
del ByteTensorBase

del SparseDoubleTensorBase
del SparseFloatTensorBase
del SparseLongTensorBase
del SparseIntTensorBase
del SparseShortTensorBase
del SparseCharTensorBase
del SparseByteTensorBase

################################################################################
# Import most common subpackages
################################################################################

import torch.cuda
import torch.autograd
import torch.nn
import torch.optim
import torch.multiprocessing
import torch.sparse
import torch.utils.backcompat
import torch.onnx
import torch.random
import torch.distributions

_C._init_names(list(torch._tensor_classes) + list(torch._storage_classes))

# attach docstrings to torch and tensor functions
from . import _torch_docs, _tensor_docs, _storage_docs
del _torch_docs, _tensor_docs, _storage_docs
