import ctypes
import sys
import torch
import warnings
from torch.version import cuda
from contextlib import contextmanager

enabled = True  # set to False to globally disable cuDNN

lib = None
__cudnn_version = None
# TODO: dynamic version checks via cudnnGetVersion


def _libcudnn():
    global lib, __cudnn_version
    if lib is None:
        lib = ctypes.cdll.LoadLibrary(None)
        if hasattr(lib, 'cudnnGetErrorString'):
            lib.cudnnGetErrorString.restype = ctypes.c_char_p
            __cudnn_version = lib.cudnnGetVersion()
            compile_version = torch._C._cudnn_version()
            # Check that cuDNN major and minor versions match
            if (__cudnn_version // 100) != (compile_version // 100):
                raise RuntimeError(
                    'cuDNN version mismatch: PyTorch was compiled against {} '
                    'but linked against {}'.format(compile_version, __cudnn_version))
        else:
            lib = None
    return lib


def version():
    if _libcudnn() is None:
        return None
    return __cudnn_version


def is_acceptable(tensor):
    if not enabled:
        return False
    if not (isinstance(tensor, torch.cuda.HalfTensor) or
            isinstance(tensor, torch.cuda.FloatTensor) or
            isinstance(tensor, torch.cuda.DoubleTensor)):
        return False
    if not torch._C.has_cudnn:
        warnings.warn(
            "PyTorch was compiled without cuDNN support. To use cuDNN, rebuild "
            "PyTorch making sure the library is visible to the build system.")
        return False
    if _libcudnn() is None:
        warnings.warn('cuDNN library not found. Check your {libpath}'.format(
            libpath={
                'darwin': 'DYLD_LIBRARY_PATH',
                'win32': 'PATH'
            }.get(sys.platform, 'LD_LIBRARY_PATH')))
        return False
    return True


_handles = {}

deterministic = False
benchmark = False
verbose = False

CUDNN_DATA_FLOAT = 0
CUDNN_DATA_DOUBLE = 1
CUDNN_DATA_HALF = 2

CUDNN_TENSOR_NCHW = 0
CUDNN_TENSOR_NHWC = 1

CUDNN_RNN_RELU = 0
CUDNN_RNN_TANH = 1
CUDNN_LSTM = 2
CUDNN_GRU = 3

CUDNN_LINEAR_INPUT = 0
CUDNN_SKIP_INPUT = 1

CUDNN_RNN_ALGO_STANDARD = 0
CUDNN_RNN_ALGO_PERSIST_STATIC = 1
CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2

CUDNN_DEFAULT_MATH = 0
CUDNN_TENSOR_OP_MATH = 1


def set_flags(_enabled, _benchmark, _deterministic, _verbose):
    global enabled, benchmark, deterministic, verbose
    orig_flags = enabled, benchmark, deterministic, verbose
    enabled, benchmark, deterministic, verbose = _enabled, _benchmark, _deterministic, _verbose
    return orig_flags


@contextmanager
def flags(enabled=False, benchmark=False, deterministic=False, verbose=False):
    orig_flags = set_flags(enabled, benchmark, deterministic, verbose)
    try:
        yield
    finally:
        # recover the previous values
        set_flags(orig_flags[0], orig_flags[1], orig_flags[2], orig_flags[3])


class CuDNNHandle:
    def __init__(self):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreate(ctypes.byref(ptr)))
        self._as_parameter_ = ptr

    def __del__(self):
        check_error(lib.cudnnDestroy(self))


class CuDNNError(RuntimeError):
    def __init__(self, status):
        self.status = status
        msg = '{}: {}'.format(status, get_error_string(status))
        super(CuDNNError, self).__init__(msg)


class TensorDescriptor(object):
    def __init__(self):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateTensorDescriptor(ctypes.byref(ptr)))
        self._as_parameter_ = ptr

    def __del__(self):
        check_error(lib.cudnnDestroyTensorDescriptor(self._as_parameter_))
        del self._as_parameter_

    def set(self, tensor):
        self._type = tensor.type()
        self._size = tensor.size()
        self._stride = tensor.stride()
        check_error(lib.cudnnSetTensorNdDescriptor(
            self, _typemap[tensor.type()], tensor.dim(),
            int_array(tensor.size()), int_array(tensor.stride())))

    def as_tuple(self):
        return (self._type, tuple(self._size), tuple(self._stride))


class TensorDescriptorArray(object):
    def __init__(self, N):
        self.ptrs = (ctypes.c_void_p * N)()
        for i in range(N):
            ptr = ctypes.byref(self.ptrs, i * ctypes.sizeof(ctypes.c_void_p))
            check_error(lib.cudnnCreateTensorDescriptor(ptr))
        self._as_parameter_ = self.ptrs

    def __del__(self):
        for ptr in self.ptrs:
            check_error(lib.cudnnDestroyTensorDescriptor(ctypes.c_void_p(ptr)))

    def __getitem__(self, key):
        return ctypes.c_void_p(self.ptrs[key])

    def set_all(self, tensor):
        _type = _typemap[tensor.type()]
        _ndim = tensor.dim()
        _size = int_array(tensor.size())
        _stride = int_array(tensor.stride())
        for ptr in self.ptrs:
            check_error(lib.cudnnSetTensorNdDescriptor(
                ctypes.c_void_p(ptr), _type, _ndim, _size, _stride))

    def set_raw(self, i, _type, _ndim, _size, _stride):
        ptr = self.ptrs[i]
        check_error(lib.cudnnSetTensorNdDescriptor(
            ctypes.c_void_p(ptr), _type, _ndim, _size, _stride))


class FilterDescriptor(object):
    def __init__(self):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateFilterDescriptor(ctypes.byref(ptr)))
        self._as_parameter_ = ptr

    def __del__(self):
        check_error(lib.cudnnDestroyFilterDescriptor(self._as_parameter_))
        del self._as_parameter_

    def set(self, weight):
        self._size = weight.size()
        datatype = _typemap[weight.type()]
        check_error(lib.cudnnSetFilterNdDescriptor(
            self, datatype, CUDNN_TENSOR_NCHW, weight.ndimension(),
            int_array(weight.size())))

    def as_tuple(self):
        return tuple(self._size)


class DropoutDescriptor(object):
    def __init__(self, handle, dropout, seed):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateDropoutDescriptor(ctypes.byref(ptr)))

        self._as_parameter_ = ptr
        self.state = None
        self.dropout = dropout
        self.handle = handle

        self._set(dropout, seed)

    def set_dropout(self, dropout, seed):
        if dropout != self.dropout:
            self._set(dropout, seed)

    def _set(self, dropout, seed):
        if self.state is None and dropout > 0:
            dropout_states_size = ctypes.c_long()
            check_error(lib.cudnnDropoutGetStatesSize(
                self.handle,
                ctypes.byref(dropout_states_size)))
            self.state = torch.cuda.ByteTensor(dropout_states_size.value)
            state_ptr = self.state.data_ptr()
            state_size = self.state.size(0)
        else:
            state_ptr = None
            state_size = 0

        check_error(lib.cudnnSetDropoutDescriptor(
            self,
            self.handle,
            ctypes.c_float(dropout),
            ctypes.c_void_p(state_ptr),
            ctypes.c_size_t(state_size),
            ctypes.c_ulonglong(seed),
        ))

        self.dropout = dropout

    def __del__(self):
        check_error(lib.cudnnDestroyDropoutDescriptor(self))


class RNNDescriptor(object):
    def __init__(self, handle, hidden_size, num_layers, dropout_desc, input_mode,
                 bidirectional, mode, datatype):
        ptr = ctypes.c_void_p()
        check_error(lib.cudnnCreateRNNDescriptor(ctypes.byref(ptr)))
        self._as_parameter_ = ptr
        if version() >= 6000:
            check_error(lib.cudnnSetRNNDescriptor_v6(
                handle,
                self,
                hidden_size,
                num_layers,
                dropout_desc,
                input_mode,
                bidirectional,
                mode,
                CUDNN_RNN_ALGO_STANDARD,
                datatype
            ))
            if version() >= 7000 and int(cuda[0]) >= 9 and (
                    torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 7):
                lib.cudnnSetRNNMatrixMathType(self, CUDNN_DEFAULT_MATH)
                if datatype == CUDNN_DATA_HALF:
                    lib.cudnnSetRNNMatrixMathType(self, CUDNN_TENSOR_OP_MATH)
        else:
            check_error(lib.cudnnSetRNNDescriptor(
                self,
                hidden_size,
                num_layers,
                dropout_desc,
                input_mode,
                bidirectional,
                mode,
                datatype
            ))

    def __del__(self):
        check_error(lib.cudnnDestroyRNNDescriptor(self))


def check_error(status):
    if status is not 0:
        raise CuDNNError(status)


def get_error_string(status):
    return lib.cudnnGetErrorString(status)


def get_handle():
    if _libcudnn() is None:
        raise RuntimeError('cuDNN not available')
    current_device = torch.cuda.current_device()
    handle = _handles.get(current_device, None)
    if handle is None:
        handle = CuDNNHandle()
        _handles[current_device] = handle
    return handle


_typemap = {
    'torch.cuda.HalfTensor': CUDNN_DATA_HALF,
    'torch.cuda.FloatTensor': CUDNN_DATA_FLOAT,
    'torch.cuda.DoubleTensor': CUDNN_DATA_DOUBLE,
}

_sizeofmap = {
    CUDNN_DATA_HALF: 2,
    CUDNN_DATA_FLOAT: 4,
    CUDNN_DATA_DOUBLE: 8,
}


def c_type(tensor):
    if isinstance(tensor, torch.cuda.HalfTensor):
        return ctypes.c_float
    elif isinstance(tensor, torch.cuda.FloatTensor):
        return ctypes.c_float
    elif isinstance(tensor, torch.cuda.DoubleTensor):
        return ctypes.c_double
    else:
        raise ValueError("unknown type '{}'".format(type(tensor)))


def int_array(itr):
    array_type = ctypes.c_int * len(itr)
    return array_type(*itr)


def descriptor(tensor, N=None):
    padded_size = tensor.size() + ((1,) * (5 - tensor.dim()))
    tensor = tensor.view(padded_size)
    if N is not None:
        descriptor = TensorDescriptorArray(N)
        descriptor.set_all(tensor)
    else:
        descriptor = TensorDescriptor()
        descriptor.set(tensor)
    return descriptor


def descriptor_sequence(tensor, batch_sizes):
    descriptors = TensorDescriptorArray(len(batch_sizes))
    _type = _typemap[tensor.type()]
    _ndim = 5
    dim_pad = (1,) * (5 - tensor.dim())
    _size = int_array(tensor.size() + dim_pad)
    _stride = int_array(tensor.stride() + dim_pad)
    for i, batch_size in enumerate(batch_sizes):
        _size[0] = batch_size
        descriptors.set_raw(i, _type, _ndim, _size, _stride)
    return descriptors


def add_tensor(*args):
    check_error(lib.cudnnAddTensor(*args))
