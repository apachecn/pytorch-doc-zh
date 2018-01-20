"""
这个包增加了对 CUDA tensor (张量) 类型的支持,利用 GPUs 计算实现了与 CPU tensors 相同的类型.


这个是 lazily initialized (懒加载,延迟加载), 所以你可以一直导入它,并且可以用 :func:`is_available()` 来判断
你的系统是否支持 CUDA.

:ref:`cuda-semantics` 有更多关于使用 CUDA 的细节.
"""

import contextlib
import platform
import ctypes
import os
import torch
import traceback
import warnings
from torch._six import raise_from
from multiprocessing.util import register_after_fork as _register_after_fork

_initialized = False
_queued_calls = []  # 在初始化发生之前不要调用这个
_in_bad_fork = False  # 这个全局变量也用于 torch.manual_seed
_original_pid = False
_cudart = None


def is_available():
    """返回一个 bool 值表示 CUDA 目前是否可用."""
    if (not hasattr(torch._C, '_cuda_isDriverSufficient') or
            not torch._C._cuda_isDriverSufficient()):
        return False
    return torch._C._cuda_getDeviceCount() > 0


def _sleep(cycles):
    torch._C._cuda_sleep(cycles)


def _load_cudart():
    # 首先检查 CUDA 符号的主程序
    lib = ctypes.cdll.LoadLibrary(None)
    if hasattr(lib, 'cudaGetErrorName'):
        return lib

    raise RuntimeError(
        "couldn't find libcudart. Make sure CUDA libraries are installed in a"
        "default location, or that they're in {}."
        .format('DYLD_LIBRARY_PATH' if platform.system() == 'Darwin' else
                'LD_LIBRARY_PATH'))#找不到libcudart.确保CUDA库安装在默认路径或者在path路径


def _check_driver():
    if not hasattr(torch._C, '_cuda_isDriverSufficient'):
        raise AssertionError("Torch not compiled with CUDA enabled")
    if not torch._C._cuda_isDriverSufficient():
        if torch._C._cuda_getDriverVersion() == 0:
            # 在系统上找不到NVIDIA驱动程序
            raise AssertionError("""
Found no NVIDIA driver on your system. Please check that you
have an NVIDIA GPU and installed a driver from
http://www.nvidia.com/Download/index.aspx""")#在您的系统上找不到NVIDIA驱动程序.请检查是否有NVIDIA GPU,并从http://www.nvidia.com/Download/index.aspx 安装了驱动程序
        else:
            # TODO: 直接链接到需要安装的备用bin
            raise AssertionError("""
The NVIDIA driver on your system is too old (found version {}).
Please update your GPU driver by downloading and installing a new
version from the URL: http://www.nvidia.com/Download/index.aspx
Alternatively, go to: http://pytorch.org to install
a PyTorch version that has been compiled with your version
of the CUDA driver.""".format(str(torch._C._cuda_getDriverVersion())))#电脑上的NVIDIA驱动程序太旧了.从网页下载并安装更新你的驱动,或者去PyTorch官网安装一个与你编译的CUDA驱动匹配的版本


def _check_capability():
    error_str = """
    Found GPU%d %s which requires CUDA_VERSION >= %d for
     optimal performance and fast startup time, but your PyTorch was compiled
     with CUDA_VERSION %d. Please install the correct PyTorch binary
     using instructions from http://pytorch.org
    """
    #GPU需要CUDA_VERSION大于某个版本来获得最佳性能和快速的启动时间,但是你的PyTorch编译的CUDA_VERSION版本是xxx.请根据官网的操作说明安装正确的PyTorch编译文件.

    CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    for d in range(device_count()):
        major = get_device_capability(d)[0]
        name = get_device_name(d)
        if CUDA_VERSION < 8000 and major >= 6:
            warnings.warn(error_str % (d, name, 8000, CUDA_VERSION))
        elif CUDA_VERSION < 9000 and major >= 7:
            warnings.warn(error_str % (d, name, 8000, CUDA_VERSION))


def _lazy_call(callable):
    if _initialized:
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle(不要存储实际的回溯避免内存循环)
        _queued_calls.append((callable, traceback.format_stack()))

_lazy_call(_check_capability)


class DeferredCudaCallError(Exception):
    pass


def _lazy_init():
    global _initialized, _cudart, _original_pid, _queued_calls
    if _initialized:
        return
    if _in_bad_fork:
        from sys import version_info
        if version_info < (3, 4):
            msg = ("To use CUDA with multiprocessing, you must use Python "
                   "3.4+ and the 'spawn' start method")#要使用CUDA多线程,你的Python版本必须是3.4+,通过'spawn'启动方法
        else:
            msg = ("To use CUDA with multiprocessing, you must use the "
                   "'spawn' start method")#要使用CUDA多线程,通过'spawn'启动方法
        raise RuntimeError(
            "Cannot re-initialize CUDA in forked subprocess. " + msg)#无法重新初始化CUDA分叉子进程
    _check_driver()
    torch._C._cuda_init()
    torch._C._cuda_sparse_init()
    _cudart = _load_cudart()
    _cudart.cudaGetErrorName.restype = ctypes.c_char_p
    _cudart.cudaGetErrorString.restype = ctypes.c_char_p
    _original_pid = os.getpid()
    _initialized = True
    # 当某些队列调用时,_initialized之后很重要的去做这个
    # 或许他们叫 _lazy_init
    for queued_call, orig_traceback in _queued_calls:
        try:
            queued_call()
        except Exception as e:
            msg = ("CUDA call failed lazily at initialization with error: {}\n\n"    #CUDA调用延迟初始化失败
                   "CUDA call was originally invoked at:\n\n{}").format(str(e), orig_traceback)#CUDA最初调用在
            raise_from(DeferredCudaCallError(msg), e)


def _after_fork(arg):
    global _initialized, _in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _in_bad_fork = True
        _CudaBase.__new__ = _lazy_new


_register_after_fork(_after_fork, _after_fork)


def cudart():
    _lazy_init()
    return _cudart


class cudaStatus(object):
    SUCCESS = 0
    ERROR_NOT_READY = 34


class CudaError(RuntimeError):
    def __init__(self, code):
        msg = cudart().cudaGetErrorString(code).decode('utf-8')
        super(CudaError, self).__init__('{0} ({1})'.format(msg, code))


def check_error(res):
    if res != cudaStatus.SUCCESS:
        raise CudaError(res)


class device(object):
    """更改选定设备的上下文管理器.

    Arguments:
        idx (int): 选择设备编号. 如果参数无效,则是无效操作.
    """

    def __init__(self, idx):
        self.idx = idx
        self.prev_idx = -1

    def __enter__(self):
        if self.idx is -1:
            return
        _lazy_init()
        self.prev_idx = torch._C._cuda_getDevice()
        if self.prev_idx != self.idx:
            torch._C._cuda_setDevice(self.idx)

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch._C._cuda_setDevice(self.prev_idx)
        return False


class device_of(device):
    """将当前设备更改为给定对象的上下文管理器.

    可以使用张量和存储作为参数,如果给定的对象不是在 GPU 上分配的,这是一个无效操作.

    Arguments:
        obj (Tensor or Storage): 在选定设备上分配的对象.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_cuda else -1
        super(device_of, self).__init__(idx)


def set_device(device):
    """设置当前设备.

    不鼓励使用这个函数 :any:`device` . 
    在大多数情况下,最好使用 ``CUDA_VISIBLE_DEVICES`` 环境变量.

    Arguments:
        device (int): 选择设备. 参数无效时,则是无效操作.
    """
    if device >= 0:
        torch._C._cuda_setDevice(device)


def get_device_name(device):
    """获取设备名.

    Arguments:
        device (int): 返回设备名. 参数无效时,则是无效操作.
    """
    if device >= 0:
        return torch._C._cuda_getDeviceName(device)


def get_device_capability(device):
    """获取设备的 CUDA 算力.

    Arguments:
        device (int): 返回设备名, 参数无效时, 方法失效.
    Returns:
        tuple(int, int):设备的主次要 CUDA 算力.
    """
    if device >= 0:
        return torch._C._cuda_getDeviceCapability(device)


@contextlib.contextmanager
def stream(stream):
    """选择给定流的上下文管理器.

    在选定的流上, 所有的CUDA内核在其上下文内排队.

    Arguments:
        stream (Stream): 选择流. 如果是 ``None`` , 管理器无效.
    """
    if stream is None:
        yield
        return
    prev_stream = current_stream()
    torch._C._cuda_setStream(stream._cdata)
    try:
        yield
    finally:
        torch._C._cuda_setStream(prev_stream._cdata)


def device_count():
    """返回可用的 GPU 数量."""
    if is_available():
        _lazy_init()
        return torch._C._cuda_getDeviceCount()
    else:
        return 0


def current_device():
    """返回当前选择的设备的索引."""
    _lazy_init()
    return torch._C._cuda_getDevice()


def synchronize():
    """等待当前设备上所有流中的所有内核完成."""
    _lazy_init()
    return torch._C._cuda_synchronize()


def current_stream():
    """返回当前选择的 :class:`Stream` ."""
    _lazy_init()
    return torch.cuda.Stream(_cdata=torch._C._cuda_getCurrentStream())


def current_blas_handle():
    """返回指向当前 cuBLAS 句柄的 cublasHandle_t 指针"""
    return torch._C._cuda_getCurrentBlasHandle()


def empty_cache():
    """释放当前由缓存持有的所有未占用缓存内存分配器,以便可以在其他GPU应用程序中使用并在 `nvidia-smi` 中可见."""
    return torch._C._cuda_emptyCache()


def _host_allocator():
    _lazy_init()
    return torch._C._cuda_cudaHostAllocator()


@contextlib.contextmanager
def _free_mutex():
    torch._C._cuda_lock_mutex()
    try:
        yield
    finally:
        torch._C._cuda_unlock_mutex()


from .random import *

################################################################################
# 定义存储和张量类
################################################################################


from ..tensor import _TensorBase
from ..storage import _StorageBase


def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name))#试图实例化虚拟基类
    return type(storage_name, (object,), {"__init__": init_err})


if not hasattr(torch._C, 'CudaDoubleStorageBase'):
    # 定义虚拟基类
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half']:
        storage_name = 'Cuda{0}StorageBase'.format(t)
        tensor_name = 'Cuda{0}TensorBase'.format(t)

        torch._C.__dict__[storage_name] = _dummy_type(storage_name)
        torch._C.__dict__[tensor_name] = _dummy_type(tensor_name)

    torch._C.__dict__['_CudaStreamBase'] = _dummy_type('CudaStreamBase')


@staticmethod
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    #我们只需要这个方法的惰性init(lazy init),所以我们可以删除它。
    del _CudaBase.__new__
    return super(_CudaBase, cls).__new__(cls, *args, **kwargs)


class _CudaBase(object):
    is_cuda = True
    is_sparse = False

    def type(self, *args, **kwargs):
        with device(self.get_device()):
            return super(_CudaBase, self).type(*args, **kwargs)

    __new__ = _lazy_new


class DoubleStorage(_CudaBase, torch._C.CudaDoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_CudaBase, torch._C.CudaFloatStorageBase, _StorageBase):
    pass


class LongStorage(_CudaBase, torch._C.CudaLongStorageBase, _StorageBase):
    pass


class IntStorage(_CudaBase, torch._C.CudaIntStorageBase, _StorageBase):
    pass


class ShortStorage(_CudaBase, torch._C.CudaShortStorageBase, _StorageBase):
    pass


class CharStorage(_CudaBase, torch._C.CudaCharStorageBase, _StorageBase):
    pass


class ByteStorage(_CudaBase, torch._C.CudaByteStorageBase, _StorageBase):
    pass


class HalfStorage(_CudaBase, torch._C.CudaHalfStorageBase, _StorageBase):
    pass


class DoubleTensor(_CudaBase, torch._C.CudaDoubleTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return DoubleStorage


class FloatTensor(_CudaBase, torch._C.CudaFloatTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return FloatStorage


class LongTensor(_CudaBase, torch._C.CudaLongTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return LongStorage


class IntTensor(_CudaBase, torch._C.CudaIntTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return IntStorage


class ShortTensor(_CudaBase, torch._C.CudaShortTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return ShortStorage


class CharTensor(_CudaBase, torch._C.CudaCharTensorBase, _TensorBase):

    def is_signed(self):
        # TODO
        return False

    @classmethod
    def storage_type(cls):
        return CharStorage


class ByteTensor(_CudaBase, torch._C.CudaByteTensorBase, _TensorBase):

    def is_signed(self):
        return False

    @classmethod
    def storage_type(cls):
        return ByteStorage


class HalfTensor(_CudaBase, torch._C.CudaHalfTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type():
        return HalfStorage


torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)

torch._tensor_classes.add(DoubleTensor)
torch._tensor_classes.add(FloatTensor)
torch._tensor_classes.add(LongTensor)
torch._tensor_classes.add(IntTensor)
torch._tensor_classes.add(ShortTensor)
torch._tensor_classes.add(CharTensor)
torch._tensor_classes.add(ByteTensor)
torch._tensor_classes.add(HalfTensor)

from . import sparse
from . import profiler
from . import nvtx
from .streams import Stream, Event
