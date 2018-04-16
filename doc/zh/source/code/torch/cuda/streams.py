import ctypes
import torch
from . import cudart, check_error, cudaStatus


class Stream(torch._C._CudaStreamBase):
    """ CUDA 流的包装.

    Arguments:
        device(int, optional): 分配流的设备.
        priority(int, optional): 流的优先级. 较低的数字代表较高的优先级.
    """

    def __new__(cls, device=-1, priority=0, **kwargs):
        with torch.cuda.device(device):
            return super(Stream, cls).__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event):
        """将所有未来的工作提交到流等待事件.

        Arguments:
            event (Event): 等待的事件.
        """
        check_error(cudart().cudaStreamWaitEvent(self, event, ctypes.c_int(0)))

    def wait_stream(self, stream):
        """与另一个流同步.

        提交到此流的所有未来工作将等待直到所有核心在调用完成时提交给给定的流.

        Arguments:
            stream (Stream): 同步流.
        """
        self.wait_event(stream.record_event())

    def record_event(self, event=None):
        """记录一个事件.

        Arguments:
            event (Event, optional): 要记录的事件.如果没有给出,将分配一个新的.

        Returns:
            记录的事件.
        """
        if event is None:
            event = Event()
        check_error(cudart().cudaEventRecord(event, self))
        return event

    def query(self):
        """检查事件是否已被记录.

        Returns:
            一个 BOOL 值, 指示事件是否已被记录.
        """
        res = cudart().cudaStreamQuery(self)
        if res == cudaStatus.ERROR_NOT_READY:
            return False
        check_error(res)
        return True

    def synchronize(self):
        """等待流中的所有内核完成."""
        check_error(cudart().cudaStreamSynchronize(self))

    @staticmethod
    def priority_range():
        least_priority = ctypes.c_int()
        greatest_priority = ctypes.c_int()
        check_error(cudart().cudaDeviceGetStreamPriorityRange(
            ctypes.byref(least_priority), ctypes.byref(greatest_priority)))
        return (least_priority.value, greatest_priority.value)

    @property
    def priority(self):
        priority = ctypes.c_int()
        check_error(cudart().cudaStreamGetPriority(self, ctypes.byref(priority)))
        return priority.value

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.cuda_stream)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return o.device == self.device and o.cuda_stream == self.cuda_stream
        return False

    def __hash__(self):
        return hash((self.cuda_stream, self.device))

    def __repr__(self):
        return ('<torch.cuda.Stream device={0} cuda_stream={1:#x}>'
                .format(self.device, self.cuda_stream))


class EventHandle(ctypes.Structure):
    IPC_HANDLE_SIZE = 64
    _fields_ = [('reserved', ctypes.c_char * IPC_HANDLE_SIZE)]


class Event(object):
    """ CUDA 事件包装器.

    Arguments:
        enable_timing (bool): 指示事件是否应测量时间
            (默认: ``False``)
        blocking (bool): 如果 ``True``, :meth:`wait` 将阻塞 (默认: ``False`` )
        interprocess (bool): 如果 ``True``, 事件可以在进程之间共享
            (默认: ``False``)
    """

    DEFAULT = 0x0
    BLOCKING_SYNC = 0x1
    DISABLE_TIMING = 0x2
    INTERPROCESS = 0x4

    def __init__(self, enable_timing=False, blocking=False, interprocess=False,
                 _handle=None):
        flags = Event.DEFAULT
        if not enable_timing:
            flags |= Event.DISABLE_TIMING
        if blocking:
            flags |= Event.BLOCKING_SYNC
        if interprocess:
            flags |= Event.INTERPROCESS

        ptr = ctypes.c_void_p()
        self._cudart = cudart()
        if _handle:
            check_error(self._cudart.cudaIpcOpenEventHandle(ctypes.byref(ptr), _handle))
        else:
            check_error(self._cudart.cudaEventCreateWithFlags(ctypes.byref(ptr), ctypes.c_uint(flags)))
        self._as_parameter_ = ptr

    def __del__(self):
        if hasattr(self, '_as_parameter_'):
            check_error(self._cudart.cudaEventDestroy(self._as_parameter_))
            del self._as_parameter_

    def record(self, stream=None):
        """记录给定流中的事件."""
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.record_event(self)

    def wait(self, stream=None):
        """使给定流等待事件发生."""
        if stream is None:
            stream = torch.cuda.current_stream()
        stream.wait_event(self)

    def query(self):
        """检查事件是否已记录.

        Returns:
            一个 BOOL 值, 指示事件是否已被记录.
        """
        res = cudart().cudaEventQuery(self)
        if res == cudaStatus.ERROR_NOT_READY:
            return False
        check_error(res)
        return True

    def elapsed_time(self, end_event):
        """返回记录事件之前所经过的时间."""
        time_ms = ctypes.c_float()
        check_error(cudart().cudaEventElapsedTime(
            ctypes.byref(time_ms), self, end_event))
        return time_ms.value

    def synchronize(self):
        """与事件同步."""
        check_error(cudart().cudaEventSynchronize(self))

    def ipc_handle(self):
        """返回此事件的 IPC 句柄."""
        handle = EventHandle()
        check_error(cudart().cudaIpcGetEventHandle(ctypes.byref(handle), self))
        return handle

    def __repr__(self):
        return '<torch.cuda.Event {0:#x}>'.format(self._as_parameter_.value)
