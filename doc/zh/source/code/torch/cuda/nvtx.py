import ctypes

lib = None

__all__ = ['range_push', 'range_pop', 'mark']


def _libnvToolsExt():
    global lib
    if lib is None:
        lib = ctypes.cdll.LoadLibrary(None)
        lib.nvtxMarkA.restype = None
    return lib


def range_push(msg):
    """
    设置一个固定范围的堆栈,返回的堆栈范围深度从0开始.

    Arguments:
        msg (string): 范围(用 ASCII 编码设置)
    """
    if _libnvToolsExt() is None:
        raise RuntimeError('Unable to load nvToolsExt library')
    return lib.nvtxRangePushA(ctypes.c_char_p(msg.encode("ascii")))


def range_pop():
    """
    弹出一个固定范围的堆栈,返回的堆栈范围深度从0结束.
    """
    if _libnvToolsExt() is None:
        raise RuntimeError('Unable to load nvToolsExt library')
    return lib.nvtxRangePop()


def mark(msg):
    """
    描述在某个时刻发生的瞬间事件.

    Arguments:
        msg (string): 事件(用 ASCII 编码表示).
    """
    if _libnvToolsExt() is None:
        raise RuntimeError('Unable to load nvToolsExt library')
    return lib.nvtxMarkA(ctypes.c_char_p(msg.encode("ascii")))
