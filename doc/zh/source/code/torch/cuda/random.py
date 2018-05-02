from torch import _C
from . import _lazy_init, _lazy_call, device_count, device as device_ctx_manager


def get_rng_state(device=-1):
    r"""将当前 GPU 的随机数生成器状态作为 ByteTensor 返回.

    Args:
        device (int, optional): 设备的 RNG 状态.
            Default: -1 (i.e., 使用当前设备).

    .. warning::
        函数需要提前初始化 CUDA .
    """
    _lazy_init()
    with device_ctx_manager(device):
        return _C._cuda_getRNGState()


def get_rng_state_all():
    r"""返回 ByteTensor 的元组,表示所有设备的随机数状态."""

    results = []
    for i in range(device_count()):
        with device_ctx_manager(i):
            results.append(get_rng_state())
    return results


def set_rng_state(new_state, device=-1):
    r"""设置当前 GPU 的随机数发生器状态.

    Args:
        new_state (torch.ByteTensor): 所需的状态
    """
    new_state_copy = new_state.clone()

    # NB: 如果 device=-1?  您可能担心 "当前"
    # 设备将在我们真正调用调用延迟回调的时候发生变化
    # 但事实上, 这是不可能的:
    # 改变当前设备涉及CUDA调用, 这又会初始化状态
    # 收益 _lazy_call 将会立即执行cb
    def cb():
        with device_ctx_manager(device):
            _C._cuda_setRNGState(new_state_copy)

    _lazy_call(cb)


def set_rng_state_all(new_states):
    r"""设置所有设备的随机数生成器状态.

    Args:
        new_state (tuple of torch.ByteTensor): 每个设备的所需状态"""
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed):
    r"""设置用于当前 GPU 生成随机数的种子.
    如果 CUDA 不可用,调用这个函数是安全的;在这种情况下,它将被忽略.

    Args:
        seed (int or long): 所需的种子.

    .. warning::
        如果您正在使用多 GPU 模型,则此功能不足以获得确定性.  
        seef作用于所有 GPUs , 使用 :func:`manual_seed_all` .
    """
    _lazy_call(lambda: _C._cuda_manualSeed(seed))


def manual_seed_all(seed):
    r"""设置在所有 GPU 上生成随机数的种子.
    如果 CUDA 不可用, 调用此函数是安全的; 这种情况下,会被忽略.

    Args:
        seed (int or long): 所需的种子.
    """
    _lazy_call(lambda: _C._cuda_manualSeedAll(seed))


def seed():
    r"""将用于生成随机数的种子设置为当前 GPU 的随机数.
    如果 CUDA 不可用,则调用此函数是安全的. 在那种情况下,会被忽略.

    .. warning::
        如果您正在使用多 GPU 模型, 则此功能不足以获得确定性.  
        seef作用于所有 GPUs , 使用 :func:`seed_all`.
    """
    _lazy_call(lambda: _C._cuda_seed())


def seed_all():
    r"""在所有 GPU 上将用于生成随机数的种子设置为随机数.
    如果 CUDA 不可用,则调用此函数是安全的. 在那种情况下,会被忽略.
    """
    _lazy_call(lambda: _C._cuda_seedAll())


def initial_seed():
    r"""返回当前 GPU 的当前随机种子.

    .. warning::
        函数提前初始化 CUDA .
    """
    _lazy_init()
    return _C._cuda_initialSeed()
