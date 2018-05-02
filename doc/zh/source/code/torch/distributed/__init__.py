#coding=utf-8
"""
torch.distributed 提供类似 MPI 的前向运算机制, 支持在多台机的网络中交换数据. 支持不同的后段和初始化方法.
"""
import torch
import warnings

_INITIALIZED_PG = 1
_INITIALIZED_MW = 2
_initialized = 0
_scope = locals()


def _extend_scope(module):
    _scope.update({k: getattr(module, k) for k in dir(module) if not k.startswith('_')})


def is_available():
    return torch._C._has_distributed()


def init_process_group(backend, init_method='env://', **kwargs):
    """初始化方法.

    Arguments:
        backend (str): 使用后端的名字. 输入的有效值包括:  ``tcp`` ,  ``mpi`` and ``gloo`` .
        init_method (str, optional): 指定如何初始化的URL.
        world_size (int, optional): 参与工作的进程数量.
        rank (int, optional): 当前进程的排名.
        group_name (str, optional): 集群的名字. 请参阅init方法的描述.

    为了支持 ``backend == mpi`` , PyTorch 需要在支持 MPI 的系统上用进行源码编译安装
    """
    world_size = kwargs.pop('world_size', -1)
    group_name = kwargs.pop('group_name', '')
    rank = kwargs.pop('rank', -1)
    assert len(kwargs) == 0, "got unexpected keyword arguments: %s" % ",".join(kwargs.keys())

    if not is_available():
        raise RuntimeError("PyTorch built without distributed support")

    global _initialized
    if _initialized:
        raise RuntimeError("trying to initialize torch.distributed twice!")
    torch._C._dist_init_process_group(backend, init_method, world_size,
                                      group_name, rank)
    _initialized = _INITIALIZED_PG
    if not torch._C._dist_init_extension(False, reduce_op, group):
        raise RuntimeError("distributed module initialization failed")


def init_master_worker(backend, init_method='env://', **kwargs):
    warnings.warn("""
    ================================================================================
                                        WARNING
    ================================================================================
    Master-worker mode is still experimental. The API will change without
    notice and we're can't guarantee full correctness and expected performance yet.
    We'll announce it once it's ready.
    """)
    world_size = kwargs.pop('world_size', -1)
    group_name = kwargs.pop('group_name', '')
    rank = kwargs.pop('rank', -1)
    assert len(kwargs) == 0, "got unexpected keyword arguments: %s" % ",".join(kwargs.keys())

    if not is_available():
        raise RuntimeError("PyTorch built without distributed support")

    global _initialized
    if _initialized:
        raise RuntimeError("trying to initialize torch.distributed twice!")
    torch._C._dist_init_master_worker(backend, init_method, world_size,
                                      group_name, rank)
    _initialized = _INITIALIZED_MW
    import torch.distributed.collectives as collectives
    import torch.distributed.remote_types as remote_types
    _extend_scope(collectives)
    _extend_scope(remote_types)
    if not torch._C._dist_init_extension(True, reduce_op, group):
        raise RuntimeError("distributed module initialization failed")


class reduce_op(object):
    SUM = object()
    PRODUCT = object()
    MAX = object()
    MIN = object()


class group(object):
    WORLD = object()


class _DistributedRequest(object):
    def __init__(self, request):
        self.request = request

    def is_completed(self):
        return torch._C._dist_request_is_completed(self.request)

    def wait(self):
        torch._C._dist_request_wait(self.request)


def get_rank():
    """返回当前进程的排名.

    排名是独一无二的
    Rank（排名）是分配给分布式集群中每个进程的唯一标识符. 它们总是连续的整数, 范围从0到 ``world_size`` .
    """
    assert torch.distributed._initialized
    return torch._C._dist_get_rank()


def get_world_size():
    """返回在分布式集群中的进程数目."""
    assert torch.distributed._initialized
    return torch._C._dist_get_num_processes()


def isend(tensor, dst):
    """异步发送张量数据.

    Arguments:
        tensor (Tensor): 发送的张量的数据.
        dst (int): 指定发送到的 Rank.

    Returns:
        分布式请求对象.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return _DistributedRequest(torch._C._dist_isend(tensor, dst))


def irecv(tensor, src):
    """异步接收张量.

    Arguments:
        tensor (Tensor): 用收到的数据填充张量.
        src (int): 指定发送张量的 Rank.

    Returns:
        一个分布式请求对象.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return _DistributedRequest(torch._C._dist_irecv(tensor, src))


def send(tensor, dst):
    """同步发送张量.

    Arguments:
        tensor (Tensor): 发送的张量.
        dst (int): 指定发送的目的地的 Rank.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_send(tensor, dst)


def recv(tensor, src=None):
    """同步接收张量.

    Arguments:
        tensor (Tensor): 用收到的数据填充张量.
        src (int, optional): 发送端的Rank, 如果没有指定, 将会接收任何发送的数据.

    Returns:
        发送端的Rank.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    if src is None:
        return torch._C._dist_recv_any_source(tensor)
    return torch._C._dist_recv(tensor, src)


def broadcast(tensor, src, group=group.WORLD):
    """向某个小组内的张量广播的方法.

     ``tensor`` 在该小组处理数据的所有过程中元素的数目必须相同.

    Arguments:
        tensor (Tensor): 如果发送端 ``src`` 是当前进程的 Rank, 则发送数据, 否则使用张量保存接收的数据.
        src (int): 发送端的 Rank.
        group (optional): 集群内的小组的名字.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_broadcast(tensor, src, group)


def all_reduce(tensor, op=reduce_op.SUM, group=group.WORLD):
    """处理所有机器上的处理的张量数据, 计算最终的结果.

    在所有进程中调用  ``tensor`` 将按位相同.
    
    Arguments:
        tensor (Tensor): 集群的输入和输出.
        op (optional): "torch.distributed.reduce_op" 枚举值之一. 指定用于元素减少的操作.
        group (optional): 集群的内的小组的名字.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_all_reduce(tensor, op, group)


def reduce(tensor, dst, op=reduce_op.SUM, group=group.WORLD):
    """减少所有机器上的张量数据.

    只有级别为 ``dst`` 的进程才会收到最终结果.

    Arguments:
        tensor (Tensor): 集群的输入和输出数据. 分别在每台机器上本地处理.
        op (optional): "torch.distributed.reduce_op" 枚举值之一. 指定用于元素减少的操作.
        group (optional): 集群的内的小组的名字.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_reduce(tensor, dst, op, group)


def all_gather(tensor_list, tensor, group=group.WORLD):
    """在整个集群中收集list表格中的张量.

    Arguments:
        tensor_list (list[Tensor]): 输出列表. 它应该包含正确大小的张量以用于集体的输出.
        tensor (Tensor): 张量从当前进程中进行广播.
        group (optional): 集群的内的小组的名字.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_all_gather(tensor_list, tensor, group)


def gather(tensor, **kwargs):
    """收集一个张量列表从一个单一进程中.

    Arguments:
        tensor (Tensor): 输入的数据.
        dst (int): 目的地的 Rank. 包括除了正在接收数据的进程的所有进程.
        gather_list (list[Tensor]): 用于接收数据的适当大小的张量列表. 只在接收过程中需要.
        group (optional): 集群的内的小组的名字.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    my_rank = get_rank()
    dst = kwargs.pop('dst', my_rank)
    gather_list = kwargs.pop('gather_list', None)
    _group = kwargs.pop('group', group.WORLD)
    if kwargs:
        raise RuntimeError("got unexpected kwargs")
    if dst == my_rank:
        if gather_list is None:
            raise RuntimeError("gather_list is a required argument in gather destination")
        return torch._C._dist_gather_recv(gather_list, tensor, _group)
    else:
        if gather_list:
            raise RuntimeError("non-empty gather_list can be given only to gather destination")
        return torch._C._dist_gather_send(tensor, dst, _group)


def scatter(tensor, **kwargs):
    """将张量列表散布到小组中的所有进程.

    每个进程只会收到一个张量, 并将其数据存储在 ``tensor`` 的参数中.

    Arguments:
        tensor (Tensor): 输出的张量.
        src (int): 发送端的 Rank. 包括除了正在接收数据的进程的所有进程.
        scatter_list (list[Tensor]): 张量分散的列表. 仅在发送数据的过程中需要.
        group (optional): 集群的内的小组的名字.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    my_rank = get_rank()
    src = kwargs.pop('src', my_rank)
    scatter_list = kwargs.pop('scatter_list', None)
    _group = kwargs.pop('group', group.WORLD)
    if kwargs:
        raise RuntimeError("got unexpected kwargs")
    if src == my_rank:
        if scatter_list is None:
            raise RuntimeError("scatter_list is a required argument in scatter source")
        return torch._C._dist_scatter_send(scatter_list, tensor, _group)
    else:
        if scatter_list:
            raise RuntimeError("non-empty can be given only to scatter source")
        return torch._C._dist_scatter_recv(tensor, src, _group)


def barrier(group=group.WORLD):
    """同步所有进程.

    这个集群阻塞进程, 直到全部的小组的计算结果都输入进这个函数中.

    Arguments:
        group (optional): 集群的内的小组的名字.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_barrier(group)


def new_group(ranks=None):
    """创建一个新的分布式小组

    此函数要求主组中的所有进程（即作为分布式作业一部分的所有进程）都会输入此函数, 即使它们不是该小组的成员.
    此外, 应该在所有的进程中以相同的顺序创建新的小组.


    Arguments:
        ranks (list[int]): 小组内成员的 Rank 的列表.

    Returns:
        分配组的句柄, 以便在集群中调用.
    """
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    if ranks is None:
        ranks = list(range(get_world_size()))
    return torch._C._dist_new_group(ranks)


def _register_stream(stream):
    if not _initialized:
        raise RuntimeError("torch.distributed needs to be initialized first")
    return torch._C._dist_register_stream(stream)
