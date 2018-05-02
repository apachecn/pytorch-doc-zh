import sys
import math
import threading
import copy

import torch
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
import torch.distributed as dist

from ..modules import Module
from .replicate import replicate
from .scatter_gather import scatter_kwargs, gather
from .parallel_apply import parallel_apply

if sys.version_info[0] == 3:
    import queue
else:
    import Queue as queue


class DistributedDataParallel(Module):
    r"""在模块级别实现分布式数据并行.

    此容器通过在批次维度中分块, 将输入分割到指定设备上, 从而并行化给定模块的应用程序.
    该模块被复制到每台机器和每个设备上, 每个这样的副本处理一部分输入.在向后传递期间, 
    来自每个节点的梯度被平均.

    batch size 应该大于 GPUs 的数量.同时也应该是 GPU 数量的整数倍, 以便每个块大小
    相同（以便每个 GPU 处理相同数量的样本）.

    引用 ::ref:`distributed-basics`  和  :ref:`cuda-nn-dataparallel-instead`.
    对输入的约束和 :class:`torch.nn.DataParallel` 中一样.

    创建这个类需要分布式包已经在 process group 模式下被初始化 (引用 :func:`torch.distributed.init_process_group`).

    .. warning::
        这个模块只能和 ``gloo`` 后端一起工作.

    .. warning::
        构造器, 转发方法和输出（或者这个模块的输出功能）的区分是分布式同步点.考虑到不同的
        进程可能会执行不同的代码.

    .. warning::
        该模块假设所有参数在创建时都在模型中注册.之后不应该添加或删除参数.同样适用于缓冲区.

    .. warning::
        这个模块假定所有的缓冲区和梯度都是密集的.

    .. warning::
        这个模块不能用于 :func:`torch.autograd.grad`（即只有在参数的 ``.grad`` 属性中
        累积梯度才能使用）.

    .. note::
        参数永远不会在进程之间广播.模块在梯度上执行全部优化步骤, 并假定它们将以相同的方式在
        所有进程中进行优化.缓冲区（e.g. BatchNorm stats）在等级0的过程中从模块广播到系统
        中的每个迭代中的所有其他副本.

    Args:
        module: 需要并行的模型
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> torch.distributed.init_process_group(world_size=4, init_method='...')
        >>> net = torch.nn.DistributedDataParallel(model)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DistributedDataParallel, self).__init__()

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device

        # 同步参数和缓冲区
        for p in self.module.state_dict().values():
            dist.broadcast(p, 0)

        if len(device_ids) > 1:
            # TODO : 我们不需要在这里复制参数. 他们总是在广播中使用大块进行广播, 
            # 所以最好不要用这些小块来污染高速缓存.
            self._module_copies = replicate(self.module, self.device_ids)
            self._module_copies[0] = self.module
            for module_copy in self._module_copies[1:]:
                for param, copy_param in zip(self.module.parameters(), module_copy.parameters()):
                    copy_param.detach_()
                    copy_param.requires_grad = param.requires_grad
        else:
            self._module_copies = [self.module]

        # 将参数拆分成将会合并减少的存储桶
        # TODO :不同的类型需要不同的桶
        t = None
        for p in self.module.parameters():
            tp = type(p.data)
            if t is not None and t is not tp:
                raise ValueError("DistributedDataParallel requires all parameters' data to be of the same type")
            t = tp

        self.bucket_sizes = []
        self.bucket_map = {}
        MB = 1024 * 1024
        self.broadcast_bucket_size = 10 * MB  # 在转发之前用于参数同步
        bucket_bytes_cap = 1 * MB
        bucket_bytes = bucket_bytes_cap  # 立即启动第一个桶
        for param_tuple in zip(*map(lambda m: m.parameters(), self._module_copies)):
            if bucket_bytes >= bucket_bytes_cap:
                self.bucket_sizes.append(0)
                bucket_bytes = 0
            self.bucket_sizes[-1] += 1
            for p in param_tuple:
                self.bucket_map[p] = len(self.bucket_sizes) - 1
            bucket_bytes += p.numel() * p.element_size()

        self.buckets = [[[] for _ in range(len(self.device_ids))] for _ in range(len(self.bucket_sizes))]
        self.bucket_events = [[None] * len(self.device_ids) for _ in range(len(self.bucket_sizes))]
        self.reduced = [False] * len(self.bucket_sizes)

        self._register_grad_hooks()

        self.dispatch_lock = threading.Lock()
        self._start_reduction_threads()

    def __getstate__(self):
        attrs = copy.copy(self.__dict__)
        del attrs['_grad_accs'], attrs['_reduction_queues'], attrs['_reduction_streams'], \
            attrs['_reduction_threads'], attrs['_nccl_streams'], attrs['_default_streams']
        return attrs

    def __setstate__(self, state):
        super(DistributedDataParallel, self).__setstate__(state)
        self._register_grad_hooks()
        self._start_reduction_threads()

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        self._sync_params()
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        outputs = self.parallel_apply(self._module_copies, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        super(DistributedDataParallel, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)

    def _sync_params(self):
        params = [p.data for p in self.module.parameters()]
        result = broadcast_coalesced(params, self.device_ids, self.broadcast_bucket_size)
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, param in zip(tensors, module.parameters()):
                param.data.set_(tensor)

        buffers = list(self.module._all_buffers())
        if len(buffers) > 0:
            # 跨节点缓冲区同步
            flat_buffers = _flatten_dense_tensors(buffers)
            dist.broadcast(flat_buffers, 0)
            for buf, synced in zip(buffers, _unflatten_dense_tensors(flat_buffers, buffers)):
                buf.copy_(synced)

            # 节点内缓冲区同步
            result = broadcast_coalesced(buffers, self.device_ids, self.broadcast_bucket_size)
            for tensors, module in zip(result[1:], self._module_copies[1:]):
                for tensor, buf in zip(tensors, module._all_buffers()):
                    buf.set_(tensor)

    def _register_grad_hooks(self):
        self._grad_accs = []  # 需要保持在范围内
        for device_idx, module in enumerate(self._module_copies):
            for p in module.parameters():
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(p, device_idx))
                    self._grad_accs.append(grad_acc)

    def _make_param_hook(self, param, device_idx):
        bucket_idx = self.bucket_map[param]

        def distributed_data_parallel_hook(*unused):
            if not param.grad.volatile:
                raise RuntimeError("DistributedDataParallel only works with volatile gradients")
            bucket = self.buckets[bucket_idx][device_idx]
            bucket.append(param.grad.data)

            # 我们可以刷新这些并为副本节省内存
            if device_idx > 0:
                param.grad = None
                param.data.set_()

            # 当前设备的存储桶已满
            if len(bucket) == self.bucket_sizes[bucket_idx]:
                with torch.cuda.device(self.device_ids[device_idx]):
                    event = torch.cuda.Event()
                    event.record()
                with self.dispatch_lock:
                    self.bucket_events[bucket_idx][device_idx] = event
                    self._queue_reduction(bucket_idx)

        return distributed_data_parallel_hook

    def _queue_reduction(self, bucket_idx):
        dev_buckets = self.buckets[bucket_idx]
        dev_events = self.bucket_events[bucket_idx]

        # 检查是否准备好
        if any(evt is None for evt in dev_events):
            return

        # 排队减少, 并确保向后等待
        event = threading.Event()
        self._reduction_queues[bucket_idx].put((dev_buckets, dev_events, event))
        Variable._execution_engine.queue_callback(lambda: event.wait())

        # 重置存储桶状态
        self.buckets[bucket_idx] = [[] for _ in range(len(self.device_ids))]
        self.bucket_events[bucket_idx] = [None] * len(self.device_ids)
        self.reduced[bucket_idx] = True
        if all(self.reduced):
            self.reduced = [False] * len(self.bucket_sizes)

            def sync_reduction_streams():
                # 我们只需要与第一个同步, 但这样做更安全
                # 如果我们改变平行工作的方式
                r_streams = zip(*self._reduction_streams)
                for dev_id, default_stream, dev_r_streams in zip(self.device_ids, self._default_streams, r_streams):
                    with torch.cuda.device(dev_id):
                        for reduction_stream in dev_r_streams:
                            default_stream.wait_stream(reduction_stream)
            Variable._execution_engine.queue_callback(sync_reduction_streams)

    def _start_reduction_threads(self):
        num_buckets = len(self.bucket_sizes)
        self._reduction_queues = [queue.Queue() for _ in range(num_buckets)]
        self._reduction_threads = []
        self._reduction_streams = [[] for _ in range(num_buckets)]
        self._nccl_streams = []
        self._default_streams = []
        for dev_id in self.device_ids:
            with torch.cuda.device(dev_id):
                # TODO: 不要假设在使用默认流
                self._default_streams.append(torch.cuda.current_stream())
                self._nccl_streams.append(torch.cuda.Stream())
        for reduction_queue, reduction_streams in zip(self._reduction_queues, self._reduction_streams):
            for dev_id in self.device_ids:
                with torch.cuda.device(dev_id):
                    reduction_streams.append(torch.cuda.Stream())
            # 我们只使用第一台设备进行分布式减量
            dist._register_stream(reduction_streams[0])
            group_id = dist.new_group()

            self._reduction_threads.append(threading.Thread(
                target=self._reduction_thread_fn,
                args=(reduction_queue, group_id, self.device_ids, reduction_streams, self._nccl_streams)))
            self._reduction_threads[-1].daemon = True
            self._reduction_threads[-1].start()

    @staticmethod
    def _reduction_thread_fn(queue, group_id, device_ids, reduction_streams, nccl_streams):

        def _process_batch():
            dev_grad_batch, dev_events, job_event = queue.get()
            dev_coalesced = []
            # 合并所有设备上的张量并开始本地减少
            for dev_id, grad_batch, event, stream in zip(device_ids, dev_grad_batch, dev_events, reduction_streams):
                with torch.cuda.device(dev_id), torch.cuda.stream(stream):
                    stream.wait_event(event)
                    coalesced = _flatten_dense_tensors(grad_batch)
                    dev_coalesced.append(coalesced)
            # 在启动NCCL内核之前等待所有副本完成
            for stream in reduction_streams:
                stream.synchronize()
            nccl.reduce(dev_coalesced, root=0, streams=nccl_streams)

            # 从现在起, 我们只会在第一个设备上工作（从设备ID）
            grad_batch = dev_grad_batch[0]
            coalesced = dev_coalesced[0]
            reduce_stream = reduction_streams[0]
            with torch.cuda.stream(reduce_stream):
                reduce_stream.wait_stream(nccl_streams[0])
                coalesced /= dist.get_world_size()
                dist.all_reduce(coalesced, group=group_id)
                for grad, reduced in zip(grad_batch, _unflatten_dense_tensors(coalesced, grad_batch)):
                    grad.copy_(reduced)
            job_event.set()

        with torch.cuda.device(device_ids[0]):
            while True:
                _process_batch()  # 只是为了有一个清晰的范围
