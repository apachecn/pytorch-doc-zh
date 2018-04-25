import torch
from . import nccl
from torch._utils import _accumulate, _take_tensors, _flatten_dense_tensors, \
    _flatten_sparse_tensors, _unflatten_dense_tensors, \
    _unflatten_sparse_tensors, _reorder_tensors_as


def broadcast(tensor, devices):
    """将张量广播给多个 GPU .

    Arguments:
        tensor (Tensor): 需要广播的张量.
        devices (Iterable): 在一个可迭代设备中广播.
          请注意, 它应该像 (src, dst1, dst2, ...), 其中的第一个元素是来至其广播的源设备.

    Returns:
        一个元组, 包含 ``tensor`` 副本,放置在与设备的索引相对应的 ``设备`` 上.
    """
    tensors = [tensor]
    if nccl.is_available(tensors) and len(set(devices)) == len(devices):
        for device in devices[1:]:
            with torch.cuda.device(device):
                tensors.append(type(tensor)(tensor.size()))
        nccl.broadcast(tensors)
        return tuple(tensors)

    return tuple(tensor.cuda(gpu, async=True) for gpu in devices)


def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """将序列张量广播到指定的 GPUs .

    小张量首先合并到一个缓冲区中以减少同步的次数.

    Arguments:
        tensors (sequence): 要广播的张量.
        devices (Iterable): 在一个可迭代设备中广播.
          请注意, 它应该像 (src, dst1, dst2, ...), 其中的第一个元素是来至其广播的源设备.
        buffer_size (int): 用于合并的最大缓冲区大小.

    Returns:
        一个元组, 包含 ``tensor`` 副本,放置在与设备的索引相对应的设备上.
    """
    for tensor in tensors:
        if tensor.get_device() != devices[0]:
            raise RuntimeError('all tensors must be on devices[0]')
    outputs = [[] for _ in devices]
    # use the original tensors for the first device
    outputs[0].extend(tensors)
    for chunk in _take_tensors(tensors, buffer_size):
        if chunk[0].is_sparse:
            flat_indices, flat_values = _flatten_sparse_tensors(chunk)
            result_indices = broadcast(flat_indices, devices)
            result_values = broadcast(flat_values, devices)
            unflat_results = tuple(_unflatten_sparse_tensors(iv, chunk) for iv in zip(result_indices, result_values))
        else:
            flat = _flatten_dense_tensors(chunk)
            results = broadcast(flat, devices)
            unflat_results = tuple(_unflatten_dense_tensors(tensor, chunk) for tensor in results)
        # use the broadcasted tensors for the remaining devices
        for dst, unflat_res in zip(outputs[1:], unflat_results[1:]):
            dst.extend(unflat_res)
    for i, output in enumerate(outputs):
        outputs[i] = _reorder_tensors_as(output, tensors)
    return tuple(outputs)


def reduce_add(inputs, destination=None):
    """从多个 GPU 中收集张量.

    所有的输入应该有匹配的 shapes (形状).

    Arguments:
        inputs (Iterable[Tensor]): 添加一个可迭代的张量.
        destination (int, optional): 放置输出的设备 (默认: 当前设备).

    Returns:
        包含所有输入的元素和的张量, 存放在 ``destination(目标)`` 设备.
    """
    # TODO: 尝试在另一个 gpu 上找到一个输入, 复制它并添加到副本中.
    if destination is None:
        destination = torch.cuda.current_device()
    input_size = inputs[0].size()
    is_sparse = inputs[0].is_sparse
    nccl_root = None
    for i, inp in enumerate(inputs):
        assert inp.is_cuda, "reduce_add expects all inputs to be on GPUs"#reduce_add 希望所有的输入都在 gpu 上.
        if inp.get_device() == destination:
            nccl_root = i
        if inp.size() != input_size:
            got = 'x'.join(str(x) for x in inp.size())
            expected = 'x'.join(str(x) for x in input_size)
            raise ValueError("input {} has invalid size: got {}, but expected "
                             "{}".format(i, got, expected))#输入大小无效,期望是xx,得到的是xx
    assert nccl_root is not None, "reduce_add expects destination to be on the same GPU with one of the tensors"#reduce_add期望目标位于与张量之一相同的GPU上
    with torch.cuda.device(destination):
        result = type(inp)().resize_as_(inp).zero_()

    if nccl.is_available(inputs) and inputs[0].get_device() == destination:
        outputs = [result] + [t.new(t.size()) for t in inputs[1:]]
        nccl.reduce(inputs, outputs, root=nccl_root)
        return result
    for inp in inputs:
        input_correct_gpu = inp.cuda(result.get_device())
        result.add_(input_correct_gpu)
    return result


def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """从多个 GPU 中收集张量.

    小张量首先合并到一个缓冲区中以减少同步的次数.

    Arguments:
        inputs (Iterable[Iterable[Tensor]]): 包含张量来至单一的设备可迭代对象的迭代器.
        destination (int, optional): 放置输出的设备 (默认: 当前设备).
        buffer_size (int): 合并缓冲区的最大值

    Returns:
        张量元组包含每组输入的元素和, 放置在 ``目标`` 设备上.
    """
    dense_tensors = [[] for _ in inputs]  # shape (num_gpus, num_tensors) 
    output = []
    ref_order = []
    # 先处理稀疏问题因为他们可能有不同的大小在不同的GPU上.
    for tensor_at_gpus in zip(*inputs):
        if all(t.is_sparse for t in tensor_at_gpus):
            result = reduce_add(tensor_at_gpus, destination)
            output.append(result)
            ref_order.append(tensor_at_gpus[0])
        else:
            for coll, t in zip(dense_tensors, tensor_at_gpus):
                coll.append(t.to_dense() if t.is_sparse else t)
            ref_order.append(dense_tensors[0][-1])
    itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
    # 现在的稠密度大小一致
    for chunks in zip(*itrs):
        flat_tensors = [_flatten_dense_tensors(chunk) for chunk in chunks]
        flat_result = reduce_add(flat_tensors, destination)
        output.extend(_unflatten_dense_tensors(flat_result, chunks[0]))
    return tuple(_reorder_tensors_as(output, ref_order))


def scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None):
    """分散张量到多个 GPU.

    Arguments:
        tensor (Tensor): 需要分散的张量.
        devices (Iterable[int]): 整数的迭代,指定张量应分散在哪些设备之间.
        chunk_sizes (Iterable[int], optional): 要放在每个设备上的块的大小. 应该匹配 ``设备`` 长度和
             ``tensor.size(dim)`` 的和. 如果未指定,张量将被划分成相等的块.
        dim (int, optional): 分块张量沿着的维度

    Returns:
        一个元组包含 ``tensor`` 块, 传递给 ``devices`` .
    """
    if chunk_sizes is None:
        chunks = tensor.chunk(len(devices), dim)
    else:
        assert sum(chunk_sizes) == tensor.size(dim), "given chunk sizes " \
            "don't sum up to the tensor's size (sum(chunk_sizes) == {}, but " \
            "expected {})".format(sum(chunk_sizes), tensor.size(dim))
        assert min(chunk_sizes) > 0, "got a negative chunk_size"
        chunks = [tensor.narrow(dim, start - size, size)
                  for start, size in zip(_accumulate(chunk_sizes), chunk_sizes)]
    chunks = tuple(chunk.contiguous() for chunk in chunks)
    # TODO: 首先复制到固定缓冲区（如果从CPU复制）
    if streams is None:
        streams = [None] * len(devices)
    outputs = []
    for device, chunk, stream in zip(devices, chunks, streams):
        with torch.cuda.device(device), torch.cuda.stream(stream):
            outputs.append(chunk.cuda(device, async=True))
    return tuple(outputs)


def gather(tensors, dim=0, destination=None):
    """从多个 GPU 收集张量.

    张量尺寸在不同于 ``dim`` 的维度上都应该匹配.

    Arguments:
        tensors (Iterable[Tensor]): 张量集合的迭代器.
        dim (int): 张量被连接的维度.
        destination (int, optional): 输出设备 (-1 代表 CPU, 默认:
            当前设备)

    Returns:
        一个位于 ``目标`` 设备上的张量, 将 ``tensors`` 沿着 ``dim`` 连接起来的结果.
    """
    total_size = 0
    expected_size = list(tensors[0].size())
    for tensor in tensors:
        assert tensor.is_cuda, "gather expects all inputs to be on GPUs"#在GPUs上收集所有的预期输入
        expected_size[dim] = tensor.size(dim)
        if list(tensor.size()) != expected_size:
            got = 'x'.join(str(x) for x in tensor.size())
            expected = 'x'.join(str(x) for x in expected_size)
            raise ValueError("gather got an input of invalid size: got {}, "
                             "but expected {}".format(got, expected))#在预期输入时xx,得到的输入时xx
        total_size += tensor.size(dim)
    expected_size[dim] = total_size
    expected_size = torch.Size(expected_size)
    if destination is None:
        destination = torch.cuda.current_device()
    if destination == -1:
        result = getattr(torch, type(tensors[0]).__name__)(expected_size)
    else:
        with torch.cuda.device(destination):
            result = type(tensors[0])(expected_size)

    chunk_start = 0
    # TODO: 如果复制到CPU,分配一个固定缓冲,做异步拷贝,并将其复制到常规内存.
    for tensor in tensors:
        result.narrow(dim, chunk_start, tensor.size(dim)).copy_(tensor, True)
        chunk_start += tensor.size(dim)
    return result
