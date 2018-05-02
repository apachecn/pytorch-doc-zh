import torch
from ..modules import Module
from .scatter_gather import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply


class DataParallel(Module):
    r"""在模块级别实现数据并行性. 

    此容器通过在批次维度中分块, 将输入分割到指定设备上, 从而并行化给定模块的应用程
    序.在正向传递中, 模块被复制到每个设备上, 每个副本处理一部分输入.在向后传递期间, 
    来自每个副本的梯度变化被汇总到原始模块中.

    batch size 应该大于 GPUs 的数量.同时也应该是 GPU 数量的整数倍, 以
    便每个块大小相同（以便每个 GPU 处理相同数量的样本）.

    引用 ::ref:`cuda-nn-dataparallel-instead`

    允许将任意位置和关键字输入传入 DataParallel EXCEPT Tensors. 所有的变量将被分
    散在指定的维度（默认为0）.原始类型将被广播, 但所有其他类型将是一个浅层副本, 如
    果写入模型的正向传递, 可能会被损坏.

    Args :
        module: 并行的模型
        device_ids: CUDA devices（CUDA 驱动） (default: all devices)
        output_device: 输出设备位置 (default: device_ids[0])

    示例 ::
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""在 device_ids（设备 ID ）中给出的 GPU 上并行评估模块（输入）.

    这是数据并行模块的功能版本.

    Args :
        module: 并行评估的模型
        inputs: 模型的输入
        device_ids: 防止副本的 GPU 设备 ID
        output_device: 输出的 GPU 位置使用 -1 指示 CPU.（默认 : device_ids [0])
    return :
        包含位于输出设备上的模块（输入）结果的变量
    """

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
