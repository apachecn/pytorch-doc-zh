# 使用 RPC 的分布式管道并行化

> 原文：<https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html>

**作者**：[Shen Li](https://mrshenli.github.io/)

先决条件：

*   [PyTorch 分布式概述](../beginner/dist_overview.html)
*   [单机模型并行最佳实践](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
*   [分布式 RPC 框架](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)入门
*   RRef 辅助函数： [`RRef.rpc_sync()`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_sync)， [`RRef.rpc_async()`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_async)和 [`RRef.remote()`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.remote)

本教程使用 Resnet50 模型来演示如何使用[`torch.distributed.rpc`](https://pytorch.org/docs/master/rpc.html) API 实现分布式管道并行性。 可以将其视为[单机模型并行最佳实践](model_parallel_tutorial.html)中讨论的多 GPU 管道并行性的分布式对应物。

注意

本教程需要 PyTorch v1.6.0 或更高版本。

注意

本教程的完整源代码可以在[`pytorch/examples`](https://github.com/pytorch/examples/tree/master/distributed/rpc/pipeline)中找到。

## 基础知识

上一教程[分布式 RPC 框架入门](rpc_tutorial.html)显示了如何使用[`torch.distributed.rpc`](https://pytorch.org/docs/master/rpc.html)为 RNN 模型实现分布式模型并行性。 该教程使用一个 GPU 来托管`EmbeddingTable`，并且提供的代码可以正常工作。 但是，如果模型驻留在多个 GPU 上，则将需要一些额外的步骤来增加所有 GPU 的摊销利用率。 管道并行性是在这种情况下可以提供帮助的一种范例。

在本教程中，我们使用`ResNet50`作为示例模型，[单机模型并行最佳实践](model_parallel_tutorial.html)教程也使用了该模型。 类似地，`ResNet50`模型被分为两个碎片，输入批量被划分为多个拆分，并以流水线方式馈入两个模型碎片。 区别在于，本教程将调用异步 RPC，而不是使用 CUDA 流来并行执行。 因此，本教程中介绍的解决方案也可以跨计算机边界使用。 本教程的其余部分分四个步骤介绍了实现。

## 第 1 步：对 ResNet50 模型进行分片

这是在两个模型分片中实现`ResNet50`的准备步骤。 以下代码是从`torchvision`中的 [ResNet 实现](https://github.com/pytorch/vision/blob/7c077f6a986f05383bcb86b535aedb5a63dd5c4b/torchvision/models/resnet.py#L124)中借用的。 `ResNetBase`模块包含两个 ResNet 碎片的通用构件和属性。

```py
import threading

import torch
import torch.nn as nn

from torchvision.models.resnet import Bottleneck

num_classes = 1000

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetBase(nn.Module):
    def __init__(self, block, inplanes, num_classes=1000,
                groups=1, width_per_group=64, norm_layer=None):
        super(ResNetBase, self).__init__()

        self._lock = threading.Lock()
        self._block = block
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * self._block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * self._block.expansion, stride),
                norm_layer(planes * self._block.expansion),
            )

        layers = []
        layers.append(self._block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * self._block.expansion
        for _ in range(1, blocks):
            layers.append(self._block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def parameter_rrefs(self):
        return [RRef(p) for p in self.parameters()]

```

现在，我们准备定义两个模型碎片。 对于构造器，我们只需将所有 ResNet50 层分为两部分，然后将每个部分移至提供的设备中。 两个分片的`forward`函数获取输入数据的`RRef`，在本地获取数据，然后将其移至所需的设备。 将所有层应用于输入后，它将输出移至 CPU 并返回。 这是因为当调用方和被调用方中的设备数量不匹配时，RPC API 要求张量驻留在 CPU 上，以避免无效的设备错误。

```py
class ResNetShard1(ResNetBase):
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(
            Bottleneck, 64, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 3),
            self._make_layer(128, 4, stride=2)
        ).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out =  self.seq(x)
        return out.cpu()

class ResNetShard2(ResNetBase):
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(
            Bottleneck, 512, num_classes=num_classes, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            self._make_layer(256, 6, stride=2),
            self._make_layer(512, 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).to(self.device)

        self.fc =  nn.Linear(512 * self._block.expansion, num_classes).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()

```

## 第 2 步：将 ResNet50 模型片段拼接到一个模块中

然后，我们创建一个`DistResNet50`模块来组装两个分片并实现流水线并行逻辑。 在构造器中，我们使用两个`rpc.remote`调用分别将两个分片放在两个不同的 RPC 工作器上，并保持`RRef`到两个模型部分，以便可以在正向传播中引用它们。 `forward`函数将输入批量分为多个微批量，并将这些微批量以流水线方式馈送到两个模型部件。 它首先使用`rpc.remote`调用将第一个分片应用于微批量，然后将返回的中间输出`RRef`转发到第二个模型分片。 之后，它将收集所有微输出的`Future`，并在循环后等待所有它们。 请注意，`remote()`和`rpc_async()`都立即返回并异步运行。 因此，整个循环是非阻塞的，并将同时启动多个 RPC。 中间输出`y_rref`保留了两个模型零件上一个微批量的执行顺序。 微批量的执行顺序无关紧要。 最后，正向函数将所有微批量的输出连接到一个单一的输出张量中并返回。 `parameter_rrefs`函数是简化分布式优化器构造的助手，将在以后使用。

```py
class DistResNet50(nn.Module):
    def __init__(self, num_split, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()

        self.num_split = num_split

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            ResNetShard1,
            args = ("cuda:0",) + args,
            kwargs = kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            ResNetShard2,
            args = ("cuda:1",) + args,
            kwargs = kwargs
        )

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params

```

## 步骤 3：定义训练循环

定义模型后，让我们实现训练循环。 我们使用专门的“主”工作器来准备随机输入和标签，并控制分布式反向传递和分布式优化器步骤。 它首先创建`DistResNet50`模块的实例。 它指定每个批量的微批数量，并提供两个 RPC 工作程序的名称（即`worker1`和`worker2`）。 然后，它定义损失函数，并使用`parameter_rrefs()`帮助器创建`DistributedOptimizer`以获取参数`RRefs`的列表。 然后，主训练循环与常规本地训练非常相似，除了它使用`dist_autograd`向后启动并为反向和优化器`step()`提供`context_id`之外。

```py
import torch.distributed.autograd as dist_autograd
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128

def run_master(num_split):
    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(num_split, ["worker1", "worker2"])
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    one_hot_indices = torch.LongTensor(batch_size) \
                        .random_(0, num_classes) \
                        .view(batch_size, 1)

    for i in range(num_batches):
        print(f"Processing batch {i}")
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                    .scatter_(1, one_hot_indices, 1)

        with dist_autograd.context() as context_id:
            outputs = model(inputs)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)

```

## 第 4 步：启动 RPC 进程

最后，下面的代码显示了所有进程的目标函数。 主要逻辑在`run_master`中定义。 工作器被动地等待主服务器发出的命令，因此只需运行`init_rpc`和`shutdown`即可，其中默认情况下`shutdown`会阻塞，直到所有 RPC 参与者都完成。

```py
import os
import time

import torch.multiprocessing as mp

def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()

if __name__=="__main__":
    world_size = 3
    for num_split in [1, 2, 4, 8]:
        tik = time.time()
        mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
        tok = time.time()
        print(f"number of splits = {num_split}, execution time = {tok - tik}")

```

下面的输出显示通过增加每批中的拆分数量而获得的加速。

```py
$ python main.py
Processing batch 0
Processing batch 1
Processing batch 2
number of splits = 1, execution time = 16.45062756538391
Processing batch 0
Processing batch 1
Processing batch 2
number of splits = 2, execution time = 12.329529762268066
Processing batch 0
Processing batch 1
Processing batch 2
number of splits = 4, execution time = 10.164430618286133
Processing batch 0
Processing batch 1
Processing batch 2
number of splits = 8, execution time = 9.076049566268921

```