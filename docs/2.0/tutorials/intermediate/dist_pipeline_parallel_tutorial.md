


 使用 RPC 的分布式管道并行性
 [¶](#distributed-pipeline-parallelism-using-rpc "永久链接到此标题")
==========================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/dist_pipeline_parallel_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/dist_pipeline_parallel_tutorial.html>




**作者** 
 :
 [沉力](https://mrshenli.github.io/)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/intermediate_source/dist_pipeline_parallel_tutorial.rst) 
.





 先决条件:



* [PyTorch 分布式概述](../beginner/dist_overview.html)
* [单机模型并行最佳实践](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
* [获取从分布式 RPC 框架开始](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)
* RRef 辅助函数：
 [RRef.rpc_sync()](https://pytorch.org /docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_sync) 
 ,
 [RRef.rpc_async()](https://pytorch.org/docs/master/rpc.html #torch.distributed.rpc.RRef.rpc_async) 
 和
 [RRef.remote()](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.remote ）



 本教程使用 Resnet50 模型来演示使用
 [torch.distributed.rpc](https://pytorch.org/docs/master/rpc.html) API 实现分布式
管道并行性。这可以被视为多 GPU
管道并行性的分布式对应部分，请参阅
 [单机模型并行最佳实践](model_parallel_tutorial.html) 
 。





 注意




 本教程需要 PyTorch v1.6.0 或更高版本。






 没有10



 本教程的完整源代码可以在
 [pytorch/examples](https://github.com/pytorch/examples/tree/master/distributed/rpc/pipeline) 找到
 。






 基础知识
 [¶](#basics "此标题的永久链接")
-----------------------------------------------------------------



 上一篇教程，
 [分布式 RPC 框架入门](rpc_tutorial.html) 
 展示了如何使用
 [torch.distributed.rpc](https://pytorch.org/docs/master/rpc.html) 
 实现 RNN 模型的分布式模型并行性。该教程使用
一个 GPU 来托管
 `EmbeddingTable`
 ，并且提供的代码工作正常。
但是，如果模型存在于多个 GPU 上，
则需要一些额外的步骤来
提高所有 GPU 的摊销利用率。管道并行性是一种
在这种情况下可以提供帮助的范式。




 在本教程中，我们使用
 `ResNet50`
 作为示例模型，
[单机模型并行最佳实践](model_parallel_tutorial.html) 教程也使用该模型。类似地，
 `ResNet50`
 模型被分为两个分片，
输入批次被划分为多个分片，并以管道方式馈送到两个模型
分片中。不同之处在于，本教程不是使用 CUDA 流并行执行，而是调用异步 RPC。因此，
本教程中介绍的解决方案也可以跨计算机边界工作。
本教程的其余部分将分四个步骤介绍实现。






 步骤1：分区ResNet50模型
 [¶](#step-1-partition-resnet50-model "永久链接到此标题")
-------------------------------------------------------------------------------------------------------



 这是在两个模型分片中实现
 `ResNet50` 的准备步骤。
下面的代码借用
 [torchvision 中的 ResNet 实现](https://github.com/pytorch/vision/blob/7c077f6a986f05383bcb86b535aedb5a63dd5c4b/torchvision/models/resnet.py#L124) 
 。

`ResNetBase`
 模块包含两个 ResNet 分片的通用构建块和属性。






```
import threading

import torch
import torch.nn as nn

from torchvision.models.resnet import Bottleneck

num_classes = 1000


def conv1x1(in_planes，out_planes，stride=1):
 return nn.Conv2d(in_planes，out_planes，kernel_size=1，stride=stride，bias=False)


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




 现在，我们准备定义两个模型分片。对于构造函数，我们简单地将所有 ResNet50 层分成两部分，并将每个部分移动到提供的设备中。两个分片的
 `forward`
 函数采用输入数据的
 
 `RRef`
，在本地获取数据，然后将其移动到预期的设备。
将所有层应用到输入后，它将输出移至 CPU 并返回。
这是因为 RPC API 要求张量驻留在 CPU 上，以避免当调用方和被调用方的设备数量不匹配时
出现无效设备错误。






```
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






 第 2 步：将 ResNet50 模型碎片拼接到一个模块中
 [¶](#step-2-stitch-resnet50-model-shards-into-one-module "永久链接到此标题")
------------------------------------------------------------------------------------------------------------------------------------------------



 然后，我们创建一个 
 `DistResNet50`
 模块来组装两个分片并
实现管道并行逻辑。在构造函数中，我们使用两个
 `rpc.remote`
 调用将两个分片分别放在两个不同的 RPC 工作
上，并保留
 
 `RRef`
 到两个模型部分，以便它们
 n可以在前向传播中引用。 
 `forward`
 函数将输入批次分割为多个微批次，并以管道方式将这些微批次提供给两个模型部分。它首先使用
 `rpc.remote`
 调用将第一个分片应用于微批次，然后将
返回的中间输出
 
 `RRef`
 转发到第二个模型分片。之后，
收集所有微输出的
 `Future`，并在循环后等待所有
。请注意，`remote()` 和 `rpc_async()` 都会立即返回并异步运行。因此，整个循环是非阻塞的，并且会同时启动多个 RPC。两个模型部分上的一个微批次的执行顺序由中间输出保留。
 `y_rref`
 。微批次的执行顺序
并不重要。最后，前向函数将所有微批次的输出连接成一个输出张量并返回。 
 `parameter_rrefs`
 函数是简化分布式优化器构造的帮助器，稍后将使用它。






```
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
        for x in iter(xs.split(self.num_split, dim=0)):
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






 步骤 3：定义训练循环
 [¶](#step-3-define-the-training-loop "固定链接到此标题")
--------------------------------------------------------------------------------------------------------------------



 定义模型后，让我们实现训练循环。我们使用专用的 \xe2\x80\x9cmaster\xe2\x80\x9d 工作程序来准备随机输入和标签，并控制
分布式向后传递和分布式优化器步骤。它首先创建
 `DistResNet50` 模块的
 实例。它指定每个批次的
微批次数量，并提供两个 RPC 工作线程的名称
（即 \xe2\x80\x9cworker1\xe2\x80\x9d 和 \xe2\x80\x9cworker2\xe2 \x80\x9d）。然后，它定义损失函数并使用
 `parameter_rrefs()`
 帮助器创建

 `DistributedOptimizer`
 来获取参数
 
 `RRefs`
 列表。然后，主训练循环与
常规本地训练非常相似，不同之处在于它使用
 `dist_autograd`
 来启动
向后并为
向后和提供
 `context_id`
优化器
 `step()`
.






```
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






 步骤 4：启动 RPC 进程
 [¶](#step-4-launch-rpc-processes "永久链接到此标题")
------------------------------------------------------------------------------------------------



 最后，下面的代码显示了所有进程的目标函数。主要逻辑在 `run_master`
 中定义。工作线程被动地等待来自主机的命令，因此只需运行`init_rpc`和
`shutdown`，
默认情况下，
`shutdown`
将阻塞，直到所有RPC 参与者完成。






```
import os
import time

import torch.multiprocessing as mp


def run_worker(rank, world_size, num_split):
 os.environ['MASTER_ADDR'] = 'localhost'
 os.environ['MASTER\ \_PORT'] = '29500'
 options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

 如果rank == 0:
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
 rpc_backend_options=options\ n )
 pass

 # 阻塞直到所有 rpc 完成
 rpc.shutdown()


if __name__=="__main__":
    world_size = 3
    for num_split in [1, 2, 4, 8]:
        tik = time.time()
        mp.spawn(run_worker, args=(world_size, num_split), nprocs=world_size, join=True)
        tok = time.time()
        print(f"number of splits = {num_split}, execution time = {tok - tik}")

```









