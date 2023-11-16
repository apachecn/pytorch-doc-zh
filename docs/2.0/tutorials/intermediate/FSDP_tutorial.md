


# 完全分片数据并行入门 (FSDP) [¶](#getting-started-with-complete-sharded-data-parallel-fsdp "此标题的永久链接")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/FSDP_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>




**Author** 
 :
 [Hamid Shojanazeri](https://github.com/HamidShojanazeri) 
 ,
 [Yanli Zhao](https://github.com/zhaojuanmao) 
 ,
 [Shen Li](https://mrshenli.github.io/)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/intermediate_source/FSDP_tutorial.rst) 
.





 大规模训练 AI 模型是一项具有挑战性的任务，需要大量计算能力和资源。
处理这些超大型模型的训练还需要相当大的工程复杂性。
 [PyTorch FSDP]( https://pytorch.org/blog/introducing-pytorch-filled-sharded-data-parallel-api/) 
 ，在 PyTorch 1.11 中发布使这变得更容易。




 在本教程中，我们将展示如何使用
 [FSDP API](https://pytorch.org/docs/1.11/fsdp.html) 
 对于可以扩展到其他更大模型的简单 MNIST 模型，例如as
 [HuggingFace BERT 模型](https://huggingface.co/blog/zero-deepspeed-fairscale) 
 ,
 [GPT 3 模型高达 1T 参数](https://pytorch.medium.com/使用 pytorch 完全分片数据并行在 aws-3ac13aa96cff 上训练 1 万亿参数模型) 
 。示例 DDP MNIST 代码借自
 [此处](https://github.com/yqhu/mnist_examples)
 。





## FSDP 的工作原理 [¶](#how-fsdp-works "永久链接到此标题")




 在
 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
 中，(DDP) 训练，每个进程/工作线程都拥有模型并处理一批数据，最后使用 all-reduce 来总结不同工作人员的梯度。在 DDP 中，模型权重和优化器状态在所有工作线程中复制。 FSDP 是一种数据并行性，可跨 DDP 等级分片模型参数、优化器状态和梯度。




 使用 FSDP 进行训练时，GPU 内存占用量比在所有工作线程上使用 DDP 进行训练时要小。通过允许更大的模型或批量大小适合设备，这使得一些非常大的模型的训练变得可行。这是伴随着通信量增加的成本而来的。通过内部优化（例如重叠通信和计算）减少了通信开销。




[![FSDP 工作流程](https://pytorch.org/tutorials/_images/fsdp_workflow.png)](https://pytorch.org/tutorials/_images/fsdp_workflow.png)


 FSDP 工作流程
  [¶](#id1“此图像的永久链接”)





 在较高级别上，FSDP 的工作原理如下：




*在构造函数中*



* 分片模型参数，每个等级只保留自己的分片



*在前进路径中*



* 运行 all_gather 收集所有等级的所有分片，以恢复此 FSDP 单元中的完整参数
* 运行前向计算
* 丢弃刚刚收集的参数分片



*在后向路径中*



* 运行 all_gather 以收集所有等级的所有分片，以恢复此 FSDP 单元中的完整参数
* 运行反向计算
* 运行 reduce_scatter 以同步梯度
* 丢弃参数。



 查看 FSDP’s 分片的一种方法是将 DDP 梯度全归约分解为归约分散和全聚集。具体来说，在向后传递过程中，FSDP 减少并分散梯度，确保每个等级都拥有梯度碎片。然后它在优化器步骤中更新参数的相应分片。最后，在后续的前向传播中，它执行全收集操作来收集并组合更新的参数分片。




[![FSDP allreduce](https://pytorch.org/tutorials/_images/fsdp_sharding.png)](https://pytorch.org/tutorials/_images/fsdp_sharding.png)


 FSDP Allreduce
  [¶](#id2“此图像的永久链接”)





## 如何使用 FSDP [¶](#how-to-use-fsdp "永久链接到此标题")




 这里我们使用一个玩具模型在 MNIST 数据集上运行训练以进行演示。 API 和逻辑也可以应用于训练更大的模型。




*设置*




 1.1 安装 PyTorch 和 Torchvision






```
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html

```




 我们将以下代码片段添加到 python 脚本 “FSDP_mnist.py” 中。




 1.2 导入必要的包





 注意




 本教程适用于 PyTorch 版本 1.12 及更高版本。如果您使用的是早期版本，请将
 
 size_based_auto_wrap_policy
 
 的所有实例替换为
 
 default_auto_wrap_policy
 
.







```
# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

```




 1.3 分布式训练设置。正如我们提到的，FSDP 是一种数据并行性，需要分布式训练环境，因此这里我们使用两个辅助函数来初始化分布式训练和清理的过程。






```
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

```




 2.1 定义我们的手写数字分类玩具模型。






```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

```




 2.2 定义训练函数






```
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

```




 2.3 定义验证函数






```
def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))

```




 2.4 定义一个分布式训练函数，将模型包装在 FSDP 中




**注意：要保存 FSDP 模型，我们需要在每个等级上调用 state_dict，然后在等级 0 上保存整体状态。**






```
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = Net().to(rank)

    model = FSDP(model)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()

```




 2.5 最后解析参数并设置main函数






```
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)

```




 我们记录了 cuda 事件来测量 FSDP 模型细节的时间。 CUDA 事件时间为 110.85 秒。






```
python FSDP_mnist.py

CUDA event elapsed time on training loop 40.67462890625sec

```




 用 FSDP 包装模型，模型将如下所示，我们可以看到模型已被包装在一个 FSDP 单元中。
或者，我们接下来将添加 fsdp_auto_wrap_policy并将讨论差异。






```
 FullyShardedDataParallel(
 (_fsdp_wrapped_module): FlattenParamsWrapper(
 (_fpw_module): Net(
 (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
 (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
 (dropout1): Dropout(p=0.25, inplace=False)
 (dropout2): Dropout(p=0.5, inplace=False)
 (fc1): Linear(in_features=9216, out_features=128, bias=True)
 (fc2): Linear(in_features=128, out_features=10, bias=True)
 )
 )
)

```




 以下是从 PyTorch Profiler 捕获的具有 4 个 GPU 的 g4dn.12.xlarge AWS EC2 实例上 FSDP MNIST 训练的峰值内存使用情况。




[![FSDP 峰值内存](https://pytorch.org/tutorials/_images/FSDP_memory.gif)](https://pytorch.org/tutorials/_images/FSDP_memory.gif)


 FSDP 峰值内存使用
  [¶](#id3“此图像的永久链接”)





 在 FSDP 中应用
 *fsdp_auto_wrap_policy* 
 否则，FSDP 会将整个模型放在一个 FSDP 单元中，这会降低计算效率和内存效率。
其工作方式是，假设您的模型包含 100 个线性层。如果您执行 FSDP(model)，则只会有一个 FSDP 单元包裹整个模型。
在这种情况下，allgather 将收集所有 100 个线性层的完整参数，因此不会’t 保存 CUDA用于参数分片的内存。
而且，所​​有 100 个线性层只有一次阻塞 allgather 调用，层与层之间不会出现通信和计算重叠。




 为了避免这种情况，您可以传入一个 fsdp_auto_wrap_policy，它将密封当前的 FSDP 单元，并在满足指定条件（例如大小限制）时自动启动一个新的 FSDP 单元。
这样您将拥有多个 FSDP 单元，并且一次只有一个 FSDP 单元需要收集完整参数。例如，假设您有 5 个 FSDP 单元，每个单元包含 20 个线性层。
然后，在前向中，第一个 FSDP 单元将收集前 20 个线性层的参数，进行计算，丢弃参数，然后继续进行下一个20 个线性层。因此，在任何时间点，每个等级仅具体化 20 个线性层而不是 100 个线性层的参数/梯度。




 为此，在 2.4 中我们定义了 auto_wrap_policy 并将其传递给 FSDP 包装器，在下面的示例中，my_auto_wrap_policy 定义了一个层可以由 FSDP 包装或分片如果该层中的参数数量大于 100。
如果该层中的参数数量小于 100，它将通过 FSDP 与其他小层一起包装。
找到最佳的自动包装策略具有挑战性，PyTorch将来会为此配置添加自动调整。如果没有自动调整工具，最好通过实验使用不同的自动换行策略来分析您的工作流程并找到最佳策略。






```
my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )
torch.cuda.set_device(rank)
model = Net().to(rank)

model = FSDP(model,
    fsdp_auto_wrap_policy=my_auto_wrap_policy)

```




 应用 fsdp_auto_wrap_policy，模型将如下所示：






```
 FullyShardedDataParallel(
(_fsdp_wrapped_module): FlattenParamsWrapper(
 (_fpw_module): Net(
 (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
 (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
 (dropout1): Dropout(p=0.25, inplace=False)
 (dropout2): Dropout(p=0.5, inplace=False)
 (fc1): FullyShardedDataParallel(
 (_fsdp_wrapped_module): FlattenParamsWrapper(
 (_fpw_module): Linear(in_features=9216, out_features=128, bias=True)
 )
 )
 (fc2): Linear(in_features=128, out_features=10, bias=True)
 )
)

```






```
python FSDP_mnist.py

CUDA event elapsed time on training loop 41.89130859375sec

```




 以下是从 PyTorch Profiler 捕获的具有 4 个 GPU 的 g4dn.12.xlarge AWS EC2 实例上使用 MNIST 训练的 auto_wrap 策略的 FSDP 的峰值内存使用情况。
可以观察到峰值内存使用情况与未应用自动换行策略的 FSDP 相比，每台设备上的数据较小，从约 75 MB 到 66 MB。




[![FSDP 峰值内存](https://pytorch.org/tutorials/_images/FSDP_autowrap.gif)](https://pytorch.org/tutorials/_images/FSDP_autowrap.gif)


 使用 Auto_wrap 策略的 FSDP 峰值内存使用
  [¶](#id4“永久链接到此图像”)





*CPU 卸载* 
 ：如果模型非常大，即使使用 FSDP 也无法’ 适合 GPU，那么 CPU 卸载在这里会很有帮助。




 目前仅支持参数和梯度 CPU 卸载。可以通过传入 cpu_offload=CPUOffload(offload_params=True) 来启用。




 请注意，这当前隐式启用了将梯度卸载到 CPU 的功能，以便参数和梯度能够在同一设备上与优化器一起使用。此 API 可能会发生变化。默认值为“无”，在这种情况下不会进行卸载。




 由于频繁地将张量从主机复制到设备，使用此功能可能会大大减慢训练速度，但它可以帮助提高内存效率并训练更大规模的模型。




 在 2.4 中我们只是将其添加到 FSDP 包装器






```
model = FSDP(model,
    fsdp_auto_wrap_policy=my_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True))

```




 与 DDP 相比，如果在 2.4 中我们只是将模型正常包装在 DPP 中，将更改保存在 “DDP_mnist.py” 中。






```
model = Net().to(rank)
model = DDP(model)

```






```
python DDP_mnist.py

CUDA event elapsed time on training loop 39.77766015625sec

```




 以下是从 PyTorch 分析器捕获的具有 4 个 GPU 的 g4dn.12.xlarge AWS EC2 实例上 DDP MNIST 训练的峰值内存使用量。




[![FSDP 峰值内存](https://pytorch.org/tutorials/_images/DDP_memory.gif)](https://pytorch.org/tutorials/_images/DDP_memory.gif)


 使用 Auto_wrap 策略的 DDP 峰值内存使用
  [¶](#id5 "永久链接到此图像")





 考虑到我们在这里定义的玩具示例和微型 MNIST 模型，我们可以观察到 DDP 和 FSDP 的峰值内存使用量之间的差异。
在 DDP 中，每个进程都保存模型的副本，因此与对模型进行分片的 FSDP 相比，内存占用更高。模型参数、优化器状态和 DDP 等级上的梯度。
使用具有 auto_wrap 策略的 FSDP 的峰值内存使用量最低，其次是 FSDP 和 DDP。




 另外，从时间角度来看，考虑到小模型并在单台机器上运行训练，无论是否使用 auto_wrap 策略，FSDP 的执行速度几乎与 DDP 一样快。
这个示例并不代表大多数实际应用程序，因为DDP和FSDP的详细分析和比较请参考此
[博文](https://pytorch.medium.com/6c8da2be180d)
.









