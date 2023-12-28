


# 分布式数据并行入门 [¶](#getting-started-with-distributed-data-parallel "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/ddp_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>




**作者** 
 :
 [沉力](https://mrshenli.github.io/)




**编辑者** 
 :
 [Joe Zhu](https://github.com/gunandrose4u)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/intermediate_source/ddp_tutorial.rst) 
.





 先决条件:



* [PyTorch 分布式概述](../beginner/dist_overview.html)
* [DistributedDataParallel API 文档](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)\ n* [DistributedDataParallel 注释](https://pytorch.org/docs/master/notes/ddp.html)



[DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#module-torch.nn.parallel) 
 (DDP) 在模块级别实现数据并行，可以跨多台机器运行。使用 DDP 的应用程序应生成多个进程并
为每个进程创建一个 DDP 实例。 DDP 使用torch.distributed 包中的集体通信来同步梯度和缓冲区。更具体地说，DDP 为“model.parameters()”给出的每个参数注册 autograd 钩子，并且当在向后传递中计算相应的梯度时，该钩子将触发。然后，DDP 使用该信号触发跨进程的梯度同步。请参阅
 [DDP 设计说明](https://pytorch.org/docs/master/notes/ddp.html) 
 了解更多详细信息。




 建议使用 DDP 的方法是为每个模型副本生成一个进程，
其中模型副本可以跨越多个设备。 DDP 进程可以放置在同一台计算机上或跨计算机，但 GPU 设备不能跨进程共享。本教程从基本的 DDP 用例开始，
然后演示更高级的用例，包括检查点模型和
将 DDP 与模型并行相结合。





 注意




 本教程中的代码在 8-GPU 服务器上运行，但它可以轻松
推广到其他环境。





## `DataParallel`
 和 
 `DistributedDataParallel` 之间的比较 [¶](#comparison- Between-dataparallel-and-distributeddataparallel "永久链接到此标题")




 在我们深入讨论之前，让’s 澄清为什么尽管增加了复杂性，
你还是会考虑使用
 `DistributedDataParallel`
 而不是
 `DataParallel`
 :



* 首先，
 `DataParallel`
 是单进程、多线程，且仅适用于
单机，而
 `DistributedDataParallel`
 是多进程，适用于
单机和多机
机器训练。
“DataParallel”
 通常
低于
“DistributedDataParallel”
，即使在单台机器上也是如此，因为线程间的 GIL
争用、每次迭代复制模型以及分散输入和引入的额外
开销收集输出。
* 回想一下
 [之前的教程](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
 如果您的模型太大而无法适应单个 GPU，则必须使用
 **模型并行** 
 将其拆分到多个 GPU 上。
 `DistributedDataParallel`
 适用于
 **模型并行** 
 ;
 `DataParallel`
 目前不适用于。当 DDP 与模型并行结合时，每个 DDP 进程都将使用模型并行，所有进程共同使用数据并行。如果您的模型需要跨多台机器，或者您的用例不适合数据并行，
范式，请参阅
 [RPC API](https://pytorch.org/docs/stable/rpc.html) 
 以获取更通用的分布式训练支持。





## 基本用例 [¶](#basic-use-case "永久链接到此标题")




 要创建 DDP 模块，必须首先正确设置进程组。更多详细信息可以在
[使用 PyTorch 编写分布式应用程序](https://pytorch.org/tutorials/intermediate/dist_tuto.html) 中找到
 。






```
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
# "gloo",
# rank=rank,
# init_method=init_method,
# world_size=world_size)
# For TcpStore, same way as on Linux.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

```




 现在，让’s 创建一个玩具模块，用 DDP 包装它，并为其提供一些虚拟
输入数据。请注意，由于 DDP 在 DDP 构造函数中将模型状态从等级 0 进程广播到
所有其他进程，因此您无需担心
不同的 DDP 进程从不同的初始模型参数值开始。






```
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
 print(f"在rank {rank}上运行基本DDP示例。")
 setup(rank, world_size)

 # 创建模型并将其移动到GPU，id为rank
 model = ToyModel().to(rank)
 ddp_model = DDP(model, device_ids=[rank])

 loss_fn = nn.MSELoss()
 优化器 = optim.SGD(ddp_model.parameters(), lr=0.001)

 优化器.zero_grad()
 输出 = ddp_model(torch.randn( 20, 10))
 labels = torch.randn(20, 5).to(rank)
 loss_fn(outputs, labels).backward()
 optimizationr.step()

 cleanup( )


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

```


正如您所看到的，DDP 包装了较低级别的分布式通信细节，并提供了干净的 API，就像本地模型一样。梯度同步通信发生在向后传递期间，并与向后计算重叠。当 `backward()` 返回时，
 `param.grad` 已经包含同步的梯度tensor。对于基本用例，DDP 仅
需要更多几个 LoC 来设置进程组。将 DDP 应用于更
高级用例时，需要小心一些注意事项。





## 倾斜的处理速度 [¶](#skewed-processing-speeds "永久链接到此标题")




 在 DDP 中，构造函数、前向传递和后向传递是
分布式同步点。不同的进程应启动
相同数量的同步，并以相同的顺序
到达这些同步点，并在大致相同的时间进入每个同步点。
否则，快速进程可能会提早到达，并在等待
掉队进程时超时。因此，用户有责任平衡进程之间的工作负载分配。有时，由于网络延迟、资源争用或不可预测的工作负载峰值，处理速度的偏差是不可避免的。为了避免在这些情况下超时，请确保在调用 [init_process_group](https://pytorch.org/docs/stable) 时传递足够大的
 `timeout`
 值/distributed.html#torch.distributed.init_process_group) 
.





## 保存和加载检查点 [¶](#save-and-load-checkpoints "永久链接到此标题")




 在训练和从检查点恢复期间，
 通常使用
 `torch.save` 和 
 `torch.load` 来检查模块。有关更多详细信息，请参阅
 [保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
。使用 DDP 时，一种优化是将模型保存在
仅一个进程中，然后将其加载到所有进程，从而减少写入开销。
这是正确的，因为所有进程都从相同的参数开始，并且
梯度在向后传递中同步，并且因此优化器应该保持
将参数设置为相同的值。如果您使用此优化，请确保在保存完成之前没有进程启动
加载。此外，加载模块时，您需要提供适当的“map_location”参数，以防止进程进入其他’ 设备。如果
 `map_location`
 缺失，
 `torch.load`
 将首先将模块加载到 CPU，然后将每个
参数复制到保存的位置，这将导致
同一台机器使用同一组设备。如需更高级的故障恢复
和弹性支持，请参阅
 [TorchElastic](https://pytorch.org/elastic)
 。






```
def demo_checkpoint(rank, world_size):
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])


    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Not necessary to use a dist.barrier() to guard the file deletion below
    # as the AllReduce ops in the backward pass of DDP already served as
    # a synchronization.

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()

```





## 将 DDP 与模型并行性相结合 [¶](#combining-ddp-with-model-parallelism "永久链接到此标题")




 DDP 还适用于多 GPU 型号。当训练具有大量数据的大型模型时，DDP 封装多 GPU 模型尤其有用。






```
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

```




 将多 GPU 模型传递到 DDP 时，
 `device_ids`
 和
 `output_device`
 不得设置。输入和输出数据将通过
 应用程序或模型
 `forward()`
 方法放置在适当的设备中。






```
def demo_model_parallel(rank, world_size):
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    run_demo(demo_checkpoint, world_size)
    world_size = n_gpus//2
    run_demo(demo_model_parallel, world_size)

```





## 使用 torch.distributed.run/torchrun 初始化 DDP [¶](#initialize-ddp-with-torch-distributed-run-torchrun "永久链接到此标题")




 我们可以利用 PyTorch Elastic 来简化 DDP 代码并更轻松地初始化作业。
让’s 仍然使用 Toymodel 示例并创建一个名为
 `elastic_ddp.py`
 的文件。 






```
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    demo_basic()

```




 然后可以在所有节点上运行
 [torch elastic/torchrun](https://pytorch.org/docs/stable/elastic/quickstart.html) 
 命令来初始化上面创建的 DDP 作业:






```
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py

```




 我们在两台主机上运行 DDP 脚本，每台主机运行 8 个进程，也就是说，我们在 16 个 GPU 上运行它。请注意，
 `$MASTER_ADDR`
 在所有节点上必须相同。




 这里 torchrun 将启动 8 个进程，并在其启动的节点上的每个进程上调用
 `elastic_ddp.py`
，但用户还需要应用集群
管理工具(例如 slurm)来实际运行此命令在 2 个节点上。




 例如，在启用 SLURM 的集群上，我们可以编写一个脚本来运行上面的命令
并将
 `MASTER_ADDR`
 设置为：






```
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

```




 然后我们可以使用 SLURM 命令运行此脚本：
 `srun
 

 --nodes=2
 

./torchrun_script.sh`
.\当然，这只是一个例子；您可以选择自己的集群调度工具
来启动 torchrun 作业。




 有关 Elastic run 的更多信息，可以查看此
 [快速入门文档](https://pytorch.org/docs/stable/elastic/quickstart.html) 
 了解更多信息。









