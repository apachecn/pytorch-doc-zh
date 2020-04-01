# 分布式数据并行(DDP)入门

> **作者**：[Shen Li](https://mrshenli.github.io/)
>
> **译者**：[Hamish](https://sherlockbear.github.io)
>
> **校验**：[Hamish](https://sherlockbear.github.io)

[DistributedDataParallel](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html)(DDP)在模块级别实现数据并行性。它使用[torch.distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html)包中的通信集合体来同步梯度，参数和缓冲区。并行性在流程内和跨流程均可用。在一个过程中，DDP将输入模块复制到device_ids中指定的设备，相应地沿批处理维度分散输入，并将输出收集到output_device，这与[DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)相似。在整个过程中，DDP在正向传递中插入必要的参数同步，在反向传递中插入梯度同步。用户可以将进程映射到可用资源，只要进程不共享GPU设备即可。推荐的方法(通常是最快的方法）是为每个模块副本创建一个过程，即在一个过程中不进行任何模块复制。本教程中的代码在8-GPU服务器上运行，但可以轻松地推广到其他环境。

## `DataParallel`和`DistributedDataParallel`之间的比较

在深入研究之前，让我们澄清一下为什么，尽管增加了复杂性，您还是会考虑使用`DistributedDataParallel`而不是`DataParallel`：

- 首先，回想一下[之前的教程](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)，如果模型太大，无法被单个GPU容纳，则必须使用**模型并行化**将其拆分至多个GPU。`DistributedDataParallel`可以与**模型并行化**一起工作；`DataParallel`此时不工作。
- `DataParallel`是单进程、多线程的，并且只在一台机器上工作；而`DistributedDataParallel`是多进程的，可用于单机和多机训练。因此，即使对于单机训练，数据足够小，可以放在一台机器上，`DistributedDataParallel`也会比`DataParallel`更快。`DistributedDataParallel`还可以预先复制模型，而不是在每次迭代时复制模型，从而可以避免全局解释器锁定。
- 如果您的数据太大，无法在一台机器上容纳，**并且**您的模型也太大，无法在单个GPU上容纳，则可以将模型并行化(跨多个GPU拆分单个模型）与`DistributedDataParallel`结合起来。在这种机制下，每个`DistributedDataParallel`进程都可以使用模型并行化，同时所有进程都可以使用数据并行。

## 基本用例

要创建DDP模块，请首先正确设置进程组。更多细节可以在[使用PyTorch编写分布式应用程序](https://pytorch.org/tutorials/intermediate/dist_tuto.html)中找到。


```python
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()
```

现在，让我们创建一个玩具模块，用DDP包装它，并用一些虚拟输入数据给它输入。请注意，如果训练是从随机参数开始的，您可能需要确保所有DDP进程使用相同的初始值。否则，全局梯度同步将没有意义。

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

如您所见，DDP包装了较低级别的分布式通信细节，并提供了一个干净的API，就好像它是一个本地模型一样。对于基本用例，DDP只需要几个loc来设置流程组。在将DDP应用于更高级的用例时，需要注意一些注意事项。

## 不均衡的处理速度

在DDP中，构造函数、前向方法和输出的微分是分布式同步点。不同的进程将以相同的顺序到达同步点，并在大致相同的时间进入每个同步点。否则，快速进程可能会提前到达，并在等待散乱的进程时超时。因此，用户需要负责跨进程平衡工作负载的分配。有时，由于网络延迟、资源竞争、不可预测的工作量高峰，不均衡的处理速度是不可避免的。要避免在这些情况下超时，请确保在调用[init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)时传递足够大的`timeout`值。

## 保存和载入检查点

在训练过程中，经常使用`torch.save`和`torch.load`为模块创建检查点，以及从检查点恢复。有关的详细信息，请参见[保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)。在使用DDP时，一种优化方法是只在一个进程中保存模型，然后将其加载到所有进程中，从而减少写开销。这是正确的，因为所有进程都是从相同的参数开始的，并且梯度在反向过程中是同步的，因此优化器应该将参数设置为相同的值。如果使用这种优化方法，请确保在保存完成之前，所有进程都不会开始加载。此外，加载模块时，需要提供适当的`map_location`参数，以防止进程进入其他设备。如果缺少`map_location`，`torch.load`将首先将模块加载到CPU，然后将每个参数复制到其保存的位置，这将导致同一台计算机上的所有进程使用同一组设备。

```python
def demo_checkpoint(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

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
    rank0_devices = [x - rank * len(device_ids) for x in device_ids]
    device_pairs = zip(rank0_devices, device_ids)
    map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```

## 结合DDP与模型并行化

DDP也适用于多GPU模型，但不支持进程内的复制。您需要为每个模块副本创建一个进程，这通常会比每个进程创建多个副本带来更好的性能。当使用大量数据训练大型模型时，DDP包装多GPU模型尤其有用。使用此功能时，需要小心地实现多GPU模型，以避免硬编码设备，因为不同的模型副本将被放置到不同的设备上。

```python
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

将多GPU模型传递给DDP时，**不能**设置`device_ids`和`output_device`。输入和输出数据将由应用程序或模型`forward()`方法放置在适当的设备中。

```python
def demo_model_parallel(rank, world_size):
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
    run_demo(demo_basic, 2)
    run_demo(demo_checkpoint, 2)

    if torch.cuda.device_count() >= 8:
        run_demo(demo_model_parallel, 4)
```
