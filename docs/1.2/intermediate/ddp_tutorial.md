# 2.入门与分布式数据并行

**作者** ：[沉莉](https://mrshenli.github.io/)

[ DistributedDataParallel
](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html)（DDP）实现在模块级数据并行性。它使用通信集体在[
torch.distributed
](https://pytorch.org/tutorials/intermediate/dist_tuto.html)包同步梯度，参数，和缓冲剂。并行可既是一个过程内和跨流程。内的方法，DDP复制输入模块在`
device_ids`，散射沿相应的批量尺寸的输入，并收集输出到`output_device指定装置
`，这类似于[数据并行](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)。跨进程，DDP插入在正向通行证和在向后穿过梯度同步必要的参数同步。它是由用户来映射进程可用的资源，只要工艺不共享GPU设备。推荐的（通常是最快）的方法是创建一个进程的每一个模块的副本，即，在一个进程中没有模块复制。本教程中的代码运行的8
GPU服务器上，但它可以被容易地推广到其他的环境中。

## 之间`数据并行 `和`DistributedDataParallel`比较

在我们深入，让我们澄清为什么尽管增加的复杂性，你会考虑使用`DistributedDataParallel`在`数据并行 `：

  * 首先，从[之前教程](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)如果你的模型是太大，无法在单GPU，您必须使用 **模型召回平行** 将其跨多个GPU分裂。 `DistributedDataParallel`与 **模型作品平行** ;此时`数据并行 `没有。
  * `数据并行 `是单进程，多线程，并且只能在单个机器上，而`DistributedDataParallel`是多进程和作品两个单和多机训练。因此，即使是单机训练，你的 **数据** 是足够小，适合在一台机器上，`DistributedDataParallel`预计比`[快HTG15]数据并行 `。 `DistributedDataParallel`也复制模型，而不是前期在每次迭代并得到全局解释器锁的方式进行。
  * 如果这两个数据是太大，无法一体机 **和** 你的模型是太大，无法在单GPU，您可以用`模式并行（分割跨越多GPU的单一模式）相结合 DistributedDataParallel`。在该机制下，各`DistributedDataParallel`过程可以使用模型平行，并且所有过程统称将使用数据并行。

## 基本用例

要创建DDP模块，首先设置进程组正常。更多详情可与PyTorch
[编写分布式应用程序被发现。](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

    
    
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
    
    

现在，让我们创建一个玩具模块，与DDP纸包起来，然后用一些虚拟的输入数据给它。请注意，如果从培训随机参数开始，您可能希望确保所有DDP进程使用相同的初始值。否则，全球同步的梯度将没有意义。

    
    
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
    
    

正如你所看到的，DDP包下级分布式通信的细节，并提供了一个干净的API，就好像它是一个局部模型。对于基本用例，DDP只需要几个LOCS建立进程组。当应用DDP到更高级的使用情况下，有一些注意事项需要注意事项。

## 歪斜的处理速度

在DDP，构造，方法前进，并输出的分化分布同步点。不同的过程，预计以相同的顺序到达同步点和在大致相同的时间输入每个同步点。否则，快速的过程可能会提前到达，超时的等待掉队。因此，用户是负责整个流程平衡工作负载分布。有时候，歪斜的处理速度是不可避免的，由于，例如，网络延迟，资源冲突，不可预测的工作负载高峰。为了避免在这些情况下超时，请确保您调用[
init_process_group
](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)当传递一个足够大的`
超时 `值。

## 保存和载入关卡

它通常使用`torch.save`和`torch.load
`训练期间检查站模块和从检查点恢复。参见[保存和载入模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)了解更多详情。当使用DDP，一个优化的模型保存只有一个进程，然后将其加载到所有进程，从而减少写入开销。这是正确的，因为所有处理从相同的参数开始和梯度在向后经过同步，并因此优化应保持设定参数相同的值。如果你使用这种优化，确保减排完成之前所有的进程不会开始加载。此外，加载模块时，需要提供适当的`
map_location`参数来防止处理踏进别人的设备。如果`map_location`丢失，`torch.load
`模块将首先加载到CPU，然后复制每个参数保存它，这将使用相同的一组设备导致在同一台机器上的所有进程。

    
    
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
    
    

## 结合DDP与型号并行

DDP还与多GPU模式，而是一个过程中的重复，不支持。您需要创建每个模块的副本，这通常会导致更好的性能相比，每个进程的多个副本一个过程。
DDP包装多GPU模式培养具有巨大的数据量较大的模型时特别有用。使用此功能时，多GPU模式需要谨慎实施，以避免硬编码的设备，因为不同型号的副本将被放置到不同的设备。

    
    
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
    
    

当通过一个多GPU模型到DDP，`device_ids`和`output_device`必须不被设置。输入和输出的数据将被应用程序或模型`
向前（） `方法被放置在适当的设备。

    
    
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
    
    

[Next ![](../_static/images/chevron-right-orange.svg)](dist_tuto.html "3.
Writing Distributed Applications with PyTorch")
[![](../_static/images/chevron-right-orange.svg)
Previous](model_parallel_tutorial.html "1. Model Parallel Best Practices")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * [HTG0 入门分布式数据并行
    * 数据并行 和`DistributedDataParallel`之间`比较`
    * 基本用例
    * 歪斜的处理速度
    * 保存和载入关卡
    * 与模型并行联合DDP 

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



