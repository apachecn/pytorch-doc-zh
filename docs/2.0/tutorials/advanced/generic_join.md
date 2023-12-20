


# 使用连接上下文管理器进行不均匀输入的分布式训练 [¶](#distributed-training-with-uneven-inputs-using-the-join-context-manager "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/generic_join>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/generic_join.html>




**作者** 
 :
 [Andrew Gu](https://github.com/andwgu)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/advanced_source/generic_join.rst) 
.






 没有10



`Join`
 在 PyTorch 1.10 中作为原型功能引入。此
API 可能会更改。





 在本教程中，您将看到：



* 上下文管理器概述。
* 如何使用上下文管理器的示例
 n `DistributedDataParallel`
.
* 如何将上下文管理器与
 `DistributedDataParallel`
 和
 `ZeroRedundancyOptimizer`一起使用的示例
.
* 将关键字参数传递到上下文的示例
* 深入了解
 [加入](https://pytorch.org/docs/master/distributed.algorithms.join.html) 
 上下文管理器的工作原理。
* 展示如何使玩具类与上下文管理器兼容。


## 要求 [¶](#requirements "永久链接到此标题")



* PyTorch 1.10+
* [分布式数据并行入门](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
* [使用 ZeroRedundancyOptimizer 的分片优化器状态](https://pytorch.org /tutorials/recipes/zero_redundancy_optimizer.html）





## 什么是
 `Join`
 ？ [¶](#what-is-join "永久链接到此标题")




 在
 [分布式数据并行入门 - 基本用例](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case) 
 中，您看到
的一般框架使用 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 执行数据并行训练。这隐式地在每个向后传递中安排全归约，以同步跨等级的梯度。这种[集体通信](https://pytorch.org/docs/stable/distributed.html)需要进程组中所有等级的参与，因此如果一个等级的输入较少，那么其他等级的
将挂起或出错（取决于后端）。更一般而言，对于执行每次迭代同步集体通信的任何类，
此问题都会持续存在。




`Join`
 是一个上下文管理器，用于每个等级的训练循环，
以促进输入不均匀的训练。上下文管理器允许尽早耗尽其输入的队列（即尽早加入*加入）来隐藏尚未加入的群体所执行的集体通信。隐藏通信的方式
由挂钩指定。





## 使用
 `Join`
 和
 `DistributedDataParallel` [¶](#using-join-with-distributeddataparallel "永久链接到此标题")




 PyTorch’s
 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
 开箱即用
 `Join`
 上下文管理器。下面是一个用法示例：






```
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP

BACKEND = "nccl"
WORLD_SIZE = 2
NUM_INPUTS = 5

def worker(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

    model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
    # Rank 1 gets one more input than rank 0
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]

    num_inputs = 0
    with Join([model]):
        for input in inputs:
            num_inputs += 1
            loss = model(input).sum()
            loss.backward()

    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")

def main():
    mp.spawn(worker, nprocs=WORLD_SIZE, join=True)

if __name__ == "__main__":
    main()

```




 这会产生以下输出（其中 
 `print()`
 的排名 0 和
排名 1 可以任意排序）：






```
Rank 0 has exhausted all 5 of its inputs!
Rank 1 has exhausted all 6 of its inputs!

```





 没有10



[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
 提供了自己的
 [join()](https://pytorch.org /docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.join) 
 上下文管理器
在引入此通用
 `Join`
 上下文管理器之前。在上面的示例中，使用
 `with
 

 Join([model]):`
 相当于使用
 `with
 

 model.join():`
 。现有的
 `DistributedDataParallel.join()` 的一个限制是它不允许多个
参与类，例如
 `DistributedDataParallel`
 和
 [ZeroRedundancyOptimizer](https://pytorch.org/docs/stable/distributed.optim.html) 
 一起。





## 使用
 `Join`
 与
 `DistributedDataParallel`
 和
 `ZeroRedundancyOptimizer` [¶](#using-join-with-distributeddataparallel-and-zeroredundancyoptimizer "此标题的永久链接")




 `Join`
 上下文管理器不仅适用于单个类，也适用于
多个类。 PyTorch’s
 `ZeroRedundancyOptimizer`
 也与上下文管理器
兼容，因此在这里，我们研究如何修改
前面的示例以同时使用
 `DistributedDataParallel`
 和
 `ZeroRedundancyOptimizer `
 :






```
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.optim import Adam

def worker(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

    model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
    optim = ZeRO(model.parameters(), Adam, lr=0.01)
    # Rank 1 gets one more input than rank 0
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]

    num_inputs = 0
    # Pass both `model` and `optim` into `Join()`
    with Join([model, optim]):
        for input in inputs:
            num_inputs += 1
            loss = model(input).sum()
            loss.backward()
            optim.step()

    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")

```




 这将产生与之前相同的输出。显着的变化是
另外将
 `ZeroRedundancyOptimizer`
 实例传入
 
 `Join()`
 。





## 传递关键字参数 [¶](#passing-keyword-arguments "永久链接到此标题")




 类可以提供关键字参数，以在运行时修改其在上下文管理器中的行为。例如，
 `DistributedDataParallel`
 提供了
 `divide_by_initial_world_size`
 ，它确定梯度是除以初始世界大小还是除以有效大小世界规模（即未加入队伍的数量）。此类关键字参数可以直接传递到
上下文管理器。






```
with Join([model, optim], divide_by_initial_world_size=False):
    for input in inputs:
        ...

```





!!! warning "警告"

    传递到上下文管理器的关键字参数在所有参与的类之间共享。这不应该是一个限制，因为我们不希望出现多个“Joinable”需要同一参数的不同设置的情况。尽管如此，这一点值得牢记。





## `Join` 如何工作？ [¶](#how-does-join-work "永久链接到此标题")




 现在我们已经了解了如何使用
 `Join` 上下文管理器的一些初步示例，让我们更深入地研究它的工作原理。这将让您更深入地了解它所提供的全部功能，并为您自己的自定义类的兼容做好准备。在这里，我们将介绍
 `Join`
 类以及
支持类
 `Joinable`
 和
 `JoinHook`
 。




### `Joinable` [¶](#joinable "永久链接到此标题")



 首先，与
 `Join`
 上下文管理器兼容的类必须继承
抽象基类
 
 `Joinable`
 。特别是，
 `Joinable`
 必须
实现：



* `join_hook(self,
 

 **kwargs)
 

 ->
 

 JoinHook`



 这将返回
 `Joinable`
 的
 `JoinHook`
 实例，确定已加入的进程
应如何隐藏
由
 `Joinable`
 执行的每迭代集体通信。



* `join_device(self)
 

 ->
 

 torch.device`



 这将返回一个设备，供
 `Join`
 上下文管理器用于执行
集体通信，例如
 `torch.device("cuda:0")`
 或
 `torch.device (“CPU”)`
.



* `join_process_group(self)
 

 ->
 

 ProcessGroup`



 这将返回由
 `Join` 上下文管理器使用的进程组，

执行集体通信。




 特别是，
 `join_device`
 和
 `join_process_group`
 是必需的
属性，以确保上下文管理器可以
安排已加入和未加入的
之间的集体通信。加入的流程。一种用法是使用 all-reduce 计算
每次迭代中未加入进程的数量。
另一种用法是实现
 `throw_on_early_termination=True` 所需的机制
 ，我们将在下面解释。




`DistributedDataParallel`
 和
 `ZeroRedundancyOptimizer`
 已经继承
自`Joinable`
 并实现了上述方法，这就是为什么我们可以
在前面的示例中直接使用它们。




`Joinable`
 类应确保调用
 `Joinable`
 构造函数
，因为它初始化
 `JoinConfig`
 实例，该实例由上下文管理器在内部使用
以确保正确性。这将作为字段保存在每个
 `Joinable`
 中
 `_join_config`
 。





### `JoinHook` [¶](#joinhook "此标题的永久链接")



 接下来，让我们分解
 `JoinHook`
 类。 
 `JoinHook`
 为上下文管理器提供了两个
入口点:



* `main_hook(self)
 

 ->
 

 无`



 当存在尚未加入的排名时，每个加入的排名都会重复调用此挂钩。它的目的是在每次训练迭代中（例如，在一次前向传递、后向传递和优化器步骤中）
 隐藏由
“Joinable”
 执行的集体通信。



* `post_hook(self,
 

 is_last_joiner:
 

 bool)
 

 ->
 

 None`



 一旦所有队伍加入，就会调用这个钩子。它被传递一个额外的
 `bool`
 参数
 `is_last_joiner`
 ，它指示该排名是否是最后加入的
之一。该参数对于同步可能有用。




 为了给出这些钩子的具体示例，提供的
 `ZeroRedundancyOptimizer`
 主钩子按正常情况执行优化器步骤
因为加入的rank仍负责更新和同步其
参数分片，并且提供的
 `DistributedDataParallel`
 后挂钩
从最后加入的队列之一广播最终更新的模型，以确保
它在所有队列中都是相同的。





### `加入` [¶](#join "此标题的永久链接")



 最后，让我们检查一下它们如何适应
 `Join`
 类本身。



* `__init__(self,
 

 可连接:
 

 列表[可连接],
 

 启用:
 

 bool
 

 =
 

 True,
 

 throw_on_early_termination:
 

 bool
 

 =
 \ n
 错误)`


正如我们在前面的示例中看到的，构造函数接受参与训练循环的“Joinable”列表。这些应该是
在每次迭代中执行集体通信的类。




`enable`
 是一个
 `bool`
，如果您知道
不会有不均匀的输入，则可以将其设置为
 `False`
，在这种情况下，上下文管理器会变得空洞
类似到
 `contextlib.nullcontext()`
 。这也可能会禁用参与
“Joinable”
中与连接相关的
计算。




`throw_on_early_termination`
 是一个
 `bool`
，可以设置为
 `True`
，
让每个等级在输入不均匀时引发异常被检测到。
这对于不符合上下文管理器’s
要求的情况很有用，最常见的是当存在来自可能任意交错的不同类的集体通信
时，例如当使用\ n `DistributedDataParallel`
 具有具有
 `SyncBatchNorm`
 层的模型。在这种情况下，此参数应设置为“True”，以便应用程序逻辑可以捕获异常并确定如何继续。



* 核心逻辑出现在
 `__exit__()`
方法中，当存在
未连接的rank时，该方法会循环，调用每个
 `Joinable`
 ‘s 主钩子，
然后一旦所有队伍都加入，调用他们的后钩子。主挂钩和后挂钩均按照传入“Joinable”的顺序进行迭代。
* 上下文管理器需要来自非联接进程的检测信号。因此，
每个
 `Joinable`
 类应该在其每次迭代集体通信之前调用
 
 `Join.notify_join_context()`。上下文管理器将确保只有第一个传入的“Joinable”实际发送
心跳。




!!! warning "警告"

    正如上面提到的
 `throw_on_early_termination`
 ，
 `Join` 上下文管理器与
类的某些组合不兼容。 
 `Joinable`
 ‘s
 `JoinHook`
 必须是可序列化的，因为每个钩子在继续下一个钩子之前都已完全执行。换句话说，两个钩子不能重叠。
此外，目前，主钩子和后钩子都以相同的确定性顺序进行迭代。如果这看起来
是一个主要限制，我们可能会修改 API 以允许
自定义排序。








## 使玩具类与
 `Join` 一起工作 [¶](#making-a-toy-class-work-with-join "永久链接到此标题")




 由于上一节介绍了几个概念，让我们通过一个玩具示例来实际看看它们。在这里，我们将实现一个类，用于计算在其排名加入之前在所有排名中看到的
输入数量。这
应该提供一个基本概念，说明如何使自己的类与
“Join”上下文管理器兼容。




 具体来说，以下代码让每个等级打印出 (1) 在加入之前看到的
所有等级中
输入的数量，以及 (2) 所有等级中
输入的总数。






```
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join, Joinable, JoinHook

BACKEND = "nccl"
WORLD_SIZE = 2
NUM_INPUTS = 5

class CounterJoinHook(JoinHook):
 r"""
 Join hook for :class:`Counter`.

 Arguments:
 counter (Counter): the :class:`Counter` object using this hook.
 sync_max_count (bool): whether to sync the max count once all ranks
 join.
 """
    def __init__(
        self,
        counter,
        sync_max_count
    ):
        self.counter = counter
        self.sync_max_count = sync_max_count

    def main_hook(self):
 r"""
 Shadows the counter's all-reduce by all-reducing a dim-1 zero tensor.
 """
        t = torch.zeros(1, device=self.counter.device)
        dist.all_reduce(t)

    def post_hook(self, is_last_joiner: bool):
 r"""
 Synchronizes the max count across all :class:`Counter` s if
 ``sync_max_count=True``.
 """
        if not self.sync_max_count:
            return
        rank = dist.get_rank(self.counter.process_group)
        common_rank = self.counter.find_common_rank(rank, is_last_joiner)
        if rank == common_rank:
            self.counter.max_count = self.counter.count.detach().clone()
        dist.broadcast(self.counter.max_count, src=common_rank)

class Counter(Joinable):
 r"""
 Example :class:`Joinable` that counts the number of training iterations
 that it participates in.
 """
    def __init__(self, device, process_group):
        super(Counter, self).__init__()
        self.device = device
        self.process_group = process_group
        self.count = torch.tensor([0], device=device).float()
        self.max_count = torch.tensor([0], device=device).float()

    def __call__(self):
 r"""
 Counts the number of inputs processed on this iteration by all ranks
 by all-reducing a dim-1 one tensor; increments its own internal count.
 """
        Join.notify_join_context(self)
        t = torch.ones(1, device=self.device).float()
        dist.all_reduce(t)
        self.count += t

    def join_hook(self, **kwargs) -> JoinHook:
 r"""
 Return a join hook that shadows the all-reduce in :meth:`__call__`.

 This join hook supports the following keyword arguments:
 sync_max_count (bool, optional): whether to synchronize the maximum
 count across all ranks once all ranks join; default is ``False``.
 """
        sync_max_count = kwargs.get("sync_max_count", False)
        return CounterJoinHook(self, sync_max_count)

    @property
    def join_device(self) -> torch.device:
        return self.device

    @property
    def join_process_group(self):
        return self.process_group

    def find_common_rank(self, rank, to_consider):
 r"""
 Returns the max rank of the ones to consider over the process group.
 """
        common_rank = torch.tensor([rank if to_consider else -1], device=self.device)
        dist.all_reduce(common_rank, op=dist.ReduceOp.MAX, group=self.process_group)
        common_rank = common_rank.item()
        return common_rank

def worker(rank):
    assert torch.cuda.device_count() >= WORLD_SIZE
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)

    counter = Counter(torch.device(f"cuda:{rank}"), dist.group.WORLD)
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]

    with Join([counter], sync_max_count=True):
        for _ in inputs:
            counter()

    print(f"{int(counter.count.item())} inputs processed before rank {rank} joined!")
    print(f"{int(counter.max_count.item())} inputs processed across all ranks!")

def main():
    mp.spawn(worker, nprocs=WORLD_SIZE, join=True)

if __name__ == "__main__":
    main()

```




 由于排名 0 看到 5 个输入，排名 1 看到 6 个输入，因此产生输出：






```
10 inputs processed before rank 0 joined!
11 inputs processed across all ranks!
11 inputs processed before rank 1 joined!
11 inputs processed across all ranks!

```




 需要强调的一些要点：



* A
 `Counter`
 实例每次迭代执行一次全归约，因此主钩子也执行一次全归约来隐藏它。
* 
 `Counter`
 类创建了一个在其
 `__call__()`
 方法的
 开头调用
 `Join.notify_join_context()`
 方法，因为这是一个位于其每迭代集体通信（即其全归约）之前。
* `is_last_joiner`
 参数用于确定后挂钩中的
广播源。\ n* 我们将
 `sync_max_count`
 关键字参数传递给上下文管理器，
然后将其转发到
 `Counter`
 ‘s 连接挂钩。








