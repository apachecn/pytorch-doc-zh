# 分布式数据并行 [¶](#distributed-data-parallel "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/ddp>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/ddp.html>


!!! warning "警告"

    [`torch.nn.parallel.DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 的实现随着时间的推移而演变。本设计说明是根据 v1.4 的状态编写的。


[`torch.nn.parallel.DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") (DDP) 透明地执行分布式数据并行训练。本页描述了它的工作原理并揭示了实现细节。


## 示例 [¶](#example "此标题的永久链接")


 让我们从一个简单的 [`torch.nn.parallel.DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 开始例子。此示例使用 [`torch.nn.Linear`](../generated/torch.nn.Linear.html#torch.nn.Linear "torch.nn.Linear") 作为本地模型，使用 DDP 进行包装，并且然后在 DDP 模型上运行一次前向传递、一次反向传递和优化器步骤。之后，本地模型上的参数将被更新，并且不同进程上的所有模型应该完全相同。


```
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()

```


 DDP 与 TorchDynamo 配合使用。与 TorchDynamo 一起使用时，在编译模型之前应用 DDP 模型包装器，以便 torchdynamo 可以基于 DDP 存储桶大小应用“DDPOptimizer”(图中断优化)。 (有关更多信息，请参阅 [TorchDynamo DDPOptimizer](./ddp.html#torchdynamo-ddpoptimizer)。)


 TorchDynamo 对 DDP 的支持当前需要设置 static_graph=False ，这是由于图跟踪过程和 DDP 观察其模块上发生的操作的机制之间的相互作用，但这应该最终得到解决。


```
ddp_model = DDP(model, device_ids=[rank])
ddp_model = torch.compile(ddp_model)

```


## 内部设计[¶](#internal-design "此标题的永久链接")


 本节揭示了它在 [`torch.nn.parallel.DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel")，深入研究一次迭代中每个步骤的细节。



* **先决条件** : DDP 依赖于 c10d `ProcessGroup` 进行通信。因此，应用程序必须在构造 DDP 之前创建 `ProcessGroup` 实例。
* **构造** : DDP 构造函数引用本地模块，并广播 ` state_dict()` 从排名为 0 的进程到组中的所有其他进程，以确保所有模型副本从完全相同的状态开始。然后，每个 DDP 进程都会创建一个本地 `Reducer` ，稍后它将在向后传递期间处理梯度同步。为了提高通信效率，Reducer将参数梯度组织到桶中，并一次减少一个桶。可以通过在 DDP 构造函数中设置 Bucket_cap_mb 参数来配置存储桶大小。从参数梯度到桶的映射是在构建时根据桶大小限制和参数大小确定的。模型参数按照给定模型中“Model.parameters()”的相反顺序(大致)分配到存储桶中。使用反向顺序的原因是因为 DDP 期望梯度在反向传递期间以大约该顺序准备好。下图显示了一个示例。请注意，“grad0”和“grad1”位于“bucket1”中，其他两个梯度位于“bucket0”中。当然，这个假设可能并不总是正确的，当这种情况发生时，它可能会损害 DDP 后向速度，因为“Reducer”无法尽早启动通信。除了分桶之外，“Reducer”还在构建过程中注册 autograd 钩子，一个每个参数的钩子。当梯度准备好时，这些钩子将在后向传递过程中被触发。
* **前向传递** ：DDP 获取输入并将其传递到本地模型，然后分析本地模型的输出(如果“find_unused“) _parameters 设置为 `True` 。此模式允许在模型的子图上向后运行，DDP 通过从模型输出遍历 autograd 图并将所有未使用的参数标记为准备减少来找出哪些参数参与向后传递。在向后传递期间，Reducer只会等待未准备好的参数，但它仍然会减少所有桶。目前，将参数梯度标记为就绪并不能帮助 DDP 跳过存储桶，但可以防止 DDP 在向后传递过程中永远等待缺失的梯度。请注意，遍历 autograd 图会带来额外的开销，因此应用程序只应在必要时将 `find_unused_parameters` 设置为 `True`。
* **向后传递** ：在损失上直接调用 `backward()` 函数`Tensor` ，不受 DDP 的控制，DDP 使用在构造时注册的 autograd hook 来触发梯度同步。当一个梯度准备好时，该梯度累加器上相应的 DDP 钩子将触发，然后 DDP 将将该参数梯度标记为已准备好减少。当一个存储桶中的梯度全部准备好时，“Reducer”会在该存储桶上启动异步“allreduce”，以计算所有进程的梯度平均值。当所有桶准备就绪时，Reducer 将阻塞等待所有 allreduce 操作完成。完成后，平均梯度将写入所有参数的 param.grad 字段。所以在后向传递之后，不同DDP进程中相同对应参数的梯度场应该是相同的。
* **优化器步骤**：从优化器的角度来看，它正在优化局部模型。所有 DDP 进程上的模型副本都可以保持同步，因为它们都从相同的状态开始，并且在每次迭代中具有相同的平均梯度。


[![ddp_grad_sync.png](https://user-images.githubusercontent.com/16999635/72401724-d296d880-371a-11ea-90ab-737f86543df9.png)](https://user-images.githubusercontent.com/16999635/72401724-d296d880-371a-11ea-90ab-737f86543df9.png)


!!! note "笔记"

    DDP 要求所有进程上的“Reducer”实例以完全相同的顺序调用“allreduce”，这是通过始终按存储桶索引顺序而不是实际的存储桶就绪顺序运行“allreduce”来完成的。跨进程的“allreduce”顺序不匹配可能会导致错误的结果或 DDP 向后挂起。


## 实现[¶](#implementation "永久链接到此标题")


 以下是 DDP 实现组件的指针。堆叠图显示了代码的结构。


### ProcessGroup [¶](#processgroup "此标题的永久链接")



* [ProcessGroup.hpp](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/lib/c10d/ProcessGroup.hpp) ：包含所有进程组实现的抽象 API。 `c10d` 库提供了 3 个开箱即用的实现，即 ProcessGroupGloo 、 ProcessGroupNCCL 和 ProcessGroupMPI 。 `DistributedDataParallel` 在初始化期间使用 `ProcessGroup::broadcast()` 将模型状态从排名为 0 的进程发送到其他进程，并使用 `ProcessGroup::allreduce()` 来求和梯度。
* [Store.hpp](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/lib/c10d/Store.hpp)：协助进程组实例的集合服务找到彼此。


### DistributedDataParallel [¶](#distributeddataparallel "此标题的永久链接")



* [distributed.py](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/nn/parallel/distributed.py) ：是 DDP 的 Python 入口点。它实现了调用 C++ 库的“nn.parallel.DistributedDataParallel”模块的初始化步骤和“forward”函数。当一个 DDP 进程在多个设备上工作时，它的 `_sync_param` 函数执行进程内参数同步，并且还将排名 0 的进程的模型缓冲区广播到所有其他进程。进程间参数同步发生在 `Reducer.cpp`.
* [comm.h](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/csrc/distributed/c10d/comm.h) : 实现合并广播辅助函数，该函数在初始化期间被调用以广播模型状态，并在正向传递之前同步模型缓冲区。
* [reducer.h](https://github.com/pytorch/pytorch/blob/v1.7.0/torch/csrc/distributed/c10d/reducer.h) ：提供后向传递中梯度同步的核心实现。它具有三个入口点函数：



+ `Reducer` ：在 `distributed.py` 中调用构造函数，将 `Reducer::autograd_hook()` 注册到梯度累加器。 
+ 当渐变准备就绪时，autograd_hook() 函数将由 autograd 引擎调用。 
+ `prepare_for_backward()` 在 `distributed.py` 中的 DDP 前向传递结束时调用。当 DDP 构造函数中的“find_unused_parameters”设置为“True”时，它会遍历 autograd 图来查找未使用的参数。


[![ddp_code.png](https://user-images.githubusercontent.com/16999635/72313120-4e7c1c80-3658-11ea-9c6d-44336b2daeac.png)](https://user-images.githubusercontent.com/16999635/72313120-4e7c1c80-3658-11ea-9c6d-44336b2daeac.png)


### TorchDynamo DDPOptimizer [¶](#id1 "此标题的永久链接")


 DDP 的性能优势来自于向后计算期间将 allreduce 集合与计算重叠。当与 TorchDynamo 一起使用来编译整个前向和整个后向图时，AotAutograd 可以防止这种重叠，因为 allreduce 操作是在整个优化后向计算完成后由 autograd 钩子启动的。


 TorchDynamo 的 DDPOptimizer 通过在向后过程中在 DDP allreduce 存储桶的逻辑边界处打破前向图来提供帮助。注意：目标是在向后的过程中打破图，最简单的实现是打破向前的图，然后在每个部分上调用 AotAutograd 和编译。这允许 DDP 的 allreduce 钩子在后向部分之间触发，并安排通信与计算重叠。


 请参阅[此博客文章](https://dev-discuss.pytorch.org/t/torchdynamo-update-9-making-ddp-work-with-torchdynamo/860/1)以获取更深入的解释和实验结果，或阅读 [torch/_dynamo/optimizations/distributed.py](https://github.com/pytorch/pytorch/blob/4908a12542798a3e8641faae6b74f068fdfc6778/torch/_dynamo/optimizations/distributed.py#L56) 中的文档和代码


 要调试 DDPOptimizer，请将 torch._dynamo.config.log_level 设置为 DEBUG(用于完整图形转储)或 INFO(用于有关存储桶边界的基本信息)。要禁用 DDPOptimizer，请设置 torch._dynamo.config.optimize_ddp=False 。在没有 DDPOptimizer 的情况下，DDP 和 TorchDynamo 仍应正常工作，但性能会下降。