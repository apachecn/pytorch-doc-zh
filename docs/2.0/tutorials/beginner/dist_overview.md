


# PyTorch 分布式概述 [¶](#pytorch-distributed-overview "此标题的永久链接")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/dist_overview>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/dist_overview.html>




**作者** 
 :
 [沉力](https://mrshenli.github.io/)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/beginner_source/dist_overview.rst) 
.





 这是 
 `torch.distributed` 包的概述页面。此页面的目标是将文档分类为不同的主题并简要描述每个主题。如果这是您第一次使用 PyTorch 构建
分布式训练应用程序，建议
使用本文档导航到
最适合您的用例的技术。





## 简介 [¶](#introduction "此标题的永久链接")




 自 PyTorch v1.6.0 起，
 `torch.distributed` 中的功能可分为
三个主要组件：



* [分布式数据并行训练](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
 (DDP)是一种广泛采用的单程序多数据训练范例。使用 DDP，模型会在每个进程上进行复制，并且每个模型副本都将被提供一组不同的输入数据样本。 DDP 负责梯度通信以保持模型副本同步，并将其与梯度计算重叠以加速训练。
* [基于 RPC 的分布式训练](https://pytorch.org/docs/stable/rpc. html) 
 (RPC) 支持无法适应数据并行训练的一般训练结构，例如分布式管道并行、参数服务器范例以及 DDP 与其他训练范例的组合。它
帮助管理远程对象的生命周期并扩展
 [autograd 引擎](https://pytorch.org/docs/stable/autograd.html) 
 超出
机器边界。
* [集体通信](https://pytorch.org/docs/stable/distributed.html) 
 (c10d) 库支持在组内跨进程发送tensor。它提供了两种集体通信 API(例如，
 [all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) 
 和
 [all\ _gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather) 
 )
 和 P2P 通信 API(例如，
 [send](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send) 
 和
 [isend](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend) 
 ).\ nDDP 和 RPC (
 [ProcessGroup Backend](https://pytorch.org/docs/stable/rpc.html#process-group-backend) 
 )
 构建于 c10d 之上，前者使用集体通信
并且后者使用P2P通信。通常，开发人员不需要直接使用此原始通信 API，因为 DDP 和 RPC API 可以服务于许多分布式训练场景。但是，在某些用例中，此 API 仍然很有用。一个例子是分布式参数平均，其中
应用程序希望在向后传递之后
计算所有模型参数的平均值，而不是使用 DDP 来传递梯度。这可以将通信与计算分离，并允许对通信内容进行更细粒度的控制，但另一方面，它也放弃了 DDP 提供的性能优化。
 [使用 PyTorch 编写分布式应用程序](../middle/dist_tuto.html) 
 显示使用 c10d 通信 API 的示例。





## 数据并行训练 [¶](#data-parallel-training "永久链接到此标题")



PyTorch 提供了多种数据并行训练选项。对于从简单到复杂、从原型到生产逐渐发展的应用程序，
常见的开发轨迹是：



1. 如果数据和模型可以容纳在一个 GPU 中，则使用单设备训练，并且
训练速度不是问题。
2.使用单机多GPU
 [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
 在单机上利用多个GPU来加速使用最少的代码更改进行训练。
3.使用单机多GPU
 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
 ,
如果你想进一步加快训练速度并愿意编写
更多代码来设置它。
4.使用多机
 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
 和
 [启动脚本](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md) 
 ，
如果应用程序需要跨机器边界扩展。
5.如果预计会出现错误(例如内存不足)或者如果
资源可以在训练期间动态加入和离开。




 注意




 数据并行训练也适用于
 [自动混合精度 (AMP)](https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-gpus) 
.\ n




### `torch.nn.DataParallel` [¶](#torch-nn-dataparallel "此标题的永久链接")



 [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) 
 包以最低的编码障碍实现单机多 GPU 并行。只需要对应用程序代码进行一行更改。本教程
 [可选：数据并行性](../beginner/blitz/data_parallel_tutorial.html)
 显示了一个示例。虽然
 `DataParallel`
 非常容易使用，
但它通常无法提供最佳性能，因为它在每次前向传递中都会
复制模型，
其单进程多线程并行性
自然会受到
 [ GIL](https://wiki.python.org/moin/GlobalInterpreterLock) 
 争用。为了获得
更好的性能，请考虑使用
 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
 。





### `torch.nn.parallel.DistributedDataParallel` [¶](#torch-nn-parallel-distributeddataparallel "此标题的永久链接")



 与
 [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) 相比
 、
 [DistributedDataParallel](https://pytorch.org/docs /stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
 还需要一个步骤来设置，即调用
 [init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) 
.
DDP 使用多进程并行性，因此模型副本之间不存在 GIL 争用。此外，模型在 DDP 构建时广播，而不是在每次前向传递中广播，这也有助于加快训练速度。 DDP 附带了多种性能优化技术。如需更深入的解释，请参阅此[论文](http://www.vldb.org/pvldb/vol13/p3005-li.pdf)
(VLDB’20)。 




 DDP 材料如下所列：



1. [DDP 注释](https://pytorch.org/docs/stable/notes/ddp.html) 提供一个入门示例及其设计和实现的一些简要说明。如果这是您第一次使用 DDP，请从本文档开始。
2. [分布式数据并行入门](../intermediate/ddp_tutorial.html) 
 解释了 DDP 训练的一些常见问题，包括工作负载不平衡、检查点和多设备模型。请注意，DDP 可以与单机多设备模型并行性轻松结合，这在[单机模型并行最佳实践](../intermediate/model_parallel_tutorial.html) 教程中进行了描述。\ n3。 
 [启动和配置分布式数据并行应用程序](https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md)
 文档介绍了如何使用 DDP 启动脚本。\ n4。 
 [使用 ZeroRedundancyOptimizer 的分片优化器状态](../recipes/zero_redundancy_optimizer.html) 
 配方演示了
 [ZeroRedundancyOptimizer](https://pytorch.org/docs/stable/distributed.optim.html) 
 如何实现n 有助于减少优化器内存占用。
5. 
 [使用连接上下文管理器进行不均匀输入的分布式训练](../advanced/generic_join.html) 
 教程逐步介绍如何使用通用连接上下文进行不均匀输入的分布式训练。




### torch.distributed.elastic [¶](#torch-distributed-elastic "此标题的永久链接")



 随着应用程序复杂性和规模的增长，故障恢复
成为一项要求。有时在使用 DDP 时不可避免地会遇到内存不足 (OOM) 等错误，但 DDP 本身无法从这些错误中恢复，并且无法使用标准的“try- except”来处理它们
这是因为 DDP 要求所有进程以紧密同步的方式运行
并且在不同进程中启动的所有
 `AllReduce`
 通信必须匹配。
如果组中的一个进程
抛出异常，它可能会导致不同步(不匹配
 `AllReduce`
 操作)，从而导致崩溃或挂起。
 [torch.distributed.elastic](https://pytorch.org/docs/stable/distributed.elastic.html)
 增加了容错能力和利用动态机器池的能力(弹性)。





## 基于 RPC 的分布式训练 [¶](#rpc-based-distributed-training "永久链接到此标题")




 许多训练范例不适合数据并行，例如
参数服务器范例、分布式管道并行、具有多个观察者或代理的强化
学习应用程序等。
 [torch.distributed.rpc](https://pytorch.org/docs/stable/rpc.html)
 旨在
支持一般的分布式训练场景。




[torch.distributed.rpc](https://pytorch.org/docs/stable/rpc.html)
 有四个主要支柱：



* [RPC](https://pytorch.org/docs/stable/rpc.html#rpc) 
 支持在远程工作人员上运行
给定函数。
* [RRef](https://pytorch.org /docs/stable/rpc.html#rref) 
 有助于管理
远程对象的生命周期。引用计数协议在
 [RRef 注释](https://pytorch.org/docs/stable/rpc/rref.html#remote-reference-protocol) 中介绍。
.
* [Distributed Autograd]( https://pytorch.org/docs/stable/rpc.html#distributed-autograd-framework) 
 将 autograd 引擎扩展到机器边界之外。请参阅
 [分布式 Autograd 设计](https://pytorch.org/docs/stable/rpc/distributed_autograd.html#distributed-autograd-design)
 了解更多详细信息。
* [分布式优化器](https ://pytorch.org/docs/stable/rpc.html#module-torch.distributed.optim) 自动联系所有参与的工作人员，使用分布式 autograd 引擎计算的梯度来更新参数。



 下面列出了 RPC 教程：



1. [分布式 RPC 框架入门](../intermediate/rpc_tutorial.html) 教程首先使用一个简单的强化学习 (RL) 示例来演示 RPC 和 RRef。然后，它将基本分布式模型
并行性应用于 RNN 示例，以展示如何使用分布式自动分级和
分布式优化器。
2. 
 [使用分布式 RPC 框架实现参数服务器](../intermediate/rpc_param_server_tutorial.html) 
 教程借鉴了
 [HogWild!训练](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)
 并将其应用于异步参数服务器 (PS) 训练应用程序。
3. 
 [使用 RPC 的分布式管道并行性](../intermediate/dist_pipeline_parallel_tutorial.html) 
 教程扩展了单机管道并行示例(在
 [单机模型并行最佳实践](../intermediate /model_parallel_tutorial.html) 
 )
到分布式环境并展示如何使用 RPC 实现它。
4. 
 [使用异步执行实现批量 RPC 处理](../intermediate/rpc_async_execution.html) 
 教程演示了如何使用
 [@rpc.functions.async_execution](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.functions.async_execution) 
 装饰器，可以帮助加快推理和训练速度。它使用类似于上述教程 1 和 2 中的 RL 和 PS 示例。
5. [将分布式数据并行与分布式 RPC 框架相结合](../advanced/rpc_ddp_tutorial.html) 教程演示了如何将 DDP 与 RPC 相结合，以使用分布式数据并行性与分布式模型并行性相结合来训练模型。





## PyTorch 分布式开发人员 [¶](#pytorch-distributed-developers "此标题的永久链接")




 如果您’d 愿意为 PyTorch Distributed 做出贡献，请参阅我们的
 [开发者指南](https://github.com/pytorch/pytorch/blob/master/torch/distributed/CONTRIBUTING。 md) 
.









