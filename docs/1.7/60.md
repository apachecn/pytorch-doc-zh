# PyTorch 分布式概述

> 原文：<https://pytorch.org/tutorials/beginner/dist_overview.html>

**作者**：[Shen Li](https://mrshenli.github.io/)

这是`torch.distributed`包的概述页面。 由于在不同位置添加了越来越多的文档，示例和教程，因此不清楚要针对特定​​问题咨询哪个文档或教程，或者阅读这些内容的最佳顺序是什么。 该页面的目的是通过将文档分类为不同的主题并简要描述每个主题来解决此问题。 如果这是您第一次使用 PyTorch 构建分布式训练应用，建议使用本文档导航至最适合您的用例的技术。

## 简介

从 PyTorch v1.6.0 开始，`torch.distributed`中的功能可以分为三个主要组件：

*   [分布式数据并行训练](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)（DDP）是一种广泛采用的单程序多数据训练范例。 使用 DDP，可以在每个流程上复制模型，并且每个模型副本都将获得一组不同的输入数据样本。 DDP 负责梯度通信，以保持模型副本同步，并使其与梯度计算重叠，以加快训练速度。
*   [基于 RPC 的分布式训练](https://pytorch.org/docs/master/rpc.html)（RPC）开发来支持无法适应数据并行训练的常规训练结构，例如分布式管道并行性，参数服务器范式以及 DDP 与其他训练范式的组合。 它有助于管理远程对象的生命周期，并将自动微分引擎扩展到机器范围之外。
*   [集体通信](https://pytorch.org/docs/stable/distributed.html)（c10d）库支持跨组内的进程发送张量。 它提供了集体通信 API（例如[`all_reduce`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)和[`all_gather`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)）和 P2P 通信 API（例如[`send`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send)和 [`isend`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend)）。 从 v1.6.0 开始，DDP 和 RPC（[ProcessGroup 后端](https://pytorch.org/docs/master/rpc.html#process-group-backend)）建立在 c10d 上，其中前者使用集体通信，而后者使用 P2P 通信。 通常，开发人员无需直接使用此原始通信 API，因为上述 DDP 和 RPC 功能可以满足许多分布式训练方案的需求。 但是，在某些情况下，此 API 仍然很有帮助。 一个示例是分布式参数平均，其中应用希望在反向传播之后计算所有模型参数的平均值，而不是使用 DDP 来传递梯度。 这可以使通信与计算脱钩，并允许对通信内容进行更细粒度的控制，但另一方面，它也放弃了 DDP 提供的性能优化。 [用 PyTorch 编写分布式应用](https://pytorch.org/tutorials/intermediate/dist_tuto.html)显示了使用 c10d 通信 API 的示例。

现有的大多数文档都是为 DDP 或 RPC 编写的，本页面的其余部分将详细介绍这两个组件的材料。

## 数据并行训练

PyTorch 为数据并行训练提供了几种选择。 对于从简单到复杂以及从原型到生产逐渐增长的应用，共同的发展轨迹将是：

1.  如果数据和模型可以放在一个 GPU 中，并且不关心训练速度，请使用单设备训练。
2.  如果服务器上有多个 GPU，请使用单机多 GPU [`DataParallel`](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)，并且您希望以最少的代码更改来加快训练速度。
3.  如果您想进一步加快训练速度并愿意编写更多代码来设置它，请使用单机多 GPU [`DistributedDataParallel`](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)。
4.  如果应用需要跨计算机边界扩展，请使用多计算机[`DistributedDataParallel`](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)和[启动脚本](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md)。
5.  如果预计会出现错误（例如，OOM），或者在训练过程中资源可以动态加入和离开，请使用[扭弹性](https://pytorch.org/elastic)启动分布式训练。

注意

数据并行训练还可以与[自动混合精度（AMP）](https://pytorch.org/docs/master/notes/amp_examples.html#working-with-multiple-gpus)一起使用。

### `torch.nn.DataParallel`

[`DataParallel`](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)包以最低的编码障碍实现了单机多 GPU 并行处理。 它只需要一行更改应用代码。 教程[可选：数据并行](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)显示了一个示例。 需要注意的是，尽管`DataParallel`非常易于使用，但通常无法提供最佳性能。 这是因为`DataParallel`的实现会在每个正向传播中复制该模型，并且其单进程多线程并行性自然会遭受 GIL 争用。 为了获得更好的性能，请考虑使用[`DistributedDataParallel`](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)。

### `torch.nn.parallel.DistributedDataParallel`

与[`DataParallel`](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)相比，[`DistributedDataParallel`](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)还需要设置一个步骤，即调用[`init_process_group`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)。 DDP 使用多进程并行性，因此在模型副本之间没有 GIL 争用。 此外，该模型是在 DDP 构建时而不是在每个正向传播时广播的，这也有助于加快训练速度。 DDP 附带了几种性能优化技术。 有关更深入的说明，请参阅此 [DDP 论文](https://arxiv.org/abs/2006.15704)（VLDB'20）。

DDP 材料如下：

1.  [DDP 注解](https://pytorch.org/docs/stable/notes/ddp.html)提供了一个入门示例，并简要介绍了其设计和实现。 如果这是您第一次使用 DDP，请从本文档开始。
2.  [分布式数据并行入门](../intermediate/ddp_tutorial.html)解释了 DDP 训练的一些常见问题，包括不平衡的工作量，检查点和多设备模型。 请注意，DDP 可以轻松与[单机模型并行最佳实践](../intermediate/model_parallel_tutorial.html)教程中描述的单机多设备模型并行性结合。
3.  [启动和配置分布式数据并行应用](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md)文档显示了如何使用 DDP 启动脚本。
4.  [使用 Amazon AWS 的 PyTorch 分布式训练器](aws_distributed_training_tutorial.html)演示了如何在 AWS 上使用 DDP。

### TorchElastic

随着应用复杂性和规模的增长，故障恢复成为当务之急。 有时，使用 DDP 时不可避免地会遇到 OOM 之类的错误，但是 DDP 本身无法从这些错误中恢复，基本的`try-except`块也无法工作。 这是因为 DDP 要求所有进程以紧密同步的方式运行，并且在不同进程中启动的所有`AllReduce`通信都必须匹配。 如果组中的某个进程抛出 OOM 异常，则很可能导致不同步（`AllReduce`操作不匹配），从而导致崩溃或挂起。 如果您期望在训练过程中发生故障，或者资源可能会动态离开并加入，请使用 [Torrlastic](https://pytorch.org/elastic) 启动分布式数据并行训练。

## 通用分布式训练

许多训练范式不适合数据并行性，例如参数服务器范式，分布式管道并行性，具有多个观察者或智能体的强化学习应用等。 [`torch.distributed.rpc`](https://pytorch.org/docs/master/rpc.html)旨在支持一般的分布式训练方案 。

[`torch.distributed.rpc`](https://pytorch.org/docs/master/rpc.html)包具有四个主要支柱：

*   [RPC](https://pytorch.org/docs/master/rpc.html#rpc) 支持在远程工作器上运行给定函数
*   [RRef](https://pytorch.org/docs/master/rpc.html#rref) 帮助管理远程对象的生存期。 引用计数协议在 [RRef 注解](https://pytorch.org/docs/master/rpc/rref.html#remote-reference-protocol)中提供。
*   [分布式自动微分](https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework)将自动微分引擎扩展到机器范围之外。 有关更多详细信息，请参考[分布式 Autograd 设计](https://pytorch.org/docs/master/rpc/distributed_autograd.html#distributed-autograd-design)。
*   [分布式优化器](https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim)，它使用分布式 Autograd 引擎计算的梯度自动与所有参与的工作器联系以更新参数。

RPC 教程如下：

1.  [分布式 RPC 框架入门](../intermediate/rpc_tutorial.html)教程首先使用一个简单的强化学习（RL）示例来演示 RPC 和 RRef。 然后，它对 RNN 示例应用了基本的分布式模型并行性，以展示如何使用分布式 Autograd 和分布式优化器。
2.  [使用分布式 RPC 框架实现参数服务器](../intermediate/rpc_param_server_tutorial.html)教程借鉴了 [HogWild 的训练精神](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)，并将其应用于异步参数服务器（PS）训练应用。
3.  使用 RPC 的[分布式管道并行化](../intermediate/dist_pipeline_parallel_tutorial.html)教程将单机管道并行示例（在[单机模型并行最佳实践](../intermediate/model_parallel_tutorial.html)中介绍）扩展到了分布式环境，并展示了如何使用 RPC 来实现它 。
4.  [使用异步执行实现批量 RPC](../intermediate/rpc_async_execution.html) 教程演示了如何使用[`@rpc.functions.async_execution`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.functions.async_execution)装饰器实现 RPC 批量。这可以帮助加速推理和训练。 它使用了以上教程 1 和 2 中采用的类似 RL 和 PS 示例。
5.  [将分布式`DataParallel`与分布式 RPC 框架结合](../advanced/rpc_ddp_tutorial.html)教程演示了如何将 DDP 与 RPC 结合使用分布式数据并行性和分布式模型并行性来训练模型。

## PyTorch 分布式开发人员

如果您想为 PyTorch 分布式做出贡献，请参阅我们的[开发人员指南](https://github.com/pytorch/pytorch/blob/master/torch/distributed/CONTRIBUTING.md)。