# 将分布式`DataParallel`与分布式 RPC 框架相结合

> 原文：<https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html>

**作者**： [Pritam Damania](https://github.com/pritamdamania87)

本教程使用一个简单的示例演示如何将[`DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)（DDP）与[分布式 RPC 框架](https://pytorch.org/docs/master/rpc.html)结合使用，以将分布式数据并行性与分布式模型并行性结合在一起，以训练简单模型。 该示例的源代码可以在中找到[。](https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc)

先前的教程[分布式数据并行入门](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)和[分布式 RPC 框架入门](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html)分别描述了如何执行分布式数据并行训练和分布式模型并行训练。 虽然，有几种训练范例，您可能想将这两种技术结合起来。 例如：

1.  如果我们的模型具有稀疏部分（较大的嵌入表）和密集部分（FC 层），则可能需要将嵌入表放在参数服务器上，并使用[`DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)。 [分布式 RPC 框架](https://pytorch.org/docs/master/rpc.html)可用于在参数服务器上执行嵌入查找。
2.  如 [PipeDream](https://arxiv.org/abs/1806.03377) 论文中所述，启用混合并行性。 我们可以使用[分布式 RPC 框架](https://pytorch.org/docs/master/rpc.html)在多个工作程序之间流水线化模型的各个阶段，并使用[`DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)复制每个阶段（如果需要）。

在本教程中，我们将介绍上述情况 1。 我们的设置中共有 4 个工作器，如下所示：

1.  1 个主机，负责在参数服务器上创建嵌入表（`nn.EmbeddingBag`）。 主人还会在两个教练上驱动训练循环。
2.  1 参数服务器，它基本上将嵌入表保存在内存中，并响应来自主服务器和训练器的 RPC。
3.  2 个训练器，用于存储 FC 层（线性线性），并使用[`DistributedDataParallel`](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)在它们之间进行复制。 训练人员还负责执行前进，后退和优化器步骤。

整个训练过程执行如下：

1.  主服务器在参数服务器上创建一个嵌入表，并为其保留一个 [RRef](https://pytorch.org/docs/master/rpc.html#rref)。
2.  然后，主持人开始在训练器上进行训练循环，并将嵌入表 RRef 传递给训练器。
3.  训练器创建一个`HybridModel`，该`HybridModel`首先使用主机提供的嵌入表 RRef 执行嵌入查找，然后执行包装在 DDP 中的 FC 层。
4.  训练者执行模型的正向传播，并使用[分布式 Autograd](https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework) 使用损失执行反向传递。
5.  作为向后遍历的一部分，将首先计算 FC 层的梯度，并通过 DDP 中的`allreduce`将其同步到所有训练器。
6.  接下来，分布式 Autograd 将梯度传播到参数服务器，在该服务器中更新嵌入表的梯度。
7.  最后，[分布式优化器](https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim)用于更新所有参数。

注意

如果您将 DDP 和 RPC 结合使用，则应始终使用[分布式 Autograd](https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework) 进行反向传播。

现在，让我们详细介绍每个部分。 首先，我们需要先设置所有工作器，然后才能进行任何训练。 我们创建 4 个过程，使等级 0 和 1 是我们的训练器，等级 2 是主控制器，等级 3 是参数服务器。

我们使用 TCP init_method 在所有 4 个工作器上初始化 RPC 框架。 RPC 初始化完成后，主服务器使用[`rpc.remote`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.remote)在参数服务器上创建[`EmbeddingBag`](https://pytorch.org/docs/master/generated/torch.nn.EmbeddingBag.html)。 然后，主控制器通过使用[`rpc_async`](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.rpc_async)在每个教练上调用`_run_trainer`，循环遍历每个教练并开始训练循环。 最后，主人在退出之前等待所有训练结束。

训练器首先使用[`init_process_group`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)为`world_size = 2`的 DDP 初始化`ProcessGroup`（对于两个训练器）。 接下来，他们使用 TCP `init_method`初始化 RPC 框架。 请注意，RPC 初始化和`ProcessGroup`初始化中的端口不同。 这是为了避免两个框架的初始化之间的端口冲突。 初始化完成后，训练器只需等待主服务器的`_run_trainer` RPC。

参数服务器只是初始化 RPC 框架，并等待来自训练者和主服务器的 RPC。

```py
def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method='tcp://localhost:29501'

    # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
    if rank == 2:
        rpc.init_rpc(
                "master",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options)

        # Build the embedding table on the ps.
        emb_rref = rpc.remote(
                "ps",
                torch.nn.EmbeddingBag,
                args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
                kwargs={"mode": "sum"})

        # Run the training loop on trainers.
        futs = []
        for trainer_rank in [0, 1]:
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(
                    trainer_name, _run_trainer, args=(emb_rref, rank))
            futs.append(fut)

        # Wait for all training to finish.
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        # Initialize process group for Distributed DataParallel on trainers.
        dist.init_process_group(
                backend="gloo", rank=rank, world_size=2)

        # Initialize RPC.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
                trainer_name,
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options)

        # Trainer just waits for RPCs from master.
    else:
        rpc.init_rpc(
                "ps",
                rank=rank,
                world_size=world_size,
                rpc_backend_options=rpc_backend_options)
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()

if __name__=="__main__":
    # 2 trainers, 1 parameter server, 1 master.
    world_size = 4
    mp.spawn(run_worker, args=(world_size, ), nprocs=world_size, join=True)

```

在讨论训练器的详细信息之前，让我们介绍一下训练器使用的`HybridModel`。 如下所述，使用对参数服务器上嵌入表（`emb_rref`）的 RRef 和用于 DDP 的`device`初始化`HybridModel`。 模型的初始化在 DDP 中包装了[`nn.Linear`](https://pytorch.org/docs/master/generated/torch.nn.Linear.html)层，以在所有训练器之间复制和同步该层。

该模型的前进方法非常简单。 它使用 [RRef 帮助程序](https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.RRef.rpc_sync)在参数服务器上执行嵌入查找，并将其输出传递到 FC 层。

```py
class HybridModel(torch.nn.Module):
    r"""
    The model consists of a sparse part and a dense part. The dense part is an
    nn.Linear module that is replicated across all trainers using
    DistributedDataParallel. The sparse part is an nn.EmbeddingBag that is
    stored on the parameter server.

    The model holds a Remote Reference to the embedding table on the parameter
    server.
    """

    def __init__(self, emb_rref, device):
        super(HybridModel, self).__init__()
        self.emb_rref = emb_rref
        self.fc = DDP(torch.nn.Linear(16, 8).cuda(device), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.emb_rref.rpc_sync().forward(indices, offsets)
        return self.fc(emb_lookup.cuda(self.device))

```

接下来，让我们看看训练器上的设置。 训练者首先使用对参数服务器上嵌入表的 RRef 及其自身等级创建上述`HybridModel`。

现在，我们需要检索要使用[`DistributedOptimizer`](https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim)优化的所有参数的 RRef 列表。 为了从参数服务器中检索嵌入表的参数，我们定义了一个简单的辅助函数`_retrieve_embedding_parameters`，该函数基本上遍历了嵌入表的所有参数并返回 RRef 的列表。 训练器通过 RPC 在参数服务器上调用此方法，以接收所需参数的 RRef 列表。 由于`DistributedOptimizer`始终将需要优化的参数的 RRef 列表，因此我们甚至需要为 FC 层的本地参数创建 RRef。 这是通过遍历`model.parameters()`，为每个参数创建 RRef 并将其附加到列表来完成的。 请注意，`model.parameters()`仅返回本地参数，不包含`emb_rref`。

最后，我们使用所有 RRef 创建我们的`DistributedOptimizer`，并定义`CrossEntropyLoss`函数。

```py
def _retrieve_embedding_parameters(emb_rref):
    param_rrefs = []
    for param in emb_rref.local_value().parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs

def _run_trainer(emb_rref, rank):
    r"""
    Each trainer runs a forward pass which involves an embedding lookup on the
    parameter server and running nn.Linear locally. During the backward pass,
    DDP is responsible for aggregating the gradients for the dense part
    (nn.Linear) and distributed autograd ensures gradients updates are
    propagated to the parameter server.
    """

    # Setup the model.
    model = HybridModel(emb_rref, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.

    # Retrieve parameters for embedding table.
    model_parameter_rrefs = rpc.rpc_sync(
            "ps", _retrieve_embedding_parameters, args=(emb_rref,))

    # model.parameters() only includes local parameters.
    for param in model.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

```

现在，我们准备介绍在每个训练器上运行的主要训练循环。 `get_next_batch`只是一个辅助函数，用于生成随机输入和训练目标。 我们针对多个周期和每个批量运行训练循环：

1.  为分布式 Autograd 设置[分布式 Autograd 上下文](https://pytorch.org/docs/master/rpc.html#torch.distributed.autograd.context)。
2.  运行模型的正向传播并检索其输出。
3.  使用损失函数，根据我们的输出和目标计算损失。
4.  使用分布式 Autograd 使用损失执行分布式反向传递。
5.  最后，运行“分布式优化器”步骤以优化所有参数。

```py
    def get_next_batch(rank):
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # Generate offsets.
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8).cuda(rank)
            yield indices, offsets_tensor, target

    # Train for 100 epochs
    for epoch in range(100):
        # create distributed autograd context
        for indices, offsets, target in get_next_batch(rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Tun distributed optimizer
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
        print("Training done for epoch {}".format(epoch))

```

[整个示例的源代码可以在这里找到](https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc)。