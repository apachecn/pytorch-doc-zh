


# 将分布式 DataParallel 与分布式 RPC 框架相结合 [¶](#combining-distributed-dataparallel-with-distributed-rpc-framework "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/rpc_ddp_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html>




**作者** 
 :
 [Pritam Damania](https://github.com/pritamdamania87) 
 和
 [Yi Wang](https://github.com/wayi1)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/advanced_source/rpc_ddp_tutorial.rst) 
.





 本教程使用一个简单的示例来演示如何组合
 [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) 
 (DDP )
使用[分布式 RPC 框架](https://pytorch.org/docs/master/rpc.html)
 将分布式数据并行性与分布式模型并行性结合起来，
训练一个简单的模型。该示例的源代码可以在
[此处](https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc)
 找到。




 之前的教程、
 [分布式数据并行入门](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) 
 和
 [分布式 RPC 框架入门](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html) 
 ,
分别描述了如何进行分布式数据并行和分布式模型并行训练。不过，有多种训练范例
您可能希望将这两种技术结合起来。例如：



1. 如果我们有一个具有稀疏部分(大嵌入表)和密集部分(FC 层)的模型，我们可能希望将嵌入表放在参数服务器上，并使用
在多个训练器之间复制 FC 层[DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) 
.

 [分布式 RPC 框架](https://pytorch.org/docs /master/rpc.html) 
 可用于在参数服务器上执行嵌入查找。
2.按照
 [PipeDream](https://arxiv.org/abs/1806.03377)论文中所述启用混合并行性。
我们可以使用
 [分布式 RPC 框架](https://pytorch.org/docs/master/rpc.html)
 跨多个工作线程传输模型的各个阶段，并使用
 [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html 复制每个阶段(如果需要) #torch.nn.parallel.DistributedDataParallel) 
 。









 在本教程中，我们将介绍上述案例 1。我们的设置中总共有 4
工作人员，如下：



1. 1 Master，负责在参数服务器上创建嵌入表
(nn.EmbeddingBag)。主控制器还驱动两个训练器上的训练循环。
2. 1 参数服务器，它基本上将嵌入表保存在内存中并响应来自 Master 和 Trainer 的 RPC。
3. 2 个训练器，存储 FC 层 (nn.Linear)，使用 [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) 在它们之间进行复制
.
训练器还负责执行前向传递、后向传递和优化器步骤。









整个训练过程执行如下：



1. 主节点创建一个 [RemoteModule](https://pytorch.org/docs/master/rpc.html#remotemodule)，在参数服务器上保存一个嵌入表。
2.然后，主机启动训练器上的训练循环，并将
远程模块传递给训练器。
3.训练器创建一个
“HybridModel”
，它首先使用主控器提供的远程模块
执行嵌入查找，然后执行
包装在 DDP 内的
FC 层。
4.训练器执行模型的前向传递，并使用损失
使用[分布式 Autograd](https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework) 执行反向传递
 n.
5.作为向后传递的一部分，首先计算 FC 层的梯度，并通过 DDP 中的 allreduce 同步到所有训练器。
6.接下来，分布式 Autograd 将梯度传播到参数服务器，
其中嵌入表的梯度被更新。
7.最后，[分布式优化器](https://pytorch.org/docs/master/rpc.html#module-torch.distributed.optim) 用于更新所有参数。




注意




 如果您’ 组合，则应始终使用
 [分布式 Autograd](https://pytorch.org/docs/master/rpc.html#distributed-autograd-framework) 
 进行向后传递DDP 和 RPC。





 现在，让’s 详细介绍一下每个部分。首先，我们需要先设置所有
工作人员，然后才能进行任何培训。我们创建 4 个进程，
等级 0 和 1 是我们的训练器，等级 2 是主服务器，等级 3 是
参数服务器。




 我们使用 TCP init_method 在所有 4 个工作线程上初始化 RPC 框架。
一旦 RPC 初始化完成，主节点就会创建一个远程模块，其中包含
 [EmbeddingBag](https://pytorch.org/docs /master/generated/torch.nn.EmbeddingBag.html) 
 使用 [RemoteModule] 在参数服务器上分层(https://pytorch.org/docs/master/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule) 
.
然后主程序循环遍历每个训练器，并通过使用
 [rpc_async]( 在每个训练器上调用
 `_run_trainer`
 来启动训练循环https://pytorch.org/docs/master/rpc.html#torch.distributed.rpc.rpc_async) 
 。
最后，master 等待所有训练完成后退出。




 训练器首先使用 [init_process_group](https://pytorch.org) 为 DDP 初始化一个
 `ProcessGroup`
，其中 world_size=2
(对于两个训练器) /docs/stable/distributed.html#torch.distributed.init_process_group) 
.
接下来，他们使用 TCP init_method 初始化 RPC 框架。请注意
RPC 初始化和 ProcessGroup 初始化中的端口不同。
这是为了避免两个框架的初始化之间的端口冲突。
初始化完成后，训练器只需等待
 `_run_trainer` 
 来自主服务器的 RPC。




 参数服务器只是初始化 RPC 框架并等待来自
训练器和主控器的 RPC。






```
def run_worker(rank, world_size):
 r"""
 A wrapper function that initializes RPC, calls the function, and shuts down
 RPC.
 """

    # We need to use different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
    if rank == 2:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )

        # Run the training loop on trainers.
        futs = []
        for trainer_rank in [0, 1]:
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank)
            )
            futs.append(fut)

        # Wait for all training to finish.
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        # Initialize process group for Distributed DataParallel on trainers.
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # Initialize RPC.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # Trainer just waits for RPCs from master.
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    # 2 trainers, 1 parameter server, 1 master.
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)

```




 在我们讨论训练器的细节之前，让’s 介绍一下训练器使用的
 `HybridModel`
。如下所述，
 `HybridModel`
 使用远程模块进行初始化，该模块在参数服务器和
 `device` 上保存嵌入表 (
 `remote_emb_module`
 ) 
 用于 DDP。模型的初始化在 DDP 中包装了一个
 [nn.Linear](https://pytorch.org/docs/master/generated/torch.nn.Linear.html)
 层，以在所有层之间复制和同步该层培训师。




 模型的前向方法非常简单。它使用 RemoteModule’s
 `forward`
 在参数服务器上执行
嵌入查找，并将其输出传递到 FC 层。






```
class HybridModel(torch.nn.Module):
 r"""
 The model consists of a sparse part and a dense part.
 1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
 2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
 This remote model can get a Remote Reference to the embedding table on the parameter server.
 """

    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(16, 8).cuda(device), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup.cuda(self.device))

```




 接下来，让’s 看看训练器上的设置。训练器首先使用远程模块创建
上面描述的
“HybridModel”
，该模块保存
参数服务器上的嵌入表及其自己的排名。




 现在，我们需要检索我们想要使用
[DistributedOptimizer](https://pytorch.org/docs/master/rpc.html#module-torch. Distribution.optim) 
.
要从参数服务器检索嵌入表的参数，
我们可以调用 RemoteModule’s
 [remote_parameters](https://pytorch.org/docs /master/rpc.html#torch.distributed.nn.api.remote_module.RemoteModule.remote_parameters) 
 ，
它基本上遍历嵌入表的所有参数并返回
 RRef 列表。训练器通过 RPC 在参数服务器上调用此方法，以接收所需参数的 RRef 列表。由于 DistributedOptimizer 始终采用需要优化的参数的 RRef 列表，因此我们甚至需要为 FC 层的本地参数创建 RRef。这是通过执行
 `model.fc.parameters()`
 ，为每个参数创建
 RRef 并将其附加到从
 `remote_parameters()`
 返回的列表来完成的。
请注意我们不能使用
 `model.parameters()`
 ，
因为它会递归调用
 `model.remote_emb_module.parameters()`
 ，
不支持
 `远程模块`
.




 最后，我们使用所有 RRef 创建 DistributedOptimizer 并定义一个
CrossEntropyLoss 函数。






```
def _run_trainer(remote_emb_module, rank):
 r"""
 Each trainer runs a forward pass which involves an embedding lookup on the
 parameter server and running nn.Linear locally. During the backward pass,
 DDP is responsible for aggregating the gradients for the dense part
 (nn.Linear) and distributed autograd ensures gradients updates are
 propagated to the parameter server.
 """

    # Setup the model.
    model = HybridModel(remote_emb_module, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.

    # Retrieve parameters for embedding table.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    # model.fc.parameters() only includes local parameters.
    # NOTE: Cannot call model.parameters() here,
    # because this will call remote_emb_module.parameters(),
    # which supports remote_parameters() but not parameters().
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

```




 现在我们’准备好介绍在每个训练器上运行的主训练循环。
 `get_next_batch`
 只是一个用于生成随机输入的辅助函数，
 n 培训目标。我们针对多个时期和每个批次运行训练循环：



1. 为分布式 Autograd 设置
 [分布式 Autograd 上下文](https://pytorch.org/docs/master/rpc.html#torch.distributed.autograd.context)。
2.运行模型的前向传播并检索其输出。
3.使用损失函数根据我们的输出和目标计算损失。
4.使用分布式 Autograd 使用损失执行分布式向后传递。
5.最后，运行分布式优化器步骤来优化所有参数。





```
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




 整个示例的源代码可以在
 [此处](https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc) 
.








