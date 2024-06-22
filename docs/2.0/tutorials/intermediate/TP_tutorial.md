# 大规模Transformer模型训练使用张量并行（TP）

> 译者：[BrightLi](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/TP_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials//intermediate/TP_tutorial.html>

**本教程演示了如何使用Tensor Parallel和Fully Sharded Data Parallel在数百到数千个gpu上训练大型类似transformer的模型。**

先决条件：

* PyTorch 2.3.0或更高版本与CUDA/Linux一起安装
* 张量并行API
* 开始使用DeviceMesh
* 开始使用完全分片数据并行

## Tensor Parallel是如何工作的？

Tensor Parallel（TP）最初是在[Megatron-LM](https://arxiv.org/abs/1909.08053)论文中提出的，它是一种训练大规模Transformer模型的有效模型并行技术。我们在本教程中提到的[Sequence Parallel](https://arxiv.org/abs/2205.05198)（SP）是Tensor Parallel的变体，它在序列维度上对nn.LayerNorm或RMSNorm进行分片，以在训练期间进一步节省激活内存。随着模型变大，激活内存成为瓶颈，因此在Tensor Parallel训练中，它通常将Sequence Parallel应用于LayerNorm或RMSNorm层。

<img src='https://pytorch.org/tutorials/_images/megatron_lm.png' width=100% />

图1.表示变压器模型的MLP和自注意层上张量并行样式的分片，其中注意力/MLP中的矩阵乘法通过分片计算（图像源）进行

在高水平上，PyTorch Tensor Parallel的工作原理如下：

**分片初始化**

首先，要确定要应用于每个层的ParallelStyle，并通过调用parallelize_module来划分初始化模块。
并行化模块的模型参数将被交换到DTensors，DTensor将负责使用分片计算运行并行化模块。

**运行时向前/向后**

根据用户为每个ParallelStyle指定的输入/输出DTensor布局，它将运行适当的通信操作以转换输入/输出的DTensor布局（如allreduce、allgather和reduce_scatter）。然后，对于分片的层（例如nn.Linear、nn.Embedding），运行分片计算以节省计算/内存资源。

## 在什么时候以及为什么应该应用张量并行

PyTorch的完全分片数据并行（FSDP）已经能够将模型训练扩展到特定数量的GPU。然而，当涉及到从模型大小和GPU数量两方面进一步扩展模型训练时，会出现许多额外的挑战，这可能需要将张量并行与FSDP结合起来。

随着世界大小（GPU数量）变得过大（超过128/256个GPU），FSDP的集合操作（如allgather）受到环延迟的主导。通过在FSDP之上实现TP/SP，可以通过将FSDP应用于仅主机之间，从而将FSDP的世界大小减少8倍，进而将延迟成本降低相同的量。
当达到数据并行性的限制，即由于收敛和GPU内存限制，无法将全局批量大小提高到GPU数量以上时，张量/序列并行是已知的唯一方法来“大致估算”全局批量大小并继续使用更多GPU进行扩展。这意味着模型大小和GPU数量都可以继续扩展。
对于某些类型的模型，当局部批量大小变小时，TP/SP可以产生更优化于浮点运算（FLOPS）的矩阵乘法形状。
那么，在预训练时，达到这些限制有多容易？截至目前，即使使用数千个GPU，预训练一个拥有数十亿或数万亿标记的大型语言模型（LLM）也可能需要数月的时间。
在大规模训练LLM时，总是会碰到限制1。例如，使用2k个GPU训练了35天的Llama 2 70B，需要在2k的规模上使用多维并行性。
当Transformer模型变大时（如Llama2 70B），也会很快碰到限制2。由于内存和收敛约束，即使局部batch_size=1，也不能单独使用FSDP。例如，Llama 2的全局批量大小为1K，因此在2K个GPU上不能单独使用数据并行性。

## 如何应用张量并行

PyTorch张量并行API提供了一组模块级别的原语（ParallelStyle），用于配置模型每个单独层的分片，包括：

* ColwiseParallel 和 RowwiseParallel：以列或行的方式对nn.Linear 和 nn.Embedding进行分片。
* SequenceParallel：对nn.LayerNorm、nn.Dropout、RMSNormPython等执行分片计算。
* PrepareModuleInput 和 PrepareModuleOutput：使用适当的通信操作配置模块输入/输出的分片布局。

为了演示如何使用PyTorch原生张量并行API，让我们来看一个常见的Transformer模型。在这个教程中，我们参考了最近广泛使用的[Llama2 model](https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/llama2_model.py)作为Transformer模型的实现。
由于张量并行在一组设备上对单个张量进行分片，我们需要先设置分布式环境（如NCCL通信器）。张量并行是一种单程序多数据（SPMD）分片算法，类似于PyTorch DDP/FSDP，它在内部利用PyTorch DTensor进行分片。它还利用DeviceMesh抽象（在内部管理ProcessGroups）进行设备管理和分片。要了解如何利用DeviceMesh设置多维并行性，请参考[本教程](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)。张量并行通常在每个主机内工作，因此我们首先初始化一个连接主机内8个GPU的DeviceMesh。

```py
# run this via torchrun: torchrun --standalone --nproc_per_node=8 ./tp_tutorial.py

from torch.distributed.device_mesh import init_device_mesh

tp_mesh = init_device_mesh("cuda", (8,))
```

现在我们已经初始化了DeviceMesh，让我们详细看一下Llama 2模型的架构，看看我们应该如何执行张量并行分片。这里我们关注核心的TransformerBlock，其中Transformer模型通过堆叠相同的TransformerBlock s来扩展模型。
核心的TransformerBlock包括一个注意力层和一个前馈层。让我们先看一下更简单的前馈层。对于前馈层，它由三个线性层组成，在其中执行SwiGLU风格的MLP，查看其前向函数：

```py
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

layer_tp_plan = {
    # by default ColwiseParallel input layouts is replicated
    # and RowwiseParallel output layouts is replicated
    "feed_foward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

这就是我们使用PyTorch张量并行API为前馈层配置分片的方法。请注意，用户只需要指定如何对单个层进行分片，通信（例如，allreduce）将在内部自动进行。
接下来是注意力层。它由wq、wk、wv线性层组成，将输入投影到q/k/v，然后通过wo线性层执行注意力和输出投影。这里的张量并行旨在对q/k/v投影执行列式分片，对wo线性投影执行行式分片。因此，我们可以将注意力方案添加到我们刚刚草拟的tp_plan中：

```py
layer_tp_plan = {
    # by default ColwiseParallel input layouts is replicated
    # and RowwiseParallel output layouts is replicated
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

这几乎就是我们要对TransformerBlock应用张量并行所需的layer_tp_plan。然而，我们应该注意的是，当对线性层进行列式分片时，线性层的输出将在最后一个张量维度上分片，而行式分片的线性层直接接受在最后一个维度上分片的输入。如果在列式线性和行式线性之间有任何更多的张量操作（例如视图操作），我们需要调整相关的形状相关操作以适应分片形状。
对于Llama模型，在注意力层中有一些与形状相关的视图操作。特别是，对于wq/wk/wv线性层的列式并行，激活张量在num_heads维度上分片，因此我们需要将num_heads调整为本地num_heads。
最后，我们需要调用parallelize_module API，使每个TransformerBlock的计划生效。在内部，它将Attention和FeedForward层内的模型参数分发到DTensors，并在必要时为模型输入和输出（分别在每个模块之前和之后）注册通信钩子：

```py
for layer_id, transformer_block in enumerate(model.layers):
    layer_tp_plan = {...}  # i.e. the plan we just generated

    # Adjust attention module to use the local number of heads
    attn_layer = transformer_block.attention
    attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
    attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan,
    )
```

现在我们已经详细阐述了每个TransformerBlock的分片计划，通常在第一层有一个nn.Embedding和一个最终的nn.Linear投影层，用户可以为第一个nn.Embedding选择行式或列式分片，并为最后一个nn.Linear投影层指定适当的输入和输出布局进行列式分片。这里有一个示例：

```py
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
        ),
        "output": ColwiseParallel(
            output_layouts=Replicate(),
        ),
    }
)
```

> 注意:如果待划分的模型过大，无法放入CPU内存，用户可以使用元设备初始化（例如，先在元设备上初始化模型，对层进行分片，然后再实例化模型），或者在 Transformer模型初始化期间逐层并行化TransformerBlock。

## 将序列并行应用于LayerNorm/RMSNorm层

序列并行在上述张量并行的基础上工作。与仅在注意力模块和前馈模块内分片张量并保持其模块输入和输出（即前向传播中的激活和后向传播中的梯度）复制的基本张量并行相比，序列并行保持它们在序列维度上的分片。
在典型的TransformerBlock中，前向函数结合了范化层（LayerNorm或RMSNorm）、一个注意力层、一个前馈层和残差连接。例如：

```py
# forward in a TransformerBlock
def forward(self, x):
    h = x + self.attention(self.attention_norm(x))
    out = h + self.feed_forward(self.ffn_norm(h))
    return out
)
```

在大多数用例中，Attention和FeedForward模块之外的激活（和梯度）的形状为[batch size, sequence length, hidden dimension]。用DTensor的话来说，序列并行使用Shard(1)布局对模块的前向/后向进行激活计算。按照之前的代码示例，下面的代码演示了我们如何将序列并行应用于TransformerBlock内的范化层：
首先，让我们导入序列并行所需的依赖项：

```py
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    SequenceParallel,
)
```

接下来，让我们调整layer_tp_plan以在RMSNorm层上启用序列并行：

```py
layer_tp_plan = {
    # Now the input and output of SequenceParallel has Shard(1) layouts,
    # to represent the input/output tensors sharded on the sequence dimension
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "attention.wq": ColwiseParallel(),
    "attention.wk": ColwiseParallel(),
    "attention.wv": ColwiseParallel(),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm": SequenceParallel(),
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(),
}
```

我们可以看到，我们现在使用PrepareModuleInput将Attention和FeedForward层的模块输入布局从Shard(1)修改为Replicate()，并将它们的输出布局标记为Shard(1)。就像在张量并行中发生的一样，用户只需指定输入和输出的张量分片布局，层之间的通信将自动进行。
请注意，使用序列并行时，我们假设TransformerBlock的输入和输出始终在序列维度上分片，以便多个TransformerBlock可以无缝连接。这可以通过明确规定开始的nn.Embedding层的输出和最终的nn.Linear投影层的输入为Shard(1)来实现：

```py
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
)
```

###应用损失并行

损失并行是一种相关的技术，用于在计算损失函数时节省内存和通信，因为模型输出通常非常大。在损失并行中，当模型输出在（通常很大）词汇维度上分片时，可以高效地计算交叉熵损失，而无需将所有模型输出聚集到每个GPU上。这不仅显著减少了内存消耗，而且通过减少通信开销并进行分片并行计算来提高训练速度。下面的图片简要说明了损失并行如何通过进行分片计算来避免将所有模型输出聚集到每个GPU上。

<img src='https://pytorch.org/tutorials/_images/loss_parallel.png' width=100% />

图2. 在一个GPU上使用损失并行进行的交叉熵损失前向计算。蓝色代表分片张量；绿色代表复制张量；黄色代表具有部分值的张量（待全局减少）。黑色箭头是本地计算；红色箭头是GPU之间的功能集合。

在PyTorch张量并行API中，可以通过上下文管理器loss_parallel启用损失并行，使用它可以在不修改代码其他部分的情况下直接使用torch.nn.functional.cross_entropy或torch.nn.CrossEntropyLoss。
要应用损失并行，通常形状为[batch size, sequence length, vocabulary size]的模型预测应在词汇维度上进行分片。这可以通过标记最后一个线性投影层输出的输出层轻松完成：

```py
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            # use DTensor as the output
            use_local_output=False,
        ),
    },
)
```

在上面的代码中，我们还在输出前的范化层应用了序列并行。我们将use_local_output设置为False，以让输出保持为DTensor，以便与loss_parallel上下文管理器一起使用。之后，就可以像下面这样简单地调用交叉熵损失函数。请注意，后向计算也需要在上下文中进行。

```py
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel

pred = model(input_ids)
with loss_parallel():
    # assuming pred and labels are of the shape [batch, seq, vocab]
    loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
    loss.backward()
```

## 结合张量并行和完全分片数据并行

现在我们已经展示了如何将张量/序列并行应用于模型，让我们也看看张量并行和完全分片数据并行是如何一起工作的。由于张量并行化导致阻塞计算的通信，我们希望确保它在快速的通信通道内运行，例如NVLink。在实践中，我们通常在每个主机内应用张量并行，并在主机之间应用完全分片数据并行。

<img src='https://pytorch.org/tutorials/_images/fsdp_tp.png' width=100% />

图3. FSDP和TP在不同的设备维度上工作，FSDP通信发生在主机之间，而TP通信发生在主机内部。
这种二维并行模式可以通过二维DeviceMesh轻松表示，我们只需要将每个“子”DeviceMesh传递给各自的并行API：

```py
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# i.e. 2-D mesh is [dp, tp], training on 64 GPUs that performs 8 way DP and 8 way TP
mesh_2d = init_device_mesh("cuda", (8, 8))
tp_mesh = mesh_2d["tp"] # a submesh that connects intra-host devices
dp_mesh = mesh_2d["dp"] # a submesh that connects inter-host devices

model = Model(...)

tp_plan = {...}

# apply Tensor Parallel intra-host on tp_mesh
model_tp = parallelize_module(model, tp_mesh, tp_plan)
# apply FSDP inter-host on dp_mesh
model_2d = FSDP(model_tp, device_mesh=dp_mesh, use_orig_params=True, ...)
```

这将使我们能够在每个主机内轻松应用张量并行（intra-host），并在主机之间应用FSDP（inter-hosts），而对Llama模型的代码更改为零。张量（模型）并行和数据并行技术的结合，使得我们能够继续增加模型大小，并使用大量GPU高效训练。
结论
本教程演示了如何使用张量并行与完全分片数据并行相结合，在数百到数千个GPU上训练大型Transformer类模型。它解释了如何将张量并行应用于模型的不同部分，而无需对模型本身进行代码更改。张量并行是一种高效的模型并行技术，适用于大规模训练。
要查看本教程中解释的完整端到端代码示例，请参阅pytorch/examples仓库中的[Tensor Parallel examples](https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py)。