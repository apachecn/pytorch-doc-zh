# Torchec 介绍

> 译者：[方小生]()
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/torchrec_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/torchrec_tutorial.html>

**提示：**

为了充分利用本教程，我们建议使用Colab版本。这将允许您对下面提供的信息进行实验。

与下面的视频一起搭配学习更好。 [youtube](https://www.youtube.com/watch?v=cjgj41dvSeQ).

<iframe allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" frameborder="0" height="315" src="https://www.youtube.com/embed/cjgj41dvSeQ" width="560" style="box-sizing: border-box;"></iframe>

在构建推荐系统时，我们经常希望用嵌入来表示产品或页面等实体。例如，参见Meta AI的[深度学习推荐模型](https://arxiv.org/abs/1906.00091)，或DLRM。随着实体数量的增长，嵌入表的大小可能超过单个GPU的内存。一种常见的做法是跨设备对嵌入表进行分片，这是一种模型并行。为此，TorchRec引入了它的主要API [DistributedModelParallel](https://pytorch.org/torchrec/torchrec.distributed.html#torchrec.distributed.model_parallel.DistributedModelParallel)，或DMP。与PyTorch的distributeddataparlil类似，DMP封装了一个模型来支持分布式训练。

## 安装

python版本>=3.7

我们强烈建议使用TorchRec时使用CUDA(如果使用CUDA: CUDA >= 11.0)。

```bash
# install pytorch with cudatoolkit 11.3
conda install pytorch cudatoolkit=11.3 -c pytorch-nightly -y
# install TorchRec
pip3 install torchrec-nightly
```

## 概述

本教程将介绍torch的三个部分:“nn”。模块' [' EmbeddingBagCollection '](https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.embedding_modules.EmbeddingBagCollection)， [' DistributedModelParallel '](https://pytorch.org/torchrec/torchrec.distributed.html#torchrec.distributed.model_parallel.DistributedModelParallel) API，以及数据结构[' KeyedJaggedTensor '](https://pytorch.org/torchrec/torchrec.sparse.html#torchrec.sparse.jagged_tensor.JaggedTensor)。

### 分布式的设置

我们用torch.distributed设置环境。有关分布式的更多信息，请参阅本[教程](https://pytorch.org/tutorials/beginner/dist_overview.html)。

在这里，我们使用一个等级(colab进程)对应于我们的1 colab GPU。

```python
import os
import torch
import torchrec
import torch.distributed as dist

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# Note - you will need a V100 or A100 to run tutorial as as!
# If using an older GPU (such as colab free K80),
# you will need to compile fbgemm with the appripriate CUDA architecture
# or run with "gloo" on CPUs
dist.init_process_group(backend="nccl")
```

### 从EmbeddingBag到EmbeddingBagCollection

PyTorch通过[' torch.nn.Embedding '](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)和[' torch.nn.EmbeddingBag '](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html)表示嵌入。EmbeddingBag是一个集合版本的Embedding。

TorchRec通过创建嵌入集合来扩展这些模块。我们将使用[' EmbeddingBagCollection '](https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.embedding_modules.EmbeddingBagCollection)来表示一组EmbeddingBags。

在这里，我们用两个嵌入包创建了一个EmbeddingBagCollection (EBC)。每个表' product_table '和' user_table '都由大小为4096的64维嵌入表示。注意我们最初是如何在设备“meta”上分配EBC的。这将告诉EBC不分配内存。

```python
ebc = torchrec.EmbeddingBagCollection(
    device="meta",
    tables=[
        torchrec.EmbeddingBagConfig(
            name="product_table",
            embedding_dim=64,
            num_embeddings=4096,
            feature_names=["product"],
            pooling=torchrec.PoolingType.SUM,
        ),
        torchrec.EmbeddingBagConfig(
            name="user_table",
            embedding_dim=64,
            num_embeddings=4096,
            feature_names=["user"],
            pooling=torchrec.PoolingType.SUM,
        )
    ]
)
```

### 分布式模型并行-DistributedModelParallel

现在，我们准备用[' DistributedModelParallel '](https://pytorch.org/torchrec/torchrec.distributed.html#torchrec.distributed.model_parallel.DistributedModelParallel) (DMP)包装我们的模型。实例化DMP将:1。决定如何对模型进行分片。DMP将收集可用的“sharders”，并提出对嵌入表(即EmbeddingBagCollection)进行分片的最佳方式的“计划”。

2. 实际上是对模型进行分片。这包括在适当的设备上为每个嵌入表分配内存。

在这个玩具示例中，因为我们有两个EmbeddingTables和一个GPU，所以TorchRec将把它们放在单个GPU上。

```python
model = torchrec.distributed.DistributedModelParallel(ebc, device=torch.device("cuda"))
print(model)
print(model.plan)
```

### 查询vanilla nn.EmbeddingBag的输入与偏移

我们用“输入”和“偏移”来查询[' nn.Embedding '](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)和[' nn.EmbeddingBag '](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html)。输入是一个包含查找值的一维tensor。偏移量是一个一维tensor，其中序列是每个示例要池的值的数量的累积和。

让我们看一个例子，重新创建上面的产品EmbeddingBag: GPU。

```
|------------|
| product ID |
|------------|
| [101, 202] |
| []         |
| [303]      |
|------------|
```



```
product_eb = torch.nn.EmbeddingBag(4096, 64)
product_eb(input=torch.tensor([101, 202, 303]), offsets=torch.tensor([0, 2, 2]))
```

### 用KeyedJaggedTensor: GPU表示小批量。

我们需要对每个特征的任意数量的实体id的多个示例进行有效的表示。为了启用这种“参差不齐”表示，我们使用TorchRec数据结构[' KeyedJaggedTensor '](https://pytorch.org/torchrec/torchrec.sparse.html#torchrec.sparse.jagged_tensor.JaggedTensor) (KJT)。

让我们看一下如何查找两个嵌入包“product”和“user”的集合。假设minibatch由三个用户的三个示例组成。其中第一个有两个产品ID，第二个没有，第三个有一个产品ID。

```
|------------|------------|
| product ID | user ID    |
|------------|------------|
| [101, 202] | [404]      |
| []         | [505]      |
| [303]      | [606]      |
|------------|------------|
```

查询应该是:

```python
mb = torchrec.KeyedJaggedTensor(
    keys = ["product", "user"],
    values = torch.tensor([101, 202, 303, 404, 505, 606]).cuda(),
    lengths = torch.tensor([2, 0, 1, 1, 1, 1], dtype=torch.int64).cuda(),
)

print(mb.to(torch.device("cpu")))
```

注意，KJT批大小是**batch_size = len(length)//len(keys)**。在上面的例子中，batch_size为3。

### 把它们放在一起，用KJT minibatch查询我们的分布式模型

最后，我们可以使用我们的产品和用户的小批量查询我们的模型。

结果查找将包含一个KeyedTensor，其中每个键(或特征)包含一个大小为3x64 (batch_size x embedding_dim)的2Dtensor。

```python
pooled_embeddings = model(mb)
print(pooled_embeddings)
```

## 更多的资源

想了解更多信息，请参阅我们的[dlrm](https://github.com/pytorch/torchrec/tree/main/examples/dlrm)示例，其中包括使用Meta的[dlrm](https://arxiv.org/abs/1906.00091)在标准tb数据集上进行多节点训练。