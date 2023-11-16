# 一步步探索 TorchRec

> 译者：[方小生]()
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/sharding>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/sharding.html>

本教程将主要介绍通过**EmbeddingPlanner**和**DistributedModelParallel**API嵌入表的分片方案，并通过明确的配置探讨不同分片方案对嵌入表的好处。

## 安装

要求:- python >= 3.7

我们强烈推荐使用torchRec时CUDA。如果使用CUDA: - CUDA >= 11.0他们。

```bash
# install conda to make installying pytorch with cudatoolkit 11.3 easier.
sudo rm Miniconda3-py37_4.9.2-Linux-x86_64.sh Miniconda3-py37_4.9.2-Linux-x86_64.sh.*
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh
sudo chmod +x Miniconda3-py37_4.9.2-Linux-x86_64.sh
sudo bash ./Miniconda3-py37_4.9.2-Linux-x86_64.sh -b -f -p /usr/local
```



```bash
# install pytorch with cudatoolkit 11.3
sudo conda install pytorch cudatoolkit=11.3 -c pytorch-nightly -y
```

安装torchRec还将安装[FBGEMM](https://github.com/pytorch/fbgemm)，这是一个CUDA内核和GPU启用操作的集合。

```bash
# install torchrec
pip3 install torchrec-nightly
```

安装与python一起工作的multiprocess，以便在colab中进行多处理编程

```bash
pip3 install multiprocess
```

Colab运行时需要执行以下步骤来检测添加的共享库。运行时在/usr/lib中搜索共享库，因此我们复制安装在/usr/local/lib/中的库。**这是一个非常必要的步骤，仅在colab运行时**。

```bash
sudo cp /usr/local/lib/lib* /usr/lib/
```

**此时重新启动运行时，可以看到新安装的包。**重新启动后立即运行下面的步骤，以便python知道在哪里查找包。**重新启动运行时后，始终运行此步骤。**

```python
import sys
sys.path = ['', '/env/python', '/usr/local/lib/python37.zip', '/usr/local/lib/python3.7', '/usr/local/lib/python3.7/lib-dynload', '/usr/local/lib/python3.7/site-packages', './.local/lib/python3.7/site-packages']
```

## 分布式的设置

由于笔记本环境的原因，我们不能在这里运行[SPMD](https://en.wikipedia.org/wiki/SPMD)程序，但我们可以在笔记本中执行多处理来模拟设置。当使用torch时，用户应该负责设置自己的[SPMD](https://en.wikipedia.org/wiki/SPMD)启动器。我们设置了我们的环境，使torch分布式通信后端可以工作。

```python
import os
import torch
import torchrec

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
```

## 构建embedding 模型

在这里，我们使用TorchRec提供的[EmbeddingBagCollection](https://github.com/facebookresearch/torchrec/blob/main/torchrec/modules/embedding_modules.py#L59)来构建嵌入表的嵌入袋模型。

这里，我们创建了一个包含四个嵌入包的EmbeddingBagCollection (EBC)。我们有两种类型的表:大表和小表，它们的行大小不同:4096 vs 1024。每个表仍然由64维嵌入表示。

我们为表配置了“ParameterConstraints”数据结构，它为模型并行API提供了提示，以帮助决定表的分片和放置策略。在《TorchRec》中，我们支持**“table-wise”**:将整个表放在一个设备上;

*** row-wise** ：按行维度均匀地对表进行分片，并在通信世界的每个设备上放置一个分片;

***column-wise** ：通过嵌入维度均匀地对表进行分片，并在通信世界的每个设备上放置一个分片;

***table-row-wise**：针对主机内部通信优化的特殊分片，用于可用的快速机器内部设备互连，例如NVLink;

**data_parallel**：在每个设备上复制数据表;注意我们最初是如何在设备“meta”上分配EBC的。这将告诉EBC不分配内存。

```python
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import ShardingType
from typing import Dict

large_table_cnt = 2
small_table_cnt = 2
large_tables=[
  torchrec.EmbeddingBagConfig(
    name="large_table_" + str(i),
    embedding_dim=64,
    num_embeddings=4096,
    feature_names=["large_table_feature_" + str(i)],
    pooling=torchrec.PoolingType.SUM,
  ) for i in range(large_table_cnt)
]
small_tables=[
  torchrec.EmbeddingBagConfig(
    name="small_table_" + str(i),
    embedding_dim=64,
    num_embeddings=1024,
    feature_names=["small_table_feature_" + str(i)],
    pooling=torchrec.PoolingType.SUM,
  ) for i in range(small_table_cnt)
]

def gen_constraints(sharding_type: ShardingType = ShardingType.TABLE_WISE) -> Dict[str, ParameterConstraints]:
  large_table_constraints = {
    "large_table_" + str(i): ParameterConstraints(
      sharding_types=[sharding_type.value],
    ) for i in range(large_table_cnt)
  }
  small_table_constraints = {
    "small_table_" + str(i): ParameterConstraints(
      sharding_types=[sharding_type.value],
    ) for i in range(small_table_cnt)
  }
  constraints = {**large_table_constraints, **small_table_constraints}
  return constraints
```



```python
ebc = torchrec.EmbeddingBagCollection(
    device="cuda",
    tables=large_tables + small_tables
)
```

## DistributedModelParallel在多处理方面的使用

现在，我们有一个单一的进程执行函数来模拟[SPMD](https://en.wikipedia.org/wiki/SPMD)执行期间一个rank的工作。

此代码将与其他进程共同对模型进行切分，并相应地分配内存。它首先设置进程组，并使用规划器进行嵌入表放置，然后使用**DistributedModelParallel**生成分片模型。

```python
def single_rank_execution(
    rank: int,
    world_size: int,
    constraints: Dict[str, ParameterConstraints],
    module: torch.nn.Module,
    backend: str,
) -> None:
    import os
    import torch
    import torch.distributed as dist
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.types import ModuleSharder, ShardingEnv
    from typing import cast

    def init_distributed_single_host(
        rank: int,
        world_size: int,
        backend: str,
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
    ) -> dist.ProcessGroup:
        os.environ["RANK"] = f"{rank}"
        os.environ["WORLD_SIZE"] = f"{world_size}"
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        return dist.group.WORLD

    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    topology = Topology(world_size=world_size, compute_device="cuda")
    pg = init_distributed_single_host(rank, world_size, backend)
    planner = EmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints,
    )
    sharders = [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())]
    plan: ShardingPlan = planner.collective_plan(module, sharders, pg)

    sharded_model = DistributedModelParallel(
        module,
        env=ShardingEnv.from_process_group(pg),
        plan=plan,
        sharders=sharders,
        device=device,
    )
    print(f"rank:{rank},sharding plan: {plan}")
    return sharded_model
```

## 多处理执行

现在让我们在代表多个GPU等级的多进程中执行代码。

```python
import multiprocess

def spmd_sharing_simulation(
    sharding_type: ShardingType = ShardingType.TABLE_WISE,
    world_size = 2,
):
  ctx = multiprocess.get_context("spawn")
  processes = []
  for rank in range(world_size):
      p = ctx.Process(
          target=single_rank_execution,
          args=(
              rank,
              world_size,
              gen_constraints(sharding_type),
              ebc,
              "nccl"
          ),
      )
      p.start()
      processes.append(p)

  for p in processes:
      p.join()
      assert 0 == p.exitcode
```

### Table Wise分片

现在让我们在两个gpu的两个进程中执行代码。我们可以在计划打印中看到我们的表是如何跨gpu分片的。每个节点将有一个大表和一个小表，这表明我们的规划器尝试为嵌入表进行负载平衡。**Table-wise**是许多中小型的表用于设备负载平衡的去因子分片方案。

```python
spmd_sharing_simulation(ShardingType.TABLE_WISE)
```



```python
rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:0/cuda:0)])), 'large_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:0/cuda:0)])), 'small_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:1/cuda:1)]))}}
rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:0/cuda:0)])), 'large_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[0], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:0/cuda:0)])), 'small_table_1': ParameterSharding(sharding_type='table_wise', compute_kernel='batched_fused', ranks=[1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 64], placement=rank:1/cuda:1)]))}}
```

### 探索其他分片模式

我们最初探索了表分片是什么样子的，以及它如何平衡表的放置。现在我们来探讨切分模式，更细致地关注负载平衡:行方向。逐行是专门针对单个设备无法容纳的大型表，因为大型嵌入行数会增加内存大小。它可以解决模型中超大表的放置问题。用户可以在打印的计划日志中的**shard_sizes** 部分看到，表按行维减半，分发到两个gpu上。

```python
spmd_sharing_simulation(ShardingType.ROW_WISE)
```



```python
rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)]))}}
rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[2048, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[2048, 0], shard_sizes=[2048, 64], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='row_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[512, 64], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[512, 0], shard_sizes=[512, 64], placement=rank:1/cuda:1)]))}}
```

另一方面，为了解决具有大嵌入维度的表的负载不平衡问题，我们将把表格竖着分开。用户可以在打印的计划日志中的**shard_sizes**部分看到，通过将维度嵌入到两个gpu上，表被减半。

```python
spmd_sharing_simulation(ShardingType.COLUMN_WISE)
```

```python
rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)]))}}
rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'large_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[4096, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[4096, 32], placement=rank:1/cuda:1)])), 'small_table_0': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)])), 'small_table_1': ParameterSharding(sharding_type='column_wise', compute_kernel='batched_fused', ranks=[0, 1], sharding_spec=EnumerableShardingSpec(shards=[ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1024, 32], placement=rank:0/cuda:0), ShardMetadata(shard_offsets=[0, 32], shard_sizes=[1024, 32], placement=rank:1/cuda:1)]))}}
```

对于**table-row-wise**，不幸的是，由于其在多主机设置下运行的性质，我们无法模拟它。将来我们将提供一个 python [SPMD](https://en.wikipedia.org/wiki/SPMD) 示例来使用“table-row-wise”训练模型。

通过数据并行，我们将为所有设备重复该表。

```python
rank:0,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'large_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None)}}
rank:1,sharding plan: {'': {'large_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'large_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_0': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None), 'small_table_1': ParameterSharding(sharding_type='data_parallel', compute_kernel='batched_dense', ranks=[0, 1], sharding_spec=None)}}
```

