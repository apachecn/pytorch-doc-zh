# 分布式通信包 -  torch.distributed

> 译者：[univeryinli](https://github.com/univeryinli)

## 后端

`torch.distributed` 支持三个后端，每个后端具有不同的功能。下表显示哪些功能可用于CPU/CUDA张量。仅当用于构建PyTorch的实现支持时，MPI才支持CUDA。

| 后端 | `gloo` | `mpi` | `nccl` |
| --- | --- | --- | --- |
| 设备 | CPU | GPU | CPU | GPU | CPU | GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 发送 | ✓ | ✘ | ✓ | ? | ✘ | ✘ |
| 接收 | ✓ | ✘ | ✓ | ? | ✘ | ✘ |
| 广播| ✓ | ✓ | ✓ | ? | ✘ | ✓ |
| all_reduce | ✓ | ✓ | ✓ | ? | ✘ | ✓ |
| reduce | ✓ | ✘ | ✓ | ? | ✘ | ✓ |
| all_gather | ✓ | ✘ | ✓ | ? | ✘ | ✓ |
| 收集 | ✓ | ✘ | ✓ | ? | ✘ | ✘ |
| 分散 | ✓ | ✘ | ✓ | ? | ✘ | ✘ |
| 屏障 | ✓ | ✘ | ✓ | ? | ✘ | ✓ |

### PyTorch附带的后端

目前PyTorch分发版仅支持Linux。默认情况下，Gloo和NCCL后端构建并包含在PyTorch的分布之中(仅在使用CUDA构建时为NCCL）。MPI是一个可选的后端，只有从源代码构建PyTorch时才能包含它。(例如，在安装了MPI的主机上构建PyTorch）

### 哪个后端使用？

在过去，我们经常被问到：“我应该使用哪个后端？”。

*   经验法则
    *   使用NCCL后端进行分布式 **GPU** 训练。
    *   使用Gloo后端进行分布式 **CPU** 训练。
*   具有InfiniBand互连的GPU主机
    *   使用NCCL，因为它是目前唯一支持InfiniBand和GPUDirect的后端。
*   GPU主机与以太网互连
    *   使用NCCL，因为它目前提供最佳的分布式GPU训练性能，特别是对于多进程单节点或多节点分布式训练。如果您遇到NCCL的任何问题，请使用Gloo作为后备选项。(请注意，Gloo目前运行速度比GPU的NCCL慢。）
*   具有InfiniBand互连的CPU主机
    *   如果您的InfiniBand在IB上已启用IP，请使用Gloo，否则请使用MPI。我们计划在即将发布的版本中为Gloo添加InfiniBand支持。
*   具有以太网互连的CPU主机
    *   除非您有特殊原因要使用MPI，否则请使用Gloo。

### 常见的环境变量

#### 选择要使用的网络接口

默认情况下，NCCL和Gloo后端都会尝试查找用于通信的网络接口。但是，从我们的经验来看，并不总能保证这一点。因此，如果您在后端遇到任何问题而无法找到正确的网络接口。您可以尝试设置以下环境变量(每个变量适用于其各自的后端）：

*   **NCCL_SOCKET_IFNAME**, 比如 `export NCCL_SOCKET_IFNAME=eth0`
*   **GLOO_SOCKET_IFNAME**, 比如 `export GLOO_SOCKET_IFNAME=eth0`

#### 其他NCCL环境变量

NCCL还提供了许多用于微调目的的环境变量

常用的包括以下用于调试目的：

*   `export NCCL_DEBUG=INFO`
*   `export NCCL_DEBUG_SUBSYS=ALL`

有关NCCL环境变量的完整列表，请参阅[NVIDIA NCCL的官方文档](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/env.html)

## 基本

`torch.distributed`包为在一台或多台机器上运行的多个计算节点上的多进程并行性提供PyTorch支持和通信原语。类 [`torch.nn.parallel.DistributedDataParallel()`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel")基于此功能构建，以提供同步分布式训练作为包装器任何PyTorch模型。这与 [Multiprocessing package - torch.multiprocessing](multiprocessing.html) 和 [`torch.nn.DataParallel()`](nn.html#torch.nn.DataParallel "torch.nn.DataParallel") 因为它支持多个联网的机器，并且用户必须为每个进程显式启动主训练脚本的单独副本。

在单机同步的情况下，`torch.distributed` 或者 [`torch.nn.parallel.DistributedDataParallel()`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 与其他数据并行方法相比，包装器仍然具有优势，包含 [`torch.nn.DataParallel()`](nn.html#torch.nn.DataParallel "torch.nn.DataParallel"):

*   每个进程都维护自己的优化器，并在每次迭代时执行完整的优化步骤。虽然这可能看起来是多余的，但由于梯度已经聚集在一起并且在整个过程中平均，因此对于每个过程都是相同的，这意味着不需要参数广播步骤，减少了在节点之间传输张量所花费的时间。
*   每个进程都包含一个独立的Python解释器，消除了额外的解释器开销和来自单个Python进程驱动多个执行线程，模型副本或GPU的“GIL-thrashing”。这对于大量使用Python运行时的模型尤其重要，包括具有循环层或许多小组件的模型。

## 初始化

这个包在调用其他的方法之前，需要使用 [`torch.distributed.init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") 函数进行初始化。这将阻止所有进程加入。

```py
torch.distributed.init_process_group(backend, init_method='env://', timeout=datetime.timedelta(seconds=1800), **kwargs)
```

初始化默认的分布式进程组，这也将初始化分布式程序包

参数: 

*   **backend** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)") _or_ [_Backend_](#torch.distributed.Backend "torch.distributed.Backend")) – 后端使用。根据构建时配置，有效值包括 `mpi`，`gloo`和`nccl`。该字段应该以小写字符串形式给出(例如`"gloo"`)，也可以通过[`Backend`](#torch.distributed.Backend "torch.distributed.Backend")访问属性(例如`Backend.GLOO`)。
*   **init_method** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")_,_ _optional_) – 指定如何初始化进程组的URL。
*   **world_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 参与作业的进程数。
*   **rank** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 当前流程的排名。
*   **timeout** (_timedelta__,_ _optional_) – 针对进程组执行的操作超时，默认值等于30分钟，这仅适用于`gloo`后端。
*   **group_name** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")_,_ _optional__,_ _deprecated_) – 团队名字。

要启用`backend == Backend.MPI`，PyTorch需要在支持MPI的系统上从源构建，这同样适用于NCCL。

```py
class torch.distributed.Backend
```

类似枚举的可用后端类：GLOO，NCCL和MPI。

这个类的值是小写字符串，例如“gloo”。它们可以作为属性访问，例如`Backend.NCCL`。

可以直接调用此类来解析字符串，例如，`Backend(backend_str）`将检查`backend_str`是否有效，如果是，则返回解析的小写字符串。它也接受大写字符串，例如``Backend(“GLOO”）`return`“gloo”`。
注意

条目`Backend.UNDEFINED`存在但仅用作某些字段的初始值。用户既不应直接使用也不应假设存在。

```py
torch.distributed.get_backend(group=<object object>)
```

返回给定进程组的后端

| 参数: | **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。默认值是常规主进程组。如果指定了另一个特定组，则调用进程必须是`group`的一部分。
| --- | --- |
| 返回: | 给定进程组的后端作为小写字符串 |
| --- | --- |

```py
torch.distributed.get_rank(group=<object object>)
```

返回当前进程组的排名

Rank是分配给分布式进程组中每个进程的唯一标识符。它们总是从0到`world_size`的连续整数。

| 参数: | **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组|
| --- | --- |
| 返回: | 进程组-1的等级，如果不是该组的一部分 |
| --- | --- |

```py
torch.distributed.get_world_size(group=<object object>)
```

返回当前进程组中的进程数

| 参数: | **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组 |
| --- | --- |
| 返回: | 进程组-1的世界大小，如果不是该组的一部分 |
| --- | --- |

```py
torch.distributed.is_initialized()
```

检查是否已初始化默认进程组

```py
torch.distributed.is_mpi_available()
```

检查MPI是否可用

```py
torch.distributed.is_nccl_available()
```

检查NCCL是否可用

* * *

目前支持三种初始化方法：

### TCP初始化

有两种方法可以使用TCP进行初始化，这两种方法都需要从所有进程可以访问的网络地址和所需的`world_size`。第一种方法需要指定属于rank 0进程的地址。此初始化方法要求所有进程都具有手动指定的排名。

请注意，最新的分布式软件包中不再支持多播地址。`group_name`也被弃用了。

```py
import torch.distributed as dist

# 使用其中一台机器的地址
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                        rank=args.rank, world_size=4)

```

### 共享文件系统初始化

另一种初始化方法使用一个文件系统，该文件系统与组中的所有机器共享和可见，以及所需的`world_size`。URL应以`file：//`开头，并包含共享文件系统上不存在的文件(在现有目录中）的路径。如果文件不存在，文件系统初始化将自动创建该文件，但不会删除该文件。因此，下一步初始化 [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") 在相同的文件路径发生之前您有责任确保清理文件。

请注意，在最新的分布式软件包中不再支持自动排名分配，并且也不推荐使用`group_name`。

警告

此方法假定文件系统支持使用`fcntl`进行锁定 - 大多数本地系统和NFS都支持它。

警告

此方法将始终创建该文件，并尽力在程序结束时清理并删除该文件。换句话说，每次进行初始化都需要创建一个全新的空文件，以便初始化成功。如果再次使用先前初始化使用的相同文件(不会被清除），则这是意外行为，并且经常会导致死锁和故障。因此，即使此方法将尽力清理文件，如果自动删除不成功，您有责任确保在训练结束时删除该文件以防止同一文件被删除 下次再次使用。如果你打算在相同的文件系统路径下多次调用 [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") 的时候，就显得尤为重要了。换一种说法，如果那个文件没有被移除并且你再次调用 [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group")，那么失败是可想而知的。这里的经验法则是，每当调用[`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group")的时候，确保文件不存在或为空。

```py
import torch.distributed as dist

# 应始终指定等级
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                        world_size=4, rank=args.rank)

```

### 环境变量初始化

此方法将从环境变量中读取配置，从而可以完全自定义信息的获取方式。要设置的变量是：

*   `MASTER_PORT` - 需要; 必须是机器上的自由端口，等级为0。
*   `MASTER_ADDR` - 要求(0级除外）; 等级0节点的地址。
*   `WORLD_SIZE` - 需要; 可以在这里设置，也可以在调用init函数时设置。
*   `RANK` - 需要; 可以在这里设置，也可以在调用init函数时设置。

等级为0的机器将用于设置所有连接。

这是默认方法，意味着不必指定`init_method`(或者可以是`env：//`）。

## 组

默认情况下，集合体在默认组(也称为世界）上运行，并要求所有进程都进入分布式函数调用。但是，一些工作负载可以从更细粒度的通信中受益。这是分布式群体发挥作用的地方。[`new_group()`](#torch.distributed.new_group "torch.distributed.new_group") 函数可用于创建新组，具有所有进程的任意子集。它返回一个不透明的组句柄，可以作为所有集合体的“group”参数给出(集合体是分布式函数，用于在某些众所周知的编程模式中交换信息）。

目前`torch.distributed`不支持创建具有不同后端的组。换一种说法，每一个正在被创建的组都会用相同的后端，只要你在 [`init_process_group()`](#torch.distributed.init_process_group "torch.distributed.init_process_group") 里面声明清楚。

```py
torch.distributed.new_group(ranks=None, timeout=datetime.timedelta(seconds=1800))
```

创建一个新的分布式组

此功能要求主组中的所有进程(即属于分布式作业的所有进程）都进入此功能，即使它们不是该组的成员也是如此。此外，应在所有进程中以相同的顺序创建组。

参数: 

*   **ranks** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[_[_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_]_) – 小组成员的等级列表。
*   **timeout** (_timedelta__,_ _optional_) – 针对进程组执行的操作超时，默认值等于30分钟，这仅适用于`gloo`后端。


| 返回: | 分布式组的句柄，可以给予集体调用 |
| --- | --- |

## 点对点通信

```py
torch.distributed.send(tensor, dst, group=<object object>, tag=0)
```

同步发送张量

参数: 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 准备发送的张量。
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 目的地排名。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **tag** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 标记以匹配发送与远程接收。



```py
torch.distributed.recv(tensor, src=None, group=<object object>, tag=0)
```

同步接收张量

参数： 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 张量填充接收的数据。
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 来源排名。如果未指定，将从任何流程收到。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **tag** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 标记以匹配接收与远程发送。


| 返回: | 发件人排名-1，如果不是该组的一部分 |
| --- | --- |

[`isend()`](#torch.distributed.isend "torch.distributed.isend") 和 [`irecv()`](#torch.distributed.irecv "torch.distributed.irecv") 使用时返回分布式请求对象。通常，此对象的类型未指定，因为它们永远不应手动创建，但它们保证支持两种方法：

*   `is_completed()` - 如果操作已完成，则返回True。
*   `wait()` - 将阻止该过程，直到操作完成，`is_completed(）`保证一旦返回就返回True。

```py
torch.distributed.isend(tensor, dst, group=<object object>, tag=0)
```

异步发送张量

参数: 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 准本发送的张量。
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 目的地排名。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **tag** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 标记以匹配发送与远程接收。


| 返回: | 分布式请求对象。没有，如果不是该组的一部分 |
| --- | --- |

```py
torch.distributed.irecv(tensor, src, group=<object object>, tag=0)
```

异步接收张量

参数: 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 张量填充接收的数据。
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 来源排名。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **tag** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 标记以匹配接收与远程发送。


| 返回: | 分布式请求对象。没有，如果不是该组的一部分 |
| --- | --- |

## 同步和异步集合操作

每个集合操作函数都支持以下两种操作：

同步操作 - 默认模式，当`async_op`设置为False时。当函数返回时，保证执行集合操作(如果它是CUDA操作，则不一定完成，因为所有CUDA操作都是异步的），并且可以调用任何进一步的函数调用，这取决于集合操作的数据。在同步模式下，集合函数不返回任何内容。

asynchronous operation - 当`async_op`设置为True时。集合操作函数返回分布式请求对象。通常，您不需要手动创建它，并且保证支持两种方法：

*   `is_completed()` - 如果操作已完成，则返回True。
*   `wait()` - 将阻止该过程，直到操作完成。

## 集体职能

```py
torch.distributed.broadcast(tensor, src, group=<object object>, async_op=False)
```

将张量广播到整个群体

`tensor`必须在参与集合体的所有进程中具有相同数量的元素。

参数: 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 如果`src`是当前进程的等级，则发送的数据，否则用于保存接收数据的张量。
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 来源排名。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

```py
torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=<object object>, async_op=False)
```

减少所有机器上的张量数据，以便获得最终结果

调用`tensor`之后在所有进程中将按位相同。

参数: 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 集体的输入和输出。该功能就地运行。
*   **op** (_optional_) – 来自`torch.distributed.ReduceOp`枚举的值之一。指定用于逐元素减少的操作。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

```py
torch.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=<object object>, async_op=False)
```

减少所有机器的张量数据

只有排名为“dst”的进程才会收到最终结果。

参数: 

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 集体的输入和输出。该功能就地运行。
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 目的地排名。
*   **op** (_optional_) – 来自`torch.distributed.ReduceOp`枚举的值之一。指定用于逐元素减少的操作。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

```py
torch.distributed.all_gather(tensor_list, tensor, group=<object object>, async_op=False)
```

从列表中收集整个组的张量

参数：

*   **tensor_list** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – 输出列表。它应包含正确大小的张量，用于集合的输出。
*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 从当前进程广播的张量。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

```py
torch.distributed.gather(tensor, gather_list, dst, group=<object object>, async_op=False)
```

在一个过程中收集张量列表

参数：

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 输入张量。
*   **gather_list** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – 用于接收数据的适当大小的张量列表。仅在接收过程中需要。
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 目的地排名。除接收数据的进程外，在所有进程中都是必需的。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

```py
torch.distributed.scatter(tensor, scatter_list, src, group=<object object>, async_op=False)
```

将张量列表分散到组中的所有进程

每个进程只接收一个张量并将其数据存储在`tensor`参数中。

参数：

*   **tensor** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – 输出张量。
*   **scatter_list** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – 要分散的张量列表。仅在发送数据的过程中需要。
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 来源排名。除发送数据的进程外，在所有进程中都是必需的。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。


| 返回: | 异步工作句柄，如果async_op设置为True。如果不是async_op或不是组的一部分，无 |
| --- | --- |

```py
torch.distributed.barrier(group=<object object>, async_op=False)
```

同步所有进程

如果async_op为False，或者在wait(）上调用异步工作句柄，则此集合会阻止进程直到整个组进入此函数。

参数：

*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

```py
class torch.distributed.ReduceOp
```

类似枚举的可用减少操作类：`SUM`，`PRODUCT`，`MIN`和`MAX`。

该类的值可以作为属性访问，例如，`ReduceOp.SUM`。它们用于指定减少集群的战略，例如 [`reduce()`](#torch.distributed.reduce "torch.distributed.reduce"), [`all_reduce_multigpu()`](#torch.distributed.all_reduce_multigpu "torch.distributed.all_reduce_multigpu")。

成员：

> SUM
> 
> PRODUCT
> 
> MIN
> 
> MAX

```py
class torch.distributed.reduce_op
```

用于还原操作的不再使用的枚举类：`SUM`，`PRODUCT`，`MIN`和`MAX`。

建议使用[`ReduceOp`](#torch.distributed.ReduceOp "torch.distributed.ReduceOp") 代替。

## 多GPU集群功能

如果每个节点上有多个GPU，则在使用NCCL和Gloo后端时，[`broadcast_multigpu()`](#torch.distributed.broadcast_multigpu "torch.distributed.broadcast_multigpu") [`all_reduce_multigpu()`](#torch.distributed.all_reduce_multigpu "torch.distributed.all_reduce_multigpu") [`reduce_multigpu()`](#torch.distributed.reduce_multigpu "torch.distributed.reduce_multigpu") 和 [`all_gather_multigpu()`](#torch.distributed.all_gather_multigpu "torch.distributed.all_gather_multigpu") 支持每个节点内多个GPU之间的分布式集合操作。这些功能可以潜在地提高整体分布式训练性能，并通过传递张量列表轻松使用。传递的张量列表中的每个张量需要位于调用该函数的主机的单独GPU设备上。请注意，张量列表的长度在所有分布式进程中需要相同。另请注意，目前只有NCCL后端支持多GPU集合功能。

例如，如果我们用于分布式训练的系统有2个节点，每个节点有8个GPU。在16个GPU中的每一个上，都有一个我们希望减少的张量，以下代码可以作为参考：

代码在节点0上运行

```py
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl",
                        init_method="file:///distributed_test",
                        world_size=2,
                        rank=0)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)

```

代码在节点1上运行

```py
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl",
                        init_method="file:///distributed_test",
                        world_size=2,
                        rank=1)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))

dist.all_reduce_multigpu(tensor_list)

```

调用结束后，两个节点上的所有16个张量都将具有16的全减值。

```py
torch.distributed.broadcast_multigpu(tensor_list, src, group=<object object>, async_op=False, src_tensor=0)
```

使用每个节点多个GPU张量将张量广播到整个组

`tensor`必须在参与集合体的所有进程的所有GPU中具有相同数量的元素。列表中的每个张量必须位于不同的GPU上。

目前仅支持nccl和gloo后端张量应该只是GPU张量

参数：

*   **tensor_list** (_List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – 参与集群操作行动的张量。如果`src`是排名，那么``tensor_list`(`tensor_list [src_tensor]`）的`src_tensor``元素将被广播到src进程中的所有其他张量(在不同的GPU上）以及`tensor_list中的所有张量 `其他非src进程。您还需要确保调用此函数的所有分布式进程的`len(tensor_list）`是相同的。
*   **src** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 源排行。
*   **group** (_ProcessGroup__,_ _optional_) – 要被处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。
*   **src_tensor** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 源张量等级在`tensor_list`内。


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

```py
torch.distributed.all_reduce_multigpu(tensor_list, op=ReduceOp.SUM, group=<object object>, async_op=False)
```

减少所有机器上的张量数据，以便获得最终结果。此功能可减少每个节点上的多个张量，而每个张量位于不同的GPU上。因此，张量列表中的输入张量需要是GPU张量。此外，张量列表中的每个张量都需要驻留在不同的GPU上。

在调用之后，`tensor_list`中的所有`tensor`在所有进程中都是按位相同的。

目前仅支持nccl和gloo后端，张量应仅为GPU张量。

参数：

*   **list** (_tensor_) – 集体的输入和输出张量列表。该功能就地运行，并要求每个张量在不同的GPU上为GPU张量。您还需要确保调用此函数的所有分布式进程的`len(tensor_list）`是相同的。
*   **op** (_optional_) – 来自`torch.distributed.ReduceOp`枚举的值之一，并且指定一个逐元素减少的操作。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

```py
torch.distributed.reduce_multigpu(tensor_list, dst, op=ReduceOp.SUM, group=<object object>, async_op=False, dst_tensor=0)
```

减少所有计算机上多个GPU的张量数据。`tensor_list`中的每个张量应位于单独的GPU上。

只有级别为'dst`的进程中的'tensor_list [dst_tensor]`的GPU才会收到最终结果。

目前仅支持nccl后端张量应该只是GPU张量。

参数：

*   **tensor_list** (_List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – 输入和输出集体的GPU张量。该功能就地运行，您还需要确保调用此函数的所有分布式进程的`len(tensor_list）`是相同的。
*   **dst** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 目的地排名。
*   **op** (_optional_) – 来自`torch.distributed.ReduceOp`枚举的值之一。指定一个逐元素减少的操作。
*   **group** (_ProcessGroup__,_ _optional_) – The process group to work on
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作。
*   **dst_tensor** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")_,_ _optional_) – 目标张量在`tensor_list`中排名。


| 返回: | 异步工作句柄，如果async_op设置为True。没有，否则 |
| --- | --- |

```py
torch.distributed.all_gather_multigpu(output_tensor_lists, input_tensor_list, group=<object object>, async_op=False)
```

从列表中收集整个组的张量。`tensor_list`中的每个张量应位于单独的GPU上。

目前仅支持nccl后端张量应该只是GPU张量。

参数： 

*   **output_tensor_lists** (_List__[__List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]__]_) – 输出列表。它应该在每个GPU上包含正确大小的张量，以用于集合的输出。例如 `output_tensor_lists [i]`包含驻留在`input_tensor_list [i]`的GPU上的all_gather结果。请注意，`output_tensor_lists [i]`的每个元素都具有`world_size * len(input_tensor_list）`的大小，因为该函数全部收集组中每个GPU的结果。要解释`output_tensor_list [i]`的每个元素，请注意等级k的`input_tensor_list [j]`将出现在`output_tensor_list [i] [rank * world_size + j]中。还要注意`len(output_tensor_lists）`，并且`output_tensor_lists`中的每个元素的大小(每个元素都是一个列表，因此`len(output_tensor_lists [i]）`）对于调用此函数的所有分布式进程都需要相同。
*   **input_tensor_list** (_List__[_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_]_) – 从当前进程广播的张量(在不同的GPU上）的列表。请注意，调用此函数的所有分布式进程的`len(input_tensor_list）`必须相同。
*   **group** (_ProcessGroup__,_ _optional_) – 要处理的进程组。
*   **async_op** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 这个操作是否应该是异步操作


| 返回: | 异步工作句柄，如果async_op设置为True。无，如果不是async_op或不是组的一部分 |
| --- | --- |

## 启动实用程序

`torch.distributed`包还在`torch.distributed.launch`中提供了一个启动实用程序。此帮助实用程序可用于为每个节点启动多个进程以进行分布式训练。该实用程序还支持python2和python3。

`torch.distributed.launch`是一个模块，它在每个训练节点上产生多个分布式训练过程。

该实用程序可用于单节点分布式训练，其中将生成每个节点的一个或多个进程。该实用程序可用于CPU训练或GPU训练。如果该实用程序用于GPU训练，则每个分布式进程将在单个GPU上运行。这可以实现良好改进的单节点训练性能。它还可以用于多节点分布式训练，通过在每个节点上产生多个进程来获得良好改进的多节点分布式训练性能。这对于具有多个具有直接GPU支持的Infiniband接口的系统尤其有利，因为所有这些接口都可用于聚合通信带宽。

在单节点分布式训练或多节点分布式训练的两种情况下，该实用程序将为每个节点启动给定数量的进程(`--nproc_per_node`）。如果用于GPU训练，此数字需要小于或等于当前系统上的GPU数量('nproc_per_node`），并且每个进程将在单个GPU上运行，从_GPU 0到GPU(nproc_per_node - 1）_。

**如何使用这个模块：**

1.  单节点多进程分布式训练

```py
>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
 arguments of your training script)

```

1.  多节点多进程分布式训练:(例如两个节点）

节点1：_(IP：192.168.1.1，并且有一个空闲端口：1234）_

```py
>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
 --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
 and all other arguments of your training script)

```

节点2：

```py
>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
 --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
 --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
 and all other arguments of your training script)

```

1.查找此模块提供的可选参数：

```py
>>> python -m torch.distributed.launch --help

```

**重要告示：**

1\. 这种实用和多进程分布式(单节点或多节点）GPU训练目前仅使用NCCL分布式后端实现最佳性能。因此，NCCL后端是用于GPU训练的推荐后端。

2\. 在您的训练程序中，您必须解析命令行参数：`--local_rank = LOCAL_PROCESS_RANK`，这将由此模块提供。如果您的训练计划使用GPU，则应确保您的代码仅在LOCAL_PROCESS_RANK的GPU设备上运行。这可以通过以下方式完成：

解析local_rank参数

```py
>>> import argparse
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument("--local_rank", type=int)
>>> args = parser.parse_args()

```

使用其中一个将您的设备设置为本地排名

```py
>>> torch.cuda.set_device(arg.local_rank)  # before your code runs

```

或者

```py
>>> with torch.cuda.device(arg.local_rank):
>>>    # your code to run

```

3\. 在您的训练计划中，您应该在开始时调用以下函数来启动分布式后端。您需要确保init_method使用`env：//`，这是该模块唯一支持的`init_method`。

```py
torch.distributed.init_process_group(backend='YOUR BACKEND',
                                     init_method='env://')

```

4\. 在您的训练计划中，您可以使用常规分布式功能或使用 [`torch.nn.parallel.DistributedDataParallel()`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 模块。如果您的训练计划使用GPU进行训练，并且您希望使用 [`torch.nn.parallel.DistributedDataParallel()`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 模块。这里是如何配置它。

```py
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[arg.local_rank],
                                                  output_device=arg.local_rank)

```

请确保将`device_ids`参数设置为您的代码将在其上运行的唯一GPU设备ID。这通常是流程的本地排名。换句话说，`device_ids`需要是`[args.local_rank]`，`output_device`需要是'args.local_rank`才能使用这个实用程序。

警告

`local_rank`不是全局唯一的：它只对机器上的每个进程唯一。因此，不要使用它来决定是否应该，例如，写入网络文件系统，参考 [https://github.com/pytorch/pytorch/issues/12042](https://github.com/pytorch/pytorch/issues/12042) 例如，如果您没有正确执行此操作，事情可能会出错。

## Spawn实用程序

在  [`torch.multiprocessing.spawn()`](multiprocessing.html#torch.multiprocessing.spawn "torch.multiprocessing.spawn") 里面，torch.multiprocessing包还提供了一个`spawn`函数. 此辅助函数可用于生成多个进程。它通过传递您要运行的函数并生成N个进程来运行它。这也可以用于多进程分布式训练。

有关如何使用它的参考，请参阅 [PyToch example - ImageNet implementation](https://github.com/pytorch/examples/tree/master/imagenet)

请注意，此函数需要Python 3.4或更高版本。

