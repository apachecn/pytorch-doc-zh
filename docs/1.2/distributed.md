# 分布式通信包 - torch.distributed

## 后端

`torch.distributed`支持三种后端，每个具有不同的能力。下表显示了哪些功能可用于与CPU / CUDA张量使用。
MPI支持CUDA只有在执行用于构建PyTorch支持它。

后端

|

`GLOO`

|

`MPI`

|

`NCCL` 
  
---|---|---|---  
  
设备

|

中央处理器

|

GPU

|

CPU

|

GPU

|

CPU

|

GPU  
  
发送

|

✓

|

✘

|

✓

|

？

|

✘

|

✘  
  
的recv

|

✓

|

✘

|

✓

|

?

|

✘

|

✘  
  
广播

|

✓

|

✓

|

✓

|

?

|

✘

|

✓  
  
all_reduce

|

✓

|

✓

|

✓

|

?

|

✘

|

✓  
  
降低

|

✓

|

✘

|

✓

|

?

|

✘

|

✓  
  
all_gather

|

✓

|

✘

|

✓

|

?

|

✘

|

✓  
  
收集

|

✓

|

✘

|

✓

|

?

|

✘

|

✘  
  
分散

|

✓

|

✘

|

✓

|

?

|

✘

|

✘  
  
屏障

|

✓

|

✘

|

✓

|

?

|

✘

|

✓  
  
### 来与后端PyTorch

PyTorch目前仅分布支持Linux。默认情况下，GLOO和NCCL后端构建和包含在分布式PyTorch（NCCL只有CUDA建设时）。
MPI是一个可选的后端，如果你从源代码编译PyTorch只能被包括在内。 （例如构建PyTorch已安装MPI的主机上）。

### 其后端使用？

在过去，我们经常被问道：“我应该用哪个后端？”。

  * 经验法则

    * 使用NCCL后端分布式 **GPU** 培训

    * 使用GLOO后端分布式 **CPU** 培训。

  * 与InfiniBand互联GPU主机

    * 使用NCCL，因为它是目前支持InfiniBand和GPUDirect唯一的后端。

  * 与以太网互连GPU主机

    * 使用NCCL，因为它目前提供最好的分布式GPU训练的性能，特别是对于多进程单节点或多节点分布式训练。如果您遇到任何NCCL问题，使用GLOO作为后备选项。 （请注意，目前GLOO运行速度比NCCL慢于GPU的。）

  * CPU主机与InfiniBand互联

    * 如果您的InfiniBand已经启用IP超过IB，使用GLOO，否则，使用MPI来代替。我们计划在即将到来的版本中添加了对GLOO支持InfiniBand。

  * CPU主机与以太网互连

    * 使用GLOO，除非你有使用MPI具体原因。

### 常见的环境变量

#### 选择网络接口来使用

默认情况下，NCCL和GLOO后端都将尝试找到正确的网络接口使用。如果自动检测到的接口是不正确的，你可以使用下面的环境变量（适用于各自的后端）覆盖它：

  * **NCCL_SOCKET_IFNAME** ，例如`出口 NCCL_SOCKET_IFNAME = eth0的 `

  * **GLOO_SOCKET_IFNAME** ，例如`出口 GLOO_SOCKET_IFNAME = eth0的 `

如果您使用的GLOO后台，你​​可以通过用逗号隔开他们，像这样指定多个接口：`出口 GLOO_SOCKET_IFNAME
=为eth0，eth1的，ETH2，ETH3  [ HTG5。后端会通过这些接口一个循环方式调度操作。至关重要的是，所有的进程指定在此变量相同数量的接口。`

#### 其他NCCL环境变量

NCCL还提供了一些环境变量进行微调的目的。

常用的包括用于调试的目的如下：

  * `出口 NCCL_DEBUG = INFO`

  * `出口 NCCL_DEBUG_SUBSYS = ALL`

对于NCCL环境变量的完整列表，请参阅[ NVIDIA
NCCL的官方文档](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-
guide/docs/env.html)

## 基础

的 torch.distributed 包提供了跨在一个或多个计算机上运行的几个计算节点对多进程并行PyTorch支持与通信原语。类[ `
torch.nn.parallel.DistributedDataParallel（） `
](nn.html#torch.nn.parallel.DistributedDataParallel
"torch.nn.parallel.DistributedDataParallel")基于这样一种功能，以提供同步分布式训练为围绕任何PyTorch模型的包装。这不同于由[
多处理包中提供的种并行 - torch.multiprocessing  ](multiprocessing.html)和[ `
torch.nn.DataParallel（） `](nn.html#torch.nn.DataParallel
"torch.nn.DataParallel")，它支持多个网络连接的机器和在用户必须明确地启动主训练脚本的单独副本为每个进程。

在单机同步的情况下， torch.distributed 或[ `torch.nn.parallel.DistributedDataParallel（）
`](nn.html#torch.nn.parallel.DistributedDataParallel
"torch.nn.parallel.DistributedDataParallel")包装纸可能仍然有在其他的方法来数据并行的优点，包括[ `
torch.nn.DataParallel（） `](nn.html#torch.nn.DataParallel
"torch.nn.DataParallel")：

  * 每个进程维护自己的优化，并执行与每个迭代一个完整的优化步骤。虽然这可能会出现多余的，因为梯度已经聚集和跨进程平均，并且因此对于每个过程是相同的，这意味着没有参数广播步骤是需要的，减少花费的节点之间传送张量的时间。

  * 每个进程都包含一个独立的Python解释器，消除了多余的解释开销以及来自来自一个Python程序驱动多执行绪，模型复制品，或GPU“GIL-颠簸”。这是一个模型，大量使用Python运行时，包括复发层或很多小的组件模型尤为重要。

## 初始化

程序包需要调用任何其他方法之前使用 `torch.distributed.init_process_group`
（）函数被初始化。这将阻止，直到所有进程都加入。

`torch.distributed.``init_process_group`( _backend_ , _init_method=None_ ,
_timeout=datetime.timedelta(0_ , _1800)_ , _world_size=-1_ , _rank=-1_ ,
_store=None_ , _group_name=''_
)[[source]](_modules/torch/distributed/distributed_c10d.html#init_process_group)

    

初始化默认的分布式进程组，而这也将初始化分发包。

There are 2 main ways to initialize a process group:

    

  1. 指定`店 `，`位次 `和`world_size`明确。

  2. 指定`init_method`（URL字符串），其指示其中/如何发现对等体。有选择地指定`秩 `和`world_size`，或编码在URL所有必需的参数，并省略它们。

如果不指定，`init_method`假设为“ENV：//”。

Parameters

    

  * **后端** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _或_ _后端_ ） - 后端使用。取决于构建时配置中，有效的值包括`MPI`，`GLOO`和`NCCL  [ HTG23。该字段应该被给定为小写字符串（例如，`“GLOO” `），其也可以通过 `后端 [HTG32访问] `属性（例如，`Backend.GLOO`）。如果使用每个机器的多个进程使用`NCCL`后端，每个进程必须具有它使用每个GPU独占访问，如进程之间共享的GPU可以导致死锁。`

  * **init_method** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _，_ _可选_ ） - URL指定如何初始化进程组。默认值是“ENV：//”如果没有指定`init_method`或`店 `。互斥与`店 `。

  * **world_size** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 的参与工作进程数。如果`存储指定 `是必需的。

  * **秩** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 当前进程的秩。如果`存储指定 `是必需的。

  * **店** （ _存储_ _，_ _可选_ ） - 键/值存储的所有员工都可以访问，用于交换连接/地址信息。互斥与`init_method`。

  * **超时** （ _timedelta_ _，_ _可选_ ） - 超时针对进程组执行的操作。默认值等于30分钟。这是仅适用于`GLOO`后端。

  * **GROUP_NAME** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _，_ _可选_ _，_ _弃用_ ） - 组名。

为了使`后端 ==  Backend.MPI`，PyTorch需要从源内置支持MPI的系统上。这同样适用于NCCL为好。

_class_`torch.distributed.``Backend`[[source]](_modules/torch/distributed/distributed_c10d.html#Backend)

    

枚举类类可用后端的：GLOO，NCCL和MPI。

这个类的值是字符串小写，例如，`“GLOO” `。它们可以作为属性，例如被访问，`Backend.NCCL`。

这个类可以直接调用来解析字符串，如`后端（backend_str）HTG2] `将检查`backend_str
`是有效的，而回报解析小写的字符串，如果是的话。它还接受大写字符串，例如，`后端（ “GLOO”） `返回`“GLOO” `。

注意

入口`Backend.UNDEFINED`存在，但仅作为一些字段的初始值。用户应不直接使用它，也没有承担起它的存在。

`torch.distributed.``get_backend`( _group= <object
object>_)[[source]](_modules/torch/distributed/distributed_c10d.html#get_backend)

    

返回给定工艺组的后端。

Parameters

    

**组** （ _ProcessGroup_ _，_ _可选_ ） - 进程组上下工夫。默认值是一般主进程组。如果指定了另一个特定的组，调用进程必须是`组
`部分。

Returns

    

给定的处理组作为小写字符串的后端。

`torch.distributed.``get_rank`( _group= <object
object>_)[[source]](_modules/torch/distributed/distributed_c10d.html#get_rank)

    

返回当前进程组的秩

秩是分布式处理组内的分配给每个进程的唯一标识符。他们总是连续整数范围从0到`world_size  [HTG3。`

Parameters

    

**组** （ _ProcessGroup_ _，_ _可选_ ） - 进程组上下工夫

Returns

    

进程组-1的秩，如果不是组的一部分

`torch.distributed.``get_world_size`( _group= <object
object>_)[[source]](_modules/torch/distributed/distributed_c10d.html#get_world_size)

    

返回当前处理组中的进程数

Parameters

    

**group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

Returns

    

进程组-1的世界大小，如果不是组的一部分

`torch.distributed.``is_initialized`()[[source]](_modules/torch/distributed/distributed_c10d.html#is_initialized)

    

检查是否默认进程组已初始化

`torch.distributed.``is_mpi_available`()[[source]](_modules/torch/distributed/distributed_c10d.html#is_mpi_available)

    

检查该MPI后端是可用的。

`torch.distributed.``is_nccl_available`()[[source]](_modules/torch/distributed/distributed_c10d.html#is_nccl_available)

    

检查该NCCL后端是可用的。

* * *

目前有三个初始化方法的支持：

### TCP的初始化

有两种方式使用TCP来初始化，既需要网络地址从所有进程到达和期望`world_size
`。第一种方法要求指定属于等级0进程的地址。这种初始化方法要求所有的过程都有手动指定的行列。

请注意，多播地址未在最新的分布式包支持了。 `组名 `已被弃用，以及。

    
    
    import torch.distributed as dist
    
    # Use address of one of the machines
    dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                            rank=args.rank, world_size=4)
    

### 共享文件系统初始化

另一个初始化方法利用被共享的文件系统，并从可见的所有机器的基团中，与期望的`world_size`沿。 URL应以`文件开始：//
`和含有一个共享文件系统到一个不存在的文件的路径（在现有的目录中）。文件系统初始化将自动创建文件，如果它不存在，但不会删除该文件。因此，它是你的责任，以确保该文件之前清理下。
`init_process_group（） `调用同一文件路径/文件名。

需要注意的是自动排名分配没有在最新的分布式包装不再支持和`GROUP_NAME`已废弃好。

警告

此方法假定文件系统支持使用`的fcntl`锁定 - 大多数本地系统和NFS支持。

Warning

此方法将总是创建该文件，尽力清理，并在程序结束时删除该文件。换句话说，与文件init方法每次初始化需要一个全新的空文件，以便初始化成功。如果先前的初始化（恰好没有得到清理）使用相同的文件再次使用，这是意外的行为，常可引起死锁和失败。因此，即使这种方法会尽力清理文件，如果自动删除恰好是不成功的，这是你的责任，以确保该文件是在训练结束删除，以防止同一个文件是接下来的时间期间再次重复使用。如果您打算调用
`init_process_group（） `在同一个文件名多次，这一点尤为重要。换句话说，如果该文件不会被删除/清理和调用 `
init_process_group（） `[HTG11再次在该文件中，失败是期望。这里的经验法则是，请确保该文件不存在或为空，每次 `
init_process_group（） `被调用。

    
    
    import torch.distributed as dist
    
    # rank should always be specified
    dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                            world_size=4, rank=args.rank)
    

### 环境变量初始化

这种方法将读取的环境变量的配置，允许一个完全自定义如何获得的信息。要设置的变量是：

  * `MASTER_PORT`\- 所需;必须是机器上的空闲端口与秩0

  * `MASTER_ADDR`\- 需要（除了秩0）;秩0节点的地址

  * `WORLD_SIZE`\- 所需;既可以在此设置，或者在一个呼叫到INIT功能

  * `RANK`\- 所需;既可以在此设置，或者在一个呼叫到INIT功能

秩0的机器将被用来建立的所有连接。

这是默认的方法，这意味着`init_method`没有被指定的（或可以是`ENV：//`）。

## 组

默认情况下，集体的默认组（也称为世界）工作，并要求所有进程进入分布函数调用。然而，一些工作负载可以受益于更细粒度的通信。这是分布式的群体发挥作用。`
new_group（） `函数可用于创建新的组，所有进程的任意子集。它返回可以作为一个`组
`参数向所有集体的不透明基手柄（集体分布函数的某些公知的编程模式交换信息）。

`torch.distributed.``new_group`( _ranks=None_ , _timeout=datetime.timedelta(0_
, _1800)_ , _backend=None_
)[[source]](_modules/torch/distributed/distributed_c10d.html#new_group)

    

创建一个新的分布式组。

此功能要求的主要组中的所有进程（即是分布式工作的一部分的进程）进入该功能，即使他们不打算成为组的成员。此外，集团应该在所有进程以相同的顺序创建。

Parameters

    

  * **行列** （[ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _[_ [ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") __ ） - 小组成员的队伍名单。

  * **timeout** ( _timedelta_ _,_ _optional_ ) – Timeout for operations executed against the process group. Default value equals 30 minutes. This is only applicable for the `gloo`backend.

  * **后端** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)") _或_ _后端_ _，_ _可选的_ ） - 后端使用。取决于构建时的配置，有效值为`GLOO`和`NCCL`。默认情况下，使用相同的后端为全局组。该字段应该被给定为小写字符串（例如，`“GLOO” `），其也可以通过 `后端 [HTG32访问] `属性（例如，`Backend.GLOO`）。

Returns

    

分布式组的句柄，可以给集体呼吁。

## 点对点通信

`torch.distributed.``send`( _tensor_ , _dst_ , _group= <object object>_,
_tag=0_ )[[source]](_modules/torch/distributed/distributed_c10d.html#send)

    

同步发送一个张量。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来发送。

  * **DST** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 目的地等级。

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **标记** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 标签，以匹配与远程的recv发送

`torch.distributed.``recv`( _tensor_ , _src=None_ , _group= <object object>_,
_tag=0_ )[[source]](_modules/torch/distributed/distributed_c10d.html#recv)

    

同步接收一个张量。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量来填充接收的数据。

  * **SRC** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 来源秩。将收到如果未指定任何进程。

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **标记** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - 标签以匹配RECV与远程发送

Returns

    

发件人等级-1，如果不是组的一部分

`isend（） `和 `irecv（） `
当用于返回分布式请求对象。在一般情况下，该对象的类型是未指定的，因为它们不应该被手动创建，但它们保证支持两种方法：

  * `is_completed（） `\- 返回真，如果操作已完成

  * `等待（） `\- 将直至操作完成块的过程。 `is_completed（） `保证返回真一旦它返回。

`torch.distributed.``isend`( _tensor_ , _dst_ , _group= <object object>_,
_tag=0_ )[[source]](_modules/torch/distributed/distributed_c10d.html#isend)

    

异步发送一个张量。

Parameters

    

  * **tensor** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Tensor to send.

  * **dst** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Destination rank.

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **tag** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – Tag to match send with remote recv

Returns

    

分布式请求对象。无，如果不是组的一部分

`torch.distributed.``irecv`( _tensor_ , _src_ , _group= <object object>_,
_tag=0_ )[[source]](_modules/torch/distributed/distributed_c10d.html#irecv)

    

异步接收一个张量。

Parameters

    

  * **tensor** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Tensor to fill with received data.

  * **SRC** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 来源秩。

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **tag** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _,_ _optional_ ) – Tag to match recv with remote send

Returns

    

A distributed request object. None, if not part of the group

## 同步和异步共同操作

每个集体操作功能支持以下两种操作：

同步操作 - 默认模式中，当`async_op
`设定为False。当函数返回时，可以保证执行集体操作的任何进一步的函数调用取决于集体操作的数据可以被称为（不一定完成，如果它是一个CUDA运算，因为所有的CUDA
OPS是异步），和。在同步模式，集体功能不会返回任何东西

异步操作 - 当`async_op`设置为True。集体操作函数返回一个分布式请求对象。一般情况下，你不需要手动创建它，它是保证支持两种方法：

  * `is_completed()`\- returns True if the operation has finished

  * `等待（） `\- 将直至操作完成块的过程。

## 集体函数

`torch.distributed.``broadcast`( _tensor_ , _src_ , _group= <object object>_,
_async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#broadcast)

    

广播张全群。

`张量 `必须具有相同数目的参与该集体的所有进程的元素。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 如果`SRC`是当前进程的秩要发送的数据，和张量以用来保存否则所接收的数据。

  * **src** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Source rank.

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 这是否OP应该是一个异步运

Returns

    

异步工作手柄，如果async_op设置为True。无，如果没有async_op或者如果不是组的一部分

`torch.distributed.``all_reduce`( _tensor_ , _op=ReduceOp.SUM_ , _group=
<object object>_, _async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#all_reduce)

    

减少以这样的方式，所有得到最终结果在所有机器上的数据张。

呼叫`张量之后 `将被逐位在所有过程是相同的。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入和集体的输出。该函数就地操作。

  * **OP** （ _可选[HTG3） - 酮从`的值 torch.distributed.ReduceOp`枚举的。指定用于元素方面减少的操作。_

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

`torch.distributed.``reduce`( _tensor_ , _dst_ , _op=ReduceOp.SUM_ , _group=
<object object>_, _async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#reduce)

    

减少在所有机器上的数据张。

只有等级的过程`DST`将要接收的最终结果。

Parameters

    

  * **tensor** ([ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – Input and output of the collective. The function operates in-place.

  * **DST** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 目的地等级

  * **op** ( _optional_ ) – One of the values from `torch.distributed.ReduceOp`enum. Specifies an operation used for element-wise reductions.

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

`torch.distributed.``all_gather`( _tensor_list_ , _tensor_ , _group= <object
object>_, _async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#all_gather)

    

汇集了来自全团张量在列表中。

Parameters

    

  * **tensor_list** （[ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 输出列表。它应该包含用于集体的正确输出大小的张量。

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 张量从当前进程广播。

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

`torch.distributed.``gather`( _tensor_ , _gather_list_ , _dst_ , _group=
<object object>_, _async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#gather)

    

集张量在单个进程的列表。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输入张量。

  * **gather_list** （[ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 适当大小的张量清单用于接收数据。仅在接收过程中必需。

  * **DST** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 目的地等级。需要不同之处在于receiveing数据的一个所有进程。

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

`torch.distributed.``scatter`( _tensor_ , _scatter_list_ , _src_ , _group=
<object object>_, _async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#scatter)

    

散射张量的一组中的所有进程的列表。

每个进程将收到恰好一个张量并存储在`张量 `参数其数据。

Parameters

    

  * **张量** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")） - 输出张量。

  * **scatter_list** （[ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)") _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 张量清单散落一地。仅在正在发送数据的过程必需。

  * **SRC** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 来源秩。需要不同之处在于发送数据的一个所有进程。

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

`torch.distributed.``barrier`( _group= <object object>_, _async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#barrier)

    

同步所有进程。

这种集体块处理，直到全团进入该功能，如果async_op是假，或者如果异步工作手柄上调用wait（）的。

Parameters

    

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

_class_`torch.distributed.``ReduceOp`

    

枚举状类的可用的减少的操作：`SUM`，`产物 `，`MIN`和`MAX`。

这个类的值可以作为属性，例如被访问，`ReduceOp.SUM`。它们在指定为降低集体，例如策略中使用的， `减少（） `， `
all_reduce_multigpu（） `等

成员：

> 和

>

> 产品

>

> MIN

>

> MAX

_class_`torch.distributed.``reduce_op`[[source]](_modules/torch/distributed/distributed_c10d.html#reduce_op)

    

减少操作弃用枚举状类：`SUM`，`产物 `，`MIN`和`MAX`。

`ReduceOp`建议改用。

## 多GPU集体函数

如果你有每个节点在一个以上的GPU，使用NCCL和GLOO后端时， `broadcast_multigpu（） ``
all_reduce_multigpu （） ``reduce_multigpu（） `和 `all_gather_multigpu（） `
支持分布式每个节点内多个GPU之间的集体操作。这些功能可以潜在地提高整体的分布式训练的性能和很容易被路过张量的列表中。在通过张量列表中的每个张量必须在函数被调用主机的独立GPU设备上。需要注意的是，张量清单的长度需要所有的分布式进程之间是相同的。还要注意的是目前的多GPU集体功能仅由NCCL后端支持。

例如，如果我们使用用于分布式训练的系统具有2个节点，其中的每一个具有8个GPU。在每个16个GPU的，还有的是，我们希望所有减少的张量。下面的代码可以作为参考：

守则节点0运行

    
    
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
    

代码节点上运行1

    
    
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
    

通话结束后，两个节点上的所有16张量将有16的全价值降低

`torch.distributed.``broadcast_multigpu`( _tensor_list_ , _src_ , _group=
<object object>_, _async_op=False_ , _src_tensor=0_
)[[source]](_modules/torch/distributed/distributed_c10d.html#broadcast_multigpu)

    

广播张量与每节点的多个GPU张量全组。

`张量 `必须具有相同数目的在从参与集体的所有进程的所有的GPU元件。列表中的每个张量必须在不同的GPU

只有NCCL和GLOO后端目前支持张量应该只是GPU张量

Parameters

    

  * **tensor_list** （ _列表_ _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 参与集体张量操作。如果`SRC`是秩，则`tensor_list`指定`src_tensor`元素（ `tensor_list [src_tensor]`）将被广播到在src过程的所有其他张量（在不同的GPU）和所有张量在`tensor_list`的其它非SRC过程。您还需要确保`LEN（tensor_list）HTG34] `是所有分布式进程调用此函数相同。

  * **src** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Source rank.

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

  * **src_tensor** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - `tensor_list内源张量秩 `

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

`torch.distributed.``all_reduce_multigpu`( _tensor_list_ , _op=ReduceOp.SUM_ ,
_group= <object object>_, _async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#all_reduce_multigpu)

    

减少以这样的方式，所有得到最终结果在所有机器上的数据张。此功能可降低一个数量的每个节点上的张量，而每个张量驻留在不同的GPU。因此，在张量列表中输入张量需要为GPU张量。此外，在张量列表中的每个张量需要驻留在不同的GPU。

通话结束后，所有`张量 `在`tensor_list`将被逐位在所有过程是相同的。

只有NCCL和GLOO后端目前支持张量应该只是GPU张量

Parameters

    

  * **列表** （ _张量_ ） - 集体的输入和输出张量的列表。功能就地操作，并且要求每个张量，以在不同的GPU一个GPU张量。您还需要确保`LEN（tensor_list）HTG6] `是所有分布式进程调用此函数相同。

  * **op** ( _optional_ ) – One of the values from `torch.distributed.ReduceOp`enum. Specifies an operation used for element-wise reductions.

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

`torch.distributed.``reduce_multigpu`( _tensor_list_ , _dst_ ,
_op=ReduceOp.SUM_ , _group= <object object>_, _async_op=False_ ,
_dst_tensor=0_
)[[source]](_modules/torch/distributed/distributed_c10d.html#reduce_multigpu)

    

减少了对所有机器多个GPU张量数据。在`[HTG1每一张量tensor_list `应该驻留在单独的GPU

的`只有GPU tensor_list [dst_tensor]`与位次`DST`将要接收的最终结果的过程。

目前仅支持NCCL后端张量应该只是GPU张量

Parameters

    

  * **tensor_list** （ _列表_ _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 的输入和输出GPU张量集体。该函数就地操作。您还需要确保`LEN（tensor_list）HTG14] `是所有分布式进程调用此函数相同。

  * **dst** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – Destination rank

  * **op** ( _optional_ ) – One of the values from `torch.distributed.ReduceOp`enum. Specifies an operation used for element-wise reductions.

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

  * **dst_tensor** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)") _，_ _可选_ ） - `tensor_list内目的地张量秩 `

Returns

    

异步工作手柄，如果async_op设置为True。无，否则

`torch.distributed.``all_gather_multigpu`( _output_tensor_lists_ ,
_input_tensor_list_ , _group= <object object>_, _async_op=False_
)[[source]](_modules/torch/distributed/distributed_c10d.html#all_gather_multigpu)

    

汇集了来自全团张量在列表中。在`[HTG1每一张量tensor_list `应该驻留在单独的GPU

Only nccl backend is currently supported tensors should only be GPU tensors

Parameters

    

  * **output_tensor_lists** （ _列表_ _[_ _列表_ _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ __ ） - 

输出列表。它应包含在每个GPU正确大小的张量要用于集体，例如输出`output_tensor_lists [I]`包含驻留在的`
input_tensor_list [I]`GPU上的all_gather结果。

需要注意的是output_tensor_lists 的`每个元件具有的`world_size  *  LEN（input_tensor_list）的大小
`，因为该函数的所有收集来自该组中的每一个GPU的结果。为了解释的`每个元素output_tensor_lists [I]`，请注意，`
input_tensor_list [j]的 `秩k将是出现在`output_tensor_lists [I] [K  *  world_size  \+
j]的 ``

还要注意的是`LEN（output_tensor_lists） `和in `的每个元件的尺寸output_tensor_lists
`（每个元素是一个列表，因此`LEN（output_tensor_lists [I]） `）必须对所有的分布式进程调用此函数是相同的。

  * **input_tensor_list** （ _列表_ _[_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") __ ） - 张量的列表（在不同的图形处理器），以从当前进程广播。需要注意的是`LEN（input_tensor_list）HTG14] `需要为所有的分布式进程调用此函数相同。

  * **group** ( _ProcessGroup_ _,_ _optional_ ) – The process group to work on

  * **async_op** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – Whether this op should be an async op

Returns

    

Async work handle, if async_op is set to True. None, if not async_op or if not
part of the group

## 启动程序

的 torch.distributed 包还提供了在 torch.distributed.launch
发射工具。这个辅助工具可以用来启动每个节点的多个进程的分布式训练。此实用程序还支持python2和python3。

## 衍生实用程序

的 torch.multiprocessing 包还提供了`菌种 `函数在[ `torch.multiprocessing.spawn（） `
](multiprocessing.html#torch.multiprocessing.spawn
"torch.multiprocessing.spawn")。这个辅助函数可以用来产卵多个进程。其工作原理是通过在要运行，并产生数处理运行它的功能。这可以用于多进程分布式训练为好。

有关如何使用它的引用，请参考[ PyTorch例子 -
ImageNet实现](https://github.com/pytorch/examples/tree/master/imagenet)

请注意，此功能需要Python 3.4或更高版本。

[Next ![](_static/images/chevron-right-orange.svg)](distributions.html
"Probability distributions - torch.distributions")
[![](_static/images/chevron-right-orange.svg) Previous](autograd.html
"Automatic differentiation package - torch.autograd")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * [HTG0分布式通信包 - torch.distributed 
    * 后端
      * 附带PyTorch后端
      * 哪个后端使用？ 
      * 通用环境变量
        * 选择的网络接口来使用
        * [HTG0其他NCCL环境变量
    * 基础
    * 初始化
      * TCP初始化
      * 分享文件系统初始化
      * 环境变量初始化
    * 群组
    * 点对点通信
    * 同步和异步集合操作
    * 集体函数
    * 多GPU集体函数
    * 启动实用程序
    * 衍生实用程序

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

