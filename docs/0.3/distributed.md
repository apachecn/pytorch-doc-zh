# Distributed communication package - torch.distributed

> 译者：[@Mu Wu9527](https://github.com/yichuan9527)
> 
> 校对者：[@smilesboy](https://github.com/smilesboy)

torch.distributed 提供类似 MPI 的前向运算机制, 支持在多台机的网络中交换数据. 支持不同的后段和初始化方法.

目前torch.distributed支持三个后端, 每个都有不同的功能. 下表显示哪些功能可用于 CPU/CUDA 张量. 只有在设备上编译安装PyTorch, 才能在MPI的设备上支持cuda.

| Backend | `tcp` | `gloo` | `mpi` |
| --- | --- | --- | --- |
| Device | CPU | GPU | CPU | GPU | CPU | GPU |
| --- | --- | --- | --- | --- | --- | --- |
| send | ✓ | ✘ | ✘ | ✘ | ✓ | ? |
| recv | ✓ | ✘ | ✘ | ✘ | ✓ | ? |
| broadcast | ✓ | ✘ | ✓ | ✓ | ✓ | ? |
| all_reduce | ✓ | ✘ | ✓ | ✓ | ✓ | ? |
| reduce | ✓ | ✘ | ✘ | ✘ | ✓ | ? |
| all_gather | ✓ | ✘ | ✘ | ✘ | ✓ | ? |
| gather | ✓ | ✘ | ✘ | ✘ | ✓ | ? |
| scatter | ✓ | ✘ | ✘ | ✘ | ✓ | ? |
| barrier | ✓ | ✘ | ✓ | ✓ | ✓ | ? |

## Basics

`torch.distributed` 为在一台或多台机器上运行的多个计算节点提供多进程并行的通信模块和PyTorch的支持. 类 [`torch.nn.parallel.DistributedDataParallel()`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 建立在这个功能之上, 以提供任何PyTorch模型分布式训练的装饰器. 这个类和 [Multiprocessing package - torch.multiprocessing](multiprocessing.html) 和 [`torch.nn.DataParallel()`](nn.html#torch.nn.DataParallel "torch.nn.DataParallel") 并不相同, PyTorch集群分布式计算支持多台机器, 使用时用户必须在主要训练的脚本中, 明确地将每个进程复制到每台机器中.

在单机多节点计算的情况下, 使用 `torch.distributed` 和 [`torch.nn.parallel.DistributedDataParallel()`](nn.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 作为 训练的装饰器, 相比于 [`torch.nn.DataParallel()`](nn.html#torch.nn.DataParallel "torch.nn.DataParallel") 之类的数据并行计算, 任然具有优势:

*   在每次迭代中, 每个进程维护自己的优化器, 执行完整的优化步骤. 虽然这看起来可能是多余的, 但是因为梯度已经被收集在 一起, 并且计算了梯度的平均值, 因此对于每个进程梯度是相同的, 这可以减少在节点之间传递张量, 再计算参数的时间.
*   每个进程都包含一个独立的Python解释器, 消除了Python解释器的额外开销, 以及由于驱动多线程, 模型副本和GPU造成 “GIL-thrashing” . 对于需要消耗大量Python解释器运行时间 (包括具有循环图层或许多小组件的模型) 的模型来说是非常重要的.

## Initialization

在调用其他模型之前, 这个包需要使用 `torch.distributed.init_process_group()` 函数进行初始化. 在初始化单元中, 所有进程都会参与.

```py
torch.distributed.init_process_group(backend, init_method='env://', **kwargs)
```

初始化方法.

参数：

*   `backend (str)` – 使用后端的名字. 输入的有效值包括: `tcp` , `mpi` and `gloo` .
*   `init_method (str, 可选)` – 指定如何初始化的URL.
*   `world_size (int, 可选)` – 参与工作的进程数量.
*   `rank (int, 可选)` – 当前进程的排名.
*   `group_name (str, 可选)` – 集群的名字. 请参阅init方法的描述.



为了支持 `backend == mpi` , PyTorch 需要在支持 MPI 的系统上用进行源码编译安装

```py
torch.distributed.get_rank()
```

返回当前进程的排名.

排名是独一无二的 Rank(排名）是分配给分布式集群中每个进程的唯一标识符. 它们总是连续的整数, 范围从0到 `world_size` .

```py
torch.distributed.get_world_size()
```

返回在分布式集群中的进程数目.

* * *

目前支持三种初始化的方法:

### TCP initialization

提供两种TCP的初始化的方法, 两种方法都需要各台机器的网络地址和集群机器数目 `world_size` . 第一种方法需要指定属于0级进程的地址, 并且初始化时所有进程的等级都由手动指定.

第二种方法是, 地址必须是有效的IP多播地址, 在这种情况下, 可以自动分配等级. 多路通信的初始化也支持 `group_name` 参数, 它允许你为多个作业使用相同的地址, 只要它们使用不同的小组名即可.

```py
import torch.distributed as dist

# Use address of one of the machines
dist.init_process_group(init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)

# or a multicast address - rank will be assigned automatically if unspecified
dist.init_process_group(init_method='tcp://[ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7]:23456',
                        world_size=4)

```

### Shared file-system initialization

另一个初始化方法使用一个文件系统, 这个文件系统在一个组中的所有机器上共享和可见, 以及一个所需的 `world_size` 参数. URL应该以 `file://` 开头, 并包含一个可以和共享文件系统所有现有目录中的路径相区别的路径, 作为URL. 这个初始化方法也 支持 `group_name` 参数, 它允许你为多个作业使用相同的共享文件路径, 只要它们使用不同的小组名.

警告：

这种方法假设文件系统支持使用 `fcntl` 进行锁定 -大多数本地系统和NFS都支持它.

```py
import torch.distributed as dist

# Rank will be assigned automatically if unspecified
dist.init_process_group(init_method='file:///mnt/nfs/sharedfile', world_size=4,
                        group_name=args.group)

```

### Environment variable initialization

此方法将从环境变量中读取配置, 从而可以完全自定义如何获取信息. 要设置的变量是:

*   `MASTER_PORT` - 需要; 必须是0级机器上的自由端口
*   `MASTER_ADDR` - 需要 (除了等级0) ; 等级0节点的地址
*   `WORLD_SIZE` - 需要; 可以在这里设置, 或者在调用init函数
*   `RANK` - 需要; 可以在这里设置, 或者在调用init函数

等级为0的机器将用于设置所有连接.

这是默认的方法, 这意味着 `init_method` 不必被特别指定(或者可以是 `env://` )

## Groups

默认的集群 (collectives) 操作默认的小组 (group), 要求所有的进程进入分布式函数中调用. 一些工作负载可以从可以从更细粒度的通信中受益 这是分布式集群发挥作用的地方. `new_group()` 函数可以用来创建新的组, 并且包含所有进程的任意子集. 它返回一个不透明的组句柄, 它可以作为集群的 `group` 参数 (集群 collectives 是一般的编程模式中的交换信息的分布式函数) .

```py
torch.distributed.new_group(ranks=None)
```

创建一个新的分布式小组

此函数要求主组中的所有进程(即作为分布式作业一部分的所有进程）都会输入此函数, 即使它们不是该小组的成员. 此外, 应该在所有的进程中以相同的顺序创建新的小组.

参数：`ranks (list[int])` – 小组内成员的 Rank 的列表.

返回值：分配组的句柄, 以便在集群中调用.


## Point-to-point communication

```py
torch.distributed.send(tensor, dst)
```

同步发送张量.

参数：

*   `tensor (Tensor)` – 发送的张量.
*   `dst (int)` – 指定发送的目的地的 Rank.



```py
torch.distributed.recv(tensor, src=None)
```

同步接收张量.

参数：

*   `tensor (Tensor)` – 用收到的数据填充张量.
*   `src (int, 可选)` – 发送端的Rank, 如果没有指定, 将会接收任何发送的数据.


返回值：发送端的Rank.


`isend()` 和 `irecv()` 使用时返回分布式请求对象. 通常, 这个对象的类型是未指定的, 因为它们不能使用手动创建, 但是它们支持两种方法指定:

*   `is_completed()` - 如果操作完成返回True
*   `wait()` - 如果操作完成会阻塞所有的进程. `is_completed()` 如果结果返回, 保证函数返回True.

当使用MPI作为后端, `isend()` 和 `irecv()` 支持 “不超车” 式的工作方式, 这种方式可以保证消息的顺序. 更多的细节可以看 [http://mpi-forum.org/docs/mpi-2.2/mpi22-report/node54.htm#Node54](http://mpi-forum.org/docs/mpi-2.2/mpi22-report/node54.htm#Node54)

```py
torch.distributed.isend(tensor, dst)
```

异步发送张量数据.

参数：

*   `tensor (Tensor)` – 发送的张量的数据.
*   `dst (int)` – 指定发送到的 Rank.


返回值：分布式请求对象.


```py
torch.distributed.irecv(tensor, src)
```

异步接收张量.

参数：

*   `tensor (Tensor)` – 用收到的数据填充张量.
*   `src (int)` – 指定发送张量的 Rank.


返回值：一个分布式请求对象.


## Collective functions

```py
torch.distributed.broadcast(tensor, src, group=<object object>)
```

向某个小组内的张量广播的方法.

> `tensor` 在该小组处理数据的所有过程中元素的数目必须相同.

参数：

*   `tensor (Tensor)` – 如果发送端 `src` 是当前进程的 Rank, 则发送数据, 否则使用张量保存接收的数据.
*   `src (int)` – 发送端的 Rank.
*   `group (optional)` – 集群内的小组的名字.



```py
torch.distributed.all_reduce(tensor, op=<object object>, group=<object object>)
```

处理所有机器上的处理的张量数据, 计算最终的结果.

在所有进程中调用 `tensor` 将按位相同.

参数：

*   `tensor (Tensor)` – 集群的输入和输出.
*   `op (optional)` – “torch.distributed.reduce_op” 枚举值之一. 指定用于元素减少的操作.
*   `group (optional)` – 集群的内的小组的名字.



```py
torch.distributed.reduce(tensor, dst, op=<object object>, group=<object object>)
```

减少所有机器上的张量数据.

只有级别为 `dst` 的进程才会收到最终结果.

参数：

*   `tensor (Tensor)` – 集群的输入和输出数据. 分别在每台机器上本地处理.
*   `op (optional)` – “torch.distributed.reduce_op” 枚举值之一. 指定用于元素减少的操作.
*   `group (optional)` – 集群的内的小组的名字.



```py
torch.distributed.all_gather(tensor_list, tensor, group=<object object>)
```

在整个集群中收集list表格中的张量.

参数：

*   `tensor_list (list_[Tensor]_)` – 输出列表. 它应该包含正确大小的张量以用于集体的输出.
*   `tensor (Tensor)` – 张量从当前进程中进行广播.
*   `group (optional)` – 集群的内的小组的名字.



```py
torch.distributed.gather(tensor, **kwargs)
```

收集一个张量列表从一个单一进程中.

参数：

*   `tensor (Tensor)` – 输入的数据.
*   `dst (int)` – 目的地的 Rank. 包括除了正在接收数据的进程的所有进程.
*   `gather_list (list_[Tensor]_)` – 用于接收数据的适当大小的张量列表. 只在接收过程中需要.
*   `group (optional)` – 集群的内的小组的名字.



```py
torch.distributed.scatter(tensor, **kwargs)
```

将张量列表散布到小组中的所有进程.

每个进程只会收到一个张量, 并将其数据存储在 `tensor` 的参数中.

参数：

*   `tensor (Tensor)` – 输出的张量.
*   `src (int)` – 发送端的 Rank. 包括除了正在接收数据的进程的所有进程.
*   `scatter_list (list_[Tensor]_)` – 张量分散的列表. 仅在发送数据的过程中需要.
*   `group (optional)` – 集群的内的小组的名字.



```py
torch.distributed.barrier(group=<object object>)
```

同步所有进程.

这个集群阻塞进程, 直到全部的小组的计算结果都输入进这个函数中.

参数：`group (optional)` – 集群的内的小组的名字.
