


 使用 PyTorch 编写分布式应用程序
 [¶](#writing-distributed-applications-with-pytorch "永久链接到此标题")
================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/dist_tuto>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/dist_tuto.html>




**作者** 
 :
 [Séb 阿诺德](https://seba1511.com)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/intermediate_source/dist_tuto.rst) 
.





 先决条件:



* [PyTorch 分布式概述](../beginner/dist_overview.html)



 在这个简短的教程中，我们将介绍 PyTorch 的分布式包
。我们’ 将了解如何设置分布式设置、使用
不同的通信策略，并查看
包的一些内部结构。





 设置
 [¶](#setup "此标题的永久链接")
------------------------------------------------


PyTorch 中包含的分布式软件包（即“torch.distributed”）使研究人员和从业者能够轻松跨进程和机器集群并行化计算。为此，它利用消息传递语义
允许每个进程与任何其他进程通信数据。
与多处理 (
 `torch.multiprocessing`
 ) 包相反，
进程可以使用不同的通信后端，并且不
限制为在同一台计算机上执行。




 为了开始，我们需要能够同时运行多个进程。如果您有权访问计算集群，则应咨询本地系统管理员或使用您最喜欢的协调工具（例如，
 [pdsh](https://linux.die.net/man/1/pdsh)
 ,\ n [clustershell](https://cea-hpc.github.io/clustershell/) 
 或
 [其他](https://slurm.schedmd.com/) 
 )。出于本教程的目的，
我们将使用一台计算机并使用
以下模板生成多个进程。






```
"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
 """ Distributed function to be implemented later. """
    pass

def init_process(rank, size, fn, backend='gloo'):
 """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

```




 上面的脚本生成两个进程，每个进程将设置
分布式环境，初始化进程组
(
 `dist.init_process_group`
 )，最后执行给定的
 `运行`
 函数。




 让’s 看一下
 `init_process`
 函数。它确保每个进程都能够使用相同的 IP 地址和端口通过主进程进行协调。请注意，我们使用
 `gloo`
 后端，但
 其他后端可用。 (c.f.
 [第 5.1 节](#communication-backends) 
 ) 我们将在本教程的最后回顾 
`dist.init_process_group`
 中发生的魔法，
但是它本质上允许进程通过共享位置来相互通信。






 点对点通信
 [¶](#point-to-point-communication "固定链接到此标题")
------------------------------------------------------------------------------------------------



[![发送和接收](https://pytorch.org/tutorials/_images/send_recv.png)](https://pytorch.org/tutorials/_images/send_recv.png)


 发送和接收
 
[¶](#id1“此图像的永久链接”)





 从一个进程到另一个进程的数据传输称为点对点通信。这些是通过
 `send`
 和
 `recv`
 函数或其
 *立即* 
 对应部分
 `isend`
 和
 `irecv`
 来实现的。 






```
"""Blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

```




 在上面的示例中，两个进程都以零张量开始，然后
进程 0 递增张量并将其发送到进程 1，以便它们
最终都为 1.0。请注意，进程 1 需要分配内存
以便存储它将接收到的数据。




 另请注意
 `send`
 /
 `recv`
 是
 **阻塞** 
 ：两个进程都会停止
直到通信完成。另一方面，立即数是
 **非阻塞** 
 ；脚本继续执行，方法
返回一个
 `Work`
 对象，我们可以在该对象上选择
 `wait()`
 。






```
"""Non-blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

```




 使用立即数时，我们必须小心如何使用发送和接收的张量。
由于我们不知道数据何时会传递到其他进程，
我们不应该修改发送的张量，也不应该访问在`req.wait()`完成之前接收到张量。
换句话说，



* 在
 `dist.isend()`
 之后写入
 `tensor`
 将导致未定义的行为。
* 在
 `dist.irecv()` 之后读取
 `tensor`
 
 将导致未定义的行为。



 然而，在执行了
 `req.wait()`
 后，我们保证发生了通信，
并且存储在
 `tensor[0]`
 中的值是 1.0。



当我们想要更细粒度地控制进程的通信时，点对点通信非常有用。它们可用于
实现奇特的算法，例如
 [Baidu’s
DeepSpeech](https://github.com/baidu-research/baidu-allreduce)
 中使用的算法
 或\ n [Facebook’s 大规模
实验](https://research.fb.com/publications/imagenet1kin1h/) 
.(c.f.
 [第 4.1 节](#our-own-ring-全部归约) 
 )






 集体通信
 [¶](#collective-communication "永久链接到此标题")
--------------------------------------------------------------------------------------







| 
[分散](https://pytorch.org/tutorials/_images/scatter.png)


 分散
 
[¶](#id2 "此图像的永久链接") 

 | 
[收集](https://pytorch.org/tutorials/_images/gather.png)


 收集
 
[¶](#id3 "此图像的永久链接") 

 |
| 
[减少](https://pytorch.org/tutorials/_images/reduce.png)


 减少
 
[¶](#id4 "此图像的永久链接") 

 | 
[All-Reduce](https://pytorch.org/tutorials/_images/all_reduce.png)


 All-Reduce
 
[¶](#id5 "此图像的永久链接") 

 |
| 
[广播](https://pytorch.org/tutorials/_images/broadcast.png)


 广播
 
[¶](#id6 "此图像的永久链接") 

 | 
[全收集](https://pytorch.org/tutorials/_images/all_gather.png)


 All-Gather
 
[¶](#id7 "此图像的永久链接") 

 |



 与点对点通信相反，集体允许
跨
 **组** 
 中的所有进程进行通信模式。组是我们所有流程的子集。要创建组，我们可以将排名列表传递给
 `dist.new_group(group)`
 。默认情况下，集合体在所有进程中执行，也称为
 **world** 
 。例如，为了获得所有进程上所有张量的总和，我们可以使用
 `dist.all_reduce(tensor,
 

 op,
 

 group)` 
 集体。






```
""" All-Reduce example."""
def run(rank, size):
 """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])

```




 由于我们想要组中所有张量的总和，因此我们使用 
 `dist.ReduceOp.SUM`
 作为归约运算符。一般来说，任何
可交换的数学运算都可以用作运算符。
PyTorch 开箱即用地附带 4 个这样的运算符，
它们全部在元素级别工作：



* `dist.ReduceOp.SUM`
 ,
* `dist.ReduceOp.PRODUCT`
 ,
* `dist.ReduceOp.MAX`
 ,
* `dist.ReduceOp.MIN`
.



 除了 
 `dist.all_reduce(tensor,
 

 op,
 

 group)`
 之外，PyTorch 目前总共实现了 6 个集合。 



* `dist.broadcast(tensor,
 

 src,
 

 group)`
 : 将
 `tensor`
 从
 `src`
 复制到所有其他进程。\ n* `dist.reduce(tensor,
 

 dst,
 

 op,
 

 group)`
 : 将
 `op`
 应用于每个
 `张量`
 并将结果存储在
 `dst`
 中。
* `dist.all_reduce(tensor,
 

 op,
 

 group)`
 :与reduce相同，但
结果存储在所有进程中。
* `dist.scatter(tensor,
 

 scatter_list,
 

 src,
 

 group )`
 : 将
 
 \(i^{\text{th}}\)
 
 张量
 `scatter_list[i]`
 复制到
 \ n \(i^{\text{th}}\)
 
 过程。
* `dist.gather(tensor,
 

 Gather_list,
 

 dst,
 

 group)`
 : 从
 `dst`
 中的所有进程复制
 `tensor`
 。
* `dist.all_gather(tensor_list, 
 

 张量，
 

 组)`
 : 在所有进程上将
 `tensor`
 从所有进程复制到
 `tensor_list`
 。
* `dist.barrier(group)`
 : 阻止
 
 组中的所有进程
 
 直到每个进程都进入此函数。





 分布式训练
 [¶](#distributed-training "固定链接到此标题")
----------------------------------------------------------------------------------------------




**注意：** 
 您可以在
 [此
GitHub 存储库](https://github.com/seba-1511/dist_tuto.pth/) 中找到本节的示例脚本 
.



现在我们了解了分布式模块的工作原理，让我们用它编写一些有用的东西。我们的目标是复制 [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel) 的功能。
当然，这将是一个教学示例，在现实世界中
您应该使用上面链接的经过充分测试和优化的
官方版本。



很简单，我们想要实现随机梯度下降的分布式版本。我们的脚本将让所有进程根据其批量数据计算其模型的梯度，然后对其梯度进行平均。为了确保在更改进程数时
获得相似的收敛结果，我们首先必须对数据集进行分区。
（您也可以使用
 [tnt.dataset.SplitDataset](https://github.com/pytorch /tnt/blob/master/torchnet/dataset/splitdataset.py#L4) 
 ，\而不是下面的代码片段。）






```
""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

```




 通过上面的代码片段，我们现在可以使用
以下几行简单地对任何数据集进行分区：






```
""" Partitioning MNIST """
def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz

```




 假设我们有 2 个副本，那么每个进程将有一个包含 60000 /2 = 30000 个样本的 
 `train_set`
 。我们还将批次大小除以
副本数量，以保持
*总体*
 批次大小为 128。



我们现在可以编写通常的前向-后向优化训练代码，并添加一个函数调用来平均模型的梯度。 （
以下内容很大程度上受到官方
 [PyTorch MNIST
示例](https://github.com/pytorch/examples/blob/master/mnist/main.py) 
 的启发。）






```
""" Distributed Synchronous SGD Example """
def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)

```




 剩下的工作就是实现
 `average_gradients(model)`
 函数，该函数
简单地接受一个模型并在整个世界
上平均其梯度。






```
""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

```




*Et voilà* 
 !我们成功实现了分布式同步 SGD，并且
可以在大型计算机集群上训练任何模型。




**注意：** 
 虽然最后一句是
 *技术上* 
 正确的，但还有
 [更多
更多技巧](https://seba-1511.github.io/dist_blog) 
 需要
实现同步 SGD 的生产级实现。再次，
使用[已经过测试和优化](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)
 。




### 
 我们自己的 Ring-Allreduce
 [¶](#our-own-ring-allreduce "永久链接到此标题")



 作为额外的挑战，假设我们想要实现
DeepSpeech’s 高效环 allreduce。使用点对点集合
这很容易实现。






```
""" Implementation of a ring-reduce with addition. """
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]

```




 在上面的脚本中，
 `allreduce(send,
 

 recv)`
 函数的签名与 PyTorch 中的签名略有不同。它需要一个
 `recv`
 张量，并将在其中存储所有
 `send`
 张量的总和。作为留给读者的练习，我们的版本与 DeepSpeech 中的版本之间仍然存在一个差异：它们的实现将梯度张量划分为
 *块* 
 ，以便最佳地利用
通信带宽。 （提示：
 [torch.chunk](https://pytorch.org/docs/stable/torch.html#torch.chunk)
 )







 高级主题
 [¶](#advanced-topics "此标题的永久链接")
---------------------------------------------------------------------------------



 我们现在已准备好发现 
 `torch.distributed` 的一些更高级的功能。由于要介绍的内容很多，因此本节
分为两个小节：



1. 通信后端：我们在其中学习如何使用 MPI 和 Gloo 进行
GPU-GPU 通信。
2.初始化方法：我们了解如何在 `dist.init_process_group()`
 中最好地设置
初始协调阶段。



### 
 通信后端
 [¶](#communication-backends "此标题的永久链接")



 `torch.distributed` 最优雅的方面之一是它能够
 在不同后端之上进行抽象和构建。如前所述，目前 PyTorch 中实现了三个后端：Gloo、NCCL 和 MPI。它们各自具有不同的规格和权衡，具体取决于所需的用例。可以在[此处](https://pytorch.org/docs/stable/distributed.html#module-torch.distributed)
找到支持的函数的比较表
 。




**Gloo 后端**




 到目前为止，我们已经广泛使用了
 [Gloo 后端](https://github.com/facebookincubator/gloo) 
 。
作为一个开发平台，它非常方便，因为它包含在
预编译的 PyTorch 二进制文件可在 Linux（自 0.2 起）
和 macOS（自 1.3 起）上运行。它支持CPU上的所有点对点和集体操作，以及GPU上的所有集体操作。 
CUDA 张量的集体运算的实现不如
NCCL 后端提供的那样优化。




 您肯定已经注意到，如果您将
 `model`
 放在 GPU 上，我们的
分布式 SGD 示例将不起作用。
为了使用多个 GPU，我们还要进行以下
修改：



1. 使用
 `device
 

 =
 

 torch.device("cuda:{}".format(rank))`
2. `模型
 

 =
 

 Net()`

 \(\rightarrow\)
 
`模型
 

 =
 
\ n Net().to(设备)`
3.使用
 `data,
 

 target
 

 =
 

 data.to(device),
 

 target.to(device)`



 经过上述修改，我们的模型现在正在两个 GPU 上进行训练，
您可以使用
 `watch
 

 nvidia-smi`
 监控它们的利用率。




**MPI 后端**




 消息传递接口 (MPI) 是高性能计算领域的标准化工具。它允许进行点对点和
集体通信，并且是
 `torch.distributed`
 API 的主要灵感。 MPI 存在多种实现（例如
 [Open-MPI](https://www.open-mpi.org/) 
 、
 [MVAPICH2](http://mvapich.cse.ohio-state.edu /) 
 ,
 [英特尔
MPI](https://software.intel.com/en-us/intel-mpi-library) 
 ) 每个
针对不同目的进行了优化。使用 MPI 后端的优势
在于 MPI’ 在大型计算机集群上的广泛可用性和高级别的优化。
 [一些](https://developer.nvidia.com/mvapich ) 
[最近](https://developer.nvidia.com/ibm-spectrum-mpi) 
[实现](https://www.open-mpi.org/) 
 也能够采取\利用 CUDA IPC 和 GPU Direct 技术来避免
通过 CPU 进行内存复制。




 不幸的是，PyTorch’s 二进制文件无法包含 MPI 实现
，我们’ 必须手动重新编译它。幸运的是，这个过程相当简单，因为在编译时，PyTorch 将自行查找可用的 MPI 实现。以下步骤通过安装 PyTorch
 [from
source](https://github.com/pytorch/pytorch#from-source)
 来安装 MPI
后端。



1. 创建并激活 Anaconda 环境，安装以下所有先决条件
 [指南](https://github.com/pytorch/pytorch#from-source) 
 ，但执行
 **还没有** 
 run
 `python
 

 setup.py
 

 install`
 。
2.选择并安装您最喜欢的 MPI 实现。请注意，启用 CUDA 感知 MPI 可能需要一些额外的步骤。在我们的案例中，我们’ 将坚持使用 Open-MPI
 *不*
 GPU 支持：
 `conda
 

 install
 

 -c
 
 
 conda-forge
 

 openmpi`
3.现在，转到克隆的 PyTorch 存储库并执行
 `python
 

 setup.py
 

 install`
 。



 为了测试我们新安装的后端，需要进行一些修改。



1、替换
 `if
 

 __name__
 

 ==
 

 '__main\下的内容\__':`
 with
 `init_process(0,
 

 0,
 

 run,
 

 backend='mpi')` 
.
2.运行
 `mpirun
 

 -n
 

 4
 

 python
 

 myscript.py`
 。



 这些更改的原因是 MPI 需要在生成进程之前创建自己的
环境。 MPI 还将生成自己的进程并执行 [初始化方法](#initialization-methods) 中描述的握手，从而使
 `rank`
 和
 `size`
 参数
 n `init_process_group`
 多余。这实际上非常强大，因为您可以将附加参数传递给“mpirun”，以便为每个进程定制计算资源。 （例如每个进程的核心数、手动将计算机分配到特定等级，以及[一些更多](https://www.open-mpi.org/faq/?category=running#mpirun-hostfile) 
 )
这样做，您应该获得与其他
通信后端相同的熟悉输出。




**NCCL 后端**




 [NCCL 后端](https://github.com/nvidia/nccl)
 提供针对 CUDA
张量的集体操作的优化实现。如果您仅使用 CUDA 张量进行集体操作，
请考虑使用此后端以获得一流的性能。 
NCCL 后端包含在具有 CUDA 支持的预构建二进制文件中。





### 
 初始化方法
 [¶](#initialization-methods "永久链接到此标题")



 为了结束本教程，让 ’s 谈谈我们调用的第一个函数：
 `dist.init_process_group(backend,
 

 init_method) `
 。 
特别是，我们将介绍负责每个进程之间的初始协调步骤的不同初始化方法。
这些方法允许您定义如何完成此协调。
根据您的硬件设置，这些方法之一应该
自然比其他人更合适。除了以下
部分之外，您还应该查看
 [官方
文档](https://pytorch.org/docs/stable/distributed.html#initialization) 
 。




**环境变量**




 在本教程中，我们一直在使用环境变量初始化方法。通过在所有计算机上设置以下四个环境变量，
所有进程都将能够正确
连接到主机，获取有关其他进程的信息，
最后与它们握手。



* `MASTER_PORT`
 : 将托管等级为 0 的进程
的计算机上的空闲端口。
* `MASTER_ADDR`
 : 将托管进程
的计算机的 IP 地址
* `WORLD_SIZE`
 : 进程总数，以便master
知道要等待多少个worker。
* `RANK`
 : 每个进程的排名，以便它们
将知道它是否是
worker的master。



**共享文件系统**




 共享文件系统要求所有进程都可以访问共享文件系统，并将通过共享文件来协调它们。这意味着
每个进程都将打开该文件，写入其信息，然后等待
直到每个进程都这样做。此后，所有进程都可以轻松使用所有必需的信息。为了避免竞争条件，
文件系统必须支持通过
 [fcntl](http://man7.org/linux/man-pages/man2/fcntl.2.html) 锁定
 。






```
dist.init_process_group(
    init_method='file:///mnt/nfs/sharedfile',
    rank=args.rank,
    world_size=4)

```




**TCP**




 通过 TCP 初始化可以通过提供等级 0 的进程的 IP 地址和可到达的端口号来实现。
在这里，所有工作人员都能够连接到等级 0 的进程
并交换有关如何进行操作的信息。互相联系。






```
dist.init_process_group(
    init_method='tcp://10.1.1.20:23456',
    rank=args.rank,
    world_size=4)

```






**致谢**





 我’d 要感谢 PyTorch 开发人员在他们的实现、文档和测试方面做得如此出色。当代码不清楚时，我总是可以依靠
 [文档](https://pytorch.org/docs/stable/distributed.html)
 或
 [测试](https://github.com/pytorch/pytorch/tree/master/test/distributed) 
 寻找答案。我’d 特别感谢 Soumith Chintala、
Adam Paszke 和 Natalia Gimelshein 提供富有洞察力的评论
并回答有关早期草稿的问题。










