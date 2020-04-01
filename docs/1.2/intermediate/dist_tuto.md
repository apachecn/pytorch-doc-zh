# 3\. PyTorch编写分布式应用程序

**作者** ：[ SEB阿诺德](https://seba1511.com)

在这个简短的教程中，我们将要在PyTorch的分布式包。我们将看到如何建立分布式设置，使用不同的沟通策略，走在包装件的一些内部结构。

## 设定

包括在PyTorch分布式包(即，torch.distributed `
`）使研究人员和从业人员跨进程和机器的集群来容易并行化的计算。为了这样做，它利用了消息传递语义允许每个进程进行数据通信的任何其他进程的。而不是在并行处理(`
torch.multiprocessing`）包，过程可以使用不同的通信后端和不限于在同一台机器上被执行。

为了开始，我们需要同时运行多个流程的能力。如果你有机会到计算机集群，你应该用你的本地系统管理员检查或使用自己喜欢的协调工具。 (例如，[ PDSH
](https://linux.die.net/man/1/pdsh)，[ clustershell ](https://cea-
hpc.github.io/clustershell/)或[人](https://slurm.schedmd.com/)）对于本教程的目的，我们将使用一台机器和叉的多个进程使用以下模板。

    
    
    """run.py:"""
    #!/usr/bin/env python
    import os
    import torch
    import torch.distributed as dist
    from torch.multiprocessing import Process
    
    def run(rank, size):
        """ Distributed function to be implemented later. """
        pass
    
    def init_processes(rank, size, fn, backend='tcp'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=rank, world_size=size)
        fn(rank, size)
    
    
    if __name__ == "__main__":
        size = 2
        processes = []
        for rank in range(size):
            p = Process(target=init_processes, args=(rank, size, run))
            p.start()
            processes.append(p)
    
        for p in processes:
            p.join()
    
    

上述脚本派生两个过程谁将每个设置的分布式环境中，初始化处理组(`dist.init_process_group`），最后执行`运行给定的 `功能。

让我们来看看`init_processes
`功能。它确保每一道工序将能够通过一个主协调，使用相同的IP地址和端口。请注意，我们使用的是TCP后端，但我们也可以使用[ MPI
](https://en.wikipedia.org/wiki/Message_Passing_Interface)或[ GLOO
](https://github.com/facebookincubator/gloo)代替。 (参见 5.1节），我们会在魔术`
dist.init_process_group`在本教程的最后发生的事情，但它基本上可以让进程间通信其他通过分享他们的位置。

## 点对点通信

[![Send and Recv](img/send_recv.png)](img/send_recv.png)

传送和recv

数据从一个处理A转移到另一个被称为点 - 点通信。这些通过取得的`发送 `和`的recv`的功能或它们的 _立即_ 反份，`isend`和`
irecv`。

    
    
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
    
    

在上面的例子，这两个过程以零开始张量，然后处理增量0张量，并将其发送到处理1，使得它们都结束了1.0。请注意，过程1只需要以存储将接收数据分配内存。

还要注意，`发送 `/ `的recv`是 **阻断** ：两个过程停止，直到通信完成。在另一方面的立即被 **非阻塞**
;脚本将继续其执行和方法都返回一个`DistributedRequest`对象后，我们可以选择`等待(） `。

    
    
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
    
    

当使用的立即，我们必须小心我们的发送和接收的张量的使用。因为我们不知道什么时候的数据将被传递给其它工艺做的，我们不应该修改发张量也不`req.wait(）
`完成之前访问接收到的张量。换一种说法，

  * 写`张量 ``dist.isend后(） `将导致未定义的行为。
  * 从`读取张量 ``dist.irecv后(） `将导致未定义的行为。

然而，`req.wait(） `已被执行之后，我们保证了通信发生了，并且，存储在`张量的值[0]`是1.0。

点至点，当我们想在我们的流程的通信进行细粒度的控制通信是有益的。它们可以被用来实现花哨的算法，如[百度的DeepSpeech
](https://github.com/baidu-research/baidu-allreduce)或[
Facebook的大规模实验[HTG3。所使用的(c.f。](https://research.fb.com/publications/imagenet1kin1h/)第4.1节）

## 集体通信

[![Scatter](img/scatter.png)](img/scatter.png)

散点图

|

[![Gather](img/gather.png)](img/gather.png)

收集  
  
---|---  
  
[![Reduce](img/reduce.png)](img/reduce.png)

降低

|

[![All-Reduce](img/all_reduce.png)](img/all_reduce.png)

全减少  
  
[![Broadcast](img/broadcast.png)](img/broadcast.png)

广播

|

[![All-Gather](img/all_gather.png)](img/all_gather.png)

全收集  
  
相对于点对点通信电子，集体允许对 **组** 中所有进程的通信模式。 A组是我们所有进程的一个子集。要创建一个组，我们可以通过职级为`
dist.new_group(集团）名单 [HTG5。默认情况下，集体的对所有进程执行，也被称为
**世界[HTG7。例如，为了获得在所有进程都张量的总和，我们可以使用`dist.all_reduce(张量， 运算， 组） `集体。**`

    
    
    """ All-Reduce example."""
    def run(rank, size):
        """ Simple point-to-point communication. """
        group = dist.new_group([0, 1])
        tensor = torch.ones(1)
        dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
        print('Rank ', rank, ' has data ', tensor[0])
    
    

既然我们要在组中的所有张量的总和，我们使用`dist.reduce_op.SUM
`为降低运营商。一般来说，任何可交换的数学运算，可以作为运营商。外的开箱，PyTorch配备了4个这样的运营商，都在逐元素级别工作：

  * `dist.reduce_op.SUM`
  * `dist.reduce_op.PRODUCT`
  * `dist.reduce_op.MAX`
  * `dist.reduce_op.MIN`。

除了`dist.all_reduce(张量， 运算， 组） `，有一个总的目前PyTorch实现6个集体。

  * `dist.broadcast(张量， SRC， 组） `：复制`张量 `从`SRC`到所有其它过程。
  * `dist.reduce(张量， DST， 运算， 组） `：应用`OP`所有`结果张量 `，并存储在`DST`。
  * `dist.all_reduce(张量， 运算， 组） `：同降低，但其结果被存储在所有进程。
  * `dist.scatter(张量， SRC， scatter_list， 组） `：复制 \ (I ^ {\文本{第}} \）张量`scatter_list [I]`到 \(I ^ {\文本{第}} \）过程。
  * `dist.gather(张量， DST， gather_list， 组） `：复制`张量 `从`DST`所有进程。
  * `dist.all_gather(tensor_list， 张量， 组） `：复制`张量 `从所有流程，以`tensor_list`上的所有进程。
  * `dist.barrier(组） `：块组的所有进程，直至每一个已经进入该功能。

## 分布式训练

**注：** 你可以在[这个GitHub的库](https://github.com/seba-1511/dist_tuto.pth/)本节的示例脚本。

现在我们明白了分布式模块是如何工作的，让我们写的东西与它有用。我们的目标是复制的[ DistributedDataParallel
](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)的功能。当然，这将是一个说教的例子，在现实世界situtation你应该使用官方的，经过严格测试和精心优化的版本上面链接。

简单地说，我们要实现的随机梯度下降一个分布式的版本。我们的脚本将让所有的进程都计算在他们的批量数据的他们的模型的梯度，然后平均的梯度。为了改变进程的数目时，以确保类似的收敛结果，我们首先要分区我们的数据。
(你也可以使用[ tnt.dataset.SplitDataset
](https://github.com/pytorch/tnt/blob/master/torchnet/dataset/splitdataset.py#L4)，而不是片段下方。）

    
    
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
    
    

通过上述片段中，我们现在可以简单地使用下面的几行分区中的任何数据集：

    
    
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
    
    

假设我们有2个副本，那么每个进程将具有`train_set`60000/2 = 30000个样本。我们还除以副本的数量批量大小，以保持的128
_总体_ 批量大小。

现在，我们可以写我们通常前后，优化训练码，并添加一个函数调用来平均我们的模型的梯度。 (下面是从官方[ PyTorch
MNIST例如](https://github.com/pytorch/examples/blob/master/mnist/main.py)很大程度上启发。）

    
    
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
    
    

它仍然实现`average_gradients(型号） `功能，它只是发生在一个模型，在整个世界平均水平的梯度。

    
    
    """ Gradient averaging. """
    def average_gradients(model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size
    
    

_的Et瞧_ ！我们成功地实施分布式同步新元，并可能培养了大量的计算机集群上的任何模型。

**注：[HTG1虽然最后一句是 _技术上_
真实的，有[很多更多的技巧[HTG5】实行同步SGD的生产级的落实需要。再次，用什么](https://seba-1511.github.io/dist_blog)[已经过测试和优化[HTG7。](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)**

### 我们自己的戒指，Allreduce

作为一个额外的挑战，假设我们要落实DeepSpeech的高效环allreduce。这是使用点至点集体相当容易实现。

    
    
    """ Implementation of a ring-reduce with addition. """
    def allreduce(send, recv):
        rank = dist.get_rank()
        size = dist.get_world_size()
        send_buff = th.zeros(send.size())
        recv_buff = th.zeros(send.size())
        accum = th.zeros(send.size())
        accum[:] = send[:]
    
        left = ((rank - 1) + size) % size
        right = (rank + 1) % size
    
        for i in range(size - 1):
            if i % 2 == 0:
                # Send send_buff
                send_req = dist.isend(send_buff, right)
                dist.recv(recv_buff, left)
                accum[:] += recv[:]
            else:
                # Send recv_buff
                send_req = dist.isend(recv_buff, right)
                dist.recv(send_buff, left)
                accum[:] += send[:]
            send_req.wait()
        recv[:] = accum[:]
    
    

另外，在上述脚本中，`allreduce(发送， 的recv） `函数具有比PyTorch的那些稍微不同的签名。它需要一个`的recv
`张量，将所有`发 `张量的总和存储在里面。作为一个练习留给读者，还有我们的版本和一个在DeepSpeech之间的一个区别：它们的实现划分梯度张成 _块_
，从而以最佳方式利用通信带宽。 (提示：[ torch.chunk
](https://pytorch.org/docs/stable/torch.html#torch.chunk)）

## 高级主题

我们现在就可以发现一些`torch.distributed`更先进的功能性。因为有很多覆盖，本节分为两个小节：

  1. 通讯后端：我们学习如何使用MPI和GLOO的GPU-GPU通信。
  2. 初始化方法：在我们了解如何最好地设置在`dist.init_process_group初始协调阶段(） [HTG3。`

### 通信后端

其中的`最优雅的方面torch.distributed
`是它的抽象能力和建立在不同的后端之上。正如前面提到的，有目前有三个在后端实现PyTorch：TCP，MPI和GLOO。他们每个人都有不同的规格和权衡，根据所需的用例。支持的函数的比较表可以发现[这里](https://pytorch.org/docs/stable/distributed.html#module-
torch.distributed)。需要注意的是第四后端，NCCL，已自创立本教程的补充。参见[本部分](https://pytorch.org/docs/stable/distributed.html#multi-
gpu-collective-functions)中的`torch.distributed`文档有关其使用和值的详细信息的。

**TCP后端**

到目前为止，我们已经取得了TCP后端的广泛使用。这是作为一个开发平台非常方便，因为它是保证在大多数计算机和操作系统上运行。它还支持所有点至点和集体功能的CPU。然而，对于GPU和它的通信程序并不作为优化的MPI一个不支持。

**GLOO后端**

的[ GLOO后端](https://github.com/facebookincubator/gloo)提供了一种优化的实施 _集体_
通信过程，无论对CPU和GPU。它特别照在GPU的，因为它可以在不使用[ GPUDirect
](https://developer.nvidia.com/gpudirect)将数据传送到CPU的存储器进行通信。另外，也能够使用[ NCCL
](https://github.com/NVIDIA/nccl)执行快速节点内的通信，并实现其[自己的算法[HTG9用于节点间的例程。](https://github.com/facebookincubator/gloo/blob/master/docs/algorithms.md)

自从0.2.0版本中，GLOO后台自动包含PyTorch的预编译的二进制文件。正如你一定会注意到，如果你把`模型
`在GPU上我们的分布式SGD例如不工作。让我们从第一替换`后端= 'GLOO' 修复 `在`init_processes(秩， 大小， FN，
后端= 'TCP'） `。在这一点上，该脚本将仍然在CPU上运行，但使用的幕后GLOO后端。为了使用多GPU，让我们也做如下修改：

  0. `init_processes(秩， 大小， FN， 后端= 'TCP'） `\(\ RIGHTARROW \） `init_processes(秩， 大小， FN， 后端= 'GLOO'） `
  1. 使用`装置 =  torch.device (“CUDA：{}”。格式(评级）） `
  2. `模型 =  净(） `\(\ RIGHTARROW \） `模型 =  净(）。到(装置） `
  3. 使用`数据， 目标 =  data.to(装置）， target.to(装置） `

通过上述修改，我们的模型现在的训练在两个GPU和您可以监控他们与`利用观看 NVIDIA-SMI  [HTG5。`

**MPI后端**

消息传递接口(MPI）是从高性能计算领域标准化的工具。它允许做点至点和集体沟通，是为`torch.distributed
`该API的主要灵感。存在MPI的若干实施方式(例如，[开放-MPI ](https://www.open-mpi.org/)，[ MVAPICH2
](http://mvapich.cse.ohio-state.edu/)，[英特尔MPI ](https://software.intel.com/en-
us/intel-mpi-library)），每个用于不同的目的进行了优化。使用MPI后端的优势在于MPI的广泛可用性 - 和优化的高层次 -
大型计算机集群。 [HTG10一些 [最近](https://developer.nvidia.com/ibm-spectrum-mpi)
[实现](https://www.open-mpi.org/)也能够利用CUDA IPC和GPU直接的技术，以便通过CPU来避免存储副本。

不幸的是，PyTorch的可执行文件可以不包括MPI实现，我们必须手工重新编译。幸运的是，这个过程是相当简单的因为在编译时，PyTorch看起来 _本身_
一个可用的MPI实现。下面的步骤安装MPI后端，通过从源安装PyTorch
[。](https://github.com/pytorch/pytorch#from-source)

  1. 创建并激活您的蟒蛇环境，安装所有下面的[导](https://github.com/pytorch/pytorch#from-source)的先决条件，但 **不是** 运行`巨蟒 setup.py  安装 `呢。
  2. 选择并安装自己喜欢的MPI实现。请注意，启用CUDA感知MPI可能需要一些额外的步骤。在我们的例子中，我们将坚持开放MPI _无_ GPU的支持：`畅达 安装 -c  康达锻 的openmpi`
  3. 现在，去你的克隆PyTorch回购和执行`巨蟒 setup.py  安装 [HTG7。`

为了测试我们新安装的后端，则需要进行一些修改。

  1. 更换下`含量如果 __name__  ==  '__main__'： `与`init_processes (0， 0， 运行， 后端= 'MPI'） `。
  2. 运行`的mpirun  -N  4  蟒 myscript.py`。

究其原因，这些变化是，MPI需要产卵的过程之前创建自己的环境。 MPI也将产生其自己的过程，并执行在初始化方法所述的握手，使得`秩 `和`大小
`的参数`init_process_group`多余的。这实际上是相当强大的，你可以通过额外的参数`的mpirun
[HTG17为了调整计算资源，为每个进程。 (比如像每个进程内核，手工分配机器特定列数和[一些更](https://www.open-
mpi.org/faq/?category=running#mpirun-hostfile)）这样做，则应该得到相同的熟悉输出与其它通信后端。`

### 初始化方法

为了完成本教程，让我们来谈谈我们称为第一个函数：`dist.init_process_group(后端， init_method）HTG4]
[HTG5。特别是，我们会在不同的初始化方法，这是负责每道工序之间的协调最初一步。这些方法允许你定义这种协调是如何实现的。根据您的硬件设置，这些方法之一应该是自然比其他人更适合。除了下面的部分，你也应该有一个看看[官方文档[HTG7。](https://pytorch.org/docs/stable/distributed.html#initialization)`

跳水进入初始化方法之前，让我们快速浏览一下背后`init_process_group`从C / C ++的角度会发生什么。

  1. 首先，参数解析和验证。
  2. 后端经由`name2channel.at(） `功能解决。 A `频道 `类被返回，并且将用于进行该数据传输。
  3. 的GIL被丢弃，并`THDProcessGroupInit(） `被调用。此实例化信道，并增加了主节点的地址。
  4. 用列0的过程中会执行`主 `过程，而所有其他等级将是`工人 `。
  5. 大师
    1. 创建为所有工人插座。
    2. 所有工人等待连接。
    3. 发送他们有关的其他进程的位置信息。
  6. 每个工人
    1. 创建一个套接字的主人。
    2. 将自己的位置信息。
    3. 接收有关的其他工作人员的信息。
    4. 打开一个插座和握手与所有其他工人。
  7. 初始化完成后，每个人都被连接到每一个人。

**环境变量**

我们一直在使用本教程的环境变量初始化方法。通过设置所有计算机上的以下四个环境变量，所有进程将能够正确地连接到主，获取有关的其他进程的信息，并最终与他们握手。

  * `MASTER_PORT`：将与等级0宿主的过程中机器上的空闲端口。
  * `MASTER_ADDR`：将与等级0宿主的过程中机器的IP地址。
  * `WORLD_SIZE`：总数的工艺，使主知道有多少工人等待。
  * `RANK`：每个处理的等级，所以他们会知道它是否是一个工人的主人。

**共享文件系统**

共享文件系统需要的所有进程能够访问共享文件系统，并协调将通过共享文件。这意味着，每个进程将打开该文件，写入其信息，并等待，直到每个人都这样做了。以后有什么需要的所有信息将随时提供给所有的进程。为了避免竞态条件，则文件系统必须支持通过[的fcntl
](http://man7.org/linux/man-
pages/man2/fcntl.2.html)锁定。请注意，您可以手动指定行列或让流程弄清楚自己。可以定义一个独特的`组名
`每次作业你可以使用相同的文件路径为多个作业，然后安全地避免冲突。

    
    
    dist.init_process_group(init_method='file:///mnt/nfs/sharedfile', world_size=4,
                            group_name='mygroup')
    
    

**TCP初始化 &安培;组播**

通过TCP初始化可以用两种不同的方式来实现：

  1. 通过提供过程中的IP地址与等级0和世界大小。
  2. 通过提供 _任何_ 有效的IP [多播地址](https://en.wikipedia.org/wiki/Multicast_address)和世界的大小。

在第一种情况下，所有工人将能够与秩0连接至该过程，并按照上面描述的过程。

    
    
    dist.init_process_group(init_method='tcp://10.1.1.20:23456', rank=args.rank, world_size=4)
    
    

在第二种情况下，多播地址指定组节点谁可能潜在地是活性和协调可以通过允许每个进程遵循上面的程序之前，有一个初始握手处理的。此外TCP组播初始化还支持`组名
`参数(与共享文件的方法），从而允许多个作业要在同一群集中调度。

    
    
    dist.init_process_group(init_method='tcp://[ff15:1e18:5d4c:4cf0:d02d:b659:53ba:b0a7]:23456',
                            world_size=4)
    
    

**致谢**

我想感谢PyTorch开发人员就其执行，文档和测试做这样一个好工作。当代码不清楚，我总能指望[文档](https://pytorch.org/docs/stable/distributed.html)或[测试](https://github.com/pytorch/pytorch/blob/master/test/test_distributed.py)找到答案。我特别要感谢Soumith
Chintala，亚当Paszke，和Natalia Gimelshein提供有见地的意见和回答有关初稿的问题。

[Next ![](../_static/images/chevron-right-
orange.svg)](../beginner/aws_distributed_training_tutorial.html "4.
\(advanced\) PyTorch 1.0 Distributed Trainer with Amazon AWS")
[![](../_static/images/chevron-right-orange.svg) Previous](ddp_tutorial.html
"2. Getting Started with Distributed Data Parallel")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * 3\. PyTorch编写分布式应用
    * 安装
    * 点对点通讯
    * 集群通信
    * [HTG0分布式训练
      * 我们自己的戒指，Allreduce 
    * 高级主题
      * 通信后端
      * 初始化方法

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



