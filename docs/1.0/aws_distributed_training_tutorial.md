# PyTorch 1.0 使用 Amazon AWS 进行分布式训练

> 译者：[yportne13](https://github.com/yportne13)

**作者**: [Nathan Inkawhich](https://github.com/inkawhich)

**编辑**: [Teng Li](https://github.com/teng-li)

在这篇教程中我们会展示如何使用 Amazon AWS 的两个多路GPU节点来设置，编写和运行 PyTorch 1.0 分布式训练程序。首先我们会介绍 AWS 设置, 然后是 PyTorch 环境配置, 最后是分布式训练的代码。你会发现想改成分布式应用你只需要对你目前写的训练程序做很少的代码改动, 绝大多数工作都只是一次性的环境配置。

## Amazon AWS 设置

在这篇教程中我们会在两个多路 GPU 节点上运行分布式训练。在这一节中我们首先会展示如何创建节点，然后是设置安全组(security group)来让节点之间能够通信。

### 创建节点

在 Amazon AWS 上创建一个实例需要七个步骤。首先，登录并选择 **Launch Instance**.

**1: 选择 Amazon Machine Image (AMI)** - 我们选择 `Deep Learning AMI (Ubuntu) Version 14.0`。 正如介绍中所说的，这个实例安装了许多流行的深度学习框架并已经配置好了 CUDA, cuDNN 和 NCCL。 这是一个很好的开始。

**2: 选择一个实例类型** - 选择GPU计算单元 `p2.8xlarge`。 注意，每个实例的价格不同，这个实例为每个节点提供 8 个 NVIDIA Tesla K80 GPU，并且提供了适合多路 GPU 分布式训练的架构。

**3: 设置实例的细节** - 唯一需要设置的就是把 _Number of instances_ 加到 2。其他的都可以保留默认设置。

**4: 增加存储空间** - 注意, 默认情况下这些节点并没有很大的存储空间 (只有 75 GB)。对于这个教程, 我们只使用 STL-10 数据集, 存储空间是完全够用的。但如果你想要训练一个大的数据集比如 ImageNet , 你需要根据数据集和训练模型去增加存储空间。

**5: 加 Tags** - 这一步没有什么需要设置的，直接进入下一步。

**6: 设置安全组(Security Group)** - 这一步很重要。默认情况下同一安全组的两个节点无法在分布式训练设置下通信。 这里我们想要创建一个**新的**安全组并将两个节点加入组内。 但是我们没法在这一步完成这一设置。记住你设置的新的安全组名(例如 launch-wizard-12)然后进入下一步步骤7。

**7: 确认实例启动** - 接下来，确认例程并启动它。 默认情况下这会自动开始两个实例的初始化。你可以通过控制面板监视初始化的进程。

### 设置安全组

我们刚才在创建实例的时候没办法正确设置安全组。当你启动好实例后，在 EC2 的控制面板选择 _Network & Security &gt; Security Groups_ 选项。 这将显示你有权限访问的安全组列表。 选择你在第六步创建的新的安全组(也就是 launch-wizard-12), 会弹出选项 _Description, Inbound, Outbound, and Tags_。 首先，选择 _Inbound_ 的 _Edit_ 选项添加规则以允许来自 launch-wizard-12 安全组内源(“Sources”)的所有流量(“All Traffic”)。 然后选择 _Outbound_ 选项并做同样的工作。 现在，我们有效地允许了 launch-wizard-12 安全组所有类型的入站和出站流量(Inbound and Outbound traffic)。

### 必要的信息

继续下一步之前，我们必须找到并记住节点的IP地址。 在 EC2 的控制面板找到你正在运行的实例。 记下实例的 _IPv4 Public IP_ 和 _Private IPs_。 在之后的文档中我们会把这些称为 **node0-publicIP**, **node0-privateIP**, **node1-publicIP** 和 **node1-privateIP**。 其中 public IP 地址用来 SSH 登录,  private IP 用来节点间通信。

## 环境配置

下一个重要步骤是设置各个节点。 不幸的是，我们不能同时设置两个节点, 所以这一步必须每个节点分别做一遍。 然而，这是一次性的设置，一旦你的节点设置正常你就不需要再为你未来的分布式训练项目重新设置了。

第一步，登录节点，创建一个带 python 3.6 和 numpy 的 conda 环境。 创建完成后激活环境。

```py
$ conda create -n nightly_pt python=3.6 numpy
$ source activate nightly_pt

```

下一步，我们使用 pip 在 conda 环境中每日编译 (nightly build) 支持 Cuda 9.0 的 PyTorch 。

```py
$ pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html

```

我们还需要安装 torchvision 来使用 torchvision 的模型和数据集。这次我们需要从源代码构建 torchvision 因为使用 pip 安装会默认安装老版本的 PyTorch 。

```py
$ cd
$ git clone https://github.com/pytorch/vision.git
$ cd vision
$ python setup.py install

```

最后, 一步**很重要**的步骤是为 NCCL 设置网络接口名。这步通过设置环境变量 `NCCL_SOCKET_IFNAME` 来实现。 为了获得正确的名字，在节点上执行 `ifconfig` 命令并看和节点对应的 _privateIP_ (例如 ens3)接口名字。 然后设置环境变量如下

```py
$ export NCCL_SOCKET_IFNAME=ens3

```

记住，对所有节点都执行这个操作。 你也许还需要考虑对 _.bashrc_ 添加 NCCL_SOCKET_IFNAME。 注意到我们没有在节点间设置共享文件系统。 因此，每个节点都需要复制一份代码和数据集。 想要了解更多有关设置节点间共享文件系统参考[这里](https://aws.amazon.com/blogs/aws/amazon-elastic-file-system-shared-file-storage-for-amazon-ec2/).

## 分布式训练代码

实例开始运行，环境配置好了以后我们可以开始准备训练代码了。绝大多数代码是从 [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet) 来的，这些代码同样支持分布式训练。以这个代码为基础你可以搭自己的训练代码因为它有标准的训练循环，验证循环和准确率追踪函数。然而，你会注意到为了简洁起见参数解析和其他非必须的函数被去掉了。

在这个例子中我们会使用 [torchvision.models.resnet18](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet18) 模型并将会在 [torchvision.datasets.STL10](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.STL10) 数据集上训练它。 为了解决 STL-10 和 Resnet18 维度不匹配的问题, 我们将会使用一个变换把图片的尺寸改为 224x224。 注意到，对于分布式训练代码，模型和数据集的选择是正交(orthogonal)的, 你可以选择任何你想用的数据集和模型，操作的步骤是一样的。 让我们首先操作 import 和一些辅助函数。然后我们会定义 train 和 test 函数，这些都可以从 ImageNet Example 例程中大量复制出来。 结尾部分，我们会搭建代码的 main 部分来设置分布式训练。 最后我们会讨论如何让代码运行起来。

### Imports

在分布式训练中特别需要 import 的东西是 [torch.nn.parallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel), [torch.distributed](https://pytorch.org/docs/stable/distributed.html), [torch.utils.data.distributed](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) 和 [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html)。同时需要把多进程的 start 方法(multiprocessing start method) 设置为 _spawn_ 或 _forkserver_ (仅支持 Python 3), 因为默认是 _fork_ 会导致使用多进程加载数据时锁死。

```py
import time
import sys
import torch

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.multiprocessing import Pool, Process

```

### 辅助函数

我们还需要定义一些辅助函数和类来使训练更简单。 `AverageMeter` 类追踪训练的状态比如准确率和循环次数。`accuracy` 函数计算并返还模型的 top-k 准确率这样我们就可以跟踪学习进程。 这两个都是为了训练方便而不是为分布式训练特别设定的。

```py
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

```

### 训练函数

为了简化 main 循环，最好把一步 training epoch 放进 `train` 函数中。 这个函数用于训练一个 epoch 的 _train_loader_ 的输入模型。 唯一需要为分布式训练特别调整的是在前向传播前将数据和标签张量的 [non_blocking](https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers) 属性设置为 `True`。 这允许异步 GPU 复制数据也就是说计算和数据传输可以同时进行。 这个函数同时也输出训练状态这样我们就可以在整个 epoch 中跟踪进展。

另一个需要定义的函数是 `adjust_learning_rate`, 这个函数以一个固定的方式调低学习率。这也是一个标准的训练函数，有助于训练准确的模型。

```py
def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode 转到训练模式
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time 计算加载数据的时间
        data_time.update(time.time() - end)

        # Create non_blocking tensors for distributed training 为分布式训练创建 non_blocking 张量
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output 计算输出
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss 计算准确率并记录 loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradients in a backward pass 在反向传播中计算梯度
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params 调用一个 optimizer 步骤来更新模型参数
        optimizer.step()

        # measure elapsed time 计算花费的时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

```

### 验证函数

为了进一步简化 main 循环和追踪进程我们可以把验证过程放进命名为 `validate` 的函数中。 这个函数对输入的验证集数据在输入模型上执行一个完整的验证步骤并返还验证集对该模型的 top-1 准确率。 和刚才一样，你会注意到这里唯一需要为分布式训练特别设置的特性依然是在传递进模型前将训练数据和标签值设定 `non_blocking=True`。

```py
def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode 转到验证模式
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output 计算输出
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss 计算准确率并记录 loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time 计算花费时间
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

```

### 输入

随着辅助函数的出现，我们进入了有趣的部分。这里我们将会定义程序的输入部分。一些输入的参数是标准的训练模型的输入比如 batch size 和训练的 epoch 数, 而有些则是我们的分布式训练任务特别需要的。需要的输入参数是：

*   **batch_size** - 分布式训练组中_单一_进程的 batch size。 整个分布式模型总的 batch size 是 batch_size*world_size
*   **workers** - 每个进程中数据加载使用的工作进程数
*   **num_epochs** - 总的训练的 epoch 数
*   **starting_lr** - 开始训练时的学习率
*   **world_size** - 分布式训练环境的进程数
*   **dist_backend** - 分布式训练通信使用的后端框架 (也就是 NCCL, Gloo, MPI 等)。 在这篇教程中因为我们使用了多个多路 GPU 节点因此推荐 NCCL。
*   **dist_url** - 确定进程组的初始化方法的 URL。 这可能包含 IP 地址和 rank0 进程的端口或者是一个在共享文件系统中的 non-existant 文件。 这里由于我们没有共享文件系统因此是包含 **node0-privateIP** 和要使用的 node0 的端口的 url。

```py
print("Collect Inputs...")

# Batch Size for training and testing 训练和测试的 batch size
batch_size = 32

# Number of additional worker processes for dataloading 数据加载的额外工作进程数
workers = 2

# Number of epochs to train for 训练的 epoch 数
num_epochs = 2

# Starting Learning Rate 初始学习率
starting_lr = 0.1

# Number of distributed processes 分布式进程数
world_size = 4

# Distributed backend type 分布式后端类型
dist_backend = 'nccl'

# Url used to setup distributed training 设置分布式训练的 url
dist_url = "tcp://172.31.22.234:23456"

```

### 初始化进程组

在使用 PyTorch 进行分布式训练中有一个很重要的部分是正确设置进程组, 也就是初始化 `torch.distributed` 包的**第一**步。为了完成这一步我们将会使用 `torch.distributed.init_process_group` 函数，这个函数需要几个输入参数。首先，需要输入 _backend_ 参数，这个参数描述了需要什么后端(也就是 NCCL, Gloo, MPI 等)。 输入参数 _init_method_ 同时也是包含 rank0 地址和端口的 url 或是共享文件系统上的 non-existant 文件路径。注意，为了使用文件的 init_method, 所有机器必须有访问文件的权限，和使用 url 方法类似，所有机器必须要能够联网通信所以确保防火墙和网络设置正确。 _init_process_group_ 函数也接受 _rank_ 和 _world_size_ 参数，这些参数表明了进程运行时的编号并分别展示了集群内的进程数。_init_method_ 也可以是 “env://”。 在这种情况下，rank0 机器的地址和端口将会分别从以下环境变量中读出来：MASTER_ADDR, MASTER_PORT。 如果 _rank_ 和 _world_size_ 参数没有在 _init_process_group_ 函数中表示出来，他们都可以从以下环境变量中分别读出来：RANK, WORLD_SIZE。

另一个重要步骤，尤其是当一个节点使用多路 gpu 的时候，就是设置进程的 _local_rank_。 例如，如果你有两个节点，每个节点有8个 GPU 并且你希望使用所有 GPU 来训练那么设置 $$world\_size=16$$ 这样每个节点都会有一个本地编号为 0-7 的进程。 这个本地编号(local_rank) 是用来为进程配置设备 (也就是所使用的 GPU ) 并且之后用来创建分布式数据并行模型时配置设备。 在这样的假定环境下同样推荐使用 NCCL 后端因为 NCCL 更适合多路 gpu 节点。

```py
print("Initialize Process Group...")
# Initialize Process Group 初始化进程组
# v1 - init with url  使用 url 初始化
dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(sys.argv[1]), world_size=world_size)
# v2 - init with file 使用文件初始化
# dist.init_process_group(backend="nccl", init_method="file:///home/ubuntu/pt-distributed-tutorial/trainfile", rank=int(sys.argv[1]), world_size=world_size)
# v3 - init with environment variables 使用环境变量初始化
# dist.init_process_group(backend="nccl", init_method="env://", rank=int(sys.argv[1]), world_size=world_size)

# Establish Local Rank and set device on this node 设置节点的本地化编号和设备
local_rank = int(sys.argv[2])
dp_device_ids = [local_rank]
torch.cuda.set_device(local_rank)

```

### 初始化模型

下一个主要步骤是初始化训练模型。这里我们将会使用 `torchvision.models` 中的 resnet18 模型但是你可以选用任何一种模型。首先，我们初始化模型并将它放进显存中。然后，我们创建模型 `DistributedDataParallel`, 它负责分配数据进出模型，这对分布式训练很重要。 `DistributedDataParallel` 模块同时也计算整体的平均梯度, 这样我们就不需要在训练步骤计算平均梯度。

还要注意到这是一个阻塞函数 (blocking function), 也就是程序执行时会在这个函数等待直到 _world_size_ 进程加入进程组。 同时注意到，我们将我们的设备 ids 表以参数的形式传递，这个参数还包含了我们正在使用的本地编号 (也就是 GPU)。 最后，我们设定了训练模型使用的 loss function 和 optimizer。

```py
print("Initialize Model...")
# Construct Model 构建模型
model = models.resnet18(pretrained=False).cuda()
# Make model DistributedDataParallel  
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)

# define loss function (criterion) and optimizer 定义 loss 函数和 optimizer 
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4)

```

### 初始化数据加载器 (dataloader)

准备训练的最后一步是确认使用什么数据集。 这里我们使用 [torchvision.datasets.STL10](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.STL10) 中的 [STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/)。 STL10 数据集是一个 10 分类 96x96px 彩色图片集。为了在我们的模型中使用它，我们在一个变换中把图片的尺寸调整为 224x224px。 在这节中特别需要为分布式训练准备的东西是为训练集使用 `DistributedSampler`，这是设计来与 `DistributedDataParallel` 模型相结合的。 这个对象控制进入分布式环境的数据集以确保模型不是对同一个子数据集训练，以达到训练目标。最后，我们创建 `DataLoader` 负责向模型喂数据。

如果你的节点上没有 STL-10 数据集那么它会自动下载到节点上。如果你想要使用你自己的数据集那么下载你的数据集，搭建你自己的数据操作函数和加载器。

```py
print("Initialize Dataloaders...")
# Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model. 定义数据的变换。尺寸转为224x224
transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Initialize Datasets. STL10 will automatically download if not present 初始化数据集。如果没有STL10数据集则会自动下载
trainset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
valset = datasets.STL10(root='./data', split='test', download=True, transform=transform)

# Create DistributedSampler to handle distributing the dataset across nodes when training 创建分布式采样器来控制训练中节点间的数据分发
# This can only be called after torch.distributed.init_process_group is called 这个只能在 torch.distributed.init_process_group 被调用后调用
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

# Create the Dataloaders to feed data to the training and validation steps 创建数据加载器，在训练和验证步骤中喂数据
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers, pin_memory=False, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

```

### 训练循环

最后一步是定义训练循环。我们已经完成了设置分布式训练的绝大多数工作了，这一步不是特别为分布式训练做的。 唯一的细节是在 `DistributedSampler` 中记录目前的 epoch 数， 因为采样器是根据 epoch 来决定如何打乱分配数据进各个进程。 更新采样器后，循环执行一个完整的 epoch， 一个完整的验证步骤然后打印目前模型的表现并和目前表现最好的模型对比。 在训练了 num_epochs 后, 循环退出，教程结束。注意，因为这只是个例程，我们没有保存模型，但如果想要训练结束后保存表现最好的模型请看[这里](https://github.com/pytorch/examples/blob/master/imagenet/main.py#L184)。

```py
best_prec1 = 0

for epoch in range(num_epochs):
    # Set epoch count for DistributedSampler 为分布式采样器设置 epoch 数
    train_sampler.set_epoch(epoch)

    # Adjust learning rate according to schedule 调整学习率
    adjust_learning_rate(starting_lr, optimizer, epoch)

    # train for one epoch 训练1个 epoch
    print("\nBegin Training Epoch {}".format(epoch+1))
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set 在验证集上验证
    print("Begin Validation @ Epoch {}".format(epoch+1))
    prec1 = validate(val_loader, model, criterion)

    # remember best prec@1 and save checkpoint if desired 保存最佳的prec@1，如果需要的话保存检查点
    # is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    print("Epoch Summary: ")
    print("\tEpoch Accuracy: {}".format(prec1))
    print("\tBest Accuracy: {}".format(best_prec1))

```

## 运行代码

和其他 PyTorch 教程不一样, 这个代码也许不能直接以 notebook 的形式执行。 为了运行它需要以 .py 形式下载这份文件(或者使用[这个](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe)来转换它)然后复制到各个节点上。 聪明的读者也许注意到了我们写死了(硬编码，hardcode) **node0-privateIP** 和 $$world\_size=4$$ 但把 _rank_ 和 _local_rank_ 以 arg[1] 和 arg[2] 命令行参数的形式分别输入。 上传后对每个节点分别打开两个 ssh 终端。

*   对 node0 的第一个终端，运行 `$ python main.py 0 0`
*   对 node0 的第二个终端，运行 `$ python main.py 1 1`
*   对 node1 的第一个终端，运行 `$ python main.py 2 0`
*   对 node1 的第二个终端，运行 `$ python main.py 3 1`

程序会开始运行并等待直到四个进程都加入进程组后打印 “Initialize Model…” 。 注意到第一个参数不能重复因为这是进程的全局编号(唯一的)。 第二个参数可重复因为这是节点上进程的本地编号。 如果你对每个节点运行 `nvidia-smi`，你会看见每个节点上有两个进程，一个运行在 GPU0 上，另一个运行在 GPU1 上。

我们现在已经实现了一个分布式训练的范例！ 希望你可以通过这个教程学会如何在你自己的数据集上搭建你自己的模型，即使你不是使用同样的分布式环境。 如果你在使用 AWS，切记在你不使用时**关掉你的节点**不然月末你会发现你要交好多钱。

**接下来看什么**

*   看看 [launcher utility](https://pytorch.org/docs/stable/distributed.html#launch-utility) 以了解另一种启动运行的方式
*   看看 [torch.multiprocessing.spawn utility](https://pytorch.org/docs/master/multiprocessing.html#spawning-subprocesses) 以了解另一种简单的启动多路分布式进程的方式。 [PyTorch ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet) 已经实现并可以演示如何使用它。
*   如果可能，请设置一个NFS，这样你只需要一个数据集副本
