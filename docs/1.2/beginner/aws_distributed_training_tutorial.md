# (高级）PyTorch 1.0分布式训练与Amazon AWS

**作者** ：[弥敦道Inkawhich ](https://github.com/inkawhich)

**由** 编辑：[腾力](https://github.com/teng-li)

在本教程中，我们将介绍如何设置，代码，并在两个多GPU亚马逊AWS节点运行PyTorch
1.0分布式教练。我们将与描述为分布式教练的AWS设置，那么PyTorch环境配置，最后的代码开始。希望你会发现，有您当前的训练码扩展到分布式应用程序实际上需要很少的代码变化，大部分工作是在一次环境设置。

## 亚马逊AWS设定

在本教程中，我们将运行在两个多GPU节点的分布式训练。在本节中，我们将首先介绍如何创建节点，那么如何设置安全组，这样的节点可以与海誓山盟沟通。

### 创建结点

在亚马逊AWS，有七个步骤来创建一个实例。要开始，登录并选择 **启动实例[HTG1。**

[HTG0步骤1：选择一个亚马逊机器映像(AMI） - 在这里，我们将选择`深 学习 AMI  (Ubuntu的） 版 14.0
`。如上所述，这种情况下附带了许多安装了最流行的深学习框架，并进行了预配置CUDA，cuDNN和NCCL。正是由于这个教程一个很好的起点。

**步骤2：选择一个实例类型** \- 称为`现在，选择GPU计算单元p2.8xlarge
`。通知，每个这些实例都具有不同的成本，但这种情况下，每个节点提供了8个NVIDIA特斯拉K80 GPU和提供了用于多GPU分布式训练良好的体系结构。

**步骤3：配置实例详细说明** \- 改变这里的唯一设置在增加的实例的 _Number来2.所有其它配置可以在默认留。_

**第四步：添加存储** \- 请注意，默认情况下，这些节点不来了大量的存储空间(只有75
GB）。对于本教程，因为我们只使用STL-10数据集，这是充足的存储空间。但是，如果你想在一个更大的数据集，如ImageNet训练，你将不得不增加更多的存储只是为了适应数据集，并要保存任何训练的模型。

[HTG0步骤5：添加标签 - 什么可以在这里完成，只是继续前进。

[HTG0步骤6：配置安全组 - 这是在配置过程中的关键步骤。默认情况下，同一安全组中的两个节点将不能够在分布式训练环境进行通信。在这里，我们要创建一个
**新** 为两个节点安全组是在，但我们无法完成配置在这一步。现在，只记得你的新的安全组的名称(例如，推出的向导-12），然后转移到步骤7。

[HTG0步骤7：回顾实例启动 - 在这里，查看实例，然后启动它。默认情况下，会自动启动初始化两个实例。您可以监视从仪表板的初始化进度。

### 配置安全组

回想一下，我们无法创建实例时正确配置安全组。一旦你启动实例，选择 _网络 &放;安全& GT
;安全组_在EC2仪表板选项卡。这将显示您可以访问安全组的列表。选择在步骤6中创建的新的安全组(即启动的向导-12），这将打开的选项卡被称为
_说明，入站，出站，和标签_ 。首先，选择 _入境_ 选项卡和 _编辑_ 从“源”推出的向导-12安全组中添加规则允许“所有流量”。然后选择 _出境_
选项卡，然后做同样的事情。现在，我们已经有效地使推出的向导-12安全组中的节点之间的所有类型的所有入站和出站流量。

### 必要信息

在继续之前，我们必须找到并记住两个节点的IP地址。在EC2仪表板找到你正在运行的实例。对于这两种情况下，写下 _IPv4公网IP_ 和
_私人IP地址[HTG3。对于文档的其余部分，我们将把这些作为 **NODE0-publicIP** ， **NODE0-privateIP** ，
**节点1-publicIP** 和 **node1- privateIP** 。公共IP地址是我们将使用SSH连接的地址和私有地址将被用于节点间通信。_

## 环境设置

下一关键步骤是在每个节点的设置。不幸的是，我们不能同时设置两个节点，所以这个过程必须在每个节点上分别进行。然而，这是一个时间的设置，所以一旦你正确配置的节点，您将不必重新配置为未来分布式训练项目。

第一步骤中，一旦登录到节点，是与蟒3.6和numpy的创建一个新的康达环境。一旦创建启动环境。

    
    
    $ conda create -n nightly_pt python=3.6 numpy
    $ source activate nightly_pt
    

接下来，我们将安装Cuda的9.0启用PyTorch的每晚构建与在畅达环境点子。

    
    
    $ pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
    

我们还必须安装torchvision所以我们可以使用torchvision模型和数据集。此时，由于PIP安装将在默认情况下，我们刚刚安装的每晚构建的顶部安装了旧版本PyTorch的，我们必须从源代码编译torchvision。

    
    
    $ cd
    $ git clone https://github.com/pytorch/vision.git
    $ cd vision
    $ python setup.py install
    

最后， **非常重要[HTG1步骤是为NCCL插座设置网络接口的名称。这被设定为环境变量`NCCL_SOCKET_IFNAME
`。为了得到正确的名称，该节点上运行`使用ifconfig`命令，查看对应的接口名称节点的 _privateIP
[HTG11(例如ens3）。然后设置环境变量_**

    
    
    $ export NCCL_SOCKET_IFNAME=ens3
    

请记住，这样做两个节点上。您也可以考虑加入NCCL_SOCKET_IFNAME设置为你的
_.bashrc中[HTG1。一个重要的观察是，我们没有设置节点之间共享的文件系统。因此，每个节点必须有代码的副本和所述数据集的副本。有关设置节点之间的共享的网络文件系统的更多信息，请参见[这里](https://aws.amazon.com/blogs/aws/amazon-
elastic-file-system-shared-file-storage-for-amazon-ec2/)。_

## 分布式训练码

随着运行的实例和环境设置，我们现在可以进入训练码。这里的大多数代码的已采取从[ PyTorch
ImageNet实施例](https://github.com/pytorch/examples/tree/master/imagenet)这也支持分布式训练。此代码提供了一个自定义的教练一个很好的起点，因为它有很多的样板训练循环，确认循环和准确性跟踪功能。但是，您会注意到参数解析和其他非必要的功能已被剥离出来的简单性。

在这个例子中，我们将使用[ torchvision.models.resnet18
](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet18)模式，并将训练它的[
torchvision.datasets.STL10
](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.STL10)数据集。为了适应对STL-10与Resnet18维数不匹配，我们将每个图像尺寸调整到224x224通过转换。请注意，模型和数据集的选择是正交的分布式训练码，你可以使用任何你想要的数据集，模型和过程是相同的。让我们得到由第一处理进口和谈论了一些辅助功能启动。然后，我们将定义火车和测试功能，这已经在很大程度上从ImageNet实施例作出。最后，我们将构建一个处理分布式训练设置的代码的主要部分。最后，我们将讨论如何实际运行代码。

### 进口

最重要的分布式训练特定这里进口[ torch.nn.parallel
](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)，torch.distributed
[，torch.utils.data.distributed
](https://pytorch.org/docs/stable/distributed.html)
[和](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)[Torch 。并行处理](https://pytorch.org/docs/stable/multiprocessing.html)。同样重要的是，设置多启动方法为
_菌种_ 或 _forkserver_ (仅在Python 3支持），作为默认的是 _叉_ 这可能引起死锁时使用多个工作进行dataloading处理。

    
    
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
    

### 辅助函数

我们还必须定义一些辅助函数和类将会使训练更容易。在`AverageMeter`类曲目训练统计资料，例如精度和迭代次数。在`精度
`函数计算并返回模型的前k精度，所以我们可以跟踪学习进度。两者都提供了方便训练但是没有分配具体的训练。

    
    
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
    

### 列车功能

为了简化主循环中，最好是分离出训练时期步骤到一个名为`列车 `功能。该功能用于训练的 _train_loader_
一个历元输入模型。在此功能仅分布训练神器的数据和标签张量[ non_blocking
](https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-
buffers)属性设置为`前直传真
[HTG11。这使得数据传输含义异步GPU拷贝可以与计算重叠。该功能还输出沿途训练统计，所以我们可以跟踪整个时代的进步。`

其它功能在这里定义为`adjust_learning_rate
`，其衰减在一个固定的时间表初始学习速率。这又是一个样板教练功能是训练精确的模型非常有用。

    
    
    def train(train_loader, model, criterion, optimizer, epoch):
    
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
    
        # switch to train mode
        model.train()
    
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
    
            # measure data loading time
            data_time.update(time.time() - end)
    
            # Create non_blocking tensors for distributed training
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
    
            # compute output
            output = model(input)
            loss = criterion(output, target)
    
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))
    
            # compute gradients in a backward pass
            optimizer.zero_grad()
            loss.backward()
    
            # Call step of optimizer to update model params
            optimizer.step()
    
            # measure elapsed time
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
    

### 验证函数

为了跟踪推广性能和简化主回路还我们还可以提取所述验证步骤到一个名为`函数验证
`。该函数运行在输入验证的DataLoader输入模型的一个完整的验证步骤，并返回所述验证集的模型的顶部-1的精度。同样，你会发现这里唯一的分布式训练功能设置`
non_blocking =真 `训练数据和标签它们传递给模型前。

    
    
    def validate(val_loader, model, criterion):
    
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
    
        # switch to evaluate mode
        model.eval()
    
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
    
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
    
                # compute output
                output = model(input)
                loss = criterion(output, target)
    
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))
    
                # measure elapsed time
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
    

### 输入

随着辅助函数的方式进行，现在我们已经到了有趣的部分。这里我们将定义运行的输入。一些输入是标准模型的训练投入，如批量大小和训练时期的数量，有些是专门针对我们的分布式训练任务。所需的输入是：

  * **的batch_size** \- 批量大小为 _分布式训练组中的每个_ 过程。在分布式模型总批量大小的batch_size是* world_size
  * **工人** \- 中的每个进程与所述dataloaders使用的工作进程数
  * **num_epochs** \- 历元用于训练的总次数
  * **starting_lr** \- 开始进行训练学习速率
  * **world_size** \- 过程在分布式训练环境数
  * **dist_backend** \- 后端用于分布式训练的通信(即NCCL，GLOO，MPI，等）。在本教程中，由于我们使用几个多GPU节点，NCCL建议。
  * **dist_url** \- URL来指定处理组的初始化方法。这可以包含rank0处理的IP地址和端口或者是一个共享的文件系统上的不存在的文件。这里，因为我们没有一个共享文件系统，这将包括在NODE0使用 **NODE0-privateIP** 和端口。

    
    
    print("Collect Inputs...")
    
    # Batch Size for training and testing
    batch_size = 32
    
    # Number of additional worker processes for dataloading
    workers = 2
    
    # Number of epochs to train for
    num_epochs = 2
    
    # Starting Learning Rate
    starting_lr = 0.1
    
    # Number of distributed processes
    world_size = 4
    
    # Distributed backend type
    dist_backend = 'nccl'
    
    # Url used to setup distributed training
    dist_url = "tcp://172.31.22.234:23456"
    

### 初始化进程组

一个在PyTorch分布式训练的最重要的部分是正确设置进程组，这是在初始化`torch.distributed`包 **第一**
步骤。要做到这一点，我们将使用`torch.distributed.init_process_group`功能，需要几个输入。首先， _后端_
输入指定后端使用(即NCCL，GLOO，MPI，等）。一个 _init_method_
输入其是含有rank0机器的地址和端口或共享文件系统上的一个不存在的文件的路径的URL。注意，使用文件init_method，所有机器必须能够访问该文件，同样的网址的方法，所有机器必须能够在网络上进行通信，从而确保配置任何防火墙和网络设置，以适应。的
_init_process_group_ 功能也需要 _秩_ 和 _world_size_ 用于指定运行时该方法的秩和的过程中集体的数量，分别参数。的
_init_method_ 输入也可以是“ENV：//”。
MASTER_ADDR，MASTER_PORT：在这种情况下，rank0机器的地址和端口号会从以下两个环境变量分别读取。
RANK，WORLD_SIZE：如果 _位次_ 和 _world_size_ 未在 _init_process_group_
功能指定的参数，它们都可以从以下两个环境变量分别也被读取。

另一个重要的步骤，特别是当每一个节点具有多个GPU是设置 _local_rank_ 该方法的。例如，如果有两个节点，每个节点8个GPU和希望与他们的训练然后
\(世界\ _size = 16
\），并且每个节点将与本地秩0-的处理7。此local_rank用于设置该装置(即要使用的GPU）的过程和随后用于创建分布式数据并行模型时设置该装置。此外，还建议使用NCCL后端在这个假设的环境NCCL是优选的多GPU节点。

    
    
    print("Initialize Process Group...")
    # Initialize Process Group
    # v1 - init with url
    dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(sys.argv[1]), world_size=world_size)
    # v2 - init with file
    # dist.init_process_group(backend="nccl", init_method="file:///home/ubuntu/pt-distributed-tutorial/trainfile", rank=int(sys.argv[1]), world_size=world_size)
    # v3 - init with environment variables
    # dist.init_process_group(backend="nccl", init_method="env://", rank=int(sys.argv[1]), world_size=world_size)
    
    
    # Establish Local Rank and set device on this node
    local_rank = int(sys.argv[2])
    dp_device_ids = [local_rank]
    torch.cuda.set_device(local_rank)
    

### 初始化模型

下一个主要步骤是初始化进行训练的模式。在这里，我们将使用`torchvision.models
`一个resnet18模式，但可以使用任何模型。首先，我们初始化模式，并把它放在GPU内存。接下来，我们使模型`
DistributedDataParallel`，其处理数据的分布和模型，是分布式训练的关键。在`DistributedDataParallel
`模块还可以处理世界各地的梯度的平均，所以我们没有在训练步骤明确平均梯度。

要注意，这是一个阻塞功能，这意味着程序执行将在此函数等到 _world_size_
工艺已经加入了处理组是重要的。另外，还要注意我们传递的设备ID列表，其中包含了本地等级(即GPU），我们正在使用的参数。最后，我们确定损失的功能和优化训练与模型。

    
    
    print("Initialize Model...")
    # Construct Model
    model = models.resnet18(pretrained=False).cuda()
    # Make model DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4)
    

### 初始化Dataloaders

在训练准备的最后一步是指定要使用的数据集。这里我们使用[ torchvision.datasets.STL10
](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.STL10)中的[
STL-10数据集[HTG1。所述STL10数据集是96x96px彩色图像的10类数据集。对于我们的模型的使用，我们调整图像224x224px在变换。本节中的一个分布式训练特定项是使用的`
DistributedSampler`对于训练集，其被设计为与`DistributedDataParallel一起使用
`车型。这个对象处理跨分布式环境中，这样并非所有型号都在相同的数据集，这将是适得其反的训练数据集的划分。最后，我们创建`的DataLoader
`的其负责馈送的数据的处理。](https://cs.stanford.edu/~acoates/stl10/)

如果它们不存在的STL-10数据集将自动节点上下载。如果你想使用自己的数据集，你应该下载的数据，编写自己的数据集的处理程序，并在这里建立了您的数据集的DataLoader。

    
    
    print("Initialize Dataloaders...")
    # Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Initialize Datasets. STL10 will automatically download if not present
    trainset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
    valset = datasets.STL10(root='./data', split='test', download=True, transform=transform)
    
    # Create DistributedSampler to handle distributing the dataset across nodes when training
    # This can only be called after torch.distributed.init_process_group is called
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    
    # Create the Dataloaders to feed data to the training and validation steps
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers, pin_memory=False, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
    

### 训练循环

最后一步是界定训练循环。我们已经完成了大部分的工作，为建立分布式训练所以这不是分布式训练具体。唯一的细节是设置在`DistributedSampler
`，作为取样洗牌的数据要每个进程确定性地基于历元的当前历元计数。更新采样后，循环运行完整的训练时期，然后运行一个完整的验证步骤打印对表现最好的模型当前模型的性能至今。对于num_epochs训练结束后，退出循环和教程结束。请注意，因为这是我们没有保存模型的工作，但不妨一跟踪性能最佳的模型，然后将其保存在训练结束(见[此处](https://github.com/pytorch/examples/blob/master/imagenet/main.py#L184)）。

    
    
    best_prec1 = 0
    
    for epoch in range(num_epochs):
        # Set epoch count for DistributedSampler
        train_sampler.set_epoch(epoch)
    
        # Adjust learning rate according to schedule
        adjust_learning_rate(starting_lr, optimizer, epoch)
    
        # train for one epoch
        print("\nBegin Training Epoch {}".format(epoch+1))
        train(train_loader, model, criterion, optimizer, epoch)
    
        # evaluate on validation set
        print("Begin Validation @ Epoch {}".format(epoch+1))
        prec1 = validate(val_loader, model, criterion)
    
        # remember best prec@1 and save checkpoint if desired
        # is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
    
        print("Epoch Summary: ")
        print("\tEpoch Accuracy: {}".format(prec1))
        print("\tBest Accuracy: {}".format(best_prec1))
    

## 运行代码

与其他大多数PyTorch教程，这些代码可能无法直接从这款笔记本的运行。要运行，下载此文件的版本的.py(或使用[这个](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe)将其转换）并上传一份给两个节点。细心的读者会注意到，我们硬编码了
**NODE0-privateIP** 和 \(世界\ _size = 4 \）HTG5]，但输入 _位次_ 和 _local_rank_ 输入作为ARG
[1]和Arg [2]的命令行参数，分别。上传后，打开两个SSH终端到每个节点。

  * 论NODE0第一终端，运行`$  蟒 main.py  0  0`
  * 论NODE0运行第二终端`$  蟒 main.py  1  1`
  * 对节点1的第一终端，运行`$  蟒 main.py  2  0`
  * 对节点1运行第二终端`$  蟒 main.py  3  1`

该程序将启动，并打印“初始化模式......”所有四个流程加盟流程组后等待。注意不重复的第一个参数，因为这是过程的独特的全球性排名。重复第二个参数，因为这是在该节点上运行的进程的本地秩。如果你运行`
NVIDIA-SMI`每个节点上，你会看到每个节点上的两个过程，一个是关于GPU0运行，一个在GPU1。

现在我们已经完成了分布式训练的例子！希望你可以看到你将如何使用此教程，以帮助培养你自己的数据集自己的模型，即使您不使用完全相同的分布式环境不受。如果你正在使用AWS，不要忘了，如果你不使用它们
**关闭NODES** 或者你可以在月底发现的令人不安的大账单。

**下一步去哪里**

  * 退房[启动公用](https://pytorch.org/docs/stable/distributed.html#launch-utility)踢关闭运行的不同方式
  * 检查出的[ torch.multiprocessing.spawn实用程序[HTG1用于开球多个分布式方法的另一个简单的方法。 ](https://pytorch.org/docs/master/multiprocessing.html#spawning-subprocesses)[ PyTorch ImageNet实施例](https://github.com/pytorch/examples/tree/master/imagenet)已将其实现，并且可以显示如何使用它。
  * 如果可能的话，设置一个NFS所以你只需要数据集中的一个副本

**脚本的总运行时间：** (0分钟0.000秒）

[`Download Python source code:
aws_distributed_training_tutorial.py`](../_downloads/f8e87d04570b9a376652ece1006edccb/aws_distributed_training_tutorial.py)

[`Download Jupyter notebook:
aws_distributed_training_tutorial.ipynb`](../_downloads/80fe1ab73c6b2b3cefcd5ba0e4ed7609/aws_distributed_training_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../advanced/torch_script_custom_ops.html "Extending TorchScript
with Custom C++ Operators") [![](../_static/images/chevron-right-orange.svg)
Previous](../intermediate/dist_tuto.html "3. Writing Distributed Applications
with PyTorch")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * [HTG0 (高级）PyTorch 1.0分布式训练与Amazon AWS 
    * 亚马逊AWS设定
      * 创建节点
      * 配置安集团
      * 必要的信息
    * 环境设置
    * [HTG0分布式训练码
      * 进口
      * 辅助函数
      * 列车功能
      * 验证函数
      * 输入
      * 初始化处理组
      * 初始化模型
      * 初始化Dataloaders 
      * 训练环
    * 运行代码

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



