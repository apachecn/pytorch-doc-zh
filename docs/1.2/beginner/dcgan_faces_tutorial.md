# DCGAN教程

**作者** ：[弥敦道Inkawhich ](https://github.com/inkawhich)

## 简介

本教程将为通过一个例子介绍了DCGANs。我们将培养出生成对抗网络（GAN）显示它真正的许多名人照片后产生新的名人。这里的大多数代码是从[
pytorch的dcgan执行/例子](https://github.com/pytorch/examples)，而这个文件将给出实施的全面解释，并在此模型如何以及为什么工作的线索。不过不用担心，没有事先甘斯的知识是必需的，但它可能需要第一个定时器花费大约什么是引擎盖下实际发生的一段时间推理。此外，对于时间的缘故，将有助于有一个GPU，或两个。让我们从头开始。

## 生成对抗性网络

### 什么是GAN？

甘斯是教DL模型捕捉训练数据的分布，所以我们可以产生来自同一分布的新数据的框架。甘斯分别由Ian古德费洛在2014年发明并在纸[剖成对抗性篮网](https://papers.nips.cc/paper/5423-generative-
adversarial-nets.pdf)首先描述。它们是由两种不同的型号， _发生器_ 和
_鉴别[HTG5。发电机的工作是产卵，看起来像训练图像“假”的图像。鉴别的工作是看图像和输出是否是真正的训练图像或从发电机假像。在培训过程中，发电机不断尝试通过生成好假货智取鉴别，而鉴别正在努力成为一个更好的侦探和准确区分真假的图像。这个游戏的平衡是当发电机发电，看起来好像他们直接从训练数据来完善假货，并鉴别留给始终在50％的置信猜测，发电机的输出是真实的还是假的。_

现在，让我们开始定义与鉴别某些符号在整个教程中使用。让 \（X \）是表示图像的数据。  \（d（x）的\）是鉴别器的网络，其输出的（标量）概率 \（X
\）来自训练数据，而不是发电机。这里，由于我们在输入处理图像，以 \（d（x）的\）是CHW大小3x64x64的图像。直观地说，
\（d（x）的\）应该是HIGH时 \（X \）来自训练数据和LOW时 \（X \）附带从发电机。
\（d（x）的\）也可以被认为是作为一个传统的二元分类器。

用于发电机的符号，让 \（Z \）是从标准正态分布取样的潜在空间矢量。  \（G（z）的\）表示潜矢量 \（Z
\）映射到数据空间中的发电机的功能。的目标\（G \）是估计训练数据来自于分布（ \（P_ {数据} \）），所以它可以产生从该估计的假样本分布（
\（P_G \））。

因此， \（d（G（Z））\）的概率是（标量），该发电机 \（G
\）的输出是一个真实图像。如[古德费洛的论文](https://papers.nips.cc/paper/5423-generative-
adversarial-nets.pdf)中描述的， \（d \）和 \（G \）发挥极大极小的游戏中， \（d \）尝试最大化其正确分类的实数和赝品（
\（的logD（X）\））和 \（G \）尝试的概率最小化[HTG16概率] \（d \）将预测其输出是假（
\（日志（1-d（G（X）））\））。从本文中，甘损失函数是

\\[\underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}_{x\sim
p_{data}(x)}\big[logD(x)\big] + \mathbb{E}_{z\sim
p_{z}(z)}\big[log(1-D(G(z)))\big]\\]

从理论上讲，解决这一极小极大游戏是其中 \（P_G = P_ {数据}
\），和鉴别器猜测随机如果输入是真实的还是假。然而，甘斯的收敛理论仍在积极研究和现实中的模型并不总是训练到这一点。

### 什么是DCGAN？

甲DCGAN是上述GAN的直接延伸，除了它明确使用卷积和在鉴别器和发电机分别卷积转置层。它首先被拉德福德等说明。人。在文献[无监督表示学习凭借深厚的卷积剖成对抗性网络[HTG1。鉴别器由跨距](https://arxiv.org/pdf/1511.06434.pdf)[卷积](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d)的层，[批次规范](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm2d)层，和[
LeakyReLU
](https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU)激活。输入是3x64x64输入图像和输出是一个标量概率输入是来自真正的数据分布。发电机是由[卷积转置](https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose2d)层，批量规范层，和[
RELU ](https://pytorch.org/docs/stable/nn.html#relu)激活。输入是潜向量， \（Z
\），即从一个标准正态分布绘制和输出是3x64x64
RGB图像。的跨距CONV转置层允许潜矢量被变换成具有相同形状作为图像的体积。在论文中，作者还提供了有关如何设置优化，如何计算损失函数，以及如何初始化模型权重，所有这些都将在未来的章节来说明一些技巧。

    
    
    from __future__ import print_function
    #%matplotlib inline
    import argparse
    import os
    import random
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.optim as optim
    import torch.utils.data
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    

日期：

    
    
    Random Seed:  999
    

## 输入

让我们来定义运行一些投入：

  * **dataroot** \- 路径到数据集的文件夹的根目录。我们将讨论更多有关数据集在下一节
  * **工人** \- 工作线程用于与的DataLoader加载数据的数
  * **BATCH_SIZE** \- 在训练中使用的批量大小。所述DCGAN本文采用的128批量大小
  * **IMAGE_SIZE** \- 用于训练的图像的空间大小。此实现默认为64×64。如果另一个尺寸是期望的，d和G的结构必须改变。参见[此处](https://github.com/pytorch/examples/issues/70)更多细节
  * **NC** \- 在输入图像中的颜色通道的数量。对于彩色图像，这是3
  * **新西兰** \- 长度潜矢量的
  * **NGF** \- 涉及特征映射的通过发电机进行的深度
  * **NDF** \- 设置特征映射的通过鉴别器传播的深度
  * **num_epochs** \- 训练历元的数目来运行。更长的训练可能会带来更好的结果，但也将需要更长的时间
  * **LR** \- 学习培训率。正如DCGAN论文中描述，此数应为0.0002
  * **β1的** \- β1超参数为亚当优化。如在本文所描述的，这个数量应为0.5
  * **ngpu** \- 可用的GPU的数目。如果是0，代码将在CPU模式下运行。如果这个数字大于0，将在这一数字的GPU运行

    
    
    # Root directory for dataset
    dataroot = "data/celeba"
    
    # Number of workers for dataloader
    workers = 2
    
    # Batch size during training
    batch_size = 128
    
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64
    
    # Number of channels in the training images. For color images this is 3
    nc = 3
    
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    
    # Size of feature maps in generator
    ngf = 64
    
    # Size of feature maps in discriminator
    ndf = 64
    
    # Number of training epochs
    num_epochs = 5
    
    # Learning rate for optimizers
    lr = 0.0002
    
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1
    

## 数据

在本教程中，我们将使用[名人-
A面向数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)可在所链接的网站上下载，或者在[谷歌驱动器[HTG3。该数据集将作为下载名为
_img_align_celeba.zip_ 文件。下载完成后，创建一个名为 _celeba_ 目录和zip文件解压到该目录中。那么，这款笔记本的
_celeba刚刚创建_ 目录设置 _dataroot_
输入。生成的目录结构应该是：](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)

    
    
    /path/to/celeba
        -> img_align_celeba
            -> 188242.jpg
            -> 173822.jpg
            -> 284702.jpg
            -> 537394.jpg
               ...
    

这是因为我们将要使用的ImageFolder
DataSet类，这需要有是在数据集的根文件夹中的子目录中的重要一步。现在，我们可以创建数据集，创建的DataLoader，设置设备上运行，并最终显现的一些训练数据。

    
    
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    

![img/sphx_glr_dcgan_faces_tutorial_001.png](img/sphx_glr_dcgan_faces_tutorial_001.png)

## 实现

随着我们的输入参数设置和数据集的准备，我们现在可以进入实施。我们将与weigth初始化策略开始，再谈谈发电机，鉴别，丧失功能，并且训练循环的细节。

### 重量初始化

从DCGAN论文中，作者指定所有模型权重应从均值= 0，标准偏差= 0.02的正态分布随机初始化。的`weights_init
`函数接受一个初始化模型作为输入，并重新初始化所有卷积，卷积转置，并且批标准化层以满足这个标准。这个函数初始化后立即应用于模型。

    
    
    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    

### 发电机

的发电机， \（G \），被设计来映射潜在空间向量（ \（Z \））至数据空间。由于我们的数据是图像，转换 \（Z
\）到数据空间装置最终与相同大小的训练图像创建RGB图像（即3x64x64）。在实践中，这是通过一系列跨距二维卷积转置层，每个具有二维批次模层和RELU激活配对的实现。发电机的输出通过双曲正切函数馈送给它返回到
\输入数据范围（[ - 1,1]
\）。值得一卷积转置层之后注意到的批次范数函数的存在，因为这是DCGAN纸的重要贡献。这些层帮助梯度的培训过程中的流动。从DCGAN纸发电机的图像被如下所示。

![dcgan_generator](img/dcgan_generator.png)

通知，我们如何在输入部分设置的输入（ _新西兰_ ， _NGF_ 和 _NC_ ）在代码影响发生器体系结构。 _新西兰_ 是z输入矢量的长度， _NGF_
涉及通过发生器传播的特征地图的大小，和 _NC_ 是多少在输出图像中的通道（设置为3为RGB图像）。下面是发电机的代码。

    
    
    # Generator Code
    
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
    
        def forward(self, input):
            return self.main(input)
    

现在，我们可以实例发电机和应用`weights_init`功能。退房的打印模型来查看生成的对象是如何构成的。

    
    
    # Create the generator
    netG = Generator(ngpu).to(device)
    
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
    
    # Print the model
    print(netG)
    

Out:

    
    
    Generator(
      (main): Sequential(
        (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        (12): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (13): Tanh()
      )
    )
    

### 鉴别器

如所提到的，鉴别器， \（d \），是二元分类网络拍摄图像作为输入，并输出一个标量概率输入图像是真实的（而不是伪造的）。在此， \（d
\）取3x64x64输入图像，通过一系列Conv2d，BatchNorm2d，和LeakyReLU层进行处理，并通过乙状结肠激活函数输出最终概率。这种架构可以用更多层，如果必要对这个问题进行扩展，但意义利用跨入卷积，BatchNorm和LeakyReLUs的。该DCGAN本文提到它是用跨入卷积，而不是集中到下采样，因为它可以让网络了解自己的池功能一个很好的做法。还批次规范和漏泄RELU功能促进健康的梯度流是用于学习过程临界既
\（G \）和 \（d \）。

鉴别码

    
    
    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
    
        def forward(self, input):
            return self.main(input)
    

现在，与发电机，我们可以创建鉴别，应用`weights_init`功能，打印模型的结构。

    
    
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)
    
    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    
    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)
    
    # Print the model
    print(netD)
    

Out:

    
    
    Discriminator(
      (main): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): LeakyReLU(negative_slope=0.2, inplace=True)
        (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): LeakyReLU(negative_slope=0.2, inplace=True)
        (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (10): LeakyReLU(negative_slope=0.2, inplace=True)
        (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (12): Sigmoid()
      )
    )
    

### 损失函数和优化器

随着 \（d \）HTG1]和 \（G
\）HTG3]设置中，我们可以指定他们通过丧失功能和优化的学习方式。我们将使用二进制交叉熵损失，在PyTorch定义为（[ BCELoss
](https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss)）功能：

\\[\ell(x, y) = L = \\{l_1,\dots,l_N\\}^\top, \quad l_n = - \left[ y_n \cdot
\log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]\\]

注意这个功能如何提供在目标函数中两个日志组件的计算（即， \（日志（d（X））\）和 \（日志（1-d（G（z）的））\））。我们可以指定与 \（Y
\）HTG5]输入要使用什么公元前方程式的一部分。这是在训练环路即将来临完成，而是要了解我们如何可以选择我们希望仅通过改变 \（Y
\）[HTG7（即GT标签）来计算，其成分是很重要的。

接下来，我们定义真实标签为1和计算的的损失时，假标签为0，这些标签将被用来\（d \）和 \（G
\）这也是在原来的GAN纸使用的惯例。最后，我们建立了两个分离的优化器，一个用于 \（d \），一个用于 \（G
\）。正如DCGAN纸指定，都是亚当优化与学习率0.0002和Beta1的=
0.5。用于跟踪发生器的学习进展的，我们将产生潜在向量的固定批次被从高斯分布中抽取（即fixed_noise）。在训练循环中，我们将周期性地输入此fixed_noise到
\（G \），并且在迭代，我们将看到的图像形成了噪音。

    
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()
    
    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    

### 培训

最后，现在我们都定义的GAN框架的部分，我们可以训练它。要留意的是训练甘斯是有点一种艺术形式，是不正确的超参数设置，导致用了什么差错一点解释模式的崩溃。在这里，我们将密切从古德费洛的纸遵循算法1中，同时通过一些在[
ganhacks
](https://github.com/soumith/ganhacks)所示的最佳实践守法。即，我们将“构建体不同的小批次真假”的图像，并且还调整G公司的目标函数最大化
\（的logD（G（Z））\）。培训分成两个主要部分。第1部分更新鉴别和第2部分更新生成。

**第1部分 - 培养的鉴别**

回想一下，训练鉴别的目标是最大化的正确分类给定的输入为实或伪造的可能性。在古德费洛方面，我们希望“通过提升其随机梯度更新鉴别”。实际上，我们希望最大化
\（日志（d（X））+日志（1-d（G（Z）））\）。由于从ganhacks单独的小批量的建议，我们将分两步计算此。首先，我们将构造一个批次实际样品的从训练集，向前穿过
\（d \），计算出损耗（ \（日志（d（X））\）），然后计算在后向通的梯度。其次，我们将构造一个批次与电流发生器假样本，直传这批通过 \（d
\），计算出损耗（ \（日志（1-d（G（Z ）））\））和 _积累_
与向后通的梯度。现在，无论从所有实时和全假批次积累的梯度，我们称之为鉴别的优化的步骤。

**第2部分 - 培养发电机**

正如原文件中指出，我们希望通过最小化
\训练发生器（日志（1-d（G（Z）））\）在努力产生更好假货。如所提到的，这是通过古德费洛显示出不能提供足够的梯度，在学习过程中尤其是早期。作为一个解决方法，我们会想最大限度
\（日志（d（G（Z）））\）。在代码中我们通过实现此目的：从第1部分输出的发生器，提供鉴别分类，计算使用真实的标签为G的损失 _GT_
，计算G公司的梯度在向后通，最后用优化器更新G公司的参数步。这似乎是违反直觉的使用真正的标签为GT标签的损失函数，但这允许我们使用
\（日志（X）\）HTG7]的BCELoss（而非[HTG8的一部分] \（日志（1-X）\）HTG9]部分），这正是我们想要的东西。

最后，我们会做一些统计报告，并在每一个时代的结束，我们将通过发电机把我们fixed_noise一批视觉跟踪的G公司的培训进度。报告的训练统计数据：

  * **Loss_D** \- 鉴别器损失计算为对于所有实数和所有假批次损失（总和 \（日志（d（X））+日志（d（G（Z）））\） ）。
  * **Loss_G** \- 发电机损失计算为 \（日志（d（G（Z）））\）
  * 鉴别器用于所有实际批次的平均输出（跨批） - **d（x）的** 。这应该开始接近1，则理论上收敛到0.5当G变得更好。想想这是为什么。
  * **d（G（Z））** \- 平均鉴别器输出的所有假批次。第一个数字是d被更新之前，第二个数字是d被更新之后。这些数字应该开始接近0和收敛到0.5为G变得更好。想想这是为什么。

**注：** 此步骤可能需要一段时间，这取决于你运行了多少时代，如果你删除从数据集的一些数据。

    
    
    # Training Loop
    
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
    
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
    
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
    
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
    
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    
            iters += 1
    

Out:

    
    
    Starting Training Loop...
    [0/5][0/1583]   Loss_D: 2.0937  Loss_G: 5.2059  D(x): 0.5704    D(G(z)): 0.6680 / 0.0090
    [0/5][50/1583]  Loss_D: 0.3774  Loss_G: 13.1007 D(x): 0.9287    D(G(z)): 0.1399 / 0.0000
    [0/5][100/1583] Loss_D: 0.3890  Loss_G: 7.3600  D(x): 0.9515    D(G(z)): 0.2013 / 0.0016
    [0/5][150/1583] Loss_D: 0.8623  Loss_G: 4.8858  D(x): 0.6280    D(G(z)): 0.0546 / 0.0120
    [0/5][200/1583] Loss_D: 0.2328  Loss_G: 4.0880  D(x): 0.8727    D(G(z)): 0.0468 / 0.0342
    [0/5][250/1583] Loss_D: 0.5606  Loss_G: 6.3940  D(x): 0.8928    D(G(z)): 0.2846 / 0.0033
    [0/5][300/1583] Loss_D: 0.9473  Loss_G: 2.2100  D(x): 0.5401    D(G(z)): 0.0405 / 0.2226
    [0/5][350/1583] Loss_D: 0.5938  Loss_G: 2.3492  D(x): 0.6671    D(G(z)): 0.0787 / 0.1434
    [0/5][400/1583] Loss_D: 0.6209  Loss_G: 4.6997  D(x): 0.6428    D(G(z)): 0.0168 / 0.0245
    [0/5][450/1583] Loss_D: 0.2974  Loss_G: 4.0321  D(x): 0.8766    D(G(z)): 0.1159 / 0.0362
    [0/5][500/1583] Loss_D: 0.6701  Loss_G: 4.4486  D(x): 0.6652    D(G(z)): 0.0455 / 0.0287
    [0/5][550/1583] Loss_D: 0.4637  Loss_G: 5.2266  D(x): 0.8923    D(G(z)): 0.2620 / 0.0092
    [0/5][600/1583] Loss_D: 0.5639  Loss_G: 4.7983  D(x): 0.9016    D(G(z)): 0.3207 / 0.0173
    [0/5][650/1583] Loss_D: 0.7982  Loss_G: 5.0614  D(x): 0.5701    D(G(z)): 0.0204 / 0.0218
    [0/5][700/1583] Loss_D: 0.4445  Loss_G: 4.9462  D(x): 0.7558    D(G(z)): 0.0659 / 0.0158
    [0/5][750/1583] Loss_D: 0.5148  Loss_G: 3.5789  D(x): 0.7042    D(G(z)): 0.0432 / 0.0453
    [0/5][800/1583] Loss_D: 0.4863  Loss_G: 4.6765  D(x): 0.7542    D(G(z)): 0.0759 / 0.0231
    [0/5][850/1583] Loss_D: 0.3902  Loss_G: 5.8273  D(x): 0.9055    D(G(z)): 0.2264 / 0.0054
    [0/5][900/1583] Loss_D: 0.2873  Loss_G: 4.9891  D(x): 0.9196    D(G(z)): 0.1646 / 0.0127
    [0/5][950/1583] Loss_D: 0.3514  Loss_G: 5.7773  D(x): 0.8035    D(G(z)): 0.0290 / 0.0187
    [0/5][1000/1583]        Loss_D: 0.2073  Loss_G: 4.6480  D(x): 0.8781    D(G(z)): 0.0526 / 0.0179
    [0/5][1050/1583]        Loss_D: 0.3943  Loss_G: 3.9658  D(x): 0.8101    D(G(z)): 0.1151 / 0.0375
    [0/5][1100/1583]        Loss_D: 0.4837  Loss_G: 7.8827  D(x): 0.9326    D(G(z)): 0.2947 / 0.0007
    [0/5][1150/1583]        Loss_D: 0.8206  Loss_G: 5.7468  D(x): 0.7890    D(G(z)): 0.3709 / 0.0070
    [0/5][1200/1583]        Loss_D: 0.3523  Loss_G: 5.2779  D(x): 0.9274    D(G(z)): 0.1794 / 0.0170
    [0/5][1250/1583]        Loss_D: 0.4778  Loss_G: 3.7886  D(x): 0.8180    D(G(z)): 0.1853 / 0.0392
    [0/5][1300/1583]        Loss_D: 0.6191  Loss_G: 4.5570  D(x): 0.6579    D(G(z)): 0.0228 / 0.0329
    [0/5][1350/1583]        Loss_D: 0.9187  Loss_G: 2.3565  D(x): 0.5046    D(G(z)): 0.0160 / 0.1633
    [0/5][1400/1583]        Loss_D: 1.3850  Loss_G: 1.4330  D(x): 0.3892    D(G(z)): 0.0022 / 0.3387
    [0/5][1450/1583]        Loss_D: 1.1444  Loss_G: 1.4010  D(x): 0.4826    D(G(z)): 0.0790 / 0.3273
    [0/5][1500/1583]        Loss_D: 0.6209  Loss_G: 2.3856  D(x): 0.6477    D(G(z)): 0.0598 / 0.1366
    [0/5][1550/1583]        Loss_D: 0.3691  Loss_G: 4.1789  D(x): 0.8185    D(G(z)): 0.1073 / 0.0289
    [1/5][0/1583]   Loss_D: 1.0041  Loss_G: 6.3416  D(x): 0.9488    D(G(z)): 0.5145 / 0.0038
    [1/5][50/1583]  Loss_D: 0.3362  Loss_G: 5.1711  D(x): 0.9164    D(G(z)): 0.1905 / 0.0098
    [1/5][100/1583] Loss_D: 0.4752  Loss_G: 4.7347  D(x): 0.9064    D(G(z)): 0.2696 / 0.0158
    [1/5][150/1583] Loss_D: 0.3594  Loss_G: 4.2543  D(x): 0.8233    D(G(z)): 0.0889 / 0.0261
    [1/5][200/1583] Loss_D: 0.3224  Loss_G: 4.1060  D(x): 0.9342    D(G(z)): 0.1887 / 0.0328
    [1/5][250/1583] Loss_D: 0.3484  Loss_G: 4.1485  D(x): 0.9282    D(G(z)): 0.2083 / 0.0263
    [1/5][300/1583] Loss_D: 0.6082  Loss_G: 4.0181  D(x): 0.8497    D(G(z)): 0.3036 / 0.0301
    [1/5][350/1583] Loss_D: 0.3780  Loss_G: 3.8947  D(x): 0.8648    D(G(z)): 0.1663 / 0.0354
    [1/5][400/1583] Loss_D: 0.5670  Loss_G: 4.1670  D(x): 0.8218    D(G(z)): 0.2409 / 0.0301
    [1/5][450/1583] Loss_D: 0.5585  Loss_G: 3.1787  D(x): 0.7655    D(G(z)): 0.2057 / 0.0637
    [1/5][500/1583] Loss_D: 0.7137  Loss_G: 4.9132  D(x): 0.8824    D(G(z)): 0.3703 / 0.0148
    [1/5][550/1583] Loss_D: 0.4914  Loss_G: 5.2257  D(x): 0.9024    D(G(z)): 0.2840 / 0.0093
    [1/5][600/1583] Loss_D: 0.5191  Loss_G: 4.3694  D(x): 0.8699    D(G(z)): 0.2514 / 0.0219
    [1/5][650/1583] Loss_D: 0.5218  Loss_G: 3.0204  D(x): 0.8033    D(G(z)): 0.2015 / 0.0813
    [1/5][700/1583] Loss_D: 0.4707  Loss_G: 3.7884  D(x): 0.7416    D(G(z)): 0.0953 / 0.0498
    [1/5][750/1583] Loss_D: 0.4335  Loss_G: 3.2868  D(x): 0.7429    D(G(z)): 0.0884 / 0.0579
    [1/5][800/1583] Loss_D: 0.3846  Loss_G: 4.6926  D(x): 0.9407    D(G(z)): 0.2499 / 0.0160
    [1/5][850/1583] Loss_D: 0.5482  Loss_G: 3.6550  D(x): 0.7687    D(G(z)): 0.1835 / 0.0465
    [1/5][900/1583] Loss_D: 0.3070  Loss_G: 3.3886  D(x): 0.8349    D(G(z)): 0.0808 / 0.0542
    [1/5][950/1583] Loss_D: 0.5366  Loss_G: 4.5934  D(x): 0.9043    D(G(z)): 0.3098 / 0.0156
    [1/5][1000/1583]        Loss_D: 0.7676  Loss_G: 6.3473  D(x): 0.9307    D(G(z)): 0.4354 / 0.0033
    [1/5][1050/1583]        Loss_D: 0.2988  Loss_G: 2.8881  D(x): 0.8340    D(G(z)): 0.0837 / 0.0806
    [1/5][1100/1583]        Loss_D: 0.2307  Loss_G: 4.0665  D(x): 0.8507    D(G(z)): 0.0497 / 0.0297
    [1/5][1150/1583]        Loss_D: 0.4752  Loss_G: 3.3592  D(x): 0.7987    D(G(z)): 0.1827 / 0.0527
    [1/5][1200/1583]        Loss_D: 0.4123  Loss_G: 2.8147  D(x): 0.8577    D(G(z)): 0.1978 / 0.0855
    [1/5][1250/1583]        Loss_D: 0.6260  Loss_G: 4.0730  D(x): 0.8506    D(G(z)): 0.3111 / 0.0348
    [1/5][1300/1583]        Loss_D: 1.1704  Loss_G: 0.9039  D(x): 0.3939    D(G(z)): 0.0124 / 0.4852
    [1/5][1350/1583]        Loss_D: 0.7011  Loss_G: 2.8476  D(x): 0.5769    D(G(z)): 0.0256 / 0.1121
    [1/5][1400/1583]        Loss_D: 0.4104  Loss_G: 3.1058  D(x): 0.8774    D(G(z)): 0.2140 / 0.0639
    [1/5][1450/1583]        Loss_D: 0.6811  Loss_G: 4.2002  D(x): 0.8413    D(G(z)): 0.3494 / 0.0231
    [1/5][1500/1583]        Loss_D: 1.1317  Loss_G: 4.9345  D(x): 0.9371    D(G(z)): 0.5929 / 0.0142
    [1/5][1550/1583]        Loss_D: 0.4742  Loss_G: 3.6869  D(x): 0.8981    D(G(z)): 0.2814 / 0.0334
    [2/5][0/1583]   Loss_D: 0.7098  Loss_G: 2.2753  D(x): 0.7126    D(G(z)): 0.2353 / 0.1409
    [2/5][50/1583]  Loss_D: 0.8551  Loss_G: 4.0366  D(x): 0.9233    D(G(z)): 0.4786 / 0.0293
    [2/5][100/1583] Loss_D: 1.3078  Loss_G: 5.5286  D(x): 0.9644    D(G(z)): 0.6616 / 0.0087
    [2/5][150/1583] Loss_D: 0.5860  Loss_G: 3.0621  D(x): 0.8354    D(G(z)): 0.2879 / 0.0660
    [2/5][200/1583] Loss_D: 0.7063  Loss_G: 4.4227  D(x): 0.9211    D(G(z)): 0.4102 / 0.0214
    [2/5][250/1583] Loss_D: 0.7483  Loss_G: 4.3158  D(x): 0.9114    D(G(z)): 0.4218 / 0.0235
    [2/5][300/1583] Loss_D: 0.3818  Loss_G: 2.6245  D(x): 0.8214    D(G(z)): 0.1382 / 0.0954
    [2/5][350/1583] Loss_D: 1.0843  Loss_G: 5.0712  D(x): 0.9312    D(G(z)): 0.5778 / 0.0114
    [2/5][400/1583] Loss_D: 0.4509  Loss_G: 2.8962  D(x): 0.8141    D(G(z)): 0.1853 / 0.0809
    [2/5][450/1583] Loss_D: 1.6330  Loss_G: 0.9981  D(x): 0.2956    D(G(z)): 0.0459 / 0.4390
    [2/5][500/1583] Loss_D: 0.6487  Loss_G: 2.1938  D(x): 0.7994    D(G(z)): 0.3067 / 0.1466
    [2/5][550/1583] Loss_D: 0.9323  Loss_G: 0.9386  D(x): 0.5224    D(G(z)): 0.1030 / 0.4615
    [2/5][600/1583] Loss_D: 0.5440  Loss_G: 2.1702  D(x): 0.7386    D(G(z)): 0.1785 / 0.1451
    [2/5][650/1583] Loss_D: 1.0955  Loss_G: 4.1925  D(x): 0.8748    D(G(z)): 0.5495 / 0.0243
    [2/5][700/1583] Loss_D: 0.9323  Loss_G: 1.9101  D(x): 0.5384    D(G(z)): 0.1423 / 0.2040
    [2/5][750/1583] Loss_D: 0.5053  Loss_G: 3.0426  D(x): 0.8162    D(G(z)): 0.2322 / 0.0672
    [2/5][800/1583] Loss_D: 0.6751  Loss_G: 3.3158  D(x): 0.9154    D(G(z)): 0.3967 / 0.0506
    [2/5][850/1583] Loss_D: 0.6562  Loss_G: 3.2938  D(x): 0.8295    D(G(z)): 0.3324 / 0.0515
    [2/5][900/1583] Loss_D: 0.7118  Loss_G: 1.2240  D(x): 0.6193    D(G(z)): 0.1380 / 0.3455
    [2/5][950/1583] Loss_D: 0.8978  Loss_G: 1.6854  D(x): 0.5290    D(G(z)): 0.1213 / 0.2381
    [2/5][1000/1583]        Loss_D: 1.7309  Loss_G: 0.4199  D(x): 0.2345    D(G(z)): 0.0295 / 0.6955
    [2/5][1050/1583]        Loss_D: 1.0172  Loss_G: 2.5191  D(x): 0.7005    D(G(z)): 0.4067 / 0.1074
    [2/5][1100/1583]        Loss_D: 0.7516  Loss_G: 4.3600  D(x): 0.9211    D(G(z)): 0.4427 / 0.0188
    [2/5][1150/1583]        Loss_D: 1.1362  Loss_G: 4.1261  D(x): 0.9477    D(G(z)): 0.5982 / 0.0235
    [2/5][1200/1583]        Loss_D: 0.4525  Loss_G: 2.9000  D(x): 0.7585    D(G(z)): 0.1208 / 0.0792
    [2/5][1250/1583]        Loss_D: 0.6209  Loss_G: 2.6601  D(x): 0.6727    D(G(z)): 0.1333 / 0.0993
    [2/5][1300/1583]        Loss_D: 0.6188  Loss_G: 1.8989  D(x): 0.6197    D(G(z)): 0.0591 / 0.1911
    [2/5][1350/1583]        Loss_D: 0.5986  Loss_G: 2.2171  D(x): 0.7147    D(G(z)): 0.1789 / 0.1359
    [2/5][1400/1583]        Loss_D: 0.6236  Loss_G: 1.5753  D(x): 0.6225    D(G(z)): 0.0892 / 0.2549
    [2/5][1450/1583]        Loss_D: 1.4575  Loss_G: 4.5445  D(x): 0.9019    D(G(z)): 0.6660 / 0.0170
    [2/5][1500/1583]        Loss_D: 0.4806  Loss_G: 2.0873  D(x): 0.7311    D(G(z)): 0.1014 / 0.1669
    [2/5][1550/1583]        Loss_D: 0.6069  Loss_G: 2.4878  D(x): 0.7693    D(G(z)): 0.2556 / 0.1059
    [3/5][0/1583]   Loss_D: 0.6953  Loss_G: 1.5334  D(x): 0.5927    D(G(z)): 0.0873 / 0.2576
    [3/5][50/1583]  Loss_D: 0.5561  Loss_G: 1.6132  D(x): 0.7008    D(G(z)): 0.1354 / 0.2534
    [3/5][100/1583] Loss_D: 0.4794  Loss_G: 2.3090  D(x): 0.7693    D(G(z)): 0.1588 / 0.1250
    [3/5][150/1583] Loss_D: 1.4472  Loss_G: 4.4442  D(x): 0.9591    D(G(z)): 0.6936 / 0.0197
    [3/5][200/1583] Loss_D: 0.8359  Loss_G: 3.2797  D(x): 0.8965    D(G(z)): 0.4565 / 0.0537
    [3/5][250/1583] Loss_D: 2.0792  Loss_G: 4.2226  D(x): 0.9092    D(G(z)): 0.7681 / 0.0260
    [3/5][300/1583] Loss_D: 0.6438  Loss_G: 3.1580  D(x): 0.9164    D(G(z)): 0.3874 / 0.0598
    [3/5][350/1583] Loss_D: 1.7056  Loss_G: 0.8386  D(x): 0.2668    D(G(z)): 0.0734 / 0.5220
    [3/5][400/1583] Loss_D: 0.6288  Loss_G: 2.1909  D(x): 0.7401    D(G(z)): 0.2322 / 0.1413
    [3/5][450/1583] Loss_D: 0.5742  Loss_G: 1.9729  D(x): 0.7162    D(G(z)): 0.1722 / 0.1700
    [3/5][500/1583] Loss_D: 0.6798  Loss_G: 3.1593  D(x): 0.8591    D(G(z)): 0.3698 / 0.0543
    [3/5][550/1583] Loss_D: 0.7612  Loss_G: 1.2536  D(x): 0.5592    D(G(z)): 0.0940 / 0.3256
    [3/5][600/1583] Loss_D: 1.0874  Loss_G: 0.9601  D(x): 0.4155    D(G(z)): 0.0562 / 0.4391
    [3/5][650/1583] Loss_D: 0.7018  Loss_G: 2.5142  D(x): 0.8042    D(G(z)): 0.3334 / 0.1051
    [3/5][700/1583] Loss_D: 0.5612  Loss_G: 2.1963  D(x): 0.7554    D(G(z)): 0.2125 / 0.1376
    [3/5][750/1583] Loss_D: 0.7318  Loss_G: 1.6377  D(x): 0.6495    D(G(z)): 0.1979 / 0.2296
    [3/5][800/1583] Loss_D: 0.5621  Loss_G: 1.8894  D(x): 0.6796    D(G(z)): 0.1187 / 0.1907
    [3/5][850/1583] Loss_D: 0.6477  Loss_G: 2.5308  D(x): 0.7984    D(G(z)): 0.2913 / 0.1005
    [3/5][900/1583] Loss_D: 0.7904  Loss_G: 1.6153  D(x): 0.5864    D(G(z)): 0.1544 / 0.2314
    [3/5][950/1583] Loss_D: 0.5315  Loss_G: 2.1866  D(x): 0.7990    D(G(z)): 0.2288 / 0.1405
    [3/5][1000/1583]        Loss_D: 0.8392  Loss_G: 3.7965  D(x): 0.8504    D(G(z)): 0.4431 / 0.0332
    [3/5][1050/1583]        Loss_D: 0.8082  Loss_G: 3.7510  D(x): 0.8679    D(G(z)): 0.4384 / 0.0341
    [3/5][1100/1583]        Loss_D: 0.5648  Loss_G: 1.9762  D(x): 0.7244    D(G(z)): 0.1718 / 0.1608
    [3/5][1150/1583]        Loss_D: 0.6545  Loss_G: 3.1910  D(x): 0.8204    D(G(z)): 0.3298 / 0.0534
    [3/5][1200/1583]        Loss_D: 0.6370  Loss_G: 2.0567  D(x): 0.7406    D(G(z)): 0.2551 / 0.1560
    [3/5][1250/1583]        Loss_D: 0.6561  Loss_G: 1.8144  D(x): 0.6885    D(G(z)): 0.2035 / 0.1921
    [3/5][1300/1583]        Loss_D: 0.6860  Loss_G: 1.8726  D(x): 0.5865    D(G(z)): 0.0620 / 0.1953
    [3/5][1350/1583]        Loss_D: 0.5618  Loss_G: 1.8079  D(x): 0.7159    D(G(z)): 0.1685 / 0.1976
    [3/5][1400/1583]        Loss_D: 0.6877  Loss_G: 2.5243  D(x): 0.7913    D(G(z)): 0.3214 / 0.1004
    [3/5][1450/1583]        Loss_D: 0.6534  Loss_G: 2.6937  D(x): 0.7997    D(G(z)): 0.3071 / 0.0848
    [3/5][1500/1583]        Loss_D: 0.5443  Loss_G: 2.1160  D(x): 0.7078    D(G(z)): 0.1242 / 0.1515
    [3/5][1550/1583]        Loss_D: 1.5968  Loss_G: 4.8972  D(x): 0.9627    D(G(z)): 0.7338 / 0.0110
    [4/5][0/1583]   Loss_D: 0.7820  Loss_G: 1.8219  D(x): 0.5272    D(G(z)): 0.0467 / 0.2010
    [4/5][50/1583]  Loss_D: 0.6637  Loss_G: 1.9136  D(x): 0.6712    D(G(z)): 0.1865 / 0.1876
    [4/5][100/1583] Loss_D: 1.0259  Loss_G: 1.2513  D(x): 0.4374    D(G(z)): 0.0684 / 0.3257
    [4/5][150/1583] Loss_D: 0.5099  Loss_G: 2.4926  D(x): 0.7915    D(G(z)): 0.2024 / 0.1111
    [4/5][200/1583] Loss_D: 0.7905  Loss_G: 3.8833  D(x): 0.9060    D(G(z)): 0.4502 / 0.0309
    [4/5][250/1583] Loss_D: 0.8218  Loss_G: 1.3731  D(x): 0.5398    D(G(z)): 0.0961 / 0.3197
    [4/5][300/1583] Loss_D: 0.7159  Loss_G: 2.9385  D(x): 0.7769    D(G(z)): 0.3270 / 0.0678
    [4/5][350/1583] Loss_D: 0.5711  Loss_G: 3.2981  D(x): 0.8730    D(G(z)): 0.3232 / 0.0506
    [4/5][400/1583] Loss_D: 0.9274  Loss_G: 1.3243  D(x): 0.4666    D(G(z)): 0.0547 / 0.3089
    [4/5][450/1583] Loss_D: 1.9290  Loss_G: 5.5781  D(x): 0.9685    D(G(z)): 0.8031 / 0.0063
    [4/5][500/1583] Loss_D: 0.7317  Loss_G: 2.9507  D(x): 0.7779    D(G(z)): 0.3349 / 0.0688
    [4/5][550/1583] Loss_D: 0.3878  Loss_G: 3.0483  D(x): 0.8716    D(G(z)): 0.2052 / 0.0606
    [4/5][600/1583] Loss_D: 0.5016  Loss_G: 2.1415  D(x): 0.7794    D(G(z)): 0.1992 / 0.1496
    [4/5][650/1583] Loss_D: 0.8692  Loss_G: 4.0726  D(x): 0.9369    D(G(z)): 0.5011 / 0.0239
    [4/5][700/1583] Loss_D: 1.0189  Loss_G: 0.5405  D(x): 0.4590    D(G(z)): 0.0792 / 0.6298
    [4/5][750/1583] Loss_D: 0.6823  Loss_G: 1.8271  D(x): 0.5918    D(G(z)): 0.0876 / 0.2046
    [4/5][800/1583] Loss_D: 0.8343  Loss_G: 3.9417  D(x): 0.8795    D(G(z)): 0.4572 / 0.0283
    [4/5][850/1583] Loss_D: 0.5352  Loss_G: 2.8730  D(x): 0.8354    D(G(z)): 0.2612 / 0.0770
    [4/5][900/1583] Loss_D: 0.5948  Loss_G: 1.9490  D(x): 0.6961    D(G(z)): 0.1582 / 0.1789
    [4/5][950/1583] Loss_D: 0.6370  Loss_G: 3.2704  D(x): 0.8925    D(G(z)): 0.3600 / 0.0523
    [4/5][1000/1583]        Loss_D: 0.7010  Loss_G: 1.9136  D(x): 0.6741    D(G(z)): 0.2126 / 0.1832
    [4/5][1050/1583]        Loss_D: 0.7043  Loss_G: 1.5664  D(x): 0.6225    D(G(z)): 0.1439 / 0.2530
    [4/5][1100/1583]        Loss_D: 0.4952  Loss_G: 2.1362  D(x): 0.7396    D(G(z)): 0.1442 / 0.1535
    [4/5][1150/1583]        Loss_D: 1.1702  Loss_G: 0.9483  D(x): 0.3849    D(G(z)): 0.0445 / 0.4278
    [4/5][1200/1583]        Loss_D: 0.6114  Loss_G: 1.6389  D(x): 0.6706    D(G(z)): 0.1427 / 0.2354
    [4/5][1250/1583]        Loss_D: 0.6020  Loss_G: 1.9253  D(x): 0.7218    D(G(z)): 0.1923 / 0.1769
    [4/5][1300/1583]        Loss_D: 0.6117  Loss_G: 3.6101  D(x): 0.8724    D(G(z)): 0.3392 / 0.0371
    [4/5][1350/1583]        Loss_D: 0.8552  Loss_G: 4.2809  D(x): 0.9218    D(G(z)): 0.4932 / 0.0205
    [4/5][1400/1583]        Loss_D: 0.6170  Loss_G: 4.0999  D(x): 0.9353    D(G(z)): 0.3772 / 0.0246
    [4/5][1450/1583]        Loss_D: 0.5660  Loss_G: 2.2870  D(x): 0.6739    D(G(z)): 0.1064 / 0.1389
    [4/5][1500/1583]        Loss_D: 0.7235  Loss_G: 3.5680  D(x): 0.8678    D(G(z)): 0.3896 / 0.0403
    [4/5][1550/1583]        Loss_D: 0.8062  Loss_G: 3.8185  D(x): 0.9046    D(G(z)): 0.4511 / 0.0305
    

## 结果

最后，让我们看看我们是怎么做。在这里，我们将着眼于三个不同的结果。首先，我们将看到G公司的损失在训练中如何d和改变。其次，我们将可视化的fixed_noise批次每一个时代G公司的产量。第三，我们将着眼于一批真实数据的批量从G.假数据的旁边

**损耗与训练迭代**

下面是d &安培的曲线图; G公司的损失与训练迭代。

    
    
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    

![img/sphx_glr_dcgan_faces_tutorial_002.png](img/sphx_glr_dcgan_faces_tutorial_002.png)

**G公司的进展的可视化**

还记得我们的训练每一个时代后保存在发电机上fixed_noise批量输出。现在，我们可以想像G的训练进展与动画。按PLAY键开始播放动画。

    
    
    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    
    HTML(ani.to_jshtml())
    

![img/sphx_glr_dcgan_faces_tutorial_003.png](img/sphx_glr_dcgan_faces_tutorial_003.png)

**真实全景与假图片**

最后，让我们来看看一些真实的图像和假图像并排。

    
    
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    
    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
    

![img/sphx_glr_dcgan_faces_tutorial_004.png](img/sphx_glr_dcgan_faces_tutorial_004.png)

## 下一步是什么

我们已经达到了我们的旅程结束，但有几个地方，你可以从这里走。你可以：

  * 火车较长时间才能看到效果有多好得
  * 修改这个模型来采取不同的数据集，并有可能改变图像的大小和模型架构
  * 看看其他一些很酷的GAN项目[此处](https://github.com/nashory/gans-awesome-applications)
  * 创建生成[音乐甘斯](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

**脚本的总运行时间：** （28分钟13.763秒）

[`Download Python source code:
dcgan_faces_tutorial.py`](../_downloads/dc0e6f475c6735eb8d233374f8f462eb/dcgan_faces_tutorial.py)

[`Download Jupyter notebook:
dcgan_faces_tutorial.ipynb`](../_downloads/e9c8374ecc202120dc94db26bf08a00f/dcgan_faces_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](audio_preprocessing_tutorial.html "torchaudio Tutorial")
[![](../_static/images/chevron-right-orange.svg) Previous](fgsm_tutorial.html
"Adversarial Example Generation")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * DCGAN教程
    * 介绍
    * 剖成对抗性网络
      * 什么是甘？ 
      * 什么是DCGAN？ 
    * 输入
    * 数据
    * 实现
      * 重量初始化
      * 发生器
      * 鉴别
      * 损失函数和优化器
      * 培训
    * 结果
    * 下一步是什么

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



