# 空间变压器网络教程

**作者** ：[ Ghassen HAMROUNI ](https://github.com/GHamrouni)

![img/FSeq.png](img/FSeq.png)

在本教程中，您将学习如何使用称为空间变压器网络视觉注意机制来增强你的网络。你可以阅读更多有关在[
DeepMind纸空间变压器网](https://arxiv.org/abs/1506.02025)

空间变压器网络是微注意泛化到任何空间变换。空间变换器网络（STN的简称）允许一个神经网络学习如何以提高模型的几何不变性的输入图像上执行空间变换。例如，它可以裁剪的兴趣，规模区域和纠正图像的方向。它可以是一个有用的机制，因为细胞神经网络的不不变的旋转和缩放，更全面的仿射变换。

其中一件关于STN的最好的事情就是简单地把它用很少的修改插入到任何现有CNN的能力。

    
    
    # License: BSD
    # Author: Ghassen Hamrouni
    
    from __future__ import print_function
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.ion()   # interactive mode
    

## 加载数据

在这篇文章中，我们尝试用经典MNIST数据集。使用具有空间变换网络增加一个标准的卷积网络。

    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=64, shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=64, shuffle=True, num_workers=4)
    

日期：

    
    
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./MNIST/raw/train-images-idx3-ubyte.gz
    Extracting ./MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./MNIST/raw/train-labels-idx1-ubyte.gz
    Extracting ./MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./MNIST/raw/t10k-images-idx3-ubyte.gz
    Extracting ./MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST/raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST/raw/t10k-labels-idx1-ubyte.gz
    Extracting ./MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST/raw
    Processing...
    Done!
    

## 描绘空间变换器网络

空间变压器网络可以归结为三个主要组成部分：

  * 本地化网络是一个普通的CNN其倒退的转换参数。转型是永远不会从这个数据集显式地了解到，而非网络自动学习的空间变换，增强全球精度。
  * 网格生成器生成对应于来自所述输出图像的每个像素在输入图像中的坐标的网格。
  * 采样器，使用变换的参数，并将其应用于输入图像。

![img/stn-arch.png](img/stn-arch.png)

Note

我们需要最新版本PyTorch的包含affine_grid和grid_sample模块。

    
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
    
            # Spatial transformer localization-network
            self.localization = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
    
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 3 * 3, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )
    
            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
        # Spatial transformer network forward function
        def stn(self, x):
            xs = self.localization(x)
            xs = xs.view(-1, 10 * 3 * 3)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
    
            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)
    
            return x
    
        def forward(self, x):
            # transform the input
            x = self.stn(x)
    
            # Perform the usual forward pass
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    
    model = Net().to(device)
    

## 培养模式

现在，让我们使用SGD算法训练模型。该网络学习在监督方式的分类任务。在同一时间模型在一个终端到高端时尚自动学习STN。

    
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
    
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    #
    # A simple test procedure to measure STN the performances on MNIST.
    #
    
    
    def test():
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
    
                # sum up batch loss
                test_loss += F.nll_loss(output, target, size_average=False).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
    
            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(test_loss, correct, len(test_loader.dataset),
                          100. * correct / len(test_loader.dataset)))
    

## 可视化STN结果

现在，我们要来视察我们了解到视觉注意机制的结果。

我们以可视化，同时训练转变定义一个小助手功能。

    
    
    def convert_image_np(inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp
    
    # We want to visualize the output of the spatial transformers layer
    # after the training, we visualize a batch of input images and
    # the corresponding transformed batch using STN.
    
    
    def visualize_stn():
        with torch.no_grad():
            # Get a batch of training data
            data = next(iter(test_loader))[0].to(device)
    
            input_tensor = data.cpu()
            transformed_input_tensor = model.stn(data).cpu()
    
            in_grid = convert_image_np(
                torchvision.utils.make_grid(input_tensor))
    
            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor))
    
            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')
    
            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')
    
    for epoch in range(1, 20 + 1):
        train(epoch)
        test()
    
    # Visualize the STN transformation on some input batch
    visualize_stn()
    
    plt.ioff()
    plt.show()
    

![img/sphx_glr_spatial_transformer_tutorial_001.png](img/sphx_glr_spatial_transformer_tutorial_001.png)

Out:

    
    
    Train Epoch: 1 [0/60000 (0%)]   Loss: 2.290877
    Train Epoch: 1 [32000/60000 (53%)]      Loss: 0.910913
    
    Test set: Average loss: 0.2449, Accuracy: 9312/10000 (93%)
    
    Train Epoch: 2 [0/60000 (0%)]   Loss: 0.489534
    Train Epoch: 2 [32000/60000 (53%)]      Loss: 0.296471
    
    Test set: Average loss: 0.1443, Accuracy: 9563/10000 (96%)
    
    Train Epoch: 3 [0/60000 (0%)]   Loss: 0.410248
    Train Epoch: 3 [32000/60000 (53%)]      Loss: 0.355454
    
    Test set: Average loss: 0.1019, Accuracy: 9687/10000 (97%)
    
    Train Epoch: 4 [0/60000 (0%)]   Loss: 0.217658
    Train Epoch: 4 [32000/60000 (53%)]      Loss: 0.185522
    
    Test set: Average loss: 0.0818, Accuracy: 9751/10000 (98%)
    
    Train Epoch: 5 [0/60000 (0%)]   Loss: 0.471464
    Train Epoch: 5 [32000/60000 (53%)]      Loss: 0.591574
    
    Test set: Average loss: 0.0770, Accuracy: 9760/10000 (98%)
    
    Train Epoch: 6 [0/60000 (0%)]   Loss: 0.119462
    Train Epoch: 6 [32000/60000 (53%)]      Loss: 0.093015
    
    Test set: Average loss: 0.0817, Accuracy: 9744/10000 (97%)
    
    Train Epoch: 7 [0/60000 (0%)]   Loss: 0.074523
    Train Epoch: 7 [32000/60000 (53%)]      Loss: 0.414406
    
    Test set: Average loss: 0.0944, Accuracy: 9714/10000 (97%)
    
    Train Epoch: 8 [0/60000 (0%)]   Loss: 0.100317
    Train Epoch: 8 [32000/60000 (53%)]      Loss: 0.114539
    
    Test set: Average loss: 0.1519, Accuracy: 9510/10000 (95%)
    
    Train Epoch: 9 [0/60000 (0%)]   Loss: 0.205053
    Train Epoch: 9 [32000/60000 (53%)]      Loss: 0.135724
    
    Test set: Average loss: 0.0892, Accuracy: 9749/10000 (97%)
    
    Train Epoch: 10 [0/60000 (0%)]  Loss: 0.213368
    Train Epoch: 10 [32000/60000 (53%)]     Loss: 0.208627
    
    Test set: Average loss: 0.0634, Accuracy: 9813/10000 (98%)
    
    Train Epoch: 11 [0/60000 (0%)]  Loss: 0.078725
    Train Epoch: 11 [32000/60000 (53%)]     Loss: 0.099131
    
    Test set: Average loss: 0.0580, Accuracy: 9834/10000 (98%)
    
    Train Epoch: 12 [0/60000 (0%)]  Loss: 0.133572
    Train Epoch: 12 [32000/60000 (53%)]     Loss: 0.213358
    
    Test set: Average loss: 0.0506, Accuracy: 9854/10000 (99%)
    
    Train Epoch: 13 [0/60000 (0%)]  Loss: 0.289802
    Train Epoch: 13 [32000/60000 (53%)]     Loss: 0.165571
    
    Test set: Average loss: 0.0542, Accuracy: 9842/10000 (98%)
    
    Train Epoch: 14 [0/60000 (0%)]  Loss: 0.219281
    Train Epoch: 14 [32000/60000 (53%)]     Loss: 0.284233
    
    Test set: Average loss: 0.0505, Accuracy: 9856/10000 (99%)
    
    Train Epoch: 15 [0/60000 (0%)]  Loss: 0.218599
    Train Epoch: 15 [32000/60000 (53%)]     Loss: 0.055698
    
    Test set: Average loss: 0.0507, Accuracy: 9848/10000 (98%)
    
    Train Epoch: 16 [0/60000 (0%)]  Loss: 0.048718
    Train Epoch: 16 [32000/60000 (53%)]     Loss: 0.093410
    
    Test set: Average loss: 0.0502, Accuracy: 9855/10000 (99%)
    
    Train Epoch: 17 [0/60000 (0%)]  Loss: 0.071185
    Train Epoch: 17 [32000/60000 (53%)]     Loss: 0.053381
    
    Test set: Average loss: 0.0587, Accuracy: 9829/10000 (98%)
    
    Train Epoch: 18 [0/60000 (0%)]  Loss: 0.127790
    Train Epoch: 18 [32000/60000 (53%)]     Loss: 0.169319
    
    Test set: Average loss: 0.0484, Accuracy: 9863/10000 (99%)
    
    Train Epoch: 19 [0/60000 (0%)]  Loss: 0.224094
    Train Epoch: 19 [32000/60000 (53%)]     Loss: 0.175750
    
    Test set: Average loss: 0.0628, Accuracy: 9817/10000 (98%)
    
    Train Epoch: 20 [0/60000 (0%)]  Loss: 0.251131
    Train Epoch: 20 [32000/60000 (53%)]     Loss: 0.024119
    
    Test set: Average loss: 0.0445, Accuracy: 9869/10000 (99%)
    

**脚本的总运行时间：** （1分钟44.448秒）

[`Download Python source code:
spatial_transformer_tutorial.py`](../_downloads/8aa31a122008b8db8bbe28365db9ea47/spatial_transformer_tutorial.py)

[`Download Jupyter notebook:
spatial_transformer_tutorial.ipynb`](../_downloads/b0786fd6ca28ee4ff3f2aa27080cdf18/spatial_transformer_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../advanced/neural_style_tutorial.html "Neural Transfer Using
PyTorch") [![](../_static/images/chevron-right-orange.svg)
Previous](../beginner/finetuning_torchvision_models_tutorial.html "Finetuning
Torchvision Models")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * 空间变压器网络教程
    * 加载数据
    * 各取空间变换器网络
    * 训练模型
    * 形象化STN结果

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



