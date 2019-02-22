# 空间转换网络 (Spatial Transformer Networks) 教程

> 译者：[@Twinkle](https://github.com/kemingzeng)

**原作者**: [Ghassen HAMROUNI](https://github.com/GHamrouni)

![http://pytorch.apachecn.org/cn/tutorials/_images/FSeq.png](img/01cb117c4c0d3fcaa29ac2f3c359529a.jpg)

在这篇教程中, 你会学到如何用名为空间转换网络 (spatial transformer networks) 的视觉注意力结构来加强你的网络. 你可以从这篇论文上看到更多关于空间转换网络 (spatial transformer networks)的知识: [DeepMind paper](https://arxiv.org/abs/1506.02025)

空间转换网络 (spatial transformer networks) 是对关注空间变换可区分性的一种推广 形式. 短空间转换网络 (STN for short) 允许一个神经网络学习如何在输入图像上表现出空 间变换, 以此来增强模型的几何不变性. 例如, 它可以裁剪一个感兴趣的区域, 缩放和修正图像的方向. 由于卷积神经网络对旋转、缩放 和更普遍仿射变换并不具有不变性, 因此它相对来说是一种有用的结构.

STN (空间转换网络) 最好的一点是它能在非常小的改动之后, 被简单地嵌入到任何已存在的卷积神 经网络中.

```py
# 许可协议: BSD
# 作者: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

plt.ion()   # 交互模式

```

## 读数据

在这里我们用经典的 MNIST 数据集做试验. 使用一个被空间转换网络增强的标准卷积神经 网络.

```py
use_cuda = torch.cuda.is_available()

# 训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)

```

## 描述空间转换网络 (spatial transformer networks)

空间转换网络 (spatial transformer networks) 归纳为三个主要的部件 :

*   本地网络 (The localization network) 是一个常规CNN, 它可以回归转换参数. 这种空间转换不是简单地从数据集显式学习到的, 而是自动地学习以增强全局准确率.
*   网格生成器 (The grid generator) 在输入图像中生成对应于来自输出图像的每个像 素的坐标网格.
*   采样器 (The sampler) 将转换的参数应用于输入图像.

![http://pytorch.apachecn.org/cn/tutorials/_images/stn-arch.png](img/3864cede91518c948be774422c076cc0.jpg)

注解：

我们需要包含 affine_grid 和 grid_sample 模块的 PyTorch 最新版本.

```py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # 空间转换本地网络 (Spatial transformer localization-network)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # 3 * 2 仿射矩阵 (affine matrix) 的回归器
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 用身份转换 (identity transformation) 初始化权重 (weights) / 偏置 (bias)
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # 空间转换网络的前向函数 (Spatial transformer network forward function)
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # 转换输入
        x = self.stn(x)

        # 执行常规的正向传递
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if use_cuda:
    model.cuda()

```

## 训练模型

现在, 让我们用 SGD 算法来训练模型. 这个网络用监督学习的方式学习分类任务. 同时, 这个模型以端到端的方式自动地学习空间转换网络 (STN) .

```py
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
#
# 一个简单的测试程序来测量空间转换网络 (STN) 在 MNIST 上的表现.
#

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # 累加批loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # 得到最大对数几率 (log-probability) 的索引.
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

```

## 可视化空间转换网络 (STN) 的结果

现在, 我们要检查学到的视觉注意力机制的结果.

我们定义一个小的辅助函数, 以在训练过程中可视化转换过程.

```py
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# 我们想要在训练之后可视化空间转换层 (spatial transformers layer) 的输出, 我们
# 用 STN 可视化一批输入图像和相对于的转换后的数据.

def visualize_stn():
    # 得到一批输入数据
    data, _ = next(iter(test_loader))
    data = Variable(data, volatile=True)

    if use_cuda:
        data = data.cuda()

    input_tensor = data.cpu().data
    transformed_input_tensor = model.stn(data).cpu().data

    in_grid = convert_image_np(
        torchvision.utils.make_grid(input_tensor))

    out_grid = convert_image_np(
        torchvision.utils.make_grid(transformed_input_tensor))

    # 并行地 (side-by-side) 画出结果
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(in_grid)
    axarr[0].set_title('Dataset Images')

    axarr[1].imshow(out_grid)
    axarr[1].set_title('Transformed Images')

for epoch in range(1, 20 + 1):
    train(epoch)
    test()

# 在一些输入批次中可视化空间转换网络 (STN) 的转换
visualize_stn()

plt.ioff()
plt.show()

```
