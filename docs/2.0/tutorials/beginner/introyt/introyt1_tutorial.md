# PyTorch 入门

> 译者：[Fadegentle](https://github.com/Fadegentle)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/introyt/introyt1_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html>

请跟随下面的视频或在 [youtube](https://www.youtube.com/watch?v=IC0_FRiX-sw) 上观看。

<iframe allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" frameborder="0" height="315" src="https://www.youtube.com/embed/IC0_FRiX-sw" width="560"></iframe>

## PyTorch 张量
请从视频的 [03:50](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=230s) 开始跟随。

首先，我们要导入 pytorch。

```python
import torch
```

让我们看看张量的一些基本操作。首先，是创建张量的几种方法：

```python
z = torch.zeros(5, 3)
print(z)
print(z.dtype)
```

输出：
```shell
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
torch.float32
```

在上面，我们创建了一个填满零的 5x3 矩阵，并查询其数据类型，发现其中的零是 32 位浮点数，这是 PyTorch 的默认值。

如果您想要用整数代替呢？您可以覆盖默认值：

```python
i = torch.ones((5, 3), dtype=torch.int16)
print(i)
```


输出：
```shell
tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int16)
```

您可以看到，当我们改变默认值时，张量在打印时会有相应的报告。

常见的做法是随机初始化学习权重，通常使用特定的伪随机数生成器种子来确保结果的可重复性：

```python
torch.manual_seed(1729)
r1 = torch.rand(2, 2)
print('A random tensor:')
print(r1)

r2 = torch.rand(2, 2)
print('\nA different random tensor:')
print(r2) # new values

torch.manual_seed(1729)
r3 = torch.rand(2, 2)
print('\nShould match r1:')
print(r3) # repeats values of r1 because of re-seed
```

输出：
```shell
A random tensor:
tensor([[0.3126, 0.3791],
        [0.3087, 0.0736]])

A different random tensor:
tensor([[0.4216, 0.0691],
        [0.2332, 0.4047]])

Should match r1:
tensor([[0.3126, 0.3791],
        [0.3087, 0.0736]])
```

PyTorch 张量执行算术运算很直观。形状相似的张量可以进行加法、乘法等操作。标量运算则会分布在张量上：

```python
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # every element is multiplied by 2
print(twos)

threes = ones + twos       # addition allowed because shapes are similar
print(threes)              # tensors are added element-wise
print(threes.shape)        # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(3, 2)
# uncomment this line to get a runtime error
# r3 = r1 + r2
```

输出：
```shell
tensor([[1., 1., 1.],
        [1., 1., 1.]])
tensor([[2., 2., 2.],
        [2., 2., 2.]])
tensor([[3., 3., 3.],
        [3., 3., 3.]])
torch.Size([2, 3])
```

以下是一小部分可用的数学运算示例：

```python
r = (torch.rand(2, 2) - 0.5) * 2 # values between -1 and 1
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:
print('\nAbsolute value of r:')
print(torch.abs(r))

# ...as are trigonometric functions:
print('\nInverse sine of r:')
print(torch.asin(r))

# ...and linear algebra operations like determinant and singular value decomposition
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))

# ...and statistical and aggregate operations:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))
```


输出：
```shell
A random matrix, r:
tensor([[ 0.9956, -0.2232],
        [ 0.3858, -0.6593]])

Absolute value of r:
tensor([[0.9956, 0.2232],
        [0.3858, 0.6593]])

Inverse sine of r:
tensor([[ 1.4775, -0.2251],
        [ 0.3961, -0.7199]])

Determinant of r:
tensor(-0.5703)

Singular value decomposition of r:
torch.return_types.svd(
U=tensor([[-0.8353, -0.5497],
        [-0.5497,  0.8353]]),
S=tensor([1.1793, 0.4836]),
V=tensor([[-0.8851, -0.4654],
        [ 0.4654, -0.8851]]))

Average and standard deviation of r:
(tensor(0.7217), tensor(0.1247))

Maximum value of r:
tensor(0.9956)
```

有关 PyTorch 张量的更多信息，包括如何设置它们以在 GPU 上进行并行计算，我们将在另一个视频中进行更深入的讨论。

## PyTorch 模型
请从视频的 [10:00](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=600s) 开始跟随。

让我们来看看如何在 PyTorch 中表示模型。

```python
import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
```

![le-net-5 diagram](../../../img/mnist.png)
_图片: LeNet-5_

上面是 LeNet-5 的示意图，它是最早的卷积神经网络之一，也是深度学习爆发的推动因素之一。它被设计用于识别手写数字的小图像（MNIST 数据集），并正确分类图像中所代表的数字。

以下是它的简化版工作原理：

- C1 层是一个卷积层，它会扫描输入图像，寻找在训练过程中学习到的特征。它输出一个映射，显示了它在图像中看到的每个学习到特征的位置。这个“激活映射”在 S2 层中进行了降采样。

- C3 层是另一个卷积层，这次它扫描 C1 的激活映射，寻找特征的组合。它还输出一个描述这些特征组合空间位置的激活映射，这在 S4 层中进行了降采样。

- 最后，末端的 F5、F6 和 OUTPUT 全连接层构成一个分类器，它接收最终的激活映射，并将其分类表示为 10 个数字的十个类别之一。

我们如何用代码表示这个简单的神经网络呢？

```python
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

仔细查看这段代码，您应该能够在上面的图示中看到一些结构上的相似之处。

这展示了一个典型的 PyTorch 模型的结构：

- 它继承自 `torch.nn.Module` ，模块可以嵌套，实际上，甚至 `Conv2d` 和 `Linear` 层类都继承自 `torch.nn.Module`。

- 一个模型会有一个 `__init__()` 函数，用于实例化它的层，并加载它可能需要的任何数据（例如，NLP 模型可能会加载词汇表）。

- 一个模型会有一个 `forward()` 函数。这是实际的计算发生的地方：输入通过网络层和各种函数传递，生成输出。

- 除此之外，您可以像构建其他 Python 类一样构建模型类，添加任何属性和方法，以支持模型的计算。

现在我们来实例化这个对象，并通过它运行一个样本输入。


```python
net = LeNet()
print(net)                         # what does the object tell us about itself?

input = torch.rand(1, 1, 32, 32)   # stand-in for a 32x32 black & white image
print('\nImage batch shape:')
print(input.shape)

output = net(input)                # we don't call forward() directly
print('\nRaw output:')
print(output)
print(output.shape)
```

输出：
```shell
LeNet(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)

Image batch shape:
torch.Size([1, 1, 32, 32])

Raw output:
tensor([[ 0.0898,  0.0318,  0.1485,  0.0301, -0.0085, -0.1135, -0.0296,  0.0164,
          0.0039,  0.0616]], grad_fn=<AddmmBackward0>)
torch.Size([1, 10])
```

上述代码中发生了一些重要的事情：

首先，我们实例化了 `LeNet` 类，并打印了 `net` 对象。`torch.nn.Module` 的子类会报告它所创建的层及其形状和参数。如果您想要了解模型的处理过程，这可以提供一个方便的概述。

在此之下，我们创建了一个代表 32x32 图像且具有 1 个颜色通道的虚拟输入。通常，您会加载一个图像块并将其转换为这种形状的张量。

您可能已经注意到了张量中的额外维度 —— _批处理维度_。PyTorch 模型假定它们在 _批次_ 数据上进行操作，例如，批处理包含 16 个图像块的情况下，形状将为 (`16`, `1`, `32`, `32`)。由于我们只使用了一个图像，我们创建了一个形状为 (`1`, `1`, `32`, `32`) 的批次。

我们像调用函数一样调用该模型推断：`net(input)`。该调用的输出表示，模型对表示特定数字输入的置信度。（由于这个模型实例尚未学习任何内容，我们不应该在输出中看到任何信号。）观察 `output` 的形状，我们可以看到它也有一个批处理维度，其大小应始终与输入批处理维度相匹配。如果我们传入一个包含 16 个实例的输入批次，`output` 的形状将为 (`16`, `10`)。


## 数据集和数据加载器
请从视频的 [14:00](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=840s) 开始跟随。

接下来，我们将使用 TorchVision 中的一个可随时下载的开放访问数据集来演示，如何转换图像以供您的模型使用，以及如何使用 DataLoader 将数据批量提供给您的模型。
我们需要做的第一件事就是将输入的图像转换成 PyTorch 张量。

```python
#%matplotlib inline

import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
```

在这里，我们为输入指定了两种转换：

- `transforms.ToTensor()` 将 Pillow 加载的图像转换成 PyTorch 张量。
- `transforms.Normalize()` 调整张量值，使其平均值为零，标准差为 1.0。大多数激活函数在 x = 0 附近具有最强梯度，因此将数据集中在这里可以加快学习速度。传递给该变换的值是数据集中图像 rgb 值的均值（第一个元组）和标准差（第二个元组）。您可以通过运行以下几行代码来计算这些值：


```python
from torch.utils.data import ConcatDataset

transform = transforms.Compose([transforms.ToTensor()]) trainset = torchvision.datasets.CIFAR10(root=’./data’, train=True,
                                                                                                download=True, transform=transform)

#stack all train images together into a tensor of shape 
#(50000, 3, 32, 32)
x = torch.stack([sample[0] for sample in ConcatDataset([trainset])])

#get the mean of each channel
mean = torch.mean(x, dim=(0,2,3)) #tensor([0.4914, 0.4822, 0.4465])
std = torch.std(x, dim=(0,2,3)) #tensor([0.2470, 0.2435, 0.2616])

```

还有许多其他的变换可用，包括裁剪、居中、旋转和翻转。

接下来，我们将创建一个 CIFAR10 数据集的实例。这是一组 32x32 的彩色图像块，代表着 10 类对象：6 类动物（鸟、猫、鹿、狗、青蛙、马）和 4 类交通工具（飞机、汽车、船、卡车）：

```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
```

输出：
```shell
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz

  0%|          | 0/170498071 [00:00<?, ?it/s]
  0%|          | 458752/170498071 [00:00<00:37, 4549065.07it/s]
  4%|3         | 6094848/170498071 [00:00<00:04, 34641164.48it/s]
  9%|8         | 15269888/170498071 [00:00<00:02, 60391866.07it/s]
 15%|#4        | 24772608/170498071 [00:00<00:01, 73973649.69it/s]
 21%|##        | 35356672/170498071 [00:00<00:01, 85345426.42it/s]
 26%|##6       | 44597248/170498071 [00:00<00:01, 87704676.02it/s]
 33%|###2      | 55836672/170498071 [00:00<00:01, 95691278.35it/s]
 38%|###8      | 65437696/170498071 [00:00<00:01, 94027732.60it/s]
 45%|####4     | 76480512/170498071 [00:00<00:00, 98984247.69it/s]
 51%|#####     | 86409216/170498071 [00:01<00:00, 97370495.50it/s]
 57%|#####7    | 97353728/170498071 [00:01<00:00, 100993414.13it/s]
 63%|######3   | 107479040/170498071 [00:01<00:00, 99422133.34it/s]
 69%|######9   | 118358016/170498071 [00:01<00:00, 102151684.56it/s]
 75%|#######5  | 128614400/170498071 [00:01<00:00, 100819228.99it/s]
 82%|########1 | 139264000/170498071 [00:01<00:00, 102477663.74it/s]
 88%|########7 | 149553152/170498071 [00:01<00:00, 101979319.50it/s]
 94%|#########3| 159842304/170498071 [00:01<00:00, 102197851.28it/s]
100%|#########9| 170098688/170498071 [00:01<00:00, 102247991.64it/s]
100%|##########| 170498071/170498071 [00:01<00:00, 93026704.20it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
```

!!! note "注意"
    当您运行上面的单元格时，可能需要一些时间下载数据集。

这是一个在 PyTorch 中创建数据集对象的示例。可下载数据集（如上文的 CIFAR-10）是 `torch.utils.data.Dataset` 的子类。PyTorch 中的 `Dataset` 类包括 TorchVision、Torchtext 和 TorchAudio 中的可下载数据集，以及像 `torchvision.datasets.ImageFolder` 般的实用数据集类，它可以读取带有标签的图像文件夹。您也可以创建自己的 `Dataset` 子类。

当我们实例化数据集时，我们需要告诉它一些信息：

- 我们希望数据存放的文件系统路径。
- 是否将此数据集用于训练，大多数数据集都会分成训练子集和测试子集。
- 如果尚未下载数据集，是否要下载。
- 我们想要对数据进行的转换。

数据集准备就绪后，就可以将其交给 `DataLoader`：

```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
```

`Dataset` 子类封装了对数据的访问权限，并专门针对所服务的数据类型。 `DataLoader` 对数据一无所知，但会根据您指定的参数将 `Dataset` 提供的输入张量组织成批。

在上面的示例中，我们要求 `DataLoader` 从 `trainset` 中批量加载 4 幅图像，并随机调整它们的顺序（`shuffle=True`），我们还告诉它启动两个工作者从磁盘加载数据。

将 `DataLoader` 提供的批次可视化是一种很好的做法：

```python
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

![sphx_glr_introyt1_tutorial_001](../../../img/sphx_glr_introyt1_tutorial_001.png)

输出：
```shell
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
 ship   car horse  ship
```

运行上述单元格应该会显示给您一条包含四张图像的条带，以及每张图像的正确标签。

## 训练您的 PyTorch 模型
请从视频的 [17:10](https://www.youtube.com/watch?v=IC0_FRiX-sw&t=1030s) 开始跟随。

让我们把所有的部分都放在一起，训练一个模型：

```python

```python
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

首先，我们需要训练和测试数据集。如果您还没有，请运行下面的单元格，确保数据集已下载。（可能需要一分钟。）

```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

输出：
```shell
Files already downloaded and verified
Files already downloaded and verified
```

我们要对 DataLoader 的输出进行检查：

```python
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

![sphx_glr_introyt1_tutorial_002](../../../img/sphx_glr_introyt1_tutorial_002.png)

输出：
```shell
cat   cat  deer  frog
```

这就是我们要训练的模型。如果觉得眼熟，那是因为它是 LeNet 的一个变体，在本视频前面讨论过，适用于 3 色图像。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

最后，我们需要的是一个损失函数和一个优化器：

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

损失函数，正如本视频前面所讨论的那样，是衡量模型预测与理想输出差距的一种指标。交叉熵损失是我们这种分类模型的常用损失函数。

**优化器**是学习的驱动力。在这里，我们创建了一个实现随机梯度下降的优化器，这是一种更直接的优化算法。除了学习率（`lr`）和动量等算法参数外，我们还传入了 `net.parameters()`，这是模型中所有学习权重的集合，也是优化器要调整的内容。

最后，这些都会组成训练循环。请继续运行这个单元，因为它可能需要几分钟的时间来执行：

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

输出：
```shell
[1,  2000] loss: 2.195
[1,  4000] loss: 1.879
[1,  6000] loss: 1.656
[1,  8000] loss: 1.576
[1, 10000] loss: 1.517
[1, 12000] loss: 1.461
[2,  2000] loss: 1.415
[2,  4000] loss: 1.368
[2,  6000] loss: 1.334
[2,  8000] loss: 1.327
[2, 10000] loss: 1.318
[2, 12000] loss: 1.261
Finished Training
```

在这里，我们只进行 **2 个训练周期**（第 1 行）—— 即对训练数据集进行两次遍历。每一遍都有一个内循环，**对训练数据进行迭代**（第 4 行），提供一批转换后的输入图像及其正确标签。

**梯度归零**（第 9 行）是一个重要步骤。梯度是在一个批次中累积起来的；如果我们不在每个批次中重置梯度，它们就会不断累积，产生错误的梯度值，导致学习无法进行。

在第 12 行，我们**要求模型对这一批数据进行预测**。在接下来的第 13 行，我们计算损失 —— 即 `outputs`（模型预测）和 `labels`（正确输出）之间的差值。

在第 14 行中，我们进行 `backward()` 传递，计算梯度以指导学习。

在第 15 行中，优化器执行一个学习步骤 —— 它使用来自 `backward()` 调用的梯度，将学习权重推向它认为能减少损失的方向。

循环的余下部分会简单报告一下当前周期数、已完成的训练实例数以及在训练循环中收集到的损失。

**运行上述单元时**，您应该会看到类似下面的内容：

```shell
[1,  2000] loss: 2.235
[1,  4000] loss: 1.940
[1,  6000] loss: 1.713
[1,  8000] loss: 1.573
[1, 10000] loss: 1.507
[1, 12000] loss: 1.442
[2,  2000] loss: 1.378
[2,  4000] loss: 1.364
[2,  6000] loss: 1.349
[2,  8000] loss: 1.319
[2, 10000] loss: 1.284
[2, 12000] loss: 1.267
Finished Training
```

请注意，损失是单调下降的，这表明我们的模型在训练数据集上的性能在不断提高。

最后一步，我们应该检查模型是否真的在进行一般学习，而不是简单地 "记忆 "数据集。这就是所谓的**过拟合**，通常表明数据集太小（没有足够的例子进行一般学习），或者模型的学习参数超过了正确建模数据集所需的参数。

这也是将数据集分为训练子集和测试子集的原因 —— 为了测试模型的通用性，我们让模型对其未训练过的数据进行预测：


```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
```

输出：
```shell
Accuracy of the network on the 10000 test images: 54 %
```

如果您跟随进行了操作，您应该会发现，此时模型的准确率大约为 50%。虽然这并不是最先进的模型，但比我们预期的随机输出 10% 准确率要好得多。这表明模型中确实进行了一些一般性学习。