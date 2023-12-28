# 使用 PyTorch 训练

> 译者：[Fadegentle](https://github.com/Fadegentle)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/introyt/trainingyt>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/introyt/trainingyt.html>

请跟随下面的视频或在 [youtube](https://www.youtube.com/watch?v=jF43_wj_DCQ) 上观看。

<iframe width="560" height="315" src="https://www.youtube.com/embed/jF43_wj_DCQ" title="Training with PyTorch" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## 入门

在过去的视频中，我们讨论并演示了

- 使用 torch.nn 模块的神经网络层和功能构建模型
- 自动计算梯度的机制，这是基于梯度的模型训练的核心
- 使用 TensorBoard 可视化训练进度和其他活动

在本视频中，我们将为您的清单添加一些新工具：

- 我们将熟悉数据集和数据加载器抽象，以及它们如何简化在训练循环中向模型输入数据的过程
- 我们将讨论特定的损失函数以及何时使用它们
- 我们将了解 PyTorch 优化器，它实现了根据损失函数的结果调整模型权重的算法

最后，我们将把所有这些内容整合在一起，看看一个完整的 PyTorch 训练循环是如何运行的。

## Dataset 和 DataLoader

`Dataset` 和 `DataLoader` 类封装了从存储中提取数据并将其批量供给训练循环的过程。

`Dataset` 负责访问和处理单个数据实例。

`DataLoader` 从 `Dataset` 中提取数据实例(可自动提取，也可使用您定义的采样器)，分批收集，并返回给训练循环使用。`DataLoader` 可以处理各种数据集，无论其中数据类型如何。

在本教程中，我们将使用 TorchVision 提供的 Fashion-MNIST 数据集。我们使用 `torchvision.transforms.Normalize()` 对图块内容的分布进行零中心化和归一化处理，并下载训练和验证数据分片。

```python
import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))
```

输出：
```shell
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:13, 360703.29it/s]
  1%|          | 229376/26421880 [00:00<00:38, 678236.38it/s]
  3%|2         | 753664/26421880 [00:00<00:12, 2086208.76it/s]
  6%|5         | 1474560/26421880 [00:00<00:08, 3023702.71it/s]
 14%|#4        | 3702784/26421880 [00:00<00:02, 8041860.16it/s]
 23%|##3       | 6127616/26421880 [00:00<00:01, 10505418.63it/s]
 32%|###1      | 8355840/26421880 [00:01<00:01, 13279807.23it/s]
 41%|####1     | 10911744/26421880 [00:01<00:01, 13962012.06it/s]
 49%|####9     | 13008896/26421880 [00:01<00:00, 15452797.33it/s]
 59%|#####9    | 15695872/26421880 [00:01<00:00, 15555809.56it/s]
 67%|######7   | 17760256/26421880 [00:01<00:00, 16646214.71it/s]
 78%|#######8  | 20643840/26421880 [00:01<00:00, 16671105.27it/s]
 86%|########6 | 22740992/26421880 [00:01<00:00, 17634531.55it/s]
 97%|#########7| 25657344/26421880 [00:02<00:00, 17402442.83it/s]
100%|##########| 26421880/26421880 [00:02<00:00, 13077679.87it/s]
Extracting ./data/FashionMNIST/raw/train-images-idx3-ubyte.gz to ./data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 325110.39it/s]
Extracting ./data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 360581.21it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 677780.87it/s]
 16%|#6        | 720896/4422102 [00:00<00:01, 1977033.91it/s]
 44%|####3     | 1933312/4422102 [00:00<00:00, 4219301.79it/s]
 92%|#########1| 4063232/4422102 [00:00<00:00, 8628651.30it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 5939129.50it/s]
Extracting ./data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 45941014.88it/s]
Extracting ./data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST/raw

Training set has 60000 instances
Validation set has 10000 instances
```

和往常一样，让我们把数据可视化，检查其是否合理：

```python
import matplotlib.pyplot as plt
import numpy as np

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print('  '.join(classes[labels[j]] for j in range(4)))
```

![sphx_glr_trainingyt_001](../../../img/sphx_glr_trainingyt_001.png)

输出：
```shell
Sandal  Sneaker  Coat  Sneaker
```

## 模型
本示例中使用的模型是 LeNet-5 的变体，如果您看过本系列的前几期视频，应该对它不陌生。

```python
import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = GarmentClassifier()
```

## 损失函数

本例中，我们将使用交叉熵损失。为了演示，我们将创建一批假输出和标签值，通过损失函数运行它们，并检查结果。

```python
loss_fn = torch.nn.CrossEntropyLoss()

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 10)
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 5, 3, 7])

print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))
```

输出：
```shell
tensor([[0.7026, 0.1489, 0.0065, 0.6841, 0.4166, 0.3980, 0.9849, 0.6701, 0.4601,
         0.8599],
        [0.7461, 0.3920, 0.9978, 0.0354, 0.9843, 0.0312, 0.5989, 0.2888, 0.8170,
         0.4150],
        [0.8408, 0.5368, 0.0059, 0.8931, 0.3942, 0.7349, 0.5500, 0.0074, 0.0554,
         0.1537],
        [0.7282, 0.8755, 0.3649, 0.4566, 0.8796, 0.2390, 0.9865, 0.7549, 0.9105,
         0.5427]])
tensor([1, 5, 3, 7])
Total loss for this batch: 2.428950071334839
```

## 优化器

本例中，我们将使用简单的动量[随机梯度下降法](https://pytorch.org/docs/stable/optim.html)。

尝试这种优化方案的一些变式可能会有所启发：

- 学习率决定了优化器的步长。不同的学习率会对训练结果的准确性和收敛时间产生什么影响？
- 动量(Momentum)使优化器在多个步骤中朝着梯度最大的方向前进。改变这个值对结果有什么影响？
- 尝试一些不同的优化算法，如平均 SGD、Adagrad 或 Adam。结果有何不同？

```python
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

## 训练循环

下面，我们有一个执行一个训练周期的函数。它从 DataLoader 中枚举数据，并在循环的每次传递中执行以下操作：

- 从 DataLoader 中获取一批训练数据
- 将优化器的梯度清零
- 执行推理，即从输入批次的模型中获取预测结果
- 计算这组预测与数据集标签的损失
- 计算学习权重的后向梯度
- 告诉优化器执行一个学习步骤，即根据我们选择的优化算法和观察到的梯度调整模型的学习权重。
- 报告每 1000 个批次的损失。
- 最后，它会报告最近 1000 个批次的平均每批次损失，以便与验证运行进行比较

```python
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
```

### 周期活动

有几件事我们需要在每个周期进行一次：

- 在一组未用于训练的数据上，验证相对损失，并报告结果
- 保存模型副本

在这里，我们用 TensorBoard 进行报告。这需要通过命令行启动 TensorBoard，并在另一个浏览器标签页中打开它。

```python
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
```

输出：
```shell
EPOCH 1:
  batch 1000 loss: 1.6334228584356607
  batch 2000 loss: 0.8325267538074403
  batch 3000 loss: 0.7359380583595484
  batch 4000 loss: 0.6198329215242994
  batch 5000 loss: 0.6000315657821484
  batch 6000 loss: 0.555109024874866
  batch 7000 loss: 0.5260250487388112
  batch 8000 loss: 0.4973462742221891
  batch 9000 loss: 0.4781935699362075
  batch 10000 loss: 0.47880298678041433
  batch 11000 loss: 0.45598648857555235
  batch 12000 loss: 0.4327470133750467
  batch 13000 loss: 0.41800182418141046
  batch 14000 loss: 0.4115047634313814
  batch 15000 loss: 0.4211296908891527
LOSS train 0.4211296908891527 valid 0.414460688829422
EPOCH 2:
  batch 1000 loss: 0.3879808729066281
  batch 2000 loss: 0.35912817339546743
  batch 3000 loss: 0.38074520684120944
  batch 4000 loss: 0.3614532373107213
  batch 5000 loss: 0.36850082185724753
  batch 6000 loss: 0.3703581801643886
  batch 7000 loss: 0.38547042514081115
  batch 8000 loss: 0.37846584360170527
  batch 9000 loss: 0.3341486988377292
  batch 10000 loss: 0.3433013284947956
  batch 11000 loss: 0.35607743899174965
  batch 12000 loss: 0.3499939931873523
  batch 13000 loss: 0.33874178926000603
  batch 14000 loss: 0.35130289171106416
  batch 15000 loss: 0.3394507191307202
LOSS train 0.3394507191307202 valid 0.3581162691116333
EPOCH 3:
  batch 1000 loss: 0.3319729989422485
  batch 2000 loss: 0.29558994361863006
  batch 3000 loss: 0.3107374766407593
  batch 4000 loss: 0.3298987646112146
  batch 5000 loss: 0.30858693152241906
  batch 6000 loss: 0.33916381367447684
  batch 7000 loss: 0.3105102765217889
  batch 8000 loss: 0.3011080777524912
  batch 9000 loss: 0.3142058177240979
  batch 10000 loss: 0.31458891937109
  batch 11000 loss: 0.31527258940579483
  batch 12000 loss: 0.31501667268342864
  batch 13000 loss: 0.3011875962628328
  batch 14000 loss: 0.30012811454350596
  batch 15000 loss: 0.31833117976446373
LOSS train 0.31833117976446373 valid 0.3307691514492035
EPOCH 4:
  batch 1000 loss: 0.2786161053752294
  batch 2000 loss: 0.27965198021690596
  batch 3000 loss: 0.28595415444140965
  batch 4000 loss: 0.292985666413857
  batch 5000 loss: 0.3069892351147719
  batch 6000 loss: 0.29902250939945224
  batch 7000 loss: 0.2863366014406201
  batch 8000 loss: 0.2655441066541243
  batch 9000 loss: 0.3045048695363293
  batch 10000 loss: 0.27626545656517554
  batch 11000 loss: 0.2808379335970967
  batch 12000 loss: 0.29241049340573955
  batch 13000 loss: 0.28030834131941446
  batch 14000 loss: 0.2983542350126445
  batch 15000 loss: 0.3009556676162611
LOSS train 0.3009556676162611 valid 0.41686952114105225
EPOCH 5:
  batch 1000 loss: 0.2614263167564495
  batch 2000 loss: 0.2587047562422049
  batch 3000 loss: 0.2642477260621345
  batch 4000 loss: 0.2825975873669813
  batch 5000 loss: 0.26987933717705165
  batch 6000 loss: 0.2759250026817317
  batch 7000 loss: 0.26055969463163275
  batch 8000 loss: 0.29164007206353565
  batch 9000 loss: 0.2893096504513578
  batch 10000 loss: 0.2486029507305684
  batch 11000 loss: 0.2732803234480907
  batch 12000 loss: 0.27927226484491985
  batch 13000 loss: 0.2686819267635074
  batch 14000 loss: 0.24746483912148323
  batch 15000 loss: 0.27903492261294194
LOSS train 0.27903492261294194 valid 0.31206756830215454
```

加载模型的已保存版本：

```python
saved_model = GarmentClassifier()
saved_model.load_state_dict(torch.load(PATH))
```

加载模型后，它就可以用于任何用途——更多训练、推理或分析。

请注意，如果您的模型有影响模型结构的构造函数参数，您需要提供这些参数，并将模型配置成和保存时相同的状态。

## 其它资源

- pytorch.org 上关于[数据工具](https://pytorch.org/docs/stable/data.html)(包括 Dataset 和 DataLoader)的文档
- 关于在 GPU 训练中[使用固定内存的说明](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning)
- 关于 [TorchVision](https://pytorch.org/vision/stable/datasets.html)、[TorchText](https://pytorch.org/text/stable/datasets.html) 和 [TorchAudio](https://pytorch.org/audio/stable/datasets.html) 中可用数据集的文档
- 关于 PyTorch 中可用[损失函数](https://pytorch.org/docs/stable/nn.html#loss-functions)的文档
- 有关 [torch.optim 包](https://pytorch.org/docs/stable/optim.html)的文档，其中包括优化器和相关工具，如学习率调度等
- [保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)的详细教程
- [pytorch.org 的教程部分](https://pytorch.org/tutorials/)包含各种训练任务的教程，包括不同领域的分类、生成对抗网络、强化学习等。