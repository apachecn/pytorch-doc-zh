# 可选: 数据并行

> 译者：[@小王子](https://github.com/VPrincekin)
> 
> 校对者：[@李子文](https://github.com/liziwenzzzz)

**作者**: [Sung Kim](https://github.com/hunkim) 和 [Jenny Kang](https://github.com/jennykang)

在这个教程中, 我们将会学习如何在多个GPU上使用 `DataParallel` .

在 PyTorch 中使用 GPU 是一件很容易的事情.你可以像下面这样轻松的将一个模型分配到一个 GPU 上.

```py
model.gpu()

```

随后, 你可以将你的所有张量拷贝到上面的GPU:

```py
mytensor = my_tensor.gpu()

```

此处请注意: 如果只是调用 `mytensor.gpu()` 是不会将张量拷贝到 GPU 的.你需要将它赋给一个 新的张量, 这个张量就能在 GPU 上使用了.

在多个 GPU 上运行前向、反向传播是一件很自然的事情, 然而, PyTorch 默认情况下只会用到一个GPU, 可以通过使用 `DataParallel` 使你的模型并行运行, 在多个GPU上运行这些操作也将变得非常简单:

```py
model = nn.DataParallel(model)

```

这是教程的核心内容, 我们将在随后进行详细讲解

## 导入和参数

导入PyTorch模块和参数定义

```py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# 参数和数据加载
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

```

## 伪数据集

只需要实现 getitem 就可以轻松的生成一个(随机）伪数据集, 如下代码所示:

```py
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, 100),
                         batch_size=batch_size, shuffle=True)

```

## 简单模型

在下面的示例中, 我们的模型只需要一个输入并且完成一个线性操作, 最后得 到一个输出.当然, 你可以在任意模型 (CNN,RNN,Capsule Net等) 运用 `DataParallel`

我们在模型中设置了打印指令来监控输入和输出的张量大小, 请注意批数据次序为0时的输出.

```py
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("  In Model: input size", input.size(),
              "output size", output.size())

        return output

```

## 创建模型和 DataParallel

这是本教程的核心部分. 首先, 我们需要生成一个模型的实例并且检测我们是否拥有多个 GPU.如果有多个GPU , 我们可以使用 `nn.DataParallel` 来包装我们的模型, 然后我们 就可以将我们的模型通过 `model.gpu()` 施加于这些GPU上.

```py
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

if torch.cuda.is_available():
   model.cuda()

```

## 运行模型

现在我们可以看到输入和输出张量的大小了.

```py
for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)

    output = model(input_var)
    print("Outside: input size", input_var.size(),
          "output_size", output.size())

```

## 结果

当我们将输入设置为30批, 模型也产生了30批的输出.但是当我们使用多个GPU, 然后你 会得到类似下面这样的输出.

### 2 GPUs

如果有2个GPU, 我们将会看到这样的结果:

```py
# on 2 GPUs
Let's use 2 GPUs!
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
    In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
    In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])

```

### 3 GPUs

如果有3个GPU, 我们将会看到这样的结果:

```py
Let's use 3 GPUs!
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
    In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])

```

### 8 GPUs

如果有8个GPU, 我们将会看到这样的结果:

```py
Let's use 8 GPUs!
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
    In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])

```

## 总结

DataParallel 自动地将数据分割并且将任务送入多个GPU上的多个模型中进行处理. 在每个模型完成任务后, DataParallel 采集和合并所有结果, 并将最后的结果呈现给你.

想了解更多信息, 请点击: [https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html).
