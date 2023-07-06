# 快速入门
> 译者：[Daydaylight](https://github.com/Daydaylight)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/Introduction_to_PyTorch/quickstart_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html>


本节介绍机器学习中常见任务的 API。请参考每个部分中的链接，以便进一步深入。



## 处理数据
PyTorch有两个[处理数据的基本操作](https://pytorch.org/docs/stable/data.html)：``torch.utils.data.DataLoader``和``torch.utils.data.Dataset``。``Dataset``存储样本及其相应的标签，而``Dataset``则围绕``Dataset``包装了一个可迭代的数据。

```py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```

PyTorch 提供特定于领域的库，如 [TorchText](https://pytorch.org/text/stable/index.html),
[TorchVision](https://pytorch.org/vision/stable/index.html) 和 [TorchAudio](https://pytorch.org/audio/stable/index.html),所有这些库都包含数据集。对于本教程，我们将使用 TorchVision 数据集。

``torchvision.datasets``模块包含了许多真实世界视觉数据的 ``Dataset``对象，比如 CIFAR、 COCO ([完整列表在这里](https://pytorch.org/vision/stable/datasets.html))。在本教程中，我们使用 FashionMNIST 数据集。每个 TorchVision ``Dataset``都包含两个参数: ``transform`` 和 ``target_transform``，分别用于转换样本和标签。


```py
# 从开源数据集下载训练数据。
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 从开源数据集下载测试数据。
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```
输出：
```py
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:12, 365718.31it/s]
  1%|          | 229376/26421880 [00:00<00:38, 685682.68it/s]
  3%|3         | 884736/26421880 [00:00<00:10, 2498938.52it/s]
  7%|7         | 1933312/26421880 [00:00<00:05, 4141475.37it/s]
 19%|#8        | 4915200/26421880 [00:00<00:01, 10854978.12it/s]
 26%|##5       | 6782976/26421880 [00:00<00:01, 11037400.65it/s]
 37%|###7      | 9797632/26421880 [00:01<00:01, 15568756.79it/s]
 44%|####4     | 11730944/26421880 [00:01<00:01, 14184748.16it/s]
 55%|#####5    | 14647296/26421880 [00:01<00:00, 17510568.70it/s]
 63%|######3   | 16777216/26421880 [00:01<00:00, 15834704.91it/s]
 75%|#######4  | 19693568/26421880 [00:01<00:00, 18759775.35it/s]
 83%|########2 | 21889024/26421880 [00:01<00:00, 16780435.96it/s]
 94%|#########3| 24772608/26421880 [00:01<00:00, 19391805.01it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 13914460.04it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 326673.50it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 362354.20it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 684627.79it/s]
 21%|##        | 917504/4422102 [00:00<00:01, 2626211.85it/s]
 44%|####3     | 1933312/4422102 [00:00<00:00, 4103892.12it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6109664.51it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 61868988.52it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```



We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
in the dataloader iterable will return a batch of 64 features and labels.


我们将``Dataset``作为参数传递给``DataLoader``。这在我们的数据集上包裹了一个可迭代的数据集，并支持自动批处理、采样、随机打乱和多进程数据加载。
在这里，我们定义了一个大小为64的批处理，即dataloader迭代器中的每个元素都会返回一个由64个特征和标签组成的批次数据。



```py
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```
输出：
```py

Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```

阅读有关[在 PyTorch 中加载数据](data_tutorial.html).的更多信息。


## 创建模型
To define a neural network in PyTorch, we create a class that inherits
from [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We define the layers of the network
in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate
operations in the neural network, we move it to the GPU or MPS if available.



为了在 PyTorch 中定义一个神经网络，我们创建了一个继承自  [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)的类。我们在 ``__init__``函数中定义网络的层，并在 forward 函数中指定数据将如何通过网络。为了加速神经网络中的操作，我们将其移动到 GPU 或 MPS (如果有的话)。
```py
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```


输出：
```py
Using cuda device
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)

```
了解更多关于[在 PyTorch 中构建神经网络](buildmodel_tutorial.html)的信息。
