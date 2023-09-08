# 构建神经网络模型

> 译者：[runzhi214](https://github.com/runzhi214)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/basics/buildmodel_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html>

神经网络由在数据上完成操作的层/模块构成。[`torch.nn`](https://pytorch.org/docs/stable/nn.html) 命名空间提供了所有你用来构建你自己的神经网络所需的的东西。PyTorch 中每个模块都是 [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) 的子类。一个由其他模块(层)组成的神经网络自身也是一个模块。这种嵌套的结构让构建和管理复杂的结构更轻松。

在下面的章节中，我们将构建一个神经网络来给 FashionMNIST 数据集的图片分类。

```py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

## 获取训练的设备

我们希望能够在一个硬件加速设备比如 GPU 或者 MPS 上（如果有的话）训练我们的模型。让我们检查 `torch.cuda` 和 `torch.backend.mps` 是否可用，否则我们使用 CPU。

```py
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

输出:

```py
Using cuda device
```

## 定义类

我们通过子类化 `nn.Module` 来定义我们的神经网络，并在 `__init__` 中初始化神经网络。每个 `nn.Module` 子类都会在 `forward` 方法中实现对输入数据的操作。

```py
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

我们创建一个 `NeuralNetwork` 实例，并将它发送到 `device` ，然后打印它的结构.

```py
model = NeuralNetwork().to(device)
print(model)
```

输出:

```py
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

为了使用这个模型，我们给它传递输入数据。这将会执行模型的 `forward`，伴随着一些[幕后工作](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866)。不要直接调用 `model.forward()` !

将数据传递给模型并调用后返回一个 2 维张量（第0维对应一组 10 个代表每种类型的原始预测值，第1维对应该类型对应的原始预测值）。我们将它传递给一个 `nn.Softmax` 模块的实例来来获得预测概率。

```py
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

输出:

```py
Predicted class: tensor([7], device='cuda:0')
```

## 模型层

让我们分析这个 FashionMNIST 模型层。为了说明，我们会取一个由 3 张 28x28 的图片数据组成的样例数据，并看看当我们将它传递给模型后会发生什么。

```py
input_image = torch.rand(3,28,28)
print(input_image.size())
```

输出:

```py
torch.Size([3, 28, 28])
```

### nn.Flatten

我们初始化 [`nn.Flatten`(展平层)](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) 层来将每个 2 维的 28x28 图像转换成一个包含 784 像素值的连续数组（微批数据的维度(第0维)保留了）.

```py
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

输出:

```py
torch.Size([3, 784])
```

### nn.Linear

nn.Linear（线性层）是一个对输入值使用自己存储的权重 (w) 和偏差 (b) 来做线性转换的模块。

```py
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

输出:

```py
torch.Size([3, 20])
```

### nn.ReLU

非线性的激活函数创造了模型的输入值和输出值之间的复杂映射。它们在线性转换之后应用来引入*非线性*，帮助神经网络学习更广阔范围的现象。

在这个模型中，我们在线性层之间使用 [`nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)，不过还有其他的激活函数可以用在你的模型中引入非线性。

(译者注：ReLU 即 Rectified Linear Unit，译为线性整流函数或者修正线性单元)

```py
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

输出:

```py
Before ReLU: tensor([[ 0.4158, -0.0130, -0.1144,  0.3960,  0.1476, -0.0690, -0.0269,  0.2690,
          0.1353,  0.1975,  0.4484,  0.0753,  0.4455,  0.5321, -0.1692,  0.4504,
          0.2476, -0.1787, -0.2754,  0.2462],
        [ 0.2326,  0.0623, -0.2984,  0.2878,  0.2767, -0.5434, -0.5051,  0.4339,
          0.0302,  0.1634,  0.5649, -0.0055,  0.2025,  0.4473, -0.2333,  0.6611,
          0.1883, -0.1250,  0.0820,  0.2778],
        [ 0.3325,  0.2654,  0.1091,  0.0651,  0.3425, -0.3880, -0.0152,  0.2298,
          0.3872,  0.0342,  0.8503,  0.0937,  0.1796,  0.5007, -0.1897,  0.4030,
          0.1189, -0.3237,  0.2048,  0.4343]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.4158, 0.0000, 0.0000, 0.3960, 0.1476, 0.0000, 0.0000, 0.2690, 0.1353,
         0.1975, 0.4484, 0.0753, 0.4455, 0.5321, 0.0000, 0.4504, 0.2476, 0.0000,
         0.0000, 0.2462],
        [0.2326, 0.0623, 0.0000, 0.2878, 0.2767, 0.0000, 0.0000, 0.4339, 0.0302,
         0.1634, 0.5649, 0.0000, 0.2025, 0.4473, 0.0000, 0.6611, 0.1883, 0.0000,
         0.0820, 0.2778],
        [0.3325, 0.2654, 0.1091, 0.0651, 0.3425, 0.0000, 0.0000, 0.2298, 0.3872,
         0.0342, 0.8503, 0.0937, 0.1796, 0.5007, 0.0000, 0.4030, 0.1189, 0.0000,
         0.2048, 0.4343]], grad_fn=<ReluBackward0>)
```

### nn.Sequential

[`nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) 是一个模块的有序容器。数据会沿着模块定义的顺序流动。你可以使用 sequential container(译者注：有序容器,也有的书称之为线性容器)来组成一个快速网络，比如`seq_modules`。

```py
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

### nn.Softmax

模型的最后一层返回 *logits*(介于[负无穷,正无穷]之间的原始值)，然后被传递给 [`nn.Softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) 模块。这些 logits 值被缩放到 [0,1]，代表模型对与每种类型的预测概率. `dim` 参数代表沿着该维度数值应该加总为 1.

```py
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

## 模型参数

神经网络内的许多层都是参数化的，比如有可以在训练中优化的关联的权重值和偏差值。子类化 `nn.Module` 会自动追踪所有你定义在模型对象中的字段，并通过你模型的 `parameters()` 或者 `named_parameters()` 访问所有参数。

在这个例子中，我们在每个参数上遍历，然后打印出它的大小(size)并预测数值。

```py
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n"
```

输出:

```py
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0273,  0.0296, -0.0084,  ..., -0.0142,  0.0093,  0.0135],
        [-0.0188, -0.0354,  0.0187,  ..., -0.0106, -0.0001,  0.0115]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0155, -0.0327], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0116,  0.0293, -0.0280,  ...,  0.0334, -0.0078,  0.0298],
        [ 0.0095,  0.0038,  0.0009,  ..., -0.0365, -0.0011, -0.0221]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0148, -0.0256], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0147, -0.0229,  0.0180,  ..., -0.0013,  0.0177,  0.0070],
        [-0.0202, -0.0417, -0.0279,  ..., -0.0441,  0.0185, -0.0268]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0070, -0.0411], device='cuda:0', grad_fn=<SliceBackward0>)
```

## 进一步阅读

- [`torch.nn API`](https://pytorch.org/docs/stable/nn.html)

