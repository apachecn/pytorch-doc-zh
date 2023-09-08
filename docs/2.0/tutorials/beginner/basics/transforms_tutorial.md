# TRANSFORMS

> 译者：[Daydaylight](https://github.com/Daydaylight)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/basics/transforms_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html>

数据并不总是以训练机器学习算法所需的最终处理形式出现。我们使用变换来对数据进行一些处理，使其适合训练。

所有的 TorchVision 数据集都有两个参数: ``transform`` 用于修改特征和 ``target_transform`` 用于修改标签，它们接受包含转换逻辑的 callables。[torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) 模块提供了几个常用的转换算法，开箱即用。

FashionMNIST 的特征是 PIL 图像格式，而标签是整数。对于训练，我们需要将特征作为归一化的张量，将标签作为独热编码的张量。
为了进行这些转换，我们使用 ``ToTensor`` 和 ``Lambda``。

```py
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

输出：

```py
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:12, 361690.02it/s]
  1%|          | 229376/26421880 [00:00<00:38, 679756.53it/s]
  2%|2         | 655360/26421880 [00:00<00:14, 1775435.30it/s]
  7%|6         | 1736704/26421880 [00:00<00:06, 3785228.35it/s]
 15%|#4        | 3833856/26421880 [00:00<00:02, 8223694.86it/s]
 21%|##1       | 5570560/26421880 [00:00<00:02, 9088903.43it/s]
 32%|###1      | 8454144/26421880 [00:01<00:01, 13772389.09it/s]
 39%|###9      | 10420224/26421880 [00:01<00:01, 13068367.31it/s]
 50%|#####     | 13238272/26421880 [00:01<00:00, 16440554.97it/s]
 58%|#####7    | 15269888/26421880 [00:01<00:00, 14938744.03it/s]
 68%|######8   | 18055168/26421880 [00:01<00:00, 17703674.30it/s]
 76%|#######6  | 20119552/26421880 [00:01<00:00, 15854480.37it/s]
 87%|########6 | 22904832/26421880 [00:01<00:00, 18366169.37it/s]
 95%|#########4| 25034752/26421880 [00:01<00:00, 16404116.31it/s]
100%|##########| 26421880/26421880 [00:02<00:00, 13106029.06it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 326257.67it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:12, 362747.74it/s]
  5%|5         | 229376/4422102 [00:00<00:06, 681864.40it/s]
 15%|#4        | 655360/4422102 [00:00<00:02, 1798436.42it/s]
 40%|####      | 1769472/4422102 [00:00<00:00, 3872995.18it/s]
 79%|#######9  | 3506176/4422102 [00:00<00:00, 7404355.18it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 5422111.79it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 35867569.75it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## ToTensor()

[ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor) 将 PIL 图像或 NumPy 的 ``ndarray`` 转换为 ``FloatTensor``。图像的像素强度值在 [0., 1.] 范围内缩放。

## Lambda Transforms

Lambda transforms 应用任何用户定义的 lambda 函数。在这里，我们定义了一个函数来把整数变成一个独热编码的张量。
它首先创建一个大小为10（我们数据集中的标签数量）的零张量，然后传递参数 ``value=1`` 在标签 ``y`` 所给的索引上调用 [scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html) 。

```py
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```

### 阅读更多
- [torchvision.transforms API](https://pytorch.org/vision/stable/transforms.html)
