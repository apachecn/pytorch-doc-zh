# torchvision.models

> 译者：[@那伊抹微笑](https://github.com/wangyangting)、@dawenzi123、[@LeeGeong](https://github.com/LeeGeong)、@liandongze
> 
> 校对者：[@咸鱼](https://github.com/Watermelon233)

torchvision.models 模块的子模块中包含以下模型结构:

*   [AlexNet](https://arxiv.org/abs/1404.5997)
*   [VGG](https://arxiv.org/abs/1409.1556)
*   [ResNet](https://arxiv.org/abs/1512.03385)
*   [SqueezeNet](https://arxiv.org/abs/1602.07360)
*   [DenseNet](https://arxiv.org/abs/1608.06993)
*   [Inception](https://arxiv.org/abs/1512.00567) v3

你可以使用随机初始化的权重来创建这些模型:

```py
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()

```

我们提供使用PyTorch [`torch.utils.model_zoo`](../model_zoo.html#module-torch.utils.model_zoo "torch.utils.model_zoo") 预训练 (pre-train)的模型, 可以通过参数 `pretrained=True` 来构造这些预训练模型.

```py
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)

```

所有预训练 (pre-train) 模型要求输入图像使用相同的标准化处理, 例如: mini-batches 中 RGB 三通道图像的 shape (3 x H x W), H 和 W 需要至少为 224, 图像必须被加载在 [0, 1] 的范围内 然后使用 `mean = [0.485, 0.456, 0.406]` 和 `std = [0.229, 0.224, 0.225]` 进行标准化处理. 你可以使用以下转换进行预标准化预处理:

```py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

```

一个使用这种标准化处理的 imagenet 样例 [here](https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101)

ImageNet 1-crop error rates (224x224)

| Network | Top-1 error | Top-5 error |
| --- | --- | --- |
| AlexNet | 43.45 | 20.91 |
| VGG-11 | 30.98 | 11.37 |
| VGG-13 | 30.07 | 10.75 |
| VGG-16 | 28.41 | 9.62 |
| VGG-19 | 27.62 | 9.12 |
| VGG-11 with batch normalization | 29.62 | 10.19 |
| VGG-13 with batch normalization | 28.45 | 9.63 |
| VGG-16 with batch normalization | 26.63 | 8.50 |
| VGG-19 with batch normalization | 25.76 | 8.15 |
| ResNet-18 | 30.24 | 10.92 |
| ResNet-34 | 26.70 | 8.58 |
| ResNet-50 | 23.85 | 7.13 |
| ResNet-101 | 22.63 | 6.44 |
| ResNet-152 | 21.69 | 5.94 |
| SqueezeNet 1.0 | 41.90 | 19.58 |
| SqueezeNet 1.1 | 41.81 | 19.38 |
| Densenet-121 | 25.35 | 7.83 |
| Densenet-169 | 24.00 | 7.00 |
| Densenet-201 | 22.80 | 6.43 |
| Densenet-161 | 22.35 | 6.20 |
| Inception v3 | 22.55 | 6.44 |

## Alexnet

```py
torchvision.models.alexnet(pretrained=False, **kwargs)
```

AlexNet 模型结构论文地址 [“One weird trick…”](https://arxiv.org/abs/1404.5997) .

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


## VGG

```py
torchvision.models.vgg11(pretrained=False, **kwargs)
```

VGG 11层模型 (configuration “A”)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.vgg11_bn(pretrained=False, **kwargs)
```

带有批标准化(batch normalization) 的VGG 11层模型 (configuration “A”)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.vgg13(pretrained=False, **kwargs)
```

VGG 13层模型 (configuration “B”)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.vgg13_bn(pretrained=False, **kwargs)
```

带有批标准化(batch normalization) 的 VGG 13层模型 (configuration “B”)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.vgg16(pretrained=False, **kwargs)
```

VGG 16层模型 (configuration “D”)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.vgg16_bn(pretrained=False, **kwargs)
```

带有批标准化(batch normalization) 的 VGG 16层模型 (configuration “D”)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.vgg19(pretrained=False, **kwargs)
```

VGG 19层模型 (configuration “E”)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.vgg19_bn(pretrained=False, **kwargs)
```

带有批标准化(batch normalization) 的 VGG 19层模型 (configuration ‘E’)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


## ResNet

```py
torchvision.models.resnet18(pretrained=False, **kwargs)
```

构造一个 ResNet-18 模型.

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.resnet34(pretrained=False, **kwargs)
```

构造一个 ResNet-34 模型.

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.resnet50(pretrained=False, **kwargs)
```

构造一个 ResNet-50 模型.

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.resnet101(pretrained=False, **kwargs)
```

构造一个 ResNet-101 模型.

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.resnet152(pretrained=False, **kwargs)
```

构造一个 ResNet-152 模型.

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


## SqueezeNet

```py
torchvision.models.squeezenet1_0(pretrained=False, **kwargs)
```

SqueezeNet 模型结构源于论文: [“SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and &lt;0.5MB model size”](https://arxiv.org/abs/1602.07360)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.squeezenet1_1(pretrained=False, **kwargs)
```

SqueezeNet 1.1 模型源于论文: [official SqueezeNet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1). SqueezeNet 1.1 比 SqueezeNet 1.0 减少了 2.4倍的运算量, 并在不损伤准确率的基础上减少了少许参数.

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


## DenseNet

```py
torchvision.models.densenet121(pretrained=False, **kwargs)
```

Densenet-121 模型源自于: [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.densenet169(pretrained=False, **kwargs)
```

Densenet-169 模型源自于: [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.densenet161(pretrained=False, **kwargs)
```

Densenet-161 模型源自于: [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


```py
torchvision.models.densenet201(pretrained=False, **kwargs)
```

Densenet-201 模型源自于: [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.


## Inception v3

```py
torchvision.models.inception_v3(pretrained=False, **kwargs)
```

Inception v3 模型结构源自于 [“Rethinking the Inception Architecture for Computer Vision”](http://arxiv.org/abs/1512.00567).

参数：`pretrained (bool)` – True, 返回一个在 ImageNet 上预训练的模型.
