

# torchvision.models

> 译者：[BXuan694](https://github.com/BXuan694)

models子包定义了以下模型架构：

*   [AlexNet](https://arxiv.org/abs/1404.5997)
*   [VGG](https://arxiv.org/abs/1409.1556)
*   [ResNet](https://arxiv.org/abs/1512.03385)
*   [SqueezeNet](https://arxiv.org/abs/1602.07360)
*   [DenseNet](https://arxiv.org/abs/1608.06993)
*   [Inception](https://arxiv.org/abs/1512.00567) v3

你可以通过调用以下构造函数构造随机权重的模型：

```py
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()

```

我们在[`torch.utils.model_zoo`](../model_zoo.html#module-torch.utils.model_zoo "torch.utils.model_zoo")中提供了预训练模型。预训练模型可以通过传递参数`pretrained=True`构造：

```py
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)

```

定义预训练模型时会把权值下载到一个缓存文件夹中，这个缓存文件可以通过环境变量`TORCH_MODEL_ZOO`来指定。更多细节见[`torch.utils.model_zoo.load_url()`](../model_zoo.html#torch.utils.model_zoo.load_url "torch.utils.model_zoo.load_url")。

有些模型在训练和测试阶段用到了不同的模块，例如批标准化(batch normalization）。使用`model.train()`或`model.eval()`可以切换到相应的模式。更多细节见[`train()`](../nn.html#torch.nn.Module.train "torch.nn.Module.train")或[`eval()`](../nn.html#torch.nn.Module.eval "torch.nn.Module.eval")。

所有的预训练模型都要求输入图片以相同的方式进行标准化，即：小批(mini-batch）三通道RGB格式(3 x H x W），其中H和W不得小于224。图片加载时像素值的范围应在[0, 1]内，然后通过指定`mean = [0.485, 0.456, 0.406]`和`std = [0.229, 0.224, 0.225]`进行标准化，例如：

```py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

```

在[imagenet的示例](https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101)中可以看到标准化的一个应用。

下表是ImageNet单次224x224中心裁剪的错误率。

| 网络 | Top-1错误率(%） | Top-5错误率(%） |
| --- | --- | --- |
| AlexNet | 43.45 | 20.91 |
| VGG-11 | 30.98 | 11.37 |
| VGG-13 | 30.07 | 10.75 |
| VGG-16 | 28.41 | 9.62 |
| VGG-19 | 27.62 | 9.12 |
| 带有批标准化的VGG-11 | 29.62 | 10.19 |
| 带有批标准化的VGG-13 | 28.45 | 9.63 |
| 带有批标准化的VGG-16 | 26.63 | 8.50 |
| 带有批标准化的VGG-19 | 25.76 | 8.15 |
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

AlexNet模型，参见论文[《One weird trick…》](https://arxiv.org/abs/1404.5997) 。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

## VGG

```py
torchvision.models.vgg11(pretrained=False, **kwargs)
```

VGG11模型。(论文中的“A”模型）

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.vgg11_bn(pretrained=False, **kwargs)
```

VGG11模型，带有批标准化。(论文中的“A”模型）

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.vgg13(pretrained=False, **kwargs)
```

VGG13模型。(论文中的“B”模型）

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.vgg13_bn(pretrained=False, **kwargs)
```

VGG13模型，带有批标准化。(论文中的“B”模型）

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.vgg16(pretrained=False, **kwargs)
```

VGG16模型。(论文中的“D”模型）

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.vgg16_bn(pretrained=False, **kwargs)
```

VGG16模型，带有批标准化。(论文中的“D”模型）

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.vgg19(pretrained=False, **kwargs)
```

VGG19模型。(论文中的“E”模型）

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.vgg19_bn(pretrained=False, **kwargs)
```

VGG19模型，带有批标准化。(论文中的“E”模型）

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

## ResNet

```py
torchvision.models.resnet18(pretrained=False, **kwargs)
```

构造ResNet-18模型。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.resnet34(pretrained=False, **kwargs)
```

构造ResNet-34模型。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.resnet50(pretrained=False, **kwargs)
```

构造ResNet-50模型。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.resnet101(pretrained=False, **kwargs)
```

构造ResNet-101模型。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.resnet152(pretrained=False, **kwargs)
```

构造ResNet-152模型。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

## SqueezeNet

```py
torchvision.models.squeezenet1_0(pretrained=False, **kwargs)
```

SqueezeNet模型，参见论文[《SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and &lt;0.5MB model size》](https://arxiv.org/abs/1602.07360)。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.squeezenet1_1(pretrained=False, **kwargs)
```

SqueezeNet 1.1模型，参见[SqueezeNet官方仓库](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1)。SqueezeNet 1.1比SqueezeNet 1.0节约2.4倍的计算量，参数也略少，然而精度未做牺牲。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

## DenseNet

```py
torchvision.models.densenet121(pretrained=False, **kwargs)
```

Densenet-121模型，参见[《Densely Connected Convolutional Networks》](https://arxiv.org/pdf/1608.06993.pdf)。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.densenet169(pretrained=False, **kwargs)
```

Densenet-169模型，参见[《Densely Connected Convolutional Networks》](https://arxiv.org/pdf/1608.06993.pdf)。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.densenet161(pretrained=False, **kwargs)
```

Densenet-161模型，参见[《Densely Connected Convolutional Networks》](https://arxiv.org/pdf/1608.06993.pdf)。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

```py
torchvision.models.densenet201(pretrained=False, **kwargs)
```

Densenet-201模型，参见[《Densely Connected Convolutional Networks》](https://arxiv.org/pdf/1608.06993.pdf)。

 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

## Inception v3

```py
torchvision.models.inception_v3(pretrained=False, **kwargs)
```

Inception v3模型，参见[《Rethinking the Inception Architecture for Computer Vision》](http://arxiv.org/abs/1512.00567)。
 
| 参数： | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 如果设置为True，返回ImageNet预训练模型 |
| --- | --- |

