

# torchvision.models

The models subpackage contains definitions for the following model architectures:

*   [AlexNet](https://arxiv.org/abs/1404.5997)
*   [VGG](https://arxiv.org/abs/1409.1556)
*   [ResNet](https://arxiv.org/abs/1512.03385)
*   [SqueezeNet](https://arxiv.org/abs/1602.07360)
*   [DenseNet](https://arxiv.org/abs/1608.06993)
*   [Inception](https://arxiv.org/abs/1512.00567) v3

You can construct a model with random weights by calling its constructor:

```py
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()

```

We provide pre-trained models, using the PyTorch [`torch.utils.model_zoo`](../model_zoo.html#module-torch.utils.model_zoo "torch.utils.model_zoo"). These can be constructed by passing `pretrained=True`:

```py
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)

```

Instancing a pre-trained model will download its weights to a cache directory. This directory can be set using the `TORCH_MODEL_ZOO` environment variable. See [`torch.utils.model_zoo.load_url()`](../model_zoo.html#torch.utils.model_zoo.load_url "torch.utils.model_zoo.load_url") for details.

Some models use modules which have different training and evaluation behavior, such as batch normalization. To switch between these modes, use `model.train()` or `model.eval()` as appropriate. See [`train()`](../nn.html#torch.nn.Module.train "torch.nn.Module.train") or [`eval()`](../nn.html#torch.nn.Module.eval "torch.nn.Module.eval") for details.

All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`. You can use the following transform to normalize:

```py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

```

An example of such normalization can be found in the imagenet example [here](https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101)

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

AlexNet model architecture from the [“One weird trick…”](https://arxiv.org/abs/1404.5997) paper.

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

## VGG

```py
torchvision.models.vgg11(pretrained=False, **kwargs)
```

VGG 11-layer model (configuration “A”)

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.vgg11_bn(pretrained=False, **kwargs)
```

VGG 11-layer model (configuration “A”) with batch normalization

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.vgg13(pretrained=False, **kwargs)
```

VGG 13-layer model (configuration “B”)

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.vgg13_bn(pretrained=False, **kwargs)
```

VGG 13-layer model (configuration “B”) with batch normalization

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.vgg16(pretrained=False, **kwargs)
```

VGG 16-layer model (configuration “D”)

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.vgg16_bn(pretrained=False, **kwargs)
```

VGG 16-layer model (configuration “D”) with batch normalization

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.vgg19(pretrained=False, **kwargs)
```

VGG 19-layer model (configuration “E”)

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.vgg19_bn(pretrained=False, **kwargs)
```

VGG 19-layer model (configuration ‘E’) with batch normalization

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

## ResNet

```py
torchvision.models.resnet18(pretrained=False, **kwargs)
```

Constructs a ResNet-18 model.

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.resnet34(pretrained=False, **kwargs)
```

Constructs a ResNet-34 model.

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.resnet50(pretrained=False, **kwargs)
```

Constructs a ResNet-50 model.

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.resnet101(pretrained=False, **kwargs)
```

Constructs a ResNet-101 model.

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.resnet152(pretrained=False, **kwargs)
```

Constructs a ResNet-152 model.

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

## SqueezeNet

```py
torchvision.models.squeezenet1_0(pretrained=False, **kwargs)
```

SqueezeNet model architecture from the [“SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and &lt;0.5MB model size”](https://arxiv.org/abs/1602.07360) paper.

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.squeezenet1_1(pretrained=False, **kwargs)
```

SqueezeNet 1.1 model from the [official SqueezeNet repo](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1). SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

## DenseNet

```py
torchvision.models.densenet121(pretrained=False, **kwargs)
```

Densenet-121 model from [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.densenet169(pretrained=False, **kwargs)
```

Densenet-169 model from [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.densenet161(pretrained=False, **kwargs)
```

Densenet-161 model from [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

```py
torchvision.models.densenet201(pretrained=False, **kwargs)
```

Densenet-201 model from [“Densely Connected Convolutional Networks”](https://arxiv.org/pdf/1608.06993.pdf)

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

## Inception v3

```py
torchvision.models.inception_v3(pretrained=False, **kwargs)
```

Inception v3 model architecture from [“Rethinking the Inception Architecture for Computer Vision”](http://arxiv.org/abs/1512.00567).

 
| Parameters: | **pretrained** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – If True, returns a model pre-trained on ImageNet |
| --- | --- |

