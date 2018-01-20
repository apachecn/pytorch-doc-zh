torchvision.models
==================

torchvision.models 模块的子模块中包含以下模型结构:

-  `AlexNet`_
-  `VGG`_
-  `ResNet`_
-  `SqueezeNet`_
-  `DenseNet`_
-  `Inception`_ v3

你可以使用随机初始化的权重来创建这些模型：

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18()
    alexnet = models.alexnet()
    vgg16 = models.vgg16()
    squeezenet = models.squeezenet1_0()
    densenet = models.densenet161()
    inception = models.inception_v3()

我们提供使用PyTorch:mod:'torch.util.model_zoo'预训练（pre-train)的模型，
可以通过参数``pretrained=True``来构造这些预训练模型。

.. code:: python

    import torchvision.models as models
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)

所有预训练（pre-train）模型要求输入图像使用相同的标准化处理，
例如： mini-batches 中 RGB 三通道图像的 shape （3 x H x W),
H 和 W 需要至少为 224， 图像必须被加载在 [0, 1] 的范围内
然后使用``mean = [0.485, 0.456, 0.406]`` 和 ``std = [0.229, 0.224, 0.225]``进行标准化处理。
你可以使用以下转换进行预标准化预处理：：

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

一个使用这种标准化处理的 imagenet 样例
`here <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>`_

ImageNet 1-crop error rates (224x224)

================================  =============   =============
Network                           Top-1 error     Top-5 error
================================  =============   =============
AlexNet                           43.45           20.91
VGG-11                            30.98           11.37
VGG-13                            30.07           10.75
VGG-16                            28.41           9.62
VGG-19                            27.62           9.12
VGG-11 with batch normalization   29.62           10.19
VGG-13 with batch normalization   28.45           9.63
VGG-16 with batch normalization   26.63           8.50
VGG-19 with batch normalization   25.76           8.15
ResNet-18                         30.24           10.92
ResNet-34                         26.70           8.58
ResNet-50                         23.85           7.13
ResNet-101                        22.63           6.44
ResNet-152                        21.69           5.94
SqueezeNet 1.0                    41.90           19.58
SqueezeNet 1.1                    41.81           19.38
Densenet-121                      25.35           7.83
Densenet-169                      24.00           7.00
Densenet-201                      22.80           6.43
Densenet-161                      22.35           6.20
Inception v3                      22.55           6.44
================================  =============   =============


.. _AlexNet: https://arxiv.org/abs/1404.5997
.. _VGG: https://arxiv.org/abs/1409.1556
.. _ResNet: https://arxiv.org/abs/1512.03385
.. _SqueezeNet: https://arxiv.org/abs/1602.07360
.. _DenseNet: https://arxiv.org/abs/1608.06993
.. _Inception: https://arxiv.org/abs/1512.00567

.. currentmodule:: torchvision.models

Alexnet
-------

.. autofunction:: alexnet

VGG
---

.. autofunction:: vgg11
.. autofunction:: vgg11_bn
.. autofunction:: vgg13
.. autofunction:: vgg13_bn
.. autofunction:: vgg16
.. autofunction:: vgg16_bn
.. autofunction:: vgg19
.. autofunction:: vgg19_bn


ResNet
------

.. autofunction:: resnet18
.. autofunction:: resnet34
.. autofunction:: resnet50
.. autofunction:: resnet101
.. autofunction:: resnet152

SqueezeNet
----------

.. autofunction:: squeezenet1_0
.. autofunction:: squeezenet1_1

DenseNet
---------

.. autofunction:: densenet121
.. autofunction:: densenet169
.. autofunction:: densenet161
.. autofunction:: densenet201

Inception v3
------------

.. autofunction:: inception_v3