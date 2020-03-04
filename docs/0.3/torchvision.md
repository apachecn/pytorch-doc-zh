# torchvision

> 译者：[@那伊抹微笑](https://github.com/wangyangting)、@dawenzi123、[@LeeGeong](https://github.com/LeeGeong)、@liandongze
> 
> 校对者：[@咸鱼](https://github.com/Watermelon233)

模块 `torchvision` 库包含了计算机视觉中一些常用的数据集, 模型架构以及图像变换方法.

Package Reference

*   [torchvision.datasets](datasets.html)
    *   [MNIST](datasets.html#mnist)
    *   [Fashion-MNIST](datasets.html#fashion-mnist)
    *   [COCO](datasets.html#coco)
    *   [LSUN](datasets.html#lsun)
    *   [ImageFolder](datasets.html#imagefolder)
    *   [Imagenet-12](datasets.html#imagenet-12)
    *   [CIFAR](datasets.html#cifar)
    *   [STL10](datasets.html#stl10)
    *   [SVHN](datasets.html#svhn)
    *   [PhotoTour](datasets.html#phototour)
*   [torchvision.models](models.html)
    *   [Alexnet](models.html#id1)
    *   [VGG](models.html#id2)
    *   [ResNet](models.html#id3)
    *   [SqueezeNet](models.html#id4)
    *   [DenseNet](models.html#id5)
    *   [Inception v3](models.html#inception-v3)
*   [torchvision.transforms](transforms.html)
    *   [PIL Image 上的变换](transforms.html#pil-image)
    *   [torch.*Tensor 上的变换](transforms.html#torch-tensor)
    *   [转换类型的变换](transforms.html#id1)
    *   [通用的变换](transforms.html#id2)
*   [torchvision.utils](utils.html)

```py
torchvision.get_image_backend()
```

获取用于加载图像的包的名称

```py
torchvision.set_image_backend(backend)
```

指定用于加载图像的包.

参数：`backend (string)` – 图像处理后端的名称. {‘PIL’, ‘accimage’} 之一. `accimage` 使用 Intel IPP library(高性能图像加载和增强程序模拟的程序）.通常比PIL库要快, 但是不支持许多操作.
