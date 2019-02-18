

# torchvision 参考

> 译者：[BXuan694](https://github.com/BXuan694)

[`torchvision`](#module-torchvision "torchvision") 包收录了若干重要的公开数据集、网络模型和计算机视觉中的常用图像变换

包参考

*   [torchvision.datasets](datasets.html)
    *   [MNIST](datasets.html#mnist)
    *   [Fashion-MNIST](datasets.html#fashion-mnist)
    *   [EMNIST](datasets.html#emnist)
    *   [COCO](datasets.html#coco)
    *   [LSUN](datasets.html#lsun)
    *   [ImageFolder](datasets.html#imagefolder)
    *   [DatasetFolder](datasets.html#datasetfolder)
    *   [Imagenet-12](datasets.html#imagenet-12)
    *   [CIFAR](datasets.html#cifar)
    *   [STL10](datasets.html#stl10)
    *   [SVHN](datasets.html#svhn)
    *   [PhotoTour](datasets.html#phototour)
    *   [SBU](datasets.html#sbu)
    *   [Flickr](datasets.html#flickr)
    *   [VOC](datasets.html#voc)
*   [torchvision.models](models.html)
    *   [Alexnet](models.html#id1)
    *   [VGG](models.html#id2)
    *   [ResNet](models.html#id3)
    *   [SqueezeNet](models.html#id4)
    *   [DenseNet](models.html#id5)
    *   [Inception v3](models.html#inception-v3)
*   [torchvision.transforms](transforms.html)
    *   [Transforms on PIL Image](transforms.html#transforms-on-pil-image)
    *   [Transforms on torch.*Tensor](transforms.html#transforms-on-torch-tensor)
    *   [Conversion Transforms](transforms.html#conversion-transforms)
    *   [Generic Transforms](transforms.html#generic-transforms)
    *   [Functional Transforms](transforms.html#functional-transforms)
*   [torchvision.utils](utils.html)

```py
torchvision.get_image_backend()
```

查看载入图片的包的名称

```py
torchvision.set_image_backend(backend)
```

指定用于载入图片的包

| 参数: | **backend** (_string_) – 图片处理后端的名称，须为{‘PIL’, ‘accimage’}中的一个。`accimage`包使用了英特尔IPP库。这个库通常比PIL快，但是支持的操作比PIL要少。|
| --- | --- |
