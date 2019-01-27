

# torchvision Reference

The [`torchvision`](#module-torchvision "torchvision") package consists of popular datasets, model architectures, and common image transformations for computer vision.

Package Reference

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

Gets the name of the package used to load images

```py
torchvision.set_image_backend(backend)
```

Specifies the package used to load images.

| Parameters: | **backend** (_string_) – Name of the image backend. one of {‘PIL’, ‘accimage’}. The `accimage` package uses the Intel IPP library. It is generally faster than PIL, but does not support as many operations. |
| --- | --- |
