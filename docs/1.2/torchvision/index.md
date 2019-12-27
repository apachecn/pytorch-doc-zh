# [TORCHVISION](https://pytorch.org/docs/stable/torchvision/index.html#torchvision)

该[torchvision](https://pytorch.org/docs/stable/torchvision/index.html#module-torchvision)软件包包括流行的数据集，模型体系结构和用于计算机视觉的常见图像转换。

Package Reference

* [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html)
    * [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)
    * [Fashion-MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist)
    * [KMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#kmnist)
    * [EMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#emnist)
    * [QMNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#qmnist)
    * [FakeData](https://pytorch.org/docs/stable/torchvision/datasets.html#fakedata)
    * [COCO](https://pytorch.org/docs/stable/torchvision/datasets.html#coco)
    * [LSUN](https://pytorch.org/docs/stable/torchvision/datasets.html#lsun)
    * [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)
    * [DatasetFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder)
    * [ImageNet](https://pytorch.org/docs/stable/torchvision/datasets.html#imagenet)
    * [CIFAR](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar)
    * [STL10](https://pytorch.org/docs/stable/torchvision/datasets.html#stl10)
    * [SVHN](https://pytorch.org/docs/stable/torchvision/datasets.html#svhn)
    * [PhotoTour](https://pytorch.org/docs/stable/torchvision/datasets.html#phototour)
    * [SBU](https://pytorch.org/docs/stable/torchvision/datasets.html#sbu)
    * [Flickr](https://pytorch.org/docs/stable/torchvision/datasets.html#flickr)
    * [VOC](https://pytorch.org/docs/stable/torchvision/datasets.html#voc)
    * [Cityscapes](https://pytorch.org/docs/stable/torchvision/datasets.html#cityscapes)
    * [SBD](https://pytorch.org/docs/stable/torchvision/datasets.html#sbd)
    * [USPS](https://pytorch.org/docs/stable/torchvision/datasets.html#usps)
    * [Kinetics-400](https://pytorch.org/docs/stable/torchvision/datasets.html#kinetics-400)
    * [HMDB51](https://pytorch.org/docs/stable/torchvision/datasets.html#hmdb51)
    * [UCF101](https://pytorch.org/docs/stable/torchvision/datasets.html#ucf101)
* [torchvision.io](https://pytorch.org/docs/stable/torchvision/io.html)
    * [Video](https://pytorch.org/docs/stable/torchvision/io.html#video)
* [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)
    * [Classification](https://pytorch.org/docs/stable/torchvision/models.html#classification)
    * [Semantic Segmentation](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation)
    * [Object Detection, Instance Segmentation and Person Keypoint Detection](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
    * [Video classification](https://pytorch.org/docs/stable/torchvision/models.html#video-classification)
* [torchvision.ops](https://pytorch.org/docs/stable/torchvision/ops.html)
* [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html)
    * [Transforms on PIL Image](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-pil-image)
    * [Transforms on torch.*Tensor](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor)
    * [Conversion Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#conversion-transforms)
    * [Generic Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#generic-transforms)
    * [Functional Transforms](https://pytorch.org/docs/stable/torchvision/transforms.html#functional-transforms)
* [torchvision.utils](https://pytorch.org/docs/stable/torchvision/utils.html)



`torchvision.``get_image_backend`()[[SOURCE]](https://pytorch.org/docs/stable/_modules/torchvision.html#get_image_backend)[](https://pytorch.org/docs/stable/torchvision/index.html#torchvision.get_image_backend)

Gets the name of the package used to load images


`torchvision.``set_image_backend`(*backend*)[[SOURCE]](https://pytorch.org/docs/stable/_modules/torchvision.html#set_image_backend)[](https://pytorch.org/docs/stable/torchvision/index.html#torchvision.set_image_backend)

Specifies the package used to load images.


Parameters

backend (string) – Name of the image backend. one of {‘PIL’, ‘accimage’}. The accimage package uses the Intel IPP library. It is generally faster than PIL, but does not support as many operations.
