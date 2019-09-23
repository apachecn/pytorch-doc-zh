# torchvision

的 `torchvision`包由流行的数据集，模型体系结构，以及用于计算机视觉共同图像变换。

Package Reference

  * [ torchvision.datasets ](datasets.html)
    * [ MNIST ](datasets.html#mnist)
    * [时装-MNIST ](datasets.html#fashion-mnist)
    * [ KMNIST ](datasets.html#kmnist)
    * [ EMNIST ](datasets.html#emnist)
    * [ QMNIST ](datasets.html#qmnist)
    * [ FakeData ](datasets.html#fakedata)
    * [ COCO ](datasets.html#coco)
    * [ LSUN ](datasets.html#lsun)
    * [ ImageFolder ](datasets.html#imagefolder)
    * [ DatasetFolder ](datasets.html#datasetfolder)
    * [ ImageNet ](datasets.html#imagenet)
    * [ CIFAR ](datasets.html#cifar)
    * [ STL10 ](datasets.html#stl10)
    * [ SVHN ](datasets.html#svhn)
    * [ PhotoTour ](datasets.html#phototour)
    * [ SBU ](datasets.html#sbu)
    * [的Flickr ](datasets.html#flickr)
    * [ VOC ](datasets.html#voc)
    * [都市风景](datasets.html#cityscapes)
    * [ SBD ](datasets.html#sbd)
    * [ USPS ](datasets.html#usps)
    * [动力学-400 ](datasets.html#kinetics-400)
    * [ HMDB51 ](datasets.html#hmdb51)
    * [ UCF101 ](datasets.html#ucf101)
  * [ torchvision.io ](io.html)
    * [HTG0视频
  * [ torchvision.models ](models.html)
    * [分类](models.html#classification)
    * [语义分割](models.html#semantic-segmentation)
    * [对象检测，实例分割和Person关键点检测](models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
    * [HTG0视频分类
  * [ torchvision.ops ](ops.html)
  * [ torchvision.transforms ](transforms.html)
    * [上PIL图像变换](transforms.html#transforms-on-pil-image)
    * [来变换的Torch 。*张量](transforms.html#transforms-on-torch-tensor)
    * [转换变换](transforms.html#conversion-transforms)
    * [HTG0】通用变换
    * [功能变换](transforms.html#functional-transforms)
  * [ torchvision.utils ](utils.html)

`torchvision.``get_image_backend`()[[source]](../_modules/torchvision.html#get_image_backend)

    

获取包的用于加载图像的名称

`torchvision.``set_image_backend`( _backend_
)[[source]](../_modules/torchvision.html#set_image_backend)

    

指定用于加载图像包。

Parameters

    

**后端** （ _串_ ） - 图像后端的名称。 {“PIL”，“accimage”}中的一个。的`accimage
`软件包使用英特尔IPP库。它通常比PIL快，但不支持尽可能多的操作。

[Next ![](../_static/images/chevron-right-orange.svg)](datasets.html
"torchvision.datasets") [![](../_static/images/chevron-right-orange.svg)
Previous](../__config__.html "torch.__config__")

* * *

©版权所有2019年，Torch 贡献者。