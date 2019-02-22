# torchvision.datasets

> 译者：[@那伊抹微笑](https://github.com/wangyangting)、@dawenzi123、[@LeeGeong](https://github.com/LeeGeong)、@liandongze
> 
> 校对者：[@咸鱼](https://github.com/Watermelon233)

所有的数据集都是 [`torch.utils.data.Dataset`](../data.html#torch.utils.data.Dataset "torch.utils.data.Dataset") 类的子类, 也就是说, 他们内部都实现了 `__getitem__` 和 `__len__` 这两个方法. 同时, 他们也都可以传递给类 [`torch.utils.data.Dataset`](../data.html#torch.utils.data.Dataset "torch.utils.data.Dataset"), 它可以使用 `torch.multiprocessing` 工作器来并行的加载多个样本.

示例：

```py
imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)

```

可用的数据集如下所示:

Datasets

*   MNIST
*   Fashion-MNIST
*   COCO
    *   Captions
    *   Detection
*   LSUN
*   ImageFolder
*   Imagenet-12
*   CIFAR
*   STL10
*   SVHN
*   PhotoTour

所有数据集都有几乎相似的 API, 它们有两个普通的参数: `transform` 和 `target_transform` 可分别的对输入和目标数据集进行变换. - `transform`: 输入原始图片, 返回转换后的图片. - `target_transform`: 输入为 target, 返回转换后的 target.

## MNIST

```py
class torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
```

[MNIST](http://yann.lecun.com/exdb/mnist/) Dataset.

参数：

*   `root (string)` – `processed/training.pt` 和 `processed/test.pt` 存在的主目录.
*   `train (bool, 可选)` – 如果 True, 数据来自训练集 `training.pt` , 如果 False, 数据来自测试集 `test.pt` .
*   `download (bool, 可选)` – 如果 true, 就从网上下载数据集并且放到 root 目录下. 如果数据集已经下载, 那么不会再次下载.
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.RandomCrop`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.



## Fashion-MNIST

```py
class torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)
```

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) Dataset.

参数：

*   `root (string)` – `processed/training.pt` 和 `processed/test.pt` 存在的主目录.
*   `train (bool, 可选)` – 如果 True, 数据来自训练集 `training.pt` , 如果 False, 数据来自测试集 `test.pt` .
*   `download (bool, 可选)` – 如果 true, 就从网上下载数据集并且放到 root 目录下. 如果数据集已经下载, 那么不会再次下载.
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.RandomCrop`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.



## COCO

注解：

需要安装 [COCO API](https://github.com/pdollar/coco/tree/master/PythonAPI)

### Captions

```py
class torchvision.datasets.CocoCaptions(root, annFile, transform=None, target_transform=None)
```

[MS Coco Captions](http://mscoco.org/dataset/#captions-challenge2015) Dataset.

参数：

*   `root (string)` – 数据集下载存放的主目录.
*   `annFile (string)` – json 注释文件存放的路径
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.ToTensor`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.



示例：

```py
import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = 'dir where images are',
                        annFile = 'json annotation file',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

print("Image Size: ", img.size())
print(target)

```

Output:

```py
Number of samples: 82783
Image Size: (3L, 427L, 640L)
[u'A plane emitting smoke stream flying over a mountain.',
u'A plane darts across a bright blue sky behind a mountain covered in snow',
u'A plane leaves a contrail above the snowy mountain top.',
u'A mountain that has a plane flying overheard in the distance.',
u'A mountain view with a plume of smoke in the background']

```

```py
__getitem__(index)
```

参数：`index (int)` – Index

返回值：`Tuple (image, target)`. 目标是一个图像标注的列表.

返回类型：`tuple`

### Detection

```py
class torchvision.datasets.CocoDetection(root, annFile, transform=None, target_transform=None)
```

[MS Coco Detection](http://mscoco.org/dataset/#detections-challenge2016) Dataset.

参数：

*   `root (string)` – 数据集下载存放的主目录.
*   `annFile (string)` – json 注释文件存放的路径
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.ToTensor`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.



```py
__getitem__(index)
```

参数：`index (int)` – Index

返回值：`Tuple (image, target)`. 目标是由 `coco.loadAnns` 返回的对象.

返回类型：`tuple`

## LSUN

```py
class torchvision.datasets.LSUN(db_path, classes='train', transform=None, target_transform=None)
```

[LSUN](http://lsun.cs.princeton.edu) dataset.

参数：

*   `db_path (string)` – 数据集文件存放的主目录.
*   `classes (string 或 list)` – {‘train’, ‘val’, ‘test’} 中的一个, 或者是一个要载入种类的列表. e,g. [‘bedroom_train’, ‘church_train’].
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.RandomCrop`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.



```py
__getitem__(index)
```

参数：`index (int)` – Index

返回值：`Tuple (image, target)` 目标是目标类别的索引.

返回类型：`tuple`

## ImageFolder

```py
class torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader at 0x432aa28>)
```

一个通用的数据加载器, 数据集中的数据以以下方式组织:

```py
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png

```

参数：

*   `root (string)` – 主目录.
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.RandomCrop`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.
*   `loader` – 一个从给定路径载入图像的函数.



```py
__getitem__(index)
```

参数：`index (int)` – Index

返回值：`(image, target)` 目标是目标类别的class_index.

返回类型：`tuple`

## Imagenet-12

这可以通过一个 `ImageFolder` 数据集轻易实现. 该数据预处理过程如 [这里描述的](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) 所示

[这里是一个预处理示例](https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py#L48-L62).

## CIFAR

```py
class torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
```

[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset.

参数：

*   `root (string)` – `cifar-10-batches-py` 存在的主目录.
*   `train (bool, 可选)` – 如果 True, 数据来自训练集, 如果 False, 数据来自测试集.
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.RandomCrop`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.
*   `download (bool, 可选)` – 如果 true, 就从网上下载数据集并且放到 root 目录下. 如果数据集已经下载, 那么不会再次下载.



```py
__getitem__(index)
```

参数：`index (int)` – Index

返回值：`(image, target)` 目标是目标分类的索引.

返回类型：`tuple`

```py
class torchvision.datasets.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)
```

[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset.

`CIFAR10` Dataset 的一个子类.

## STL10

```py
class torchvision.datasets.STL10(root, split='train', transform=None, target_transform=None, download=False)
```

[STL10](https://cs.stanford.edu/~acoates/stl10/) Dataset.

参数：

*   `root (string)` – `stl10_binary`数据集存放的主目录.
*   `split (string)` – {‘train’, ‘test’, ‘unlabeled’, ‘train+unlabeled’} 中的一个. 它是根据数据集选择的.
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.RandomCrop`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.
*   `download (bool, 可选)` – 如果 true, 就从网上下载数据集并且放到 root 目录下. 如果数据集已经下载, 那么不会再次下载.



```py
__getitem__(index)
```

参数：`index (int)` – Index

返回值：`(image, target)` 目标是目标类的索引.

返回类型：`tuple`

## SVHN

```py
class torchvision.datasets.SVHN(root, split='train', transform=None, target_transform=None, download=False)
```

[SVHN](http://ufldl.stanford.edu/housenumbers/) Dataset. Note: 原始的 SVHN 数据集把标签 `10` 分给了数字 `0`. 然而在这个数据集, 我们把标签 `0` 分给了数字 `0` 以便 和 PyTorch 的损失函数不产生冲突, 它期待的类标签的范围是 `[0, C-1]`.

参数：

*   `root (string)` – `SVHN`数据集存放的主目录.
*   `split (string)` – {‘train’, ‘test’, ‘extra’} 中的一个. 它是根据数据集选择的. ‘extra’ 是一个额外的训练集.
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本. E.g, `transforms.RandomCrop`
*   `target_transform (callable, 可选)` – 一个 transform 函数, 输入 target 并且 转换它.
*   `download (bool, 可选)` – 如果 true, 就从网上下载数据集并且放到 root 目录下. 如果数据集已经下载, 那么不会再次下载.



```py
__getitem__(index)
```

参数：`index (int)` – Index

返回值：`(image, target)` 目标是目标类的索引.

返回类型：`tuple`

## PhotoTour

```py
class torchvision.datasets.PhotoTour(root, name, train=True, transform=None, download=False)
```

[Learning Local Image Descriptors Data](http://phototour.cs.washington.edu/patches/default.htm) Dataset.

参数：

*   `root (string)` – 图像存放的主目录.
*   `name (string)` – 载入的数据集的名字.
*   `transform (callable, 可选)` – 一个 transform 函数, 它输入 PIL image 并且返回 转换后的版本.
*   `download (bool, 可选)` – 如果 true, 就从网上下载数据集并且放到 root 目录下. 如果数据集已经下载, 那么不会再次下载.



```py
__getitem__(index)
```

参数：`index (int)` – Index

返回值：`(data1, data2, matches)`

返回类型：`tuple`