# torchvision.datasets

> 译者：[BXuan694](https://github.com/BXuan694)

所有的数据集都是[`torch.utils.data.Dataset`](../data.html#torch.utils.data.Dataset "torch.utils.data.Dataset")的子类， 即：它们实现了`__getitem__`和`__len__`方法。因此，它们都可以传递给[`torch.utils.data.DataLoader`](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader")，进而通过`torch.multiprocessing`实现批数据的并行化加载。例如：

```py
imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)

```

目前为止，收录的数据集包括：

数据集

*   [MNIST](#mnist)
*   [Fashion-MNIST](#fashion-mnist)
*   [EMNIST](#emnist)
*   [COCO](#coco)
    *   [Captions](#captions)
    *   [Detection](#detection)
*   [LSUN](#lsun)
*   [ImageFolder](#imagefolder)
*   [DatasetFolder](#datasetfolder)
*   [Imagenet-12](#imagenet-12)
*   [CIFAR](#cifar)
*   [STL10](#stl10)
*   [SVHN](#svhn)
*   [PhotoTour](#phototour)
*   [SBU](#sbu)
*   [Flickr](#flickr)
*   [VOC](#voc)

以上数据集的接口基本上很相近。它们至少包括两个公共的参数`transform`和`target_transform`，以便分别对输入和和目标做变换。

```py
class torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
```

[MNIST](http://yann.lecun.com/exdb/mnist/)数据集。

 
参数： 

*   **root**(_string_）– 数据集的根目录，其中存放`processed/training.pt`和`processed/test.pt`文件。
*   **train**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True，从`training.pt`创建数据集，否则从`test.pt`创建。
*   **download**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True, 从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.RandomCrop`。
*   **target_transform** (_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。



```py
class torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)
```

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)数据集。

 
参数： 

*   **root**(_string_）– 数据集的根目录，其中存放`processed/training.pt`和`processed/test.pt`文件。
*   **train**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True，从`training.pt`创建数据集，否则从`test.pt`创建。
*   **download**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.RandomCrop`。
*   **target_transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。



```py
class torchvision.datasets.EMNIST(root, split, **kwargs)
```

[EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset/)数据集。

 
参数: 

*   **root**(_string_）– 数据集的根目录，其中存放`processed/training.pt`和`processed/test.pt`文件。
*   **split**(_string_）– 该数据集分成6种：`byclass`，`bymerge`，`balanced`，`letters`，`digits`和`mnist`。这个参数指定了选择其中的哪一种。
*   **train**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True，从`training.pt`创建数据集，否则从`test.pt`创建。
*   **download**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True, 从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.RandomCrop`。
*   **target_transform**(_可被调用_ _,_ _可选_) – 一种函数或变换，输入目标，进行变换。



注意：

以下要求预先[安装COCO API](https://github.com/pdollar/coco/tree/master/PythonAPI)。

```py
class torchvision.datasets.CocoCaptions(root, annFile, transform=None, target_transform=None)
```

[MS Coco Captions](http://mscoco.org/dataset/#captions-challenge2015)数据集。

 
参数： 

*   **root**(_string_）– 下载数据的目标目录。
*   **annFile**(_string_）– json标注文件的路径。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.ToTensor`。
*   **target_transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。



示例

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

输出：

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

 
| 参数： | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | 元组(image, target)，其中target是列表类型，包含了对图片image的描述。 |
| --- | --- |
| 返回类型： | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.CocoDetection(root, annFile, transform=None, target_transform=None)
```

[MS Coco Detection](http://mscoco.org/dataset/#detections-challenge2016)数据集。

 
参数： 

*   **root**(_string_）– 下载数据的目标目录。
*   **annFile**(_string_）– json标注文件的路径。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.ToTensor`。
*   **target_transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。



```py
__getitem__(index)
```

 
| 参数: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | 元组(image, target)，其中target是`coco.loadAnns`返回的对象。 |
| --- | --- |
| 返回类型： | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.LSUN(root, classes='train', transform=None, target_transform=None)
```

[LSUN](http://lsun.cs.princeton.edu)数据集。

 
参数：

*   **root**(_string_）– 存放数据文件的根目录。
*   **classes**(_string_ _或_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")）– {‘train’, ‘val’, ‘test’}之一，或要加载类别的列表，如[‘bedroom_train’, ‘church_train’]。
*   **transform**(_可被调用_ _,_ _可选_) – 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.RandomCrop`。
*   **target_transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。



```py
__getitem__(index)
```

 
| 参数： | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | 元组(image, target)，其中target是目标类别的索引。 |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)
```

一种通用数据加载器，其图片应该按照如下的形式保存：

```py
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png

```

 
参数： 

*   **root**(_string_）– 根目录路径。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.RandomCrop`。
*   **target_transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。
*   **loader** – 一种函数，可以由给定的路径加载图片。



```py
__getitem__(index)
```

 
| 参数： | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | (sample, target)，其中target是目标类的类索引。 |
| --- | --- |
| 返回类型： | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.DatasetFolder(root, loader, extensions, transform=None, target_transform=None)
```

一种通用数据加载器，其数据应该按照如下的形式保存：

```py
root/class_x/xxx.ext
root/class_x/xxy.ext
root/class_x/xxz.ext

root/class_y/123.ext
root/class_y/nsdf3.ext
root/class_y/asd932_.ext

```

 
参数: 

*   **root**(_string_）– 根目录路径。
*   **loader**(_可被调用_）– 一种函数，可以由给定的路径加载数据。
*   **extensions**([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[__string__]_）– 列表，包含允许的扩展。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入数据，返回变换之后的数据。如：对于图片有`transforms.RandomCrop`。
*   **target_transform** – 一种函数或变换，输入目标，进行变换。



```py
__getitem__(index)
```

 
| 参数： | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | (sample, target)，其中target是目标类的类索引. |
| --- | --- |
| 返回类型： | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

这个类可以很容易地实现`ImageFolder`数据集。数据预处理见[此处](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)。

[示例](https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126)。

```py
class torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
```

[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)数据集。

 
参数：

*   **root**(_string_）– 数据集根目录，要么其中应存在`cifar-10-batches-py`文件夹，要么当download设置为True时`cifar-10-batches-py`文件夹保存在此处。
*   **train**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True, 从训练集中创建，否则从测试集中创建。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.RandomCrop`。
*   **target_transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。
*   **download**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。



```py
__getitem__(index)
```

 
| 参数： | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | (image, target)，其中target是目标类的类索引。 |
| --- | --- |
| 返回类型： | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)
```

[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)数据集。

这是`CIFAR10`数据集的一个子集。

```py
class torchvision.datasets.STL10(root, split='train', transform=None, target_transform=None, download=False)
```

[STL10](https://cs.stanford.edu/~acoates/stl10/)数据集。

 
参数：

*   **root**(_string_）– 数据集根目录，应该包含`stl10_binary`文件夹。
*   **split**(_string_）– {‘train’, ‘test’, ‘unlabeled’, ‘train+unlabeled’}之一，选择相应的数据集。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.RandomCrop`。
*   **target_transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。
*   **download**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。



```py
__getitem__(index)
```

 
| 参数： | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | (image, target)，其中target应是目标类的类索引。 |
| --- | --- |
| 返回类型： | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.SVHN(root, split='train', transform=None, target_transform=None, download=False)
```

[SVHN](http://ufldl.stanford.edu/housenumbers/)数据集。注意：SVHN数据集将`10`指定为数字`0`的标签。然而，这里我们将`0`指定为数字`0`的标签以兼容PyTorch的损失函数，因为损失函数要求类标签在`[0, C-1]`的范围内。

 
参数：

*   **root**(_string_）– 数据集根目录，应包含`SVHN`文件夹。
*   **split**(_string_）– {‘train’, ‘test’, ‘extra’}之一，相应的数据集会被选择。‘extra’是extra训练集。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：`transforms.RandomCrop`。
*   **target_transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入目标，进行变换。
*   **download**([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。



```py
__getitem__(index)
```

 
| 参数： | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | (image, target)，其中target是目标类的类索引。 |
| --- | --- |
| 返回类型： | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.PhotoTour(root, name, train=True, transform=None, download=False)
```

[Learning Local Image Descriptors Data](http://phototour.cs.washington.edu/patches/default.htm)数据集。

 
参数：

*   **root**(_string_）– 保存图片的根目录。
*   **name**(_string_）– 要加载的数据集。
*   **transform**(_可被调用_ _,_ _可选_）– 一种函数或变换，输入PIL图片，返回变换之后的数据。
*   **download** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。



```py
__getitem__(index)
```

 
| 参数： | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – 索引 |
| --- | --- |
| 返回： | (data1, data2, matches) |
| --- | --- |
| 返回类型： | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

