# torchvision.datasets

All datasets are subclasses of [`torch.utils.data.Dataset`](../data.html#torch.utils.data.Dataset "torch.utils.data.Dataset") i.e, they have `__getitem__` and `__len__` methods implemented. Hence, they can all be passed to a [`torch.utils.data.DataLoader`](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") which can load multiple samples parallelly using `torch.multiprocessing` workers. For example:

```py
imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)

```

The following datasets are available:

Datasets

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

All the datasets have almost similar API. They all have two common arguments: `transform` and `target_transform` to transform the input and target respectively.

```py
class torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
```

[MNIST](http://yann.lecun.com/exdb/mnist/) Dataset.

 
Parameters: 

*   **root** (_string_) – Root directory of dataset where `processed/training.pt` and `processed/test.pt` exist.
*   **train** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, creates dataset from `training.pt`, otherwise from `test.pt`.
*   **download** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.



```py
class torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)
```

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) Dataset.

 
Parameters: 

*   **root** (_string_) – Root directory of dataset where `processed/training.pt` and `processed/test.pt` exist.
*   **train** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, creates dataset from `training.pt`, otherwise from `test.pt`.
*   **download** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.



```py
class torchvision.datasets.EMNIST(root, split, **kwargs)
```

[EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset/) Dataset.

 
Parameters: 

*   **root** (_string_) – Root directory of dataset where `processed/training.pt` and `processed/test.pt` exist.
*   **split** (_string_) – The dataset has 6 different splits: `byclass`, `bymerge`, `balanced`, `letters`, `digits` and `mnist`. This argument specifies which one to use.
*   **train** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, creates dataset from `training.pt`, otherwise from `test.pt`.
*   **download** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.



Note

These require the [COCO API to be installed](https://github.com/pdollar/coco/tree/master/PythonAPI)

```py
class torchvision.datasets.CocoCaptions(root, annFile, transform=None, target_transform=None)
```

[MS Coco Captions](http://mscoco.org/dataset/#captions-challenge2015) Dataset.

 
Parameters: 

*   **root** (_string_) – Root directory where images are downloaded to.
*   **annFile** (_string_) – Path to json annotation file.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.ToTensor`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.



Example

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

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | Tuple (image, target). target is a list of captions for the image. |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.CocoDetection(root, annFile, transform=None, target_transform=None)
```

[MS Coco Detection](http://mscoco.org/dataset/#detections-challenge2016) Dataset.

 
Parameters: 

*   **root** (_string_) – Root directory where images are downloaded to.
*   **annFile** (_string_) – Path to json annotation file.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.ToTensor`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.



```py
__getitem__(index)
```

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | Tuple (image, target). target is the object returned by `coco.loadAnns`. |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.LSUN(root, classes='train', transform=None, target_transform=None)
```

[LSUN](http://lsun.cs.princeton.edu) dataset.

 
Parameters: 

*   **root** (_string_) – Root directory for the database files.
*   **classes** (_string_ _or_ [_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")) – One of {‘train’, ‘val’, ‘test’} or a list of categories to load. e,g. [‘bedroom_train’, ‘church_train’].
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.



```py
__getitem__(index)
```

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | Tuple (image, target) where target is the index of the target category. |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)
```

A generic data loader where the images are arranged in this way:

```py
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png

```

 
Parameters: 

*   **root** (_string_) – Root directory path.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.
*   **loader** – A function to load an image given its path.



```py
__getitem__(index)
```

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | (sample, target) where target is class_index of the target class. |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.DatasetFolder(root, loader, extensions, transform=None, target_transform=None)
```

A generic data loader where the samples are arranged in this way:

```py
root/class_x/xxx.ext
root/class_x/xxy.ext
root/class_x/xxz.ext

root/class_y/123.ext
root/class_y/nsdf3.ext
root/class_y/asd932_.ext

```

 
Parameters: 

*   **root** (_string_) – Root directory path.
*   **loader** (_callable_) – A function to load a sample given its path.
*   **extensions** ([_list_](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.7)")_[__string__]_) – A list of allowed extensions.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in a sample and returns a transformed version. E.g, `transforms.RandomCrop` for images.
*   **target_transform** – A function/transform that takes in the target and transforms it.



```py
__getitem__(index)
```

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | (sample, target) where target is class_index of the target class. |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

This should simply be implemented with an `ImageFolder` dataset. The data is preprocessed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

[Here is an example](https://github.com/pytorch/examples/blob/e0d33a69bec3eb4096c265451dbb85975eb961ea/imagenet/main.py#L113-L126).

```py
class torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
```

[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset.

 
Parameters: 

*   **root** (_string_) – Root directory of dataset where directory `cifar-10-batches-py` exists or will be saved to if download is set to True.
*   **train** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If True, creates dataset from training set, otherwise creates from test set.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.
*   **download** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.



```py
__getitem__(index)
```

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | (image, target) where target is index of the target class. |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.CIFAR100(root, train=True, transform=None, target_transform=None, download=False)
```

[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset.

This is a subclass of the `CIFAR10` Dataset.

```py
class torchvision.datasets.STL10(root, split='train', transform=None, target_transform=None, download=False)
```

[STL10](https://cs.stanford.edu/~acoates/stl10/) Dataset.

 
Parameters: 

*   **root** (_string_) – Root directory of dataset where directory `stl10_binary` exists.
*   **split** (_string_) – One of {‘train’, ‘test’, ‘unlabeled’, ‘train+unlabeled’}. Accordingly dataset is selected.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.
*   **download** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.



```py
__getitem__(index)
```

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | (image, target) where target is index of the target class. |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.SVHN(root, split='train', transform=None, target_transform=None, download=False)
```

[SVHN](http://ufldl.stanford.edu/housenumbers/) Dataset. Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset, we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which expect the class labels to be in the range `[0, C-1]`

 
Parameters: 

*   **root** (_string_) – Root directory of dataset where directory `SVHN` exists.
*   **split** (_string_) – One of {‘train’, ‘test’, ‘extra’}. Accordingly dataset is selected. ‘extra’ is Extra training set.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version. E.g, `transforms.RandomCrop`
*   **target_transform** (_callable__,_ _optional_) – A function/transform that takes in the target and transforms it.
*   **download** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.



```py
__getitem__(index)
```

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | (image, target) where target is index of the target class. |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

```py
class torchvision.datasets.PhotoTour(root, name, train=True, transform=None, download=False)
```

[Learning Local Image Descriptors Data](http://phototour.cs.washington.edu/patches/default.htm) Dataset.

 
Parameters: 

*   **root** (_string_) – Root directory where images are.
*   **name** (_string_) – Name of the dataset to load.
*   **transform** (_callable__,_ _optional_) – A function/transform that takes in an PIL image and returns a transformed version.
*   **download** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _optional_) – If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.



```py
__getitem__(index)
```

 
| Parameters: | **index** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – Index |
| --- | --- |
| Returns: | (data1, data2, matches) |
| --- | --- |
| Return type: | [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.7)") |
| --- | --- |

