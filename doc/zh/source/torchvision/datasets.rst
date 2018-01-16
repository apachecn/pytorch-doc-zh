torchvision.datasets
====================
torchvision.datasets中包含了以下数据集

.. contents:: Datasets
    :local:


Datasets 拥有以下的API:
__getitem__ 和 __len__

由于以上Datasets都是 torch.utils.data.Dataset的子类，所以他们也
可以通过torch.utils.data.DataLoader使用多线程(python的多进程).

举例说明: ::
    
    imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
    data_loader = torch.utils.data.DataLoader(imagenet_data, 
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=args.nThreads)

在构造函数中，不同的数据集直接的构造函数会有些许不同, 但是他们都拥有下面的 keyword 参数:
``transform``: 输入原始图片，返回转换后的图片.
``target_transform``: 输入为 target, 返回转换后的 target.


.. currentmodule:: torchvision.datasets 

MNIST
~~~~~

.. autoclass:: MNIST

Fashion-MNIST
~~~~~~~~~~~~~

.. autoclass:: FashionMNIST

COCO
~~~~

.. note ::
    需要安装 `COCO API`_

.. _COCO API to be installed: https://github.com/pdollar/coco/tree/master/PythonAPI


Captions
^^^^^^^^

.. autoclass:: CocoCaptions
  :members: __getitem__
  :special-members:


Detection
^^^^^^^^^

.. autoclass:: CocoDetection
  :members: __getitem__
  :special-members:

LSUN
~~~~

.. autoclass:: LSUN
  :members: __getitem__
  :special-members:

ImageFolder
~~~~~~~~~~~

.. autoclass:: ImageFolder
  :members: __getitem__
  :special-members:


Imagenet-12
~~~~~~~~~~~

This should simply be implemented with an ``ImageFolder`` dataset.
The data is preprocessed `as described
here <https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset>`__

`Here is an
example <https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py#L48-L62>`__.

CIFAR
~~~~~

.. autoclass:: CIFAR10
  :members: __getitem__
  :special-members:

.. autoclass:: CIFAR100

STL10
~~~~~


.. autoclass:: STL10
  :members: __getitem__
  :special-members:

SVHN
~~~~~


.. autoclass:: SVHN
  :members: __getitem__
  :special-members:

PhotoTour
~~~~~~~~~


.. autoclass:: PhotoTour
  :members: __getitem__
  :special-members: