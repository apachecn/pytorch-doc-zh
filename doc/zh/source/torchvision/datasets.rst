torchvision.datasets
====================

所有的数据集都是:class:`torch.utils.data.Dataset`类的子类，也就是说，
他们内部都实现了``__getitem__``和``__len__``这两个方法。
同时，他们也都可以传递给类:class:`torch.utils.data.Dataset`，而我们
可以通过类:class:`torch.utils.data.Dataset`以及``torch.multiprocessing``
来并行地读入多个样本。
例如： ::
    
    imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
    data_loader = torch.utils.data.DataLoader(imagenet_data, 
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=args.nThreads)

可用的数据集如下：

.. contents:: Datasets
    :local:

All the datasets have almost similar API. They all have two common arguments:
``transform`` and  ``target_transform`` to transform the input and target respectively.


.. currentmodule:: torchvision.datasets 


MNIST
~~~~~

.. autoclass:: MNIST

Fashion-MNIST
~~~~~~~~~~~~~

.. autoclass:: FashionMNIST

COCO
~~~~

.. 注意 ::
    需要预先安装 `COCO API to be installed`_
    
.. _COCO API to be installed 预先安装地址: https://github.com/pdollar/coco/tree/master/PythonAPI


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

这个可以通过一个``ImageFolder``数据集轻易实现。
数据预处理过程如下：
<https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset>`__

`示例程序 
<https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py#L48-L62>`__.

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
