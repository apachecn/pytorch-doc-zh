torchvision.datasets
====================

所有的数据集都是 :class:`torch.utils.data.Dataset` 类的子类,
也就是说, 他们内部都实现了 ``__getitem__`` 和 ``__len__`` 这两个方法.
同时, 他们也都可以传递给类 :class:`torch.utils.data.Dataset`,
它可以使用 ``torch.multiprocessing`` 工作器来并行的加载多个样本.

Example: ::
    
    imagenet_data = torchvision.datasets.ImageFolder('path/to/imagenet_root/')
    data_loader = torch.utils.data.DataLoader(imagenet_data, 
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=args.nThreads)
可用的数据集如下所示:

.. contents:: Datasets
    :local:

所有数据集都有几乎相似的 API, 它们有两个普通的参数:
``transform`` 和  ``target_transform`` 可分别的对输入和目标数据集进行变换.
-  ``transform``: 输入原始图片, 返回转换后的图片.
-  ``target_transform``: 输入为 target, 返回转换后的 target.

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

.. _COCO API: https://github.com/pdollar/coco/tree/master/PythonAPI


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

这可以通过一个 ``ImageFolder`` 数据集轻易实现.
该数据预处理过程如 `这里描述的 <https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset>`__ 所示

`这里是一个预处理示例 <https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py#L48-L62>`__.

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