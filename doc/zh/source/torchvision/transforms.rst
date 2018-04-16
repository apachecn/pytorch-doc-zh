torchvision.transforms
======================

.. currentmodule:: torchvision.transforms

Transforms (变换) 是常见的 image transforms (图像变换) .他们可以使用 :class:`Compose` 类以链在一起来进行操作.

.. autoclass:: Compose

PIL Image 上的变换
-----------------------------

.. autoclass:: Resize

.. autoclass:: Scale

.. autoclass:: CenterCrop

.. autoclass:: RandomCrop

.. autoclass:: RandomHorizontalFlip

.. autoclass:: RandomVerticalFlip

.. autoclass:: RandomResizedCrop

.. autoclass:: RandomSizedCrop

.. autoclass:: Grayscale

.. autoclass:: RandomGrayscale

.. autoclass:: FiveCrop

.. autoclass:: TenCrop

.. autoclass:: Pad

.. autoclass:: ColorJitter

torch.\*Tensor 上的变换
----------------------------

.. autoclass:: Normalize
	:members: __call__
	:special-members:


转换类型的变换
---------------------

.. autoclass:: ToTensor
	:members: __call__
	:special-members:

.. autoclass:: ToPILImage
	:members: __call__
	:special-members:

通用的变换
------------------

.. autoclass:: Lambda