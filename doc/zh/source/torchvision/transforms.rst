torchvision.transforms
======================

.. currentmodule:: torchvision.transforms

Transforms are common image transforms. They can be chained together using :class:`Compose`

.. autoclass:: Compose

Transforms on PIL Image
-----------------------

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

Transforms on torch.\*Tensor
----------------------------

.. autoclass:: Normalize
	:members: __call__
	:special-members:


Conversion Transforms
---------------------

.. autoclass:: ToTensor
	:members: __call__
	:special-members:

.. autoclass:: ToPILImage
	:members: __call__
	:special-members:

Generic Transforms
------------------

.. autoclass:: Lambda