# What is PyTorch?

It’s a Python-based scientific computing package targeted at two sets of audiences:

*   A replacement for NumPy to use the power of GPUs
*   a deep learning research platform that provides maximum flexibility and speed

## Getting Started

### Tensors

Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

```py
from __future__ import print_function
import torch

```

Construct a 5x3 matrix, uninitialized:

```py
x = torch.empty(5, 3)
print(x)

```

Out:

```py
tensor([[3.9855e-28, 4.5831e-41, 4.5183e-26],
        [4.5831e-41, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]])

```

Construct a randomly initialized matrix:

```py
x = torch.rand(5, 3)
print(x)

```

Out:

```py
tensor([[0.3753, 0.0231, 0.8850],
        [0.8283, 0.4600, 0.7222],
        [0.0634, 0.3449, 0.3077],
        [0.6987, 0.0143, 0.5651],
        [0.7482, 0.2355, 0.6162]])

```

Construct a matrix filled zeros and of dtype long:

```py
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

```

Out:

```py
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])

```

Construct a tensor directly from data:

```py
x = torch.tensor([5.5, 3])
print(x)

```

Out:

```py
tensor([5.5000, 3.0000])

```

or create a tensor based on an existing tensor. These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user

```py
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size

```

Out:

```py
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
tensor([[ 0.2854,  0.9206,  0.9174],
        [ 0.3367,  1.7474, -0.5835],
        [-1.4511,  0.1129, -0.7632],
        [ 1.2742,  0.6461,  0.7472],
        [ 0.0703,  0.1842, -0.7891]])

```

Get its size:

```py
print(x.size())

```

Out:

```py
torch.Size([5, 3])

```

Note

`torch.Size` is in fact a tuple, so it supports all tuple operations.

### Operations

There are multiple syntaxes for operations. In the following example, we will take a look at the addition operation.

Addition: syntax 1

```py
y = torch.rand(5, 3)
print(x + y)

```

Out:

```py
tensor([[ 0.6824,  1.6858,  1.6162],
        [ 0.7983,  2.4377, -0.4856],
        [-0.9769,  0.7002, -0.7210],
        [ 1.7927,  1.0902,  1.2557],
        [ 0.3852,  0.7108, -0.1949]])

```

Addition: syntax 2

```py
print(torch.add(x, y))

```

Out:

```py
tensor([[ 0.6824,  1.6858,  1.6162],
        [ 0.7983,  2.4377, -0.4856],
        [-0.9769,  0.7002, -0.7210],
        [ 1.7927,  1.0902,  1.2557],
        [ 0.3852,  0.7108, -0.1949]])

```

Addition: providing an output tensor as argument

```py
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

```

Out:

```py
tensor([[ 0.6824,  1.6858,  1.6162],
        [ 0.7983,  2.4377, -0.4856],
        [-0.9769,  0.7002, -0.7210],
        [ 1.7927,  1.0902,  1.2557],
        [ 0.3852,  0.7108, -0.1949]])

```

Addition: in-place

```py
# adds x to y
y.add_(x)
print(y)

```

Out:

```py
tensor([[ 0.6824,  1.6858,  1.6162],
        [ 0.7983,  2.4377, -0.4856],
        [-0.9769,  0.7002, -0.7210],
        [ 1.7927,  1.0902,  1.2557],
        [ 0.3852,  0.7108, -0.1949]])

```

Note

Any operation that mutates a tensor in-place is post-fixed with an `_`. For example: `x.copy_(y)`, `x.t_()`, will change `x`.

You can use standard NumPy-like indexing with all bells and whistles!

```py
print(x[:, 1])

```

Out:

```py
tensor([0.9206, 1.7474, 0.1129, 0.6461, 0.1842])

```

Resizing: If you want to resize/reshape tensor, you can use `torch.view`:

```py
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

```

Out:

```py
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

```

If you have a one element tensor, use `.item()` to get the value as a Python number

```py
x = torch.randn(1)
print(x)
print(x.item())

```

Out:

```py
tensor([0.1011])
0.10109155625104904

```

**Read later:**

> 100+ Tensor operations, including transposing, indexing, slicing, mathematical operations, linear algebra, random numbers, etc., are described [here](https://pytorch.org/docs/torch).

## NumPy Bridge

Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

The Torch Tensor and NumPy array will share their underlying memory locations, and changing one will change the other.

### Converting a Torch Tensor to a NumPy Array

```py
a = torch.ones(5)
print(a)

```

Out:

```py
tensor([1., 1., 1., 1., 1.])

```

```py
b = a.numpy()
print(b)

```

Out:

```py
[1\. 1\. 1\. 1\. 1.]

```

See how the numpy array changed in value.

```py
a.add_(1)
print(a)
print(b)

```

Out:

```py
tensor([2., 2., 2., 2., 2.])
[2\. 2\. 2\. 2\. 2.]

```

### Converting NumPy Array to Torch Tensor

See how changing the np array changed the Torch Tensor automatically

```py
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

```

Out:

```py
[2\. 2\. 2\. 2\. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

```

All the Tensors on the CPU except a CharTensor support converting to NumPy and back.

## CUDA Tensors

Tensors can be moved onto any device using the `.to` method.

```py
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

```

Out:

```py
tensor([1.1011], device='cuda:0')
tensor([1.1011], dtype=torch.float64)

```

**Total running time of the script:** ( 0 minutes 6.512 seconds)

[`Download Python source code: tensor_tutorial.py`](../../_downloads/092fba3c36cb2ab226bfdaa78248b310/tensor_tutorial.py)[`Download Jupyter notebook: tensor_tutorial.ipynb`](../../_downloads/3c2b25b8a9f72db7780a6bf9b5fc9f62/tensor_tutorial.ipynb)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.readthedocs.io)

