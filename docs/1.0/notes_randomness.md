

# Reproducibility

Completely reproducible results are not guaranteed across PyTorch releases, individual commits or different platforms. Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.

However, in order to make computations deterministic on your specific problem on one specific platform and PyTorch release, there are a couple of steps to take.

There are two pseudorandom number generators involved in PyTorch, which you will need to seed manually to make runs reproducible. Furthermore, you should ensure that all other libraries your code relies on an which use random numbers also use a fixed seed.

## PyTorch

You can use [`torch.manual_seed()`](../torch.html#torch.manual_seed "torch.manual_seed") to seed the RNG for all devices (both CPU and CUDA):

```py
import torch
torch.manual_seed(0)

```

There are some PyTorch functions that use CUDA functions that can be a source of non-determinism. One class of such CUDA functions are atomic operations, in particular `atomicAdd`, where the order of parallel additions to the same value is undetermined and, for floating-point variables, a source of variance in the result. PyTorch functions that use `atomicAdd` in the forward include [`torch.Tensor.index_add_()`](../tensors.html#torch.Tensor.index_add_ "torch.Tensor.index_add_"), [`torch.Tensor.scatter_add_()`](../tensors.html#torch.Tensor.scatter_add_ "torch.Tensor.scatter_add_"), [`torch.bincount()`](../torch.html#torch.bincount "torch.bincount").

A number of operations have backwards that use `atomicAdd`, in particular [`torch.nn.functional.embedding_bag()`](../nn.html#torch.nn.functional.embedding_bag "torch.nn.functional.embedding_bag"), [`torch.nn.functional.ctc_loss()`](../nn.html#torch.nn.functional.ctc_loss "torch.nn.functional.ctc_loss") and many forms of pooling, padding, and sampling. There currently is no simple way of avoiding non-determinism in these functions.

## CuDNN

When running on the CuDNN backend, two further options must be set:

```py
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

```

Warning

Deterministic mode can have a performance impact, depending on your model.

## Numpy

If you or any of the libraries you are using rely on Numpy, you should seed the Numpy RNG as well. This can be done with:

```py
import numpy as np
np.random.seed(0)

```

