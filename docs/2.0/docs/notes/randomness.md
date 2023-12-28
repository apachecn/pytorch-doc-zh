# 再现性 [¶](#reproducibility "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/randomness>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/randomness.html>


 不保证在 PyTorch 版本、单独提交或不同平台上获得完全可重现的结果。此外，即使使用相同的种子，CPU 和 GPU 执行之间的结果也可能无法重现。


 但是，您可以采取一些步骤来限制特定平台、设备和 PyTorch 版本的不确定性行为来源的数量。首先，您可以控制可能导致应用程序的多次执行行为不同的随机性来源。其次，您可以配置 PyTorch 以避免对某些操作使用不确定性算法，以便在给定相同输入的情况下多次调用这些操作将产生相同的结果。


!!! warning "警告"

    确定性操作通常比非确定性操作慢，因此模型的单次运行性能可能会降低。然而，确定性可以通过促进实验、调试和回归测试来节省开发时间。


## 控制随机源 [¶](#controlling-sources-of-randomness "此标题的永久链接")


### PyTorch 随机数生成器 [¶](#pytorch-random-number-generator "此标题的永久链接")


 您可以使用 [`torch.manual_seed()`](../generated/torch.manual_seed.html#torch.manual_seed "torch.manual_seed") 为所有设备(CPU 和 CUDA)播种 RNG：


```
import torch
torch.manual_seed(0)

```


 某些 PyTorch 操作可能在内部使用随机数。例如， [`torch.svd_lowrank()`](../generated/torch.svd_lowrank.html#torch.svd_lowrank "torch.svd_lowrank") 就是这样做的。因此，使用相同的输入参数连续多次调用它可能会产生不同的结果。但是，只要 [`torch.manual_seed()`](../generated/torch.manual_seed.html#torch.manual_seed "torch.manual_seed") 在应用程序开始时设置为常量，并且所有其他不确定性的来源已被消除，每次应用程序在相同的环境中运行时都会生成相同的随机数序列。


 通过将 [`torch.manual_seed()`](../generated/torch.manual_seed.html#torch.manual_seed "torch.manual_seed") 设置为，也可以从使用随机数的操作中获得相同的结果后续调用之间的值相同。


### Python [¶](#python "此标题的永久链接")


 对于自定义运算符，您可能还需要设置 python 种子：


```
import random
random.seed(0)

```


### 其他库中的随机数生成器 [¶](#random-number-generators-in-other-libraries "永久链接到此标题")


 如果您或您使用的任何库依赖于 NumPy，您可以使用以下命令为 globalNumPy RNG 播种：


```
import numpy as np
np.random.seed(0)

```


 但是，某些应用程序和库可能使用 NumPy 随机生成器对象，而不是全局 RNG( <https://numpy.org/doc/stable/reference/random/generator.html>)，并且这些应用程序和库也需要一致地播种。


 如果您正在使用任何其他使用随机数生成器的库，请参阅这些库的文档以了解如何为它们设置一致的种子。


### CUDA 卷积基准测试 [¶](#cuda-卷积-基准测试“此标题的永久链接”)


 CUDA 卷积运算使用的 cuDNN 库可能是应用程序多次执行中不确定性的来源。当使用一组新的尺寸参数调用 cuDNN 卷积时，可选功能可以运行多个卷积算法，对它们进行基准测试以找到最快的算法。然后，在剩余的过程中将一致地使用最快的算法来处理相应的大小参数集。由于基准测试噪声和不同的硬件，基准测试可能会在后续运行中选择不同的算法，即使在同一台机器上也是如此。


 使用 `torch.backends.cudnn.benchmark = False` 禁用基准测试功能会导致 cuDNN 确定性地选择算法，这可能会降低性能。


 但是，如果您不需要应用程序多次执行的重现性，那么如果使用 `torch.backends.cudnn.benchmark = True` 启用基准测试功能，性能可能会提高。


 请注意，此设置与下面讨论的 `torch.backends.cudnn.definistic` 设置不同。


## 避免非确定性算法 [¶](#avoiding-nondefinitive-algorithms "永久链接到此标题")


[`torch.use_definistic_algorithms()`](../generated/torch.use_definistic_algorithms.html#torch.use_definistic_algorithms "torch.use_definistic_algorithms") 允许您将 PyTorch 配置为使用确定性算法而不是可用的非确定性算法，并抛出如果已知操作是不确定的(并且没有确定的替代方案)，则会出现错误。


 请检查 [`torch.use_definistic_algorithms()`](../generated/torch.use_definistic_algorithms.html#torch.use_definistic_algorithms "torch.use_definistic_algorithms") 的文档，以获取受影响操作的完整列表。如果根据文档，某个操作未正确执行，或者您需要确定性地实现没有的操作，请提交问题：<https://github.com/pytorch/pytorch/issues?q=label:%22module:%20determinism%22>


 例如，运行 [`torch.Tensor.index_add_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_ "torch.Tensor.index_add_") 的非确定性 CUDA 实现将抛出错误：


```
>>> import torch
>>> torch.use_deterministic_algorithms(True)
>>> torch.randn(2, 2).cuda().index_add_(0, torch.tensor([0, 1]), torch.randn(2, 2))
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
RuntimeError: index_add_cuda_ does not have a deterministic implementation, but you set
'torch.use_deterministic_algorithms(True)'. ...

```


 当使用稀疏密集 CUDA tensor调用 [`torch.bmm()`](../generated/torch.bmm.html#torch.bmm "torch.bmm") 时，它通常使用非确定性算法，但是当确定性标志打开时，将使用其替代确定性实现：


```
>>> import torch
>>> torch.use_deterministic_algorithms(True)
>>> torch.bmm(torch.randn(2, 2, 2).to_sparse().cuda(), torch.randn(2, 2, 2).cuda())
tensor([[[ 1.1900, -2.3409],
 [ 0.4796, 0.8003]],
 [[ 0.1509, 1.8027],
 [ 0.0333, -1.1444]]], device='cuda:0')

```


 此外，如果您使用 CUDA tensor，并且 CUDA 版本为 10.2 或更高版本，则应根据 CUDA 文档设置环境变量 CUBLAS_WORKSPACE_CONFIG：<https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility>


### CUDA 卷积决定论 [¶](#cuda-volving-determinism "永久链接到此标题")


 虽然禁用 CUDA 卷积基准测试(如上所述)可确保 CUDA 在每次运行应用程序时选择相同的算法，但该算法本身可能是不确定的，除非设置 `torch.use_definistic_algorithms(True)` 或 `torch.backends.cudnn.definistic = True` 。 后一个设置仅控制此行为，与 [`torch.use_definistic_algorithms()`](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms "torch.use_deterministic_algorithms") 不同，它将使其他 PyTorch 操作也具有确定性行为。


### CUDA RNN 和 LSTM [¶](#cuda-rnn-and-lstm "此标题的永久链接")


 在 CUDA 的某些版本中，RNN 和 LSTM 网络可能具有不确定性行为。请参阅 [`torch.nn.RNN()`](../generated/torch.nn.RNN.html#torch.nn.RNN "torch.nn.RNN") 和 [`torch.nn.LSTM()`](../generated/torch.nn.LSTM.html#torch.nn.LSTM "torch.nn.LSTM") 了解详细信息和解决方法。


## DataLoader [¶](#dataloader "此标题的永久链接")


 DataLoader 将按照[多进程数据加载中的随机性](../data.html#data-loading-randomness) 算法重新设定工作线程。使用 `worker_init_fn()` 和生成器来保持再现性：


```
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g,
)

```