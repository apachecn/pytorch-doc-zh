# 分析您的 PyTorch 模块 [¶](#profiling-your-pytorch-module "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/profiler>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/profiler.html>

**作者：** :[Suraj Subramanian](https://github.com/suraj813)

PyTorch 包含一个探查器 API，可用于识别代码中各种 PyTorch 操作的时间和内存成本。探查器可以\轻松集成到您的代码中，并且结果可以作为表格打印\或在 JSON 跟踪文件中返回。

 !!!  Profiler 支持多线程模型。探查器在与操作相同的线程中运行，但它也会分析可能在另一个线程中运行的子运算符。并发运行的探查器将被限定在它们自己的线程范围内，以防止结果混合。

 !!! PyTorch 1.8 引入了新的 API，它将在未来的版本中取代旧的探查器 API。请在[此页面](https://pytorch.org/docs/master/profiler.html) 检查新 API。

 请前往[此食谱](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 以更快地了解 Profiler API 的使用情况。

---

```python
import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler

```

## 使用 Profiler 进行性能调试 [¶](#performance-debugging-using-profiler "永久链接到此标题")

 探查器可用于识别模型中的性能瓶颈。在此示例中，我们构建一个执行两个子任务的自定义模块：

* 对输入进行线性变换，并且
* 使用变换结果来获取掩码张量的索引。

 我们使用 `profiler.record_function("label")` 将每个子任务的代码包装在单独的标记上下文管理器中。。在探查器输出中，子任务中所有操作的聚合性能指标将显示在其相应的标签下。

 请注意，使用探查器会产生一些开销，最好仅用于调查代码。如果您正在对运行时进行基准测试，请记住将其删除。

```python
class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx

```

## 分析正向传递 [¶](#profile-the-forward-pass "永久链接到此标题")

 我们初始化随机输入和掩码张量以及模型。

 在运行探查器之前，我们会预热 CUDA 以确保准确的性能基准测试。我们将模块的前向传递包装在`profiler.profile`上下文管理器中。 `with_stack=True`参数附加跟踪中操作的文件和行号。

 警告

`with_stack=True`会产生额外的开销，并且更适合研究代码。如果您正在对性能进行基准测试，请记住将其删除。

```python
model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

```

## 打印探查器结果 [¶](#print-profiler-results "永久链接到此标题")

 最后，我们打印探查器结果。`profiler.key_averages`按运算符名称聚合结果，也可以选择按输入形状和/或堆栈跟踪事件聚合结果。按输入形状分组对于确定模型使用哪些张量形状。

 在这里，我们使用`group_by_stack_n=5`它通过操作及其回溯（截断为最近的 5 个事件）聚合运行时，并在他们注册的顺序。该表还可以通过传递 `sort_by`参数进行排序（请参阅 [文档](https://pytorch.org/docs/stable/autograd.html#profiler) n 表示有效的排序键）。

 注意

 在笔记本中运行探查器时，您可能会在堆栈跟踪中看到类似 `<ipython-input-18-193a910735e8>(13):forward`的条目，而不是文件名。这些对应于 `<notebook-cell>(line number)calling-function`。

```python
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

"""
(Some columns are omitted)

------------- ------------ ------------ ------------ ---------------------------------
 Name Self CPU % Self CPU Self CPU Mem Source Location
------------- ------------ ------------ ------------ ---------------------------------
 MASK INDICES 87.88% 5.212s -953.67 Mb /mnt/xarfuse/.../torch/au
 <ipython-input-...>(10): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/

 aten::copy_ 12.07% 715.848ms 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 LINEAR PASS 0.01% 350.151us -20 b /mnt/xarfuse/.../torch/au
 <ipython-input-...>(7): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/

 aten::addmm 0.00% 293.342us 0 b /mnt/xarfuse/.../torch/nn
 /mnt/xarfuse/.../torch/nn
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(8): forward
 /mnt/xarfuse/.../torch/nn

 aten::mean 0.00% 235.095us 0 b <ipython-input-...>(11): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

----------------------------- ------------ ---------- ----------------------------------
Self CPU time total: 5.931s

"""

```

## 提高内存性能 [¶](#improve-memory-performance "永久链接到此标题")

 请注意，最昂贵的操作 - 就内存和时间而言 - 位于`forward(10)`表示 MASK INDICES 内的操作。让’s 首先尝试解决内存消耗问题。我们可以看到第 12 行的`.to()`操作消耗了 953.67 Mb。此操作将`mask`复制到CPU。`mask`使用`torch.double`数据类型进行初始化。我们可以通过将转换为`torch.float`来减少内存占用吗？

```python
model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

"""
(Some columns are omitted)

----------------- ------------ ------------ ------------ --------------------------------
 Name Self CPU % Self CPU Self CPU Mem Source Location
----------------- ------------ ------------ ------------ --------------------------------
 MASK INDICES 93.61% 5.006s -476.84 Mb /mnt/xarfuse/.../torch/au
 <ipython-input-...>(10): forward
 /mnt/xarfuse/ /torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/

 aten::copy_ 6.34% 338.759ms 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 aten::as_strided 0.01% 281.808us 0 b <ipython-input-...>(11): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 aten::addmm 0.01% 275.721us 0 b /mnt/xarfuse/.../torch/nn
 /mnt/xarfuse/.../torch/nn
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(8): forward
 /mnt/xarfuse/.../torch/nn

 aten::_local 0.01% 268.650us 0 b <ipython-input-...>(11): forward
 _scalar_dense /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(9): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

----------------- ------------ ------------ ------------ --------------------------------
Self CPU time total: 5.347s

"""

```

 此操作的 CPU 内存占用量已减半。

## 提高时间性能 [¶](#improve-time-performance "永久链接到此标题")

 虽然消耗的时间也减少了一点，但 ’s 仍然太高。事实证明，将矩阵从 CUDA 复制到 CPU 的成本相当昂贵！

 `aten::copy_` `forward(12)`中的运算符将`mask`复制到 CPU，以便它可以使用 NumPy`argwhere`
 函数。` aten::copy_` at `forward(13)` 将数组作为张量复制回 CUDA。如果我们在这里使用 `torch`函数 `nonzero()` 来代替，我们就可以消除这两个问题。

```python
class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return out, hi_idx


model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.float).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)

print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

"""
(Some columns are omitted)

-------------- ------------ ------------ ------------ ---------------------------------
 Name Self CPU % Self CPU Self CPU Mem Source Location
-------------- ------------ ------------ ------------ ---------------------------------
 aten::gt 57.17% 129.089ms 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 aten::nonzero 37.38% 84.402ms 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 INDEX SCORE 3.32% 7.491ms -119.21 Mb /mnt/xarfuse/.../torch/au
 <ipython-input-...>(10): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/

aten::as_strided 0.20% 441.587us 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/

 aten::nonzero
 _numpy 0.18% 395.602us 0 b <ipython-input-...>(12): forward
 /mnt/xarfuse/.../torch/nn
 <ipython-input-...>(25): <module>
 /mnt/xarfuse/.../IPython/
 /mnt/xarfuse/.../IPython/
-------------- ------------ ------------ ------------ ---------------------------------
Self CPU time total: 225.801ms

"""

```

## 进一步阅读 [¶](#further-reading "此标题的永久链接")

 我们已经了解了如何使用 Profiler 来研究 PyTorch 模型中的时间和内存瓶颈。
在此处了解有关 Profiler 的更多信息：

* [分析器使用方法](https://pytorch.org/tutorials/recipes/recipes/profiler.html)
* [分析基于 RPC 的工作负载](https://pytorch.org/tutorials/recipes/distributed_rpc_profiling. html)
* [Profiler API 文档](https://pytorch.org/docs/stable/autograd.html?highlight=profiler#profiler)

**脚本的总运行时间:** 
 ( 0 分 0.000 秒)
