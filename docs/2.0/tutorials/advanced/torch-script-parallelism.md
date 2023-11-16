


# TorchScript 中的动态并行性 [¶](#dynamic-parallelism-in-torchscript "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/torch-script-parallelism>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/torch-script-parallelism.html>




 在本教程中，我们介绍了在 TorchScript 中执行
 *动态互操作并行* 
 的语法。这种并行性具有以下属性:



* 动态 - 创建的并行任务数量及其工作负载取决于程序的控制流。
* 互操作 - 并行性涉及并行运行 TorchScript 程序片段。这与
 *操作内并行* 
 不同，后者涉及拆分各个运算符并并行运行运算符’ 的子集。


## 基本语法 [¶](#basic-syntax "此标题的永久链接")




 动态并行的两个重要 API 是：



* `torch.jit.fork(fn
 

 :
 

 可调用[...,
 

 T],
 

 *args,
 

 **kwargs)
 

 ->
 

 torch.jit.Future[T]`
* `torch.jit.wait(fut
 
 
 :
 

 torch.jit.Future[T])
 

 ->
 

 T`



 通过示例来演示这些工作原理的一个好方法是：






```
import torch

def foo(x):
    return torch.neg(x)

@torch.jit.script
def example(x):
    # Call `foo` using parallelism:
    # First, we "fork" off a task. This task will run `foo` with argument `x`
    future = torch.jit.fork(foo, x)

    # Call `foo` normally
    x_normal = foo(x)

    # Second, we "wait" on the task. Since the task may be running in
    # parallel, we have to "wait" for its result to become available.
    # Notice that by having lines of code between the "fork()" and "wait()"
    # call for a given Future, we can overlap computations so that they
    # run in parallel.
    x_parallel = torch.jit.wait(future)

    return x_normal, x_parallel

print(example(torch.ones(1))) # (-1., -1.)

```




`fork()`
 接受可调用
 `fn`
 以及该可调用
 `args`
 和
 `kwargs`
 的参数，并创建一个异步任务来执行
 `fn`
.
 `fn`
 可以是函数、方法或模块实例。
 `fork()`
 返回对此执行结果值的引用，称为a
 `Future`
 。
因为
 `fork`
 在创建异步任务后立即返回，
 `fn`
 可能
在该任务之后的代码行尚未执行
 `fork()`
 调用
 被执行。因此，
 `wait()`
 用于等待异步任务完成
并返回值。




 这些结构可用于重叠函数内语句的执行（如工作示例部分所示）或与其他语言
结构（如循环）组合：






```
import torch
from typing import List

def foo(x):
    return torch.neg(x)

@torch.jit.script
def example(x):
    futures : List[torch.jit.Future[torch.Tensor]] = []
    for _ in range(100):
        futures.append(torch.jit.fork(foo, x))

    results = []
    for future in futures:
        results.append(torch.jit.wait(future))

    return torch.sum(torch.stack(results))

print(example(torch.ones([])))

```





 没有10



 当我们初始化一个空的 Future 列表时，我们需要向
 `futures` 添加一个明确的
类型注释。在 TorchScript 中，空容器默认
假设它们包含 Tensor 值，因此我们将列表构造函数
# 注释为类型
 `List[torch.jit.Future[torch.Tensor]]`





 此示例使用
 `fork()`
 启动函数的 100 个实例
 `foo`
 ，
等待 100 个任务完成，然后对结果求和，返回
 `-100.0`
.





## 应用示例：双向 LSTM 集成 [¶](#applied-example-ensemble-of-bi Direction-lstms "永久链接到此标题")




 让’s 尝试将并行性应用到更实际的示例中，看看我们可以从中获得什么样的性能。首先，让’s 定义基线模型：双向 LSTM 层
的集合。






```
import torch, time

# In RNN parlance, the dimensions we care about are:
# # of time-steps (T)
# Batch size (B)
# Hidden size/number of "channels" (C)
T, B, C = 50, 50, 1024

# A module that defines a single "bidirectional LSTM". This is simply two
# LSTMs applied to the same sequence, but one in reverse
class BidirectionalRecurrentLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_f = torch.nn.LSTM(input_size=C, hidden_size=C)
        self.cell_b = torch.nn.LSTM(input_size=C, hidden_size=C)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Forward layer
        output_f, _ = self.cell_f(x)

        # Backward layer. Flip input in the time dimension (dim 0), apply the
        # layer, then flip the outputs in the time dimension
        x_rev = torch.flip(x, dims=[0])
        output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
        output_b_rev = torch.flip(output_b, dims=[0])

        return torch.cat((output_f, output_b_rev), dim=2)


# An "ensemble" of `BidirectionalRecurrentLSTM` modules. The modules in the
# ensemble are run one-by-one on the same input then their results are
# stacked and summed together, returning the combined result.
class LSTMEnsemble(torch.nn.Module):
    def __init__(self, n_models):
        super().__init__()
        self.n_models = n_models
        self.models = torch.nn.ModuleList([
            BidirectionalRecurrentLSTM() for _ in range(self.n_models)])

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        results = []
        for model in self.models:
            results.append(model(x))
        return torch.stack(results).sum(dim=0)

# For a head-to-head comparison to what we're going to do with fork/wait, let's
# instantiate the model and compile it with TorchScript
ens = torch.jit.script(LSTMEnsemble(n_models=4))

# Normally you would pull this input out of an embedding table, but for the
# purpose of this demo let's just use random data.
x = torch.rand(T, B, C)

# Let's run the model once to warm up things like the memory allocator
ens(x)

x = torch.rand(T, B, C)

# Let's see how fast it runs!
s = time.time()
ens(x)
print('Inference took', time.time() - s, ' seconds')

```




 在我的机器上，该网络运行时间为
 `2.05`
 秒。我们可以做得更好！





## 并行化前向和后向层 [¶](#parallelizing-forward-and-backward-layers "永久链接到此标题")




 我们可以做的一件非常简单的事情就是在 `Bi DirectionRecurrentLSTM`
 内并行化前向层和后向层。为此，计算的结构\是静态的，所以我们’t实际上甚至不需要任何循环。让’s 重写
 `Bi DirectionRecurrentLSTM`
 的
 `forward`
 方法，如下所示：






```
def forward(self, x : torch.Tensor) -> torch.Tensor:
    # Forward layer - fork() so this can run in parallel to the backward
    # layer
    future_f = torch.jit.fork(self.cell_f, x)

    # Backward layer. Flip input in the time dimension (dim 0), apply the
    # layer, then flip the outputs in the time dimension
    x_rev = torch.flip(x, dims=[0])
    output_b, _ = self.cell_b(torch.flip(x, dims=[0]))
    output_b_rev = torch.flip(output_b, dims=[0])

    # Retrieve the output from the forward layer. Note this needs to happen
    # *after* the stuff we want to parallelize with
    output_f, _ = torch.jit.wait(future_f)

    return torch.cat((output_f, output_b_rev), dim=2)

```




 在此示例中，
 `forward()`
 将 
 `cell_f`
 的执行委托给另一个线程，
而它继续执行
 
 `cell_b`
 。这会导致
两个单元的执行彼此重叠。




 通过此简单修改再次运行脚本，运行时间为
 `1.71`
 秒，改进
 `17%`
 !





## 旁白：可视化并行性 [¶](#aside-visualizing-parallelism "永久链接到此标题")




 我们’ 尚未完成模型优化，但’ 值得介绍我们用于可视化性能的工具。一个重要的工具是
 [PyTorch 分析器](https://pytorch.org/docs/stable/autograd.html#profiler)
 。




 让’s 使用探查器和 Chrome 跟踪导出功能来
可视化并行模型的性能：






```
with torch.autograd.profiler.profile() as prof:
    ens(x)
prof.export_chrome_trace('parallel.json')

```




 这段代码将写出一个名为
 `parallel.json`
 的文件。如果您
将 Google Chrome 导航到
 `chrome://tracing`
 ，单击
 `加载`
 按钮，然后
加载该 JSON 文件，您应该会看到如下所示的时间线:



![https://i.imgur.com/rm5hdG9.png](https://i.imgur.com/rm5hdG9.png)

时间线的横轴代表时间，纵轴代表线程
的执行。正如我们所看到的，我们一次运行两个
 `lstm`
 实例。这是我们努力并行化双向层
的结果！





## 在集成中并行化模型 [¶](#parallelizing-models-in-the-ensemble "永久链接到此标题")




 您可能已经注意到，我们的代码中有一个进一步的并行化机会：我们还可以彼此并行运行
 `LSTMEnsemble` 中包含的模型。方法很简单，这就是我们应该改变
 `LSTMEnsemble`
 的
 `forward`
 方法的方法：






```
def forward(self, x : torch.Tensor) -> torch.Tensor:
    # Launch tasks for each model
    futures : List[torch.jit.Future[torch.Tensor]] = []
    for model in self.models:
        futures.append(torch.jit.fork(model, x))

    # Collect the results from the launched tasks
    results : List[torch.Tensor] = []
    for future in futures:
        results.append(torch.jit.wait(future))

    return torch.stack(results).sum(dim=0)

```




 或者，如果您重视简洁性，我们可以使用列表推导式：






```
def forward(self, x : torch.Tensor) -> torch.Tensor:
    futures = [torch.jit.fork(model, x) for model in self.models]
    results = [torch.jit.wait(fut) for fut in futures]
    return torch.stack(results).sum(dim=0)

```




 就像简介中所描述的那样，我们’ 使用循环来为我们的集成中的每个模型分配任务。然后我们’ve使用另一个循环来等待所有
任务完成。这提供了更多的计算重叠。




 通过这个小更新，脚本在
 `1.4`
 秒内运行，总加速

 `32%`
 ！对于两行代码来说已经很不错了。




 我们还可以再次使用 Chrome 跟踪器来查看’s 的情况：



![https://i.imgur.com/kA0gyQm.png](https://i.imgur.com/kA0gyQm.png)

 我们现在可以看到所有
 `LSTM`
 实例都是完全并行运行。





## 结论 [¶](#conclusion "此标题的永久链接")




 在本教程中，我们了解了
 `fork()`
 和
 `wait()`
 ，这是用于在 TorchScript 中执行动态、操作间并行性的基本 API。我们看到了一些典型的使用模式，使用这些函数来并行执行 TorchScript 代码中的函数、方法或“模块”。最后，我们完成了
使用此技术优化模型的示例，并探索了
PyTorch 中可用的
性能测量和可视化工具。









