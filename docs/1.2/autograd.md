# Automatic differentiation package - torch.autograd

> 译者：[gfjiangly](https://github.com/gfjiangly)

`torch.autograd` 提供类和函数，实现任意标量值函数的自动微分。 它要求对已有代码的最小改变---你仅需要用`requires_grad=True`关键字为需要计算梯度的声明`Tensor`。

```py
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None)
```

计算被给张量关于图的叶节点的梯度和。

图使用链式法则微分。如何任何`tensors`是非标量(例如他们的数据不止一个元素）并且要求梯度，函数要额外指出`grad_tensors`。它应是一个匹配长度的序列，包含可微函数关于相应张量的梯度(`None`是一个对所有张量可接受的值，不需要梯度张量）。

此函数在叶节点累积梯度 - 你可能需要在调用前把它初始化为0.

参数：

*   **tensors** (_Tensor序列_) – 计算导数的张量。
*   **grad_tensors** (_[_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _或_ [_None_](https://docs.python.org/3/library/constants.html#None "(in Python v3.7)")序列_) – 关于相应张量每个元素的梯度。标量张量或不需要梯度的可用None指定。如果None对所有grad_tensors可接受，则此参数可选。
*   **retain_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果False，用于计算梯度的图将被释放。请注意，在几乎所有情况下，不需要将此选项设置为真，而且通常可以更有效地解决问题。默认为create_graph值。
*   **create_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果True，则构造导数图，以便计算更高阶导数，默认False。



```py
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
```

计算和返回输出关于输入的梯度和。

`grad_outputs` 应是长度匹配输出的序列，包含关于输出每个元素的预计算梯度。如果一个输出不要求梯度，则梯度是`None`。

如果`only_inputs`是`True`，此函数将仅返回关于指定输入的梯度list。如果此参数是`False`，则关于其余全部叶子的梯度仍被计算，并且将累加到`.grad`属性中。

参数: 

*   **outputs** (_Tensor序列_) – 可微函数输出
*   **inputs** (_Tensor序列_) – 关于将返回梯度的输入(不累加到`.grad`)。
*   **grad_outputs** (_Tensor序列_) – 关于每个输入的梯度。标量张量或不需要梯度的可用None指定。如果None对所有grad_tensors可接受，则此参数可选。默认：`None`。
*   **retain_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果`False`，用于计算梯度的图将被释放。请注意，在几乎所有情况下，不需要将此选项设置为真，而且通常可以更有效地解决问题。默认为`create_graph`值。
*   **create_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果`True`，则构造导数图，以便计算更高阶导数，默认`False`。
*   **allow_unused** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果`False`, 当计算输出出错时指明不使用的输入 (因此它们的梯度一直是0)。 默认`False`。


## 局部禁用梯度计算

```py
class torch.autograd.no_grad
```

禁用梯度计算的上下文管理器。

当你确认不会调用 `Tensor.backward()`，对于推断禁用梯度计算是有用的。它将减少计算的内存消耗，否则会有`requires_grad=True`。在这个模式中，每个计算结果将导致`requires_grad=False`, 即便输入有`requires_grad=True`。

函数还可作为装饰器。

示例：


```py
>>> x = torch.tensor([1], requires_grad=True)
>>> with torch.no_grad():
...   y = x * 2
>>> y.requires_grad
False
>>> @torch.no_grad()
... def doubler(x):
...     return x * 2
>>> z = doubler(x)
>>> z.requires_grad
False

```

```py
class torch.autograd.enable_grad
```

使能梯度计算的上下文管理器。

在一个[`no_grad`](#torch.autograd.no_grad "torch.autograd.no_grad")上下文中使能梯度计算。在[`no_grad`](#torch.autograd.no_grad "torch.autograd.no_grad")外部此上下文管理器无影响

函数还可作为装饰器。

示例：


```py
>>> x = torch.tensor([1], requires_grad=True)
>>> with torch.no_grad():
...   with torch.enable_grad():
...     y = x * 2
>>> y.requires_grad
True
>>> y.backward()
>>> x.grad
>>> @torch.enable_grad()
... def doubler(x):
...     return x * 2
>>> with torch.no_grad():
...     z = doubler(x)
>>> z.requires_grad
True

```

```py
class torch.autograd.set_grad_enabled(mode)
```


设置梯度计算打开或关闭的上下文管理器。

`set_grad_enabled`将基于它的参数`mode`使用或禁用梯度。它也能作为一个上下文管理器或函数使用。

| 参数: | **mode** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 标记是否使能梯度(True），或使能(False）。这能被用在有条件的使能梯度。
| --- | --- |

示例：

```py
>>> x = torch.tensor([1], requires_grad=True)
>>> is_train = False
>>> with torch.set_grad_enabled(is_train):
...   y = x * 2
>>> y.requires_grad
False
>>> torch.set_grad_enabled(True)
>>> y = x * 2
>>> y.requires_grad
True
>>> torch.set_grad_enabled(False)
>>> y = x * 2
>>> y.requires_grad
False

```

## 关于Tensors的原位操作

在autograd中支持原位操作是一件很难的事，并且我们在大多数情况下不鼓励使用它们。Autograd积极的缓冲区释放和重用使其非常高效，实际上原位操作会大幅降低内存使用量的情况非常少。你可能永远不会使用它们，除非正在很大的内存压力下操作。

### 就地正确性检查

全部的Tensor保持追踪应用到它们身上的原位操作，并且如果实现检测到在任何一个函数中，一个tensor为反向传播保存，但是随后被原位修改，一旦反向传播开始将抛出一个错误。此设计确保如果你正在使用原位操作函数并且没有看到任何错误，你可以确保计算的梯度是正确的。

### Variable (弃用)

警告

Variable API已经被弃用。对张量使用自动求导不再需要Variable。Autograd自动支持`requires_grad`参数设置成`True`的张量。以下是有关更改内容的快速指南：

*   `Variable(tensor)` 和`Variable(tensor, requires_grad)`仍然和预期一样工作，但它们返回Tensors代替Variables。
*   `var.data` 和 `tensor.data`是一回事。
*   方法如`var.backward(), var.detach(), var.register_hook()`现在在tensors上使用相同的名字起作用。

此外，现在可以使用诸如[`torch.randn()`](torch.html#torch.randn "torch.randn"), [`torch.zeros()`](torch.html#torch.zeros "torch.zeros"), [`torch.ones()`](torch.html#torch.ones "torch.ones")等工厂方法创建requires_grad=True的张量，如下所示：


`autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)`

## 张量自动求导函数

```py
class torch.Tensor
```

```py
backward(gradient=None, retain_graph=None, create_graph=False)
```

计算当前张量关于图叶节点的梯度。

图使用链式反则微分。如果张量是非标量并且要求梯度，函数额外要求指梯度。它应是一个匹配类型和位置的张量，含有可微函数关于它本身的梯度。

此函数在叶节点累加梯度-你可能需要在调用前将它初始化为0。

参数：

*   **gradient** ([_Tensor_](tensors.html#torch.Tensor "torch.Tensor") _或_ [_None_](https://docs.python.org/3/library/constants.html#None "(in Python v3.7)")) – 关于张量的梯度。如果它是一个张量，它将被自动转换成不要求梯度的张量，除非`create_graph`是`True`。标量张量或不需要梯度的可用`None`指定。如果None对所有grad_tensors可接受，则此参数可选。
*   **retain_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果`False`，用于计算梯度的图将被释放。请注意，在几乎所有情况下，不需要将此选项设置为真，而且通常可以更有效地解决问题。默认为`create_graph`值。
*   **create_graph** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果`True`，则构造导数图，以便计算更高阶导数，默认`False`。

```py
detach()
```

返回一个新的Tensor，从当前图中分离出来。

结果不要求梯度。

注意

返回的张量与原始张量使用相同的数据。关于它们中任一个原位修改将被看见，并且可能在正确性检查中触发错误。


```py
detach_()
```

从创建它的图中分离张量，使其成为叶。不能就地分离视图。

```py
grad
```

此属性默认`None`，并且调用[`backward()`](#torch.Tensor.backward "torch.Tensor.backward")计算自身梯度时第一时间成为一个Tensor。此属性将含计算的梯度，以后调用[`backward()`](#torch.Tensor.backward "torch.Tensor.backward")将累加提到到自身。

```py
is_leaf
```

按惯例，所有[`requires_grad`](#torch.Tensor.requires_grad "torch.Tensor.requires_grad")=False的张量将是叶节点张量

如果张量是由用户创建，[`requires_grad`](#torch.Tensor.requires_grad "torch.Tensor.requires_grad")的张量也是叶节点张量。这意味着它们不是一个操作的结果，并且`grad_fn`是`None`。

仅叶节点张量在调用[`backward()`](#torch.Tensor.backward "torch.Tensor.backward")时填充它们的[`grad`](#torch.Tensor.grad "torch.Tensor.grad")。为得到从非叶节点张量填充的梯度，你可以使用[`retain_grad()`](#torch.Tensor.retain_grad "torch.Tensor.retain_grad").

示例：

```py
>>> a = torch.rand(10, requires_grad=True)
>>> a.is_leaf
True
>>> b = torch.rand(10, requires_grad=True).cuda()
>>> b.is_leaf
False
# b 是由cpu Tensor投入cuda Tensor的操作创建的
>>> c = torch.rand(10, requires_grad=True) + 2
>>> c.is_leaf
False
# c 是由加操作创建的
>>> d = torch.rand(10).cuda()
>>> d.is_leaf
True
# d 不要求梯度，所以没有创建它的操作 (被自动求导引擎追踪)
>>> e = torch.rand(10).cuda().requires_grad_()
>>> e.is_leaf
True
# e 要求梯度并且没有创建它的操作
>>> f = torch.rand(10, requires_grad=True, device="cuda")
>>> f.is_leaf
True
# f 要求梯度并且没有创建它的操作

```

```py
register_hook(hook)
```

注册一个反向钩子

此钩子每次在对应张量梯度被计算时调用。此钩子应有下面鲜明特征：

```py
hook(grad) -> Tensor or None

```

此钩子不应该修改它的参数，但它能可选地返回一个新的用于替代 `grad`的梯度。

此函数返回一个句柄，其句柄方法为`handle.remove()`，用于从模块中删除钩子。

示例：

```py
>>> v = torch.tensor([0., 0., 0.], requires_grad=True)
>>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
>>> v.backward(torch.tensor([1., 2., 3.]))
>>> v.grad

 2
 4
 6
[torch.FloatTensor of size (3,)]

>>> h.remove()  # removes the hook

```

```py
requires_grad
```

如果梯度需要为此张量计算则是`True`，否则为`False`

注意

事实是梯度需要为此张量计算不意味着`grad`属性将被填充，更多细节见[`is_leaf`](#torch.Tensor.is_leaf "torch.Tensor.is_leaf")。

```py
retain_grad()
```

为非叶节点张量使能`.grad`属性

## Function

```py
class torch.autograd.Function
```

记录操作历史，定义可微操作公式。

在Tensor上执行的每个操作都会创建一个新的函数对象，执行计算，记录它发生的。历史记录以函数的DAG形式保留，DAG的边表示数据的依赖性(`input &lt;- output`）。然后，当backward被调用，图按拓扑顺序被处理，通过调用每个[`Function`](#torch.autograd.Function "torch.autograd.Function")对象的backward()方法，并且传递梯度给下一个[`Function`](#torch.autograd.Function "torch.autograd.Function")。

通常，用户与函数交互的唯一方式是通过创建子类和定义新操作。这是一种被推荐的扩展`torch.autograd`的方式。

每个函数对象只能使用一次(在正向传递中）。

示例：

```py
>>> class Exp(Function):
>>>
>>>     @staticmethod
>>>     def forward(ctx, i):
>>>         result = i.exp()
>>>         ctx.save_for_backward(result)
>>>         return result
>>>
>>>     @staticmethod
>>>     def backward(ctx, grad_output):
>>>         result, = ctx.saved_tensors
>>>         return grad_output * result

```

```py
static backward(ctx, *grad_outputs)
```

定义一个公式计算操作导数。

此函数被所有子类重载。

它必须接受一个上下文`ctx`作为第一个参数，随后是`forward()`返回的大量输出，并且它应返回尽可能多的张量，作为`forward()`函数输入。每个参数是关于被给输出的梯度，并且每个返回值是关于相应输入的梯度。

`ctx`上下文可用于恢复保存在前向传播过程的梯度。它有一个`ctx.needs_input_grad`属性，作为一个代表每个输入是否需要梯度的布尔元组。

```py
static forward(ctx, *args, **kwargs)
```

执行操作。

此函数被所有子类重载。

它必须接受一个上下文`ctx`作为第一个参数，随后是任意数量的参数(tensor或其它类型）。

此上下文可被用来存储张量，随后可在反向传播过程取出。

## 数值梯度检查

```py
torch.autograd.gradcheck(func, inputs, eps=1e-06, atol=1e-05, rtol=0.001, raise_exception=True)
```

通过小的有限差分与关于浮点类型且`requires_grad=True`的输入张量来检查计算的梯度。

在数组梯度和分析梯度之间检查使用[`allclose()`](torch.html#torch.allclose "torch.allclose")。

注意

默认值为双精度输入设计。如果输入欠精度此检查有可能失败，例如，`FloatTensor`。

警告

如果在输入中任何被检查的张量有重叠的内存，换句话说，指向相同内存地址的不同切片(例如，从torch.expand()），此检查将有可能失败，因为在这个索引通过点扰动计算的数值梯度将改变在全部其它索引处共享内存地址的值。

参数

*   **func** (_function_) – 一个Python函数，输入是张量，返回一个张量或张量元组
*   **inputs** (_张量元组_ _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – func函数输入
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选_) – 有限差分的扰动
*   **atol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选_) – 绝对容差
*   **rtol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选_) – 相对容差
*   **raise_exception** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 指示如果检查失败是否抛出一个异常。此异常给出关于失败的确切性质的更多信息。这在梯度检查调试时是有用的。


| 返回: | 如果所有差都满足全部闭合条件，则为True |
| --- | --- |


```py
torch.autograd.gradgradcheck(func, inputs, grad_outputs=None, eps=1e-06, atol=1e-05, rtol=0.001, gen_non_contig_grad_outputs=False, raise_exception=True)
```

通过小的有限差分与关于在输入中张量的分析梯度，检查已计算梯度的梯度，并且在requires_grad=True 情况下，grad_outputs是浮点类型。

此函数检查通过计算到给定grad_outputs的梯度的反向传播是否正确。

在数值梯度和分析梯度之间使用[`allclose()`](torch.html#torch.allclose "torch.allclose")检查。

注意

默认值为双精度输入设计。如果输入欠精度此检查有可能失败，例如，`FloatTensor`。

警告

如果在输入中任何被检查的张量有重叠的内存，换句话说，指向相同内存地址的不同切片(例如，从`torch.expand()`），此检查将有可能失败，因为在这个索引通过点扰动计算的数值梯度将改变在全部其它索引处共享内存地址的值。

参数：

*   **func** (_function_) – 一个Python函数，输入是张量，返回一个张量或张量元组
*   **inputs** (_张量元组_ _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – func函数输入
*   **grad_outputs** (_tuple of Tensor_ _or_ [_Tensor_](tensors.html#torch.Tensor "torch.Tensor")_,_ _可选_) – The gradients with respect to the function’s outputs.
*   **eps** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选_) – 有限差分的扰动
*   **atol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选_) – 绝对容差
*   **rtol** ([_float_](https://docs.python.org/3/library/functions.html#float "(in Python v3.7)")_,_ _可选_) – 相对容差
*   **gen_non_contig_grad_outputs** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 如果 grad_outputs 是 None 并且 gen_non_contig_grad_outputs 是 True，随机生成的梯度输出是不连续的
*   **raise_exception** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) –  指示如果检查失败是否抛出一个异常。此异常给出关于失败的确切性质的更多信息。这在梯度检查调试时是有用的。

| 返回: | 如果所有差都满足全部闭合条件，则为True |
| --- | --- |


## Profiler

Autograd 包含一个事件探查器，让你洞察在你的模型中不同操作的代价-CPU和GPU中都有。现在有两种模式实现-CPU-仅使用[`profile`](#torch.autograd.profiler.profile "torch.autograd.profiler.profile")，和使用[`emit_nvtx`](#torch.autograd.profiler.emit_nvtx "torch.autograd.profiler.emit_nvtx")的nvprof(注册CPU和GPU活动）

```py
class torch.autograd.profiler.profile(enabled=True, use_cuda=False)
```

上下文管理器管理autograd事件探查器状态和保持一份汇总结果。

参数: 

*   **enabled** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 设置成False 让此上下文管理一个 no-op. 默认：`True`。
*   **use_cuda** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 使用cudaEvent API也可以启用CUDA事件的计时。 每个张量操作增加大约4us的开销。默认:  `False`

示例：

```py
>>> x = torch.randn((1, 1), requires_grad=True)
>>> with torch.autograd.profiler.profile() as prof:
...     y = x ** 2
...     y.backward()
>>> # 注意：为简洁起见，删除了一些列
... print(prof)
-------------------------------------  ---------------  ---------------
Name                                          CPU time        CUDA time
-------------------------------------  ---------------  ---------------
PowConstant                                  142.036us          0.000us
N5torch8autograd9GraphRootE                   63.524us          0.000us
PowConstantBackward                          184.228us          0.000us
MulConstant                                   50.288us          0.000us
PowConstant                                   28.439us          0.000us
Mul                                           20.154us          0.000us
N5torch8autograd14AccumulateGradE             13.790us          0.000us
N5torch8autograd5CloneE                        4.088us          0.000us

```

```py
export_chrome_trace(path)
```

将EventList导出为Chrome跟踪工具文件。

检查点随后被加载和检查在`chrome://tracing URL`。

| 参数: | **path** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – 将写入跟踪的路径。 |
| --- | --- |

```py
key_averages()
```

平均键上的所有函数事件.

| 返回: | 一个包含FunctionEventAvg对象的EventList。 |
| --- | --- |

```py
table(sort_by=None)
```

将EventList打印为格式良好的表。

| 参数: | **sort_by** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")_,_ _optional_) – 用来排序事件的属性。默认以它们被注册时顺序打印。 合法的关键字包括：`cpu_time`, `cuda_time`, `cpu_time_total`, `cuda_time_total`, `count`。 |
| --- | --- |
| 返回: | 一个包含表格的字符串。 |
| --- | --- |

```py
total_average()
```

平均化全部事件。

| 返回: | 一个FunctionEventAvg事件。 |
| --- | --- |

```py
class torch.autograd.profiler.emit_nvtx(enabled=True)
```

让每个自动求导操作发出在一个NVTX范围内的上下文管理器。

当在nvprof下运行程序是有用的：

```py
nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

```

不幸地，没有办法强制nvprof将它收集的数据输出到磁盘，所以对于CUDA分析，必须使用此上下文管理器来声明nvprof跟踪并等待进程在检查之前退出。然后，可使用NVIDIA可视化Profiler(nvvp)来显示时间线，或[`torch.autograd.profiler.load_nvprof()`](#torch.autograd.profiler.load_nvprof "torch.autograd.profiler.load_nvprof")可加载结果以供检查，例如：在Python REPL中。

| 参数: | **enabled** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")_,_ _可选_) – 设置成False 让此上下文管理一个 no-op. 默认：`True`。 |
| --- | --- |

示例：

```py
>>> with torch.cuda.profiler.profile():
...     model(x) # Warmup CUDA memory allocator and profiler
...     with torch.autograd.profiler.emit_nvtx():
...         model(x)

```

**Forward-backward correlation**

在Nvidia Visual Profiler中查看使用emit_nvtx创建的配置文件时，将每个反向传递操作与相应的前向传递操作相关联可能很困难。 为了简化此任务，emit_nvtx将序列号信息附加到它生成的范围。

在前向传递过程，每个函数范围都用`seq=&lt;N&gt;`进行修饰。 `seq`是一个运行计数器，每次创建一个新的反向Function对象时会递增，并对前向不可见。 因此，与每个前向函数范围相关联的`seq=&lt;N&gt;`注释告诉你，如果此前向函数创建了反向的Function对象，则反向对象将收到序列号N。在反向传递过程，顶层范围包装每个C++反向函数的`apply()`调用都用不可见的`stashed seq=&lt;M&gt;`进行修饰。 M是创建反向对象的序列号。 通过比较在反向不可见的序列号和在前向的序列号，你可以跟踪哪个前向操作创建了每个反向函数。

在反向传递期间执行的任何函数也用`seq=&lt;N&gt;`进行修饰。 在默认反向(使用`create_graph=False`）时，此信息无关紧要，事实上，对于所有此类函数，`N`可能只是0。 作为将这些Function对象与早期的向前传递相关联的方法，只有与反向Function对象的`apply()`方法关联的顶级范围才有用。

**Double-backward**

另一方面，如果正在进行`create_graph=True`的反向传递(换句话说，如果你设置为double-backward），则在反向期间执行每个被给一个非零，有用的`seq=&lt;N&gt;`的函数。 这些函数本身可以稍后在double-backward期间创建Function对象来执行，就像在前向传递中原始函数所做的一样。 反向和double-backward的关系在概念上与前向和反向的关系相同：函数仍然发出当前序列号标记的范围，它们创建的Function对象仍然存储那些序列号，并且在最终的double-backward期间 向后，Function对象的`apply()`范围仍然用 `stashed seq`数字标记，可以与反向传递中的`seq`数字进行比较。

```py
torch.autograd.profiler.load_nvprof(path)
```

打开一个nvprof跟踪文件并且解析autograd注释。

| 参数: | **path** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.7)")) – nvprof跟踪路径 |
| --- | --- |

## 异常检测

```py
class torch.autograd.detect_anomaly
```

上下文管理器，为自动求导引擎使能异常检测。

这做了两件事：- 在启用检测的情况下运行前向传递将允许反向传递打印创建失败的反向函数的前向操作跟踪。 - 任何生成“nan”值的反向计算都会引发错误。

示例

```py
>>> import torch
>>> from torch import autograd
>>> class MyFunc(autograd.Function):
...     @staticmethod
...     def forward(ctx, inp):
...         return inp.clone()
...     @staticmethod
...     def backward(ctx, gO):
...         # Error during the backward pass
...         raise RuntimeError("Some error in backward")
...         return gO.clone()
>>> def run_fn(a):
...     out = MyFunc.apply(a)
...     return out.sum()
>>> inp = torch.rand(10, 10, requires_grad=True)
>>> out = run_fn(inp)
>>> out.backward()
 Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "/your/pytorch/install/torch/tensor.py", line 93, in backward
 torch.autograd.backward(self, gradient, retain_graph, create_graph)
 File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
 allow_unreachable=True)  # allow_unreachable flag
 File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
 return self._forward_cls.backward(self, *args)
 File "<stdin>", line 8, in backward
 RuntimeError: Some error in backward
>>> with autograd.detect_anomaly():
...     inp = torch.rand(10, 10, requires_grad=True)
...     out = run_fn(inp)
...     out.backward()
 Traceback of forward call that caused the error:
 File "tmp.py", line 53, in <module>
 out = run_fn(inp)
 File "tmp.py", line 44, in run_fn
 out = MyFunc.apply(a)
 Traceback (most recent call last):
 File "<stdin>", line 4, in <module>
 File "/your/pytorch/install/torch/tensor.py", line 93, in backward
 torch.autograd.backward(self, gradient, retain_graph, create_graph)
 File "/your/pytorch/install/torch/autograd/__init__.py", line 90, in backward
 allow_unreachable=True)  # allow_unreachable flag
 File "/your/pytorch/install/torch/autograd/function.py", line 76, in apply
 return self._forward_cls.backward(self, *args)
 File "<stdin>", line 8, in backward
 RuntimeError: Some error in backward

```

```py
class torch.autograd.set_detect_anomaly(mode)
```

上下文管理器，为自动求导引擎设置异常检测开或关。

`set_detect_anomaly`将基于它的参数`mode`使能或禁用自动求导异常检测。它也能作为一个上下文管理器或函数使用。

异常检测行为细节见上面`detect_anomaly`。

| 参数: | **mode** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.7)")) – 标记是否使能异常检测(`True`），或禁用(`False`）。 |
| --- | --- |

