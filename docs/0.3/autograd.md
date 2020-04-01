# Automatic differentiation package - torch.autograd

> 译者：[@ZhenLei Xu](https://github.com/HadXu)
> 
> 校对者：[@青梅往事](https://github.com/2556120684)

torch.autograd 提供了类和函数用来对任意标量函数进行求导.只需要对已有的代码进行微小的改变-只需要将所有的 tensors 包含在 `Variable` 对象中即可.

```py
torch.autograd.backward(variables, grad_variables=None, retain_graph=None, create_graph=None, retain_variables=None)
```

给定图某一个的节点变量variables,计算对该变量求导的梯度和.

计算图可以通过链式法则求导.如果任何 `variables` 都是非标量(比如 他们的 data 属性中有多个元素)并且需要求导, 那么此函数需要指定 `grad_variables`. 它的长度应该和variables的长度匹配,里面保存了相关 variable 的梯度 (对于不需要 gradient tensor 的 variable, 应制定为 None).

此函数累积叶子节点 variables 计算的梯度 - 调用此函数之前应先将叶子节点 variables 梯度置零.

参数: 

* `variables (Variable 列表)`: 被求微分的叶子节点. 
* `grad_variables ((Tensor, Variable 或 None) 列表)`:对应 variable 的梯度. 任何张量将自动转换为变量除非`create_graph` 是 `True`. 没有值可以被指定为标量变量或者不需要被求导. 如果没有值被所有的grad_variables接受, 那么该参数是可以被省略的.
*   `retain_graph (bool, 可选)`: 如果是 `False`, 该图计算过的梯度被释放掉.注意的是,几乎所有情况都设置为``True``并不是必须的并且能够高效的计算.将该 `create_graph` 参数值设置为默认即可.
*   `create_graph (bool, 可选)`: 如果是 `True`, 将会建立一个梯度图, 用来求解高阶导数.默认为 `False`, 除非 `grad_variables` 拥有不止一个 易变的 Variable.

```py
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=None, only_inputs=True, allow_unused=False)
```

计算并返回给定值的梯度的和.

`grad_outputs` 是一个列表同时长度与 `output` 一样, 存放了预先计算 input 的梯度的和. 如果 output 不需要被求导, 那么梯度将为 `None`). 当不需要派生图时,可以将梯度作为张量,或者作为变量,在这种情况下,图将被创建.

如果参数 `only_inputs` 为 `True`, 该方法将会返回给定输入的梯度值列表.如果为 `False`, 那么遗留下来的所有叶子节点的梯度都会被计算, 被且会被列加到 `.grad` 参数中.

参数: 

* `outputs (变量序列)`: 梯度函数的返回值. 
* `inputs (变量序列)`: 需要计算的梯度的输入 (并且不会被累加到 `.grad` 参数中). 
* `grad_outputs (张量或变量序列)`: 每一个输出的梯度. 所有的张量都会变成变量并且是可变的除非参数 `create_graph` 为 `True`. 没有值可以被指定为标量变量或者不需要变化的值. 如果所有 grad_variabls 都可以接受 None 值,那么这个参数是可选的.
*   `retain_graph (bool, 可选)`: 如果是 `False`, 用于计算 grad 的图将被释放. 几乎所有情况都设置为``True`` 并不是必须的并且能够高效地运行. 默认与 `create_graph` 参数一样.
*   `create_graph (bool, 可选)`: 如果是 `True`, 梯度图将会被建立,用来求解高阶导数. 默认为 `False` , 除非参数 `grad_variables` 包含不只一个变量.
*   `only_inputs (bool, 可选)`: 如果是 `True`, 叶子节点的导数将会在图中, 但是不会出现在参数 `inputs` 也不会被计算以及累加. 默认为 `True`.
*   `allow_unused (bool, 可选)`: 如果是 `False`, 指定计算输出时未使用的输入(因此它们的 grad 始终为零）是错误的. 默认为 `False`.

## Variable (变量)

### API compatibility

Variable API 几乎与常规 Tensor API 相同(一些会覆盖梯度计算输入的内置方法除外). 在大多数情况下, 变量量可以安全地替换张量并且代码将保持正常工作. 因为这个, 我们没有记录变量上的所有操作, 你应该参阅 [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 文档来查看变量上的所有操作.

### In-place operations on Variables

在 autograd 支持就地操作是一件困难的事情, 在大多数情况下我们不鼓励使用. Autograd 积极的缓冲区释放和重用使得它非常高效, 而且很少有就地操作实际上大量地降低了内存使用量的情况. 除非你正在大量的的内存压力下运行, 否则你可能永远不需要使用它们.

### In-place correctness checks

所有的 `Variable` 跟踪适用于它们的就地操作, 并且如果实现检测到一个变量是否被其中一个函数后台保存, 但是之后它被就地修改了, 会在开始求导时会报出异常. 这确保了如果你在就地使用函数并没有看到任何错误, 你可以肯定的是计算变量是正确的.

```py
class torch.autograd.Variable
```

封装一个张量用来各种操作.

变量是张量对象周围的轻包装,能够拥有导数等数据, 这个引用允许回溯整个操作链创建数据. 如果变量已经由用户创建, 它的 grad_fn 为 `None` 我们称之为叶子节点.

由于 autograd 只支持标量值函数微分, grad 大小始终与数据大小匹配. 此外,导数通常只分配 叶变量,否则将始终为零.

参数: 

* `data`: 包裹任何类型的张量. 
* `grad`: 变量保持类型和位置匹配的变量 `.data`. 这个属性是懒惰的分配,不能被重新分配. 
* `requires_grad`: 指示变量是否已被使用的布尔值由包含任何变量的子图创建,需要它. 有关更多详细信息,请参阅 excluded-subgraphs.只能在叶变量上进行更改. 
* `volatile`: 布尔值表示应该使用变量推理模式,即不保存历史. 查看 [反向排除 subgraphs (子图)](notes/autograd.html#excluding-subgraphs) 更多细节. 只能在叶变量上进行更改. 
* `is_leaf`: 指示是否为叶子节点,即是否由用户创建的节点. 
* `grad_fn`: 导数函数跟踪.

参数: 

* `data (任何 tensor 类)`: 用来包装的张量. 
* `requires_grad (bool)`: 指示是否要被求导. **仅限关键字.** 
* `volatile (bool)`: 指示是否可变. **仅限关键字.**

```py
backward(gradient=None, retain_graph=None, create_graph=None, retain_variables=None)
```

给定图叶子节点计算导数.

该图使用链式规则进行计算. 如果变量是非标量(即其数据具有多个元素）并且需要 改变,该功能另外需要指定“梯度”.它应该是一个包含匹配类型和位置的张量 微分函数的梯度w.r.t. `self` .

这个功能在叶子上累积梯度 - 你可能需要调用之前将它们置零.

参数: 

*    `gradient (Tensor, Variable or None)`: 计算变量的梯度. 如果是张量,则会自动转换到一个变量,这是挥发性的,除非 `create_graph` 为真.没有值可以被指定为标量变量或那些 不要求毕业. 如果一个None值是可以接受的这个参数是可选的.

*   `retain_graph (bool, 可选)`: 如果 “False” ,则用于计算的图形导数将被释放. 请注意,在几乎所有情况下设置这个选项为 True 是不需要的,通常可以解决在一个更有效的方式. 默认值为`create_graph`.

*   `create_graph (bool, 可选)`: 如果“真”,派生图将会被构造,允许计算更高阶的导数. 默认为 `False`,除非 `gradient` 是一个volatile变量.

```py
detach()
```

将一个Variable从创建它的图中分离,并把它设置成 leaf variable.

返回变量使用与原始数据张量相同的数据张量,其中任何一个的就地修改都将被看到,并可能触发 错误在正确性检查.

```py
detach_()
```

将一个 Variable 从创建它的图中分离,并把它设置成 leaf variable.

```py
register_hook(hook)
```

注册一个backward钩子.

每次gradients被计算的时候,这个 hook 都被调用 .hook 应该拥有以下签名:

> hook(grad) -&gt; Variable or None

hook不应该修改它的输入,但是它可以选择性的返回一个替代当前梯度的新梯度.

这个函数返回一个 句柄 (handle).它有一个方法 handle.remove(),可以用这个方法将 hook 从 module 移除.

示例：

```py
>>> v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
>>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
>>> v.backward(torch.Tensor([1, 1, 1]))
>>> v.grad.data
 2
 2
 2
[torch.FloatTensor of size 3]
>>> h.remove()  # removes the hook

```

```py
retain_grad()
```

为非叶变量启用 .grad 属性.

## Function (函数)

```py
class torch.autograd.Function
```

记录操作历史记录并定义区分操作的方法.

> 每个执行在 Varaibles 上的 operation 都会创建一个 Function 对象,这个 Function 对象执行计算工作,同时记录下来.这个历史以有向无环图的形式保存下来, 有向图的节点为 functions ,有向图的边代表数据依赖关系 (input&lt;-output).之后,当 backward 被调用的时候,计算图以拓扑顺序处理,通过调用每个 Function 对象的 backward(), 同时将返回的梯度传递给下一个 Function.

通常情况下,用户能和 Functions 交互的唯一方法就是创建 Function 的子类,定义新的 operation. 这是扩展 torch.autograd 的推荐方法.

每个 Function 只被使用一次(在forward过程中).

参数: `requires_grad`: 布尔类型依赖于方法 `backward()` 会不会还会被使用.

比如:

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
>>>         result, = ctx.saved_variables
>>>         return grad_output * result

```

_static_ `backward`(_ctx_, _*grad_outputs_)[[source]](_modules/torch/autograd/function.html#Function.backward)

定义反向传播操作

这个方法将会被继承他的所有子类覆盖.

第一个参数为上下文参数, 接下来可以输入任何张量或变量 (张量或其他类型), 并且有多个返回值, 并且为函数 `forward()` 的输入. 每个参数都是给定输出的导数, 并且每一个输出都是输入的导数.

上下文可以用来检索转发过程中保存的变量.

```py
static forward(ctx, *args, **kwargs)
```

进行操作.

这个方法将会被继承他的所有子类覆盖.

第一个参数为上下文参数,接下来可以输入任何张量或变量 (张量或其他类型).

上下文可以用来存储可以在回传期间检索的变量.

## Profiler(分析器)

Autograd 包含一个分析器, 可以让你检查你的模型在CPU 和 GPU 上不同运算的成本. 目前实现有两种模式 - 只使用 CPU 的 `profile`. 和基于 nvprof (注册 CPU 和 GPU 活动) 的方式使用 `emit_nvtx`.

```py
class torch.autograd.profiler.profile(enabled=True)
```

结果的评价指标.

参数：`enabled (bool, 可选)` – 如果设置为 False ,则没有评价指标. Default: `True`.


示例：

```py
>>> x = Variable(torch.randn(1, 1), requires_grad=True)
>>> with torch.autograd.profiler.profile() as prof:
...     y = x ** 2
...     y.backward()
>>> # NOTE: some columns were removed for brevity
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

将EventList导出为Chrome跟踪工具文件.

断点能够通过 `chrome://tracing` URL来读取.

参数：`path (str)` – 制定断点写的路径.


```py
key_averages()
```

平均所有的功能指标通过他们的键.

返回值：包含 FunctionEventAvg 对象的 EventList.


```py
table(sort_by=None)
```

打印操作表

参数：`sort_by (str, 可选)` – 用来对参数进行排序. 默认情况下,它们以与登记相同的顺序打印. 有效的键: `cpu_time`, `cuda_time`, `cpu_time_total`, `cuda_time_total`, `count`.

返回值：包含表的字符串.


```py
total_average()
```

所有事件的平均指标.

返回值：一个 FunctionEventAvg 对象.


```py
class torch.autograd.profiler.emit_nvtx(enabled=True)
```

使每个autograd操作都发出一个NVTX范围的上下文管理器.

如下使用是正确的:

```py
nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

```

不幸的是,没有办法强制nvprof刷新收集到的数据到磁盘,因此对于 CUDA 分析,必须使用此上下文管理器进行注释 nvprof 跟踪并等待进程在检查之前退出. 然后,可以使用NVIDIA Visual Profiler(nvvp）来显示时间轴,或者 `torch.autograd.profiler.load_nvprof()` 可以加载检查结果.

参数：`enabled (bool, 可选)` – 如果设置为 False ,则没有评价指标. 默认: `True`.


示例：

```py
>>> with torch.cuda.profiler.profile():
...     model(x) # Warmup CUDA memory allocator and profiler
...     with torch.autograd.profiler.emit_nvtx():
...         model(x)

```

```py
torch.autograd.profiler.load_nvprof(path)
```

打开 nvprof trace 文件.

参数：`path (str)` – nvprof trace 文件路径.
