# 自动分化包 - torch.autograd

`torch.autograd`提供的类和函数执行的任意的标量值的函数的自动分化。它需要最少的改变现有的代码 - 你只需要声明`张量 `
S代表其梯度应与`requires_grad =真 [HTG11计算]关键字。`

`torch.autograd.``backward`( _tensors_ , _grad_tensors=None_ ,
_retain_graph=None_ , _create_graph=False_ , _grad_variables=None_
)[[source]](_modules/torch/autograd.html#backward)

    

计算给出张量的梯度之w.r.t.图叶。

该图是使用链式法则区分。如果有任何的`张量
`是非标量（即，它们的数据具有一个以上的元素），并且需要的梯度，然后雅可比矢量乘积将被计算，在这种情况下，函数附加地需要指定`grad_tensors
`。它应该是匹配长度的序列中，包含在雅可比矢量乘积的“载体”，通常是有区别的功能w.r.t.的梯度相应的张量（`无
`为不需要梯度张量的所有张量的可接受的值）。

此功能聚集在叶梯度 - 你可能需要调用它之前为零它们。

Parameters

    

  * **张量** （ _的张量_ 序列） - 张量，其的衍生物将被计算出来。

  * **grad_tensors** （ _的_ _（_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ [ _无序列_ ](https://docs.python.org/3/library/constants.html#None "\(in Python v3.7\)") _）_ ） - 在雅可比矢量乘积的“载体”，通常梯度WRT对应张量的每个元素。无值可以用于标量张量或那些不要求毕业生指定。如果没有值将所有grad_tensors是可以接受的，那么这种说法是可选的。

  * **retain_graph** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`假 `，用于计算所述市图表将被释放。请注意，在几乎所有情况下将此选项设置为`真 `不需要，经常可以围绕以更有效的方式来工作。默认为`create_graph`的值。

  * **create_graph** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，所述衍生物的图形将被构建，从而允许计算高阶衍生产品。默认为`假 [HTG17。`

`torch.autograd.``grad`( _outputs_ , _inputs_ , _grad_outputs=None_ ,
_retain_graph=None_ , _create_graph=False_ , _only_inputs=True_ ,
_allow_unused=False_ )[[source]](_modules/torch/autograd.html#grad)

    

计算并返回的输出梯度之和w.r.t.输入。

`grad_outputs`应该是长度匹配`
[HTG7含有在雅可比矢量乘积的“载体”，输出通常是预先计算的梯度的序列WRT每一个输出。如果输出不require_grad，则梯度可以是`无 `）。`

如果`only_inputs`是`真 `，则该函数将只返回梯度w.r.t指定输入的列表。如果是`假
`，然后梯度w.r.t.其余所有的叶子仍然会被计算，并且将累积到他们`.grad`属性。

Parameters

    

  * **输出** （ _的张量_ 序列） - 微分函数的输出。

  * **输入** （ _张量的序列_ ） - 输入w.r.t.该梯度将被返回（并且不累积到`.grad`）。

  * **grad_outputs** （ _张量_ 的序列） - 在雅可比矢量乘积的“载体”。通常梯度w.r.t.每个输出。无值可以用于标量张量或那些不要求毕业生指定。如果没有值将所有grad_tensors是可以接受的，那么这种说法是可选的。默认值：无。

  * **retain_graph** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – If `False`, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to `True`is not needed and often can be worked around in a much more efficient way. Defaults to the value of `create_graph`.

  * **create_graph** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，所述衍生物的图形将被构建，从而允许计算高阶衍生产品。默认值：`假 [HTG17。`

  * **allow_unused** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`假 `，指定计算输出时不使用的输入（和因此它们的毕业生总是零）是错误的。默认为`假 [HTG17。`

## 本地禁用梯度计算

_class_`torch.autograd.``no_grad`[[source]](_modules/torch/autograd/grad_mode.html#no_grad)

    

上下文管理器禁用梯度计算。

禁用梯度计算是推论有用，当你确信你不会叫`Tensor.backward（） [HTG3。这将减少用于计算，否则将具有 requires_grad
=真内存消耗。`

在这种模式下，每一个计算结果将具有 requires_grad =假，即使当输入具有 requires_grad =真。

使用 `enable_grad`上下文管理器时，此模式没有影响。

这个上下文管理器是线程本地;它不会在其他线程影响计算。

也可作为装饰。

例：

    
    
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
    

_class_`torch.autograd.``enable_grad`[[source]](_modules/torch/autograd/grad_mode.html#enable_grad)

    

上下文管理器，它使梯度计算。

使梯度计算，如果它已被通过 `禁用no_grad`或 set_grad_enabled ``。

This context manager is thread local; it will not affect computation in other
threads.

Also functions as a decorator.

Example:

    
    
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
    

_class_`torch.autograd.``set_grad_enabled`( _mode_
)[[source]](_modules/torch/autograd/grad_mode.html#set_grad_enabled)

    

上下文管理器，其设定梯度计算为开或关。

`set_grad_enabled`将启用或禁用根据它的自变量`模式 `梯度。它可以作为一个上下文经理或作为一个功能。

当使用 `enable_grad`上下文管理器，`set_grad_enabled（假） `没有影响。

This context manager is thread local; it will not affect computation in other
threads.

Parameters

    

**模式** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in
Python v3.7\)")） - 标志是否启用研究所（`真 `），或禁用（`假 `）。这可以用来有条件地允许梯度。

Example:

    
    
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
    

## 就地对运营张量

在autograd支持就地操作是很难的事，我们不鼓励在大多数情况下，它们的使用。
Autograd咄咄逼人的缓冲释放和再利用使得它非常有效，也有极少数场合就地操作任何显著量实际上更低的内存使用率。除非你在重内存压力工作，你可能永远需要使用它们。

### 就地正确性检查

所有`张量 `S跟踪就地操作适用于他们，如果实现检测到张在的功能之一保存落后，但它被修改IN-
事后到位，错误将引发一次向通行已启动。这确保了如果你使用就地功能，没有看到任何错误，你可以肯定的是，计算的梯度是正确的。

## 变量（不建议使用）

警告

变量API已被弃用：变量不再需要用张量使用autograd。 Autograd自动支持与`requires_grad `设置为`真 `
张量。请在下面找到发生了什么变化的快速指南：

  * `变量（张量） `和`变量（张量， requires_grad）HTG8] `仍按预期方式工作，但他们回到张量而不是变量。

  * `var.data`是同样的事情`tensor.data`。

  * 的方法，如`var.backward（）， var.detach（）， var.register_hook（） `现在在与张量的工作同样的方法名称。

此外，一个现在可以创建与`requires_grad =真 `使用工厂方法，如[ `torch.randn（） `
[张量HTG9]，](torch.html#torch.randn "torch.randn")[ `torch.zeros（） `
](torch.html#torch.zeros "torch.zeros")，[ `torch.ones（） `
](torch.html#torch.ones "torch.ones")和其他类似如下：

`autograd_tensor  =  torch.randn（（2， 3， 4）， requires_grad = TRUE） `

## 张量autograd功能

_class_`torch.``Tensor`

    

`backward`( _gradient=None_ , _retain_graph=None_ , _create_graph=False_
)[[source]](_modules/torch/tensor.html#Tensor.backward)

    

计算当前张量w.r.t.的梯度图叶。

该图是使用链式法则区分。如果张量是非标量（即，其数据具有一个以上的元素），并且需要的梯度，所述函数另外需要指定`梯度
`。它应该是匹配的类型和位置的张量，包含有区别的功能w.r.t.的梯度`自 `。

此功能聚集在叶梯度 - 你可能需要调用它之前为零它们。

Parameters

    

  * **梯度** （[ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _或_ [ _无_ ](https://docs.python.org/3/library/constants.html#None "\(in Python v3.7\)")） - 梯度w.r.t.张量。如果它是一个张量，它将被自动转换为不需要研究所除非`create_graph`为True张量。无值可以用于标量张量或那些不要求毕业生指定。如果没有值是可以接受的，然后这种说法是可选的。

  * **retain_graph** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`假 `，用于计算梯度的图表将被释放。请注意，在几乎所有情况下的设置则不需要此选项设置为True，往往可以以更有效的方式围绕工作。默认为`create_graph`的值。

  * **create_graph** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，所述衍生物的图形将被构建，从而允许计算高阶衍生产品。默认为`假 [HTG17。`

`detach`()

    

返回一个新的张量，从当前图形分离。

其结果将永远不需要梯度。

注意

回到张量股与原来相同的存储。就地对它们中的修改可以看出，并可能引发正确性检查错误。重要提示：以前，就地尺寸/步幅/存储的变化（如 resize_  /
resize_as_  /  SET_  /  transpose_
）来返回的张量也更新原有的张量。现在，这些就地变化将不再更新原有的张量，而会触发一个错误。对于稀疏张量：就地索引/值的变化（如 zero_  /
copy_  /  add_ ）发送到返回张量将不再更新原始张量，而会触发一个错误。

`detach_`()

    

分离从创建它，使其成为一个叶子图表中的张量。意见不能就地分离。

`grad`

    

该属性是`无 `缺省和成为张量在第一时间[ `呼叫向后（） `](tensors.html#torch.Tensor.backward
"torch.Tensor.backward")计算梯度为`自 `。然后，属性将包含计算出的梯度，并[ `向后（） `
](tensors.html#torch.Tensor.backward "torch.Tensor.backward")将积累（添加）梯度到它将来的呼叫。

`is_leaf`

    

有所有的张量[ `requires_grad`](tensors.html#torch.Tensor.requires_grad
"torch.Tensor.requires_grad")是`假 `将叶按约定张量。

对于张量具有[ `requires_grad`](tensors.html#torch.Tensor.requires_grad
"torch.Tensor.requires_grad")，它是`真 `，他们将叶张量如果他们被创建用户。这意味着它们不是一个操作的结果等`
grad_fn`是无。

仅叶张量人员在[ `GRAD`](tensors.html#torch.Tensor.grad "torch.Tensor.grad")至[ `
向后（在呼叫期间填充） `](tensors.html#torch.Tensor.backward
"torch.Tensor.backward")。为了得到[ `毕业 `](tensors.html#torch.Tensor.grad
"torch.Tensor.grad")填充的无叶张量，你可以使用[ `retain_grad（） `[
HTG23。](tensors.html#torch.Tensor.retain_grad "torch.Tensor.retain_grad")

Example:

    
    
    >>> a = torch.rand(10, requires_grad=True)
    >>> a.is_leaf
    True
    >>> b = torch.rand(10, requires_grad=True).cuda()
    >>> b.is_leaf
    False
    # b was created by the operation that cast a cpu Tensor into a cuda Tensor
    >>> c = torch.rand(10, requires_grad=True) + 2
    >>> c.is_leaf
    False
    # c was created by the addition operation
    >>> d = torch.rand(10).cuda()
    >>> d.is_leaf
    True
    # d does not require gradients and so has no operation creating it (that is tracked by the autograd engine)
    >>> e = torch.rand(10).cuda().requires_grad_()
    >>> e.is_leaf
    True
    # e requires gradients and has no operations creating it
    >>> f = torch.rand(10, requires_grad=True, device="cuda")
    >>> f.is_leaf
    True
    # f requires grad, has no operation creating it
    

`register_hook`( _hook_
)[[source]](_modules/torch/tensor.html#Tensor.register_hook)

    

寄存器向后钩。

钩将被称为每相对于该张量的梯度被计算的时间。钩子应该具有以下特征：

    
    
    hook(grad) -> Tensor or None
    

钩不应该修改它的参数，但它可以任选地返回一个新的梯度，这将代替[ `GRAD`](tensors.html#torch.Tensor.grad
"torch.Tensor.grad")一起使用。

该函数返回与方法`handle.remove手柄（） `，其去除从所述模块的钩。

Example:

    
    
    >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
    >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
    >>> v.backward(torch.tensor([1., 2., 3.]))
    >>> v.grad
    
     2
     4
     6
    [torch.FloatTensor of size (3,)]
    
    >>> h.remove()  # removes the hook
    

`requires_grad`

    

是`真 [HTG3如果梯度需要计算该张量，`假 [HTG7否则。``

Note

该梯度需要计算的张量，这一事实并不意味着[ `毕业 `](tensors.html#torch.Tensor.grad
"torch.Tensor.grad")属性将被填充，请参阅[ `is_leaf`
](tensors.html#torch.Tensor.is_leaf "torch.Tensor.is_leaf")的更多细节。

`retain_grad`()[[source]](_modules/torch/tensor.html#Tensor.retain_grad)

    

启用了非叶张量.grad属性。

## 函数

_class_`torch.autograd.``Function`[[source]](_modules/torch/autograd/function.html#Function)

    

记录操作历史，并定义区分OPS公式。

在`张量 `S创建一个新的函数对象，执行计算，执行的每一个操作，并记录它的发生。历史被保持在的功能的DAG的形式，与表示边缘数据依赖性（`输入 &
LT ; -  输出 `）。然后，当向后被调用时，曲线图中的拓扑排序处理时，通过调用 `向后（） `的每个 `[方法HTG20 ]函数 `
对象，并通过返回梯度到下一个 `函数 `秒。

通常情况下，用户使用的功能交互的唯一方法是通过创建子类和定义新的操作。这是延长torch.autograd的推荐方式。

每个功能对象是指仅使用一次（在正向通）。

例子：

    
    
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
    

_static_`backward`( _ctx_ , _*grad_outputs_
)[[source]](_modules/torch/autograd/function.html#Function.backward)

    

定义用于区分该操作的公式。

此功能是通过所有子类覆盖。

它必须接受上下文`CTX`作为第一个参数，其次是因为许多输出做 `向前（） `返回，并且它应该返回尽可能多的张量，因为有输入 `向前（） `
。每个参数是w.r.t给定输出的梯度，并且每个返回的值应该是梯度w.r.t.相应的输入。

上下文可以用来检索直传期间保存张量。它也有一个属性`ctx.needs_input_grad`作为表示每个输入是否需要梯度的布尔值的元组。例如， `
向后（） `将`ctx.needs_input_grad [0]  =  真 `如果第一输入为 `向前（） `需要梯度computated
WRT输出。

_static_`forward`( _ctx_ , _*args_ , _**kwargs_
)[[source]](_modules/torch/autograd/function.html#Function.forward)

    

执行该操作。

This function is to be overridden by all subclasses.

它必须接受CTX作为第一个参数的上下文，后跟任意数量的参数（张量或其它类型的）。

上下文可以用来存储可反向通期间则检索张量。

## 数值梯度检查

`torch.autograd.``gradcheck`( _func_ , _inputs_ , _eps=1e-06_ , _atol=1e-05_ ,
_rtol=0.001_ , _raise_exception=True_ , _check_sparse_nnz=False_ ,
_nondet_tol=0.0_
)[[source]](_modules/torch/autograd/gradcheck.html#gradcheck)

    

检查梯度经由针对分析梯度小有限差分计算w.r.t.在`张量能与`浮点型的和输入 `requires_grad =真 `。

数字分析梯度之间的检查使用[ `allclose（） `](torch.html#torch.allclose "torch.allclose")。

Note

的默认值被设计用于`的双精度输入 `。如果`输入 `是精度要求不高，例如，`FloatTensor`该检查很可能会失败。

Warning

如果在`任何选中的张量输入 `具有重叠的存储，即，指向相同的存储器地址（例如，不同的指数，从`torch.expand（）
`），该检查可能会失败，因为在这些指数由点扰动计算的数值梯度将在共享同一存储器地址上的所有其他指数变化的值。

Parameters

    

  * **FUNC** （ _函数_ ） - 一个Python函数，它接受张量的输入，并返回一个或张量的张量的元组

  * **输入** （ _的张量_ _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor")元组） - 函数的输入

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 摄动有限差

  * **蒂** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 绝对公差

  * **RTOL** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 相对公差

  * **raise_exception** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 指示是否如果检查失败引发异常。异常提供了有关故障的确切性质的更多信息。调试gradchecks时，这是有帮助的。

  * **check_sparse_nnz** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果为True，gradcheck允许SparseTensor输入，并且对于任何SparseTensor在输入，gradcheck将执行仅NNZ位置检查。

  * **nondet_tol** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 用于非确定性的耐受性。当通过分化运行相同的输入，结果必须或者完全匹配（默认值，0.0）或者是此公差范围内。

Returns

    

真如果所有的差值满足allclose条件

`torch.autograd.``gradgradcheck`( _func_ , _inputs_ , _grad_outputs=None_ ,
_eps=1e-06_ , _atol=1e-05_ , _rtol=0.001_ ,
_gen_non_contig_grad_outputs=False_ , _raise_exception=True_ ,
_nondet_tol=0.0_
)[[source]](_modules/torch/autograd/gradcheck.html#gradgradcheck)

    

梯度的检查梯度经由针对分析梯度小有限差分计算w.r.t.在张量`输入 `和`grad_outputs`认为是浮点型的，并通过`
requires_grad =真 [ HTG11。`

此，通过计算给定的`grad_outputs`梯度backpropagating函数检查是正确的。

The check between numerical and analytical gradients uses
[`allclose()`](torch.html#torch.allclose "torch.allclose").

Note

的默认值被设计用于`输入 `和`的双精度grad_outputs`。如果它们的精度要求不高，例如该检查很可能会失败，`FloatTensor
`。

Warning

如果在`输入 `和任何检查张量`grad_outputs`具有重叠的存储，即，指向相同的存储器地址（例如，不同的指数，从`
torch.expand（） `），该检查可能会失败，因为在这些指数由点扰动计算的数值梯度将在共享同一存储器地址上的所有其他指数变化的值。

Parameters

    

  * **func** ( _function_ ) – a Python function that takes Tensor inputs and returns a Tensor or a tuple of Tensors

  * **inputs** ( _tuple of Tensor_ _or_[ _Tensor_](tensors.html#torch.Tensor "torch.Tensor")) – inputs to the function

  * **grad_outputs** （ _张量_ _或_ [ _张量_ ](tensors.html#torch.Tensor "torch.Tensor") _，_ _可选的元组_ ） - 相对于所述功能的输出的梯度。

  * **eps** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – perturbation for finite differences

  * **atol** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – absolute tolerance

  * **rtol** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – relative tolerance

  * **gen_non_contig_grad_outputs** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`grad_outputs`是`无 `和`gen_non_contig_grad_outputs`是`真 `中，随机产生的梯度输出做成非连续

  * **raise_exception** ([ _bool_](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _,_ _optional_ ) – indicating whether to raise an exception if the check fails. The exception gives more information about the exact nature of the failure. This is helpful when debugging gradchecks.

  * **nondet_tol** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 用于非确定性的耐受性。当通过分化运行相同的输入，结果必须或者完全匹配（默认值，0.0）或者是此公差范围内。注意，非确定性的梯度少量会导致在二阶导数更大的误差。

Returns

    

True if all differences satisfy allclose condition

## 探查

Autograd包括一个分析器，它可让您检查不同运营商的成本模型中 - 无论是CPU和GPU上。有目前实施的两种模式 - 仅CPU使用 `个人资料 `
[HTG5。并使用 `emit_nvtx`nvprof基于（寄存器CPU和GPU两者的活性）。

_class_`torch.autograd.profiler.``profile`( _enabled=True_ , _use_cuda=False_
, _record_shapes=False_
)[[source]](_modules/torch/autograd/profiler.html#profile)

    

管理autograd探查状态并保持结果的摘要上下文管理。引擎盖下它只是记录在C
++中执行的功能的事件和暴露这些事件到Python。你可以用任何代码到它，它只会报告的PyTorch功能运行。

Parameters

    

  * **启用** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 此设置为false使此背景下经理无操作。默认值：`真 [HTG13。`

  * **use_cuda** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 允许使用cudaEvent API以及CUDA事件的定时。添加约4us的开销每个张量操作。默认值：`假 `

  * **record_shapes** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果形状将录制设定，关于输入尺寸信息将被收集。这允许人们看到哪些尺寸已经罩，并进一步按他们下使用利用prof.key_averages（group_by_input_shape =真）。请注意，形状记录可能会扭曲你的分析数据。它建议使用单独的运行具有和不具有形状记录来验证定时。最有可能的倾斜将是微不足道的最底层的事件（在嵌套函数调用的情况下）。但对于更高层次的功能，总的自我CPU时间可能会因为人为的形状收集的增加。

例

    
    
    >>> x = torch.randn((1, 1), requires_grad=True)
    >>> with torch.autograd.profiler.profile() as prof:
    >>>     for _ in range(100):  # any normal python code, really!
    >>>         y = x ** 2
    >>          y.backward()
    >>> # NOTE: some columns were removed for brevity
    >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    -----------------------------------  ---------------  ---------------  ---------------
    Name                                 Self CPU total   CPU time avg     Number of Calls
    -----------------------------------  ---------------  ---------------  ---------------
    mul                                  32.048ms         32.048ms         200
    pow                                  27.041ms         27.041ms         200
    PowBackward0                         9.727ms          55.483ms         100
    torch::autograd::AccumulateGrad      9.148ms          9.148ms          100
    torch::autograd::GraphRoot           691.816us        691.816us        100
    -----------------------------------  ---------------  ---------------  ---------------
    

`export_chrome_trace`( _path_
)[[source]](_modules/torch/autograd/profiler.html#profile.export_chrome_trace)

    

出口的一个EVENTLIST以及Chrome的跟踪工具文件。

检查点可以在以后加载并`铬下检查：//追踪 `URL。

Parameters

    

**路径** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in
Python v3.7\)")） - ，其中所述迹线将被写入路径。

`key_averages`( _group_by_input_shape=False_
)[[source]](_modules/torch/autograd/profiler.html#profile.key_averages)

    

平均在他们的密钥对所有功能的事件。

@参数group_by_input_shapes的关键将成为（事件名称，输入尺寸），而不仅仅是事件名称。这是有用的，看看哪些维度有助于运行最并可与维特定的优化或选择最佳候选量化帮助（又名拟合的车顶线条）

Returns

    

含有FunctionEventAvg对象的EVENTLIST。

_property_`self_cpu_time_total`

    

返回花费在所有的自我时代的所有事件的总和获得的CPU总时间。

`table`( _sort_by=None_ , _row_limit=100_ , _header=None_
)[[source]](_modules/torch/autograd/profiler.html#profile.table)

    

打印的EVENTLIST作为一个很好的格式化的表格。

Parameters

    

**sort_by** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str
"\(in Python v3.7\)") _，_ _可选_ ） - 属性用于条目进行排序。默认情况下，它们被印在为它们注册的顺序相同。有效键包括：`
CPU_TIME`，`cuda_time`，`cpu_time_total`，`cuda_time_total`，`计数 `。

Returns

    

将含有表字符串。

`total_average`()[[source]](_modules/torch/autograd/profiler.html#profile.total_average)

    

平均值的所有事件。

Returns

    

一个FunctionEventAvg对象。

_class_`torch.autograd.profiler.``emit_nvtx`( _enabled=True_ ,
_record_shapes=False_
)[[source]](_modules/torch/autograd/profiler.html#emit_nvtx)

    

情境管理，使每一个autograd操作发出NVTX范围。

下nvprof运行程序时是非常有用的：

    
    
    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>
    

不幸的是，有没有办法强迫nvprof刷新它收集到磁盘上的数据，所以对于CUDA剖析一个具有使用此背景下经理注释nvprof跟踪和等待的过程中检查他们之前退出。然后，无论是NVIDIA的视觉分析器（nvvp）可以被用于可视化的时间线，或
`torch.autograd.profiler.load_nvprof（） `可以加载对结果检查如在Python REPL。

Parameters

    

  * **启用** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ _，_ _默认=真_ ） - 设置`启用=假 `将此情况管理器无操作。默认值：`真 [HTG21。`

  * **record_shapes** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ _，_ _默认=假_ ） - 如果`record_shapes =真 `时，nvtx范围包裹每个autograd运算将追加关于由该运算所接收的张量参数的大小的信息，在下面的格式：`[[arg0.size（0）， arg0.size（1）， ...]， [arg1.size（0）， arg1.size（1）， ...]， ...]`非张量参数将被表示为`[]`。参数将在它们由后端运算收到的顺序列出。请注意，这个顺序可能不匹配，使这些参数传递的Python端的顺序。还要注意的是形状记录可能会增加nvtx范围内创建的开销。

Example

    
    
    >>> with torch.cuda.profiler.profile():
    ...     model(x) # Warmup CUDA memory allocator and profiler
    ...     with torch.autograd.profiler.emit_nvtx():
    ...         model(x)
    

**前后的相关性**

当观看使用 `中创建的简档emit_nvtx`在NVIDIA的视觉分析器，各后向通运算与相应的前向通运算关联可能是困难的。为了缓解此任务， `
emit_nvtx`附加序列数信息向它生成的范围。

在直传，每个功能范围装饰有`SEQ = & LT ; N & GT ;`。 `SEQ
`是一个运行计数器，递增每一个新的后向功能对象被创建并藏匿用于向后时间。因此，`SEQ = & LT ; N & GT ;
`与每个进功能范围相关联的注释告诉你，如果一个向后作用目的是通过此正向函数创建，后向对象将接收序号N.在向后通，顶层范围包裹每个C ++向后功能的`
申请（） `呼叫装饰与`藏匿 SEQ = & LT ; M & GT ;`。 `M`是向后对象与所创建的序列号。通过向后`比较`藏匿
序列​​ `数字序列 `正向号码，你可以跟踪哪些向前运创建的每个向后作用。

向后传递期间执行的任何功能也装饰有`SEQ = & LT ; N & GT ;`。默认向后期间（与`create_graph =假
`）这个信息是无关的，而事实上，`N`可以简单地是0对于所有此类职能。只有具有向后功能对象相关联的顶层范围`申请（）
`方法是有用的，作为方式关联与较早直传这些功能对象。

**双向后**

如果，在另一方面，后向通用`create_graph =真 `正在进行（换句话说，如果要设置为双向后）期间落后，每个功能的执行给出非零的，有用的`SEQ
= & LT ; N & GT ;
`。这些功能可以自己创建要在以后执行的函数对象双重落后，就像直传原有功能一样。向后和双向后之间的关系是概念性一样向前和向后之间的关系：该功能仍然发射电流序列号标记的范围内，功能对象他们创建仍然藏匿那些序列号，和最终双期间落后，函数对象`
申请（） `范围仍具有标记`藏匿 SEQ`的数字，其可以进行比较，以 SEQ 号码从反向通。

`torch.autograd.profiler.``load_nvprof`( _path_
)[[source]](_modules/torch/autograd/profiler.html#load_nvprof)

    

打开一个nvprof跟踪文件，并解析autograd注解。

Parameters

    

**路径** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in
Python v3.7\)")） - 路径nvprof跟踪

## 异常检测

_class_`torch.autograd.``detect_anomaly`[[source]](_modules/torch/autograd/anomaly_mode.html#detect_anomaly)

    

上下文管理器启用的autograd发动机异常检测。

这做了两两件事： - 运行直传启用检测将允许向通行打印创建失败后退功能正向操作的回溯。 - 产生任何向后计算“男”值将产生一个错误。

Example

    
    
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
    

_class_`torch.autograd.``set_detect_anomaly`( _mode_
)[[source]](_modules/torch/autograd/anomaly_mode.html#set_detect_anomaly)

    

上下文管理器设置或关闭的autograd发动机异常检测。

`set_detect_anomaly 根据它的自变量`模式 ``将启用或禁用所述autograd异常检测。它可以作为一个上下文经理或作为一个功能。

参见`detect_anomaly`上述用于异常检测的行为的细节。

Parameters

    

**模式** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in
Python v3.7\)")） - 标志是否启用异常检测（`真 `），或禁止（`假 `）。

[Next ![](_static/images/chevron-right-orange.svg)](distributed.html
"Distributed communication package - torch.distributed")
[![](_static/images/chevron-right-orange.svg) Previous](optim.html
"torch.optim")

* * *

©版权所有2019年，火炬贡献者。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * 自动微分包 - torch.autograd 
    * 局部禁用梯度计算
    * 就地上张量运算
      * [HTG0在就地正确性检查
    * 变量（不建议使用）
    * 张量autograd功能
    * 函数
    * 数值梯度检查
    * 探查
    * 异常检测

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

