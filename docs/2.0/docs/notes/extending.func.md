# 使用 autograd.Function 扩展 torch.func [¶](#extending-torch-func-with-autograd-function "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/extending.func>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/extending.func.html>


 所以你想将 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 与 [`torch.func`](../func.api.html#module-torch.func "torch.func") 转换为 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") ， [`torch.func.grad()`](../generated/torch.func.grad.html#torch.func.grad "torch.func.grad") 等


 有两个主要用例：



* 您希望调用不包含 PyTorch 操作的代码并使其与函数转换一起使用。也就是说， [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 的向前/向后等调用来自其他系统(如 C++、CUDA)的函数, numpy.
* 您希望指定自定义渐变规则，例如 JAX 的 [custom_vjp/custom_jvp](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)


 PyTorch 将这两个概念结合到 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 中。


## 基本用法 [¶](#basic-usage "此标题的永久链接")


 本指南假设您熟悉 [扩展 torch.autograd](extending.html#extending-autograd) ，它解释了如何使用 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function")。


[`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 可以有一个 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 接受 ctx 对象，或者它可以有单独的 [`forward()`](../generated/torch. autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") (不接受 `ctx` )和修改 `ctx` 的 `setup_context()` 静态方法目的。


 函数转换仅支持后者：



* [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 是执行该操作的代码，它不应接受 `ctx` 对象。
* `setup_context(ctx,inputs,output)` 是您可以在 `ctx` 上调用方法的代码。您应该在此处保存向后tensor(通过调用 `ctx.save_for_backward(*tensors)` )，或保存非tensor(通过将它们分配给 `ctx` 对象)。


 因为 `setup_context()` 只接受 `inputs` 和 `output` ，所以可以保存的唯一数量是输入或输出中的对象(例如 Tensors)或从它们派生的数量(例如 `Tensor.shape` ).如果您希望从 [`Function.forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd. Function.forward") 向后，那么您需要将其作为 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward " 的输出返回torch.autograd.Function.forward") 以便将其传递给 `setup_context()` 。


 根据变换，



* 支持反向模式 AD ( [`torch.func.grad()`](../generated/torch.func.grad.html#torch.func.grad "torch.func.grad") , [`torch.func.vjp()`](../generated/torch.func.vjp.html#torch.func.vjp "torch.func.vjp") ),[`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 需要 [`backward()`](../generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward "torch.autograd.Function.backward") staticmethod.
* 支持 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") ， [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 需要一个 [`vmap()`](../generated/torch.autograd.Function.vmap.html#torch.autograd.Function.vmap "torch.autograd.Function.vmap") staticmethod.
* 支持 [`torch.func.jvp()`](../generated/torch.func.jvp.html#torch.func.jvp "torch.func.jvp") ， [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 需要 [`jvp( )`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") staticmethod.
* 支持变换的组合(例如 [`torch. func.jacrev()`](../生成/torch.func.jacrev.html#torch.func.jacrev "torch.func.jacrev") , [`torch.func.jacfwd()`](../generated/torch.func.jacfwd.html#torch.func.jacfwd "torch.func.jacfwd") , [`torch.func.hessian()`](../generated/torch.func.hessian.html#torch.func.hessian "torch.func.hessian") ) – 您可能需要以上多个。


 为了使 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 可以与函数转换任意组合，我们建议除 [ `forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 和 `setup_context()` 必须是可转换的：也就是说，它们必须仅包含 PyTorchoperators 或调用其他 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") (可能会调用 C++/CUDA/等)。


 让我们看一些常见用例的示例。


### 示例 1：autograd.Function 调用另一个系统 [¶](#example-1-autograd-function-calls-into-another-system "Permalink to this header")


 常见的情况是 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 具有forward() 和backward() 调用另一个系统(如C++、CUDA、numpy、triton)。


```
import torch
import numpy as np

def to_numpy(tensor):
    return tensor.cpu().numpy()

class NumpySort(torch.autograd.Function):
    # Note that forward does not take ctx
    @staticmethod
    def forward(x, dim):
        device = x.device
        x = to_numpy(x)
        ind = np.argsort(x, axis=dim)
        ind_inv = np.argsort(ind, axis=dim)
        result = np.take_along_axis(x, ind, axis=dim)
        # Any intermediates to be saved in backward must be returned as
        # outputs.
        return (
            # The desired output
            torch.tensor(result, device=device),
            # intermediate to save for backward
            torch.tensor(ind, device=device),
            # intermediate to save for backward
            torch.tensor(ind_inv, device=device),
        )

    # setup_context is responsible for calling methods and/or assigning to
    # the ctx object. Please do not do additional compute (e.g. add
    # Tensors together) in setup_context.
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, dim = inputs
        # Note that output is whatever you returned from forward.
        # If you returned multiple values, then output is a Tuple of multiple values.
        # If you returned a single Tensor, then output is a Tensor.
        # If you returned a Tuple with a single Tensor, then output is a
        # Tuple with a single Tensor.
        _, ind, ind_inv = output
        ctx.mark_non_differentiable(ind, ind_inv)
        # Tensors must be saved via ctx.save_for_backward. Please do not
        # assign them directly onto the ctx object.
        ctx.save_for_backward(ind, ind_inv)
        # Non-tensors may be saved by assigning them as attributes on the ctx object.
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        # For the autograd.Function to be arbitrarily composable with function
        # transforms, all staticmethod other than forward and setup_context
        # must be implemented in a "transformable" way; that is, they must
        # only consist of PyTorch operations or autograd.Function.
        #
        # For example, this allows us to do double backwards and/or compute
        # second order gradients.
        #
        # We've written the backward pass of NumpySort in terms of another
        # autograd.Function, NumpyTake.
        ind, ind_inv = ctx.saved_tensors
        return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

class NumpyTake(torch.autograd.Function):
    @staticmethod
    def forward(x, ind, ind_inv, dim):
        device = x.device
        x = to_numpy(x)
        ind = to_numpy(ind)
        return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, ind, ind_inv, dim = inputs
        ctx.save_for_backward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        ind, ind_inv = ctx.saved_tensors
        result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
        return result, None, None, None

```


 现在，为了更容易使用“NumpySort”(隐藏中间结果作为输出，并允许默认参数和 kwargs)，我们创建一个调用它的新函数：


```
def numpy_sort(x, dim=-1):
    result, _, _ = NumpySort.apply(x, dim)
    return result

```


 这是一个健全性检查：


```
x = torch.randn(2, 3)
grad_x = torch.func.grad(lambda x: numpy_sort(x).sum())(x)
assert torch.allclose(grad_x, torch.ones_like(x))

```


### 示例 2：autograd.Function 指定自定义渐变规则 [¶](#example-2-autograd-function-species-custom-gradient-rules "Permalink to this header")


 另一个常见的情况是使用 PyTorchoperations 实现的 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function")。 PyTorch 能够自动计算 PyTorch 操作的梯度，但也许我们希望自定义梯度的计算方式。我们可能想要与 PyTorch 提供的不同的自定义向后的一些原因是：



* 提高数值稳定性 
* 改变后向的性能特征 
* 改变边缘情况的处理方式(例如 nans、inf)
* 修改梯度(例如梯度裁剪)


 这是函数 `y = x ** 3` 的 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 示例，其中我们更改性能特征(通常在后向传递过程中发生的一些计算，即计算 dx，发生在前向传递中)。


```
class MyCube(torch.autograd.Function):
    @staticmethod
    def forward(x):
        result = x ** 3
        # In regular PyTorch, if we had just run y = x ** 3, then the backward
        # pass computes dx = 3 * x ** 2. In this autograd.Function, we've done
        # that computation here in the forward pass instead.
        dx = 3 * x ** 2
        return result, dx

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        result, dx = output
        ctx.save_for_backward(x, dx)

    @staticmethod
    def backward(ctx, grad_output, grad_dx):
        x, dx = ctx.saved_tensors
        # In order for the autograd.Function to work with higher-order
        # gradients, we must add the gradient contribution of `dx`.
        result = grad_output * dx + grad_dx * 6 * x
        return result

```


 现在，为了更容易使用“NumpySort”(并隐藏中间结果作为输出)，我们创建一个调用它的新函数：


```
def my_cube(x):
    result, _ = MyCube.apply(x)
    return result

```


 这是计算二阶梯度的健全性检查：


```
x = torch.randn([])
ggx = torch.func.grad(torch.func.grad(my_cube))(x)
assert torch.allclose(ggx, 6 * x)

```


### 限制和陷阱 [¶](#limitations-and-gotchas "此标题的永久链接")


!!! warning "警告"

    请仔细阅读 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 与 torch.func 转换的这些限制。我们无法捕获许多这样的情况和错误，因此它们会导致未定义的行为。


 请不要将正在转换的tensor、haverequires_grad=True 或双tensor捕获到 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function " 的方法中torch.autograd.Function") 。完全安全的方法是确保在 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 的任何方法中使用的 onlyTensors 必须直接作为输入传递(或通过 ctx 对象)，而不是来自 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 外部。


[`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 不处理 pytree 中的tensor(任意嵌套的 Python 数据结构，可能包含也可能不包含tensor)。对于要由 autograd 跟踪的tensor，必须将它们作为参数直接传递给 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 。这与 jax.{custom_vjp, custom_jvp} 形成对比，jax.{custom_vjp, custom_jvp} 接受 pytree。


 请仅使用 [`save_for_backward()`](../generated/torch.autograd.function.FunctionCtx.save_for_backward.html#torch.autograd.function.FunctionCtx.save_for_backward "torch.autograd.function.FunctionCtx. save_for_backward") 或 `save_for_forward()` 来保存tensor。请不要将tensor或tensor集合直接分配到 ctx 对象上 
- 这些tensor将不会被跟踪


## [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") 支持 [¶](#torch-vmap-support "永久链接到此标题")


 要将 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 与 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") 一起使用，您必须：



* 提供一个 [`vmap()`](../generated/torch.autograd.Function.vmap.html#torch.autograd.Function.vmap "torch.autograd.Function.vmap") 静态方法，告诉我们 [torch.vmap()](https://pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) 下 [torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) 的行为
* 要求我们通过设置 `generate_vmap_rule=True` 来自动生成它。


### 自动生成 vmap 规则 [¶](#automatically-generate-a-vmap-rule "Permalink to this header")


 如果您的 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 满足以下附加约束，那么我们就能够为其生成 vmap 规则。如果它不满足约束或者您想要在 vmap 下自定义行为，请手动定义 vmap 静态方法(请参阅下一节)。


!!! warning "警告"

    我们不容易优雅地检查以下约束和错误。违反约束可能会导致未定义的行为。



* [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 的 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward")、[`backward()`](../generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward "torch.autograd.Function.backward")(如果存在)和 [`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp")(如果存在)静态方法必须可以通过 [torch.vmap()](https://pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap "torch.vmap") 进行转换。 也就是说，它们必须仅包含 PyTorch 操作(而不是例如 NumPy 或自定义 CUDA 内核)。


 例子：


```
class MyCube(torch.autograd.Function):
    # Set generate_vmap_rule to True to ask PyTorch to automatically generate
    # a vmap rule.
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        result = x ** 3
        dx = 3 * x ** 2
        return result, dx

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        result, dx = output
        ctx.save_for_backward(x, dx)

    @staticmethod
    def backward(ctx, grad_output, grad_dx):
        x, dx = ctx.saved_tensors
        result = grad_output * dx + grad_dx * 6 * x
        return result

def my_cube(x):
    result, dx = MyCube.apply(x)
    return result

x = torch.randn(3)
result = torch.vmap(my_cube)(x)
assert torch.allclose(result, x ** 3)

```


### 定义 vmap 静态方法 [¶](#defining-the-vmap-staticmethod "永久链接到此标题")


 如果你的 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 调用另一个系统(如 NumPy、C++、CUDA、triton)，那么得到它要与 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") 或使用它的转换一起使用，您需要手动定义 [`vmap ()`](../generated/torch.autograd.Function.vmap.html#torch.autograd.Function.vmap "torch.autograd.Function.vmap") 静态方法。


 根据您要使用的转换和您的用例，您可能不需要添加 [`vmap()`](../generated/torch.autograd.Function.vmap.html#torch.autograd.Function.vmap " torch.autograd.Function.vmap") 静态方法到你的所有 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") ：



* 例如，[`torch.func.jaacrev()`](../generated/torch.func.jaacrev.html#torch.func.jaacrev "torch.func.jaacrev") 执行 [`vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") 向后传递。所以，如果您只对使用 [`torch.func.jacrev()`](../generated/torch.func.jaacrev.html#torch.func.jacrev "torch.func.jaacrev") ，仅 [`backward()`](../generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward "torch.autograd.Function.backward") staticmethod 需要可 vmappable。


 我们建议确保您的所有 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 都支持 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") 不过，特别是如果您正在编写第三方库并且您想要 [`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 与 [`torch.func()`](../func.api.html#module-torch.func "torch.func ”)转变。


 从概念上讲， vmap staticmethod 负责定义 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward" ) 应该在 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") 下运行。也就是说，它定义了如何转换 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 来运行在具有附加维度的输入上(被映射的维度)。这类似于在 PyTorch 操作上实现 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") 的方式：对于每个操作，我们定义一个 vmap 规则(有时也称为“批处理规则”)。


 以下是如何定义 [`vmap()`](../generated/torch.autograd.Function.vmap.html#torch.autograd.Function.vmap "torch.autograd.Function.vmap") 静态方法：



* 签名是 `vmap(info, in_dims: Tuple[Optional[int]], *args)` ，其中 `*args` 与 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward").
* vmap 静态方法负责定义 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 应该在 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap") 。也就是说，给定具有附加维度的输入(由 `in_dims` 指定)，我们如何计算 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") ?
* 对于 `args` 中的每个 arg，`in_dims` 都有一个对应的 `Optional[int]` 。如果 arg 则为 `None`不是一个 Tensor，或者如果 arg 没有被 vmapped，否则，它是一个整数，指定正在 vmappedover 的 Tensor 的维度。
* `info` 是可能有用的附加元数据的集合：`info.batch_size` 指定被 vmapped 的维度的大小，而 `info.randomness` 是传递给 [`torch.vmap()`](https://pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap "torch.vmap") 的随机性选项。
* vmap 静态方法的返回是 `(output, out_dims)` 的元组。与“in_dims”类似，“out_dims”应该与“output”具有相同的结构，并且每个输出包含一个“out_dim”，用于指定输出是否具有 vmappeddimension 以及它所在的索引。


 例子：


```
def to_numpy(tensor):
    return tensor.cpu().numpy()

class NumpySort(torch.autograd.Function):
    @staticmethod
    def forward(x, dim):
        device = x.device
        x = to_numpy(x)
        ind = np.argsort(x, axis=dim)
        ind_inv = np.argsort(ind, axis=dim)
        result = np.take_along_axis(x, ind, axis=dim)
        return (
            torch.tensor(result, device=device),
            torch.tensor(ind, device=device),
            torch.tensor(ind_inv, device=device),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, dim = inputs
        _, ind, ind_inv = output
        ctx.mark_non_differentiable(ind, ind_inv)
        ctx.save_for_backward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output, _0, _1):
        return NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim), None

    # The signature of the vmap staticmethod is:
    # vmap(info, in_dims: Tuple[Optional[int]], *args)
    # where *args is the same as the arguments to `forward`.
    @staticmethod
    def vmap(info, in_dims, x, dim):
        # For every input (x and dim), in_dims stores an Optional[int]
        # that is:
        # - None if the input is not being vmapped over or if the input
        # is not a Tensor
        # - an integer if the input is being vmapped over that represents
        # the index of the dimension being vmapped over.
        x_bdim, _ = in_dims

        # A "vmap rule" is the logic of how to perform the operation given
        # inputs with one additional dimension. In NumpySort, x has an
        # additional dimension (x_bdim). The vmap rule is simply
        # to call NumpySort again but pass it a different `dim`.
        x = x.movedim(x_bdim, 0)
        # Handle negative dims correctly
        dim = dim if dim >= 0 else dim + x.dim() - 1
        result = NumpySort.apply(x, dim + 1)

        # The vmap rule must return a tuple of two things
        # 1. the output. Should be the same amount of things
        # as returned by the forward().
        # 2. one Optional[int] for each output specifying if each output
        # is being vmapped over, and if so, the index of the
        # dimension being vmapped over.
        #
        # NumpySort.forward returns a Tuple of 3 Tensors. Since we moved the
        # dimension being vmapped over to the front of `x`, that appears at
        # dimension 0 of all outputs.
        # The return is (output, out_dims) -- output is a tuple of 3 Tensors
        # and out_dims is a Tuple of 3 Optional[int]
        return NumpySort.apply(x, dim + 1), (0, 0, 0)

class NumpyTake(torch.autograd.Function):
    @staticmethod
    def forward(x, ind, ind_inv, dim):
        device = x.device
        x = to_numpy(x)
        ind = to_numpy(ind)
        return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, ind, ind_inv, dim = inputs
        ctx.save_for_backward(ind, ind_inv)
        ctx.dim = dim

    @staticmethod
    def backward(ctx, grad_output):
        ind, ind_inv = ctx.saved_tensors
        result = NumpyTake.apply(grad_output, ind_inv, ind, ctx.dim)
        return result, None, None, None

    @staticmethod
    def vmap(info, in_dims, x, ind, ind_inv, dim):
        x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims

        # The strategy is: expand {x, ind, ind_inv} to all have the dimension
        # being vmapped over.
        # Then, call back into NumpyTake(expanded_x, expanded_ind, expanded_ind_inv, new_dim).

        # Handle negative dims by wrapping them to be positive
        logical_dim = x.dim() if x_bdim is None else x_bdim - 1
        dim = dim if dim >= 0 else dim + logical_dim

        def maybe_expand_bdim_at_front(x, x_bdim):
            if x_bdim is None:
                return x.expand(info.batch_size, *x.shape)
            return x.movedim(x_bdim, 0)

        # If the Tensor doesn't have the dimension being vmapped over,
        # expand it out. Otherwise, move it to the front of the Tensor
        x = maybe_expand_bdim_at_front(x, x_bdim)
        ind = maybe_expand_bdim_at_front(ind, ind_bdim)
        ind_inv = maybe_expand_bdim_at_front(ind_inv, ind_inv_bdim)

        # The return is a tuple (output, out_dims). Since output is a Tensor,
        # then out_dims is an Optional[int](instead of being a Tuple).
        return NumpyTake.apply(x, ind, ind_inv, dim + 1), 0

def numpy_sort(x, dim=-1):
    result, _, _ = NumpySort.apply(x, dim)
    return result

x = torch.randn(2, 3)
result = torch.vmap(numpy_sort)(x)
assert torch.allclose(result, numpy_sort(result, 1))

```


!!! note "笔记"

    vmap 静态方法应该旨在保留整个 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 的语义。也就是说，(伪代码) `grad(vmap(MyFunc))` 应该可以替换为 `grad(map(MyFunc))` 。


 如果您的 autograd.Function 在向后传递中有任何自定义行为，请记住这一点。


!!! note "笔记"

    为 PyTorch 能够通过以下方式生成 vmaprule 的 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 编写自定义 vmap 静态方法是一个合法的用例`generate_vmap_rule=True` 。如果生成的 vmap 规则不具有您正在寻找的语义，您可能希望这样做。


## [`torch.func.jvp()`](../generated/torch.func.jvp.html#torch.func.jvp "torch.func.jvp") 支持 [¶](#torch-func-jvp-支持“永久链接到此标题”)


 为了支持正向模式 AD，[`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 必须有一个 [`jvp()`](。./generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") staticmethod。请参阅[Forward mode AD](https://pytorch.org/docs/stable/notes/extending.html#forward-ad-autograd-function)了解详细信息。