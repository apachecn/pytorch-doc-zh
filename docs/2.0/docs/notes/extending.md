# 扩展 PyTorch [¶](#extending-pytorch "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/extending>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/extending.html>


 在这篇文章中，我们将介绍扩展 [`torch.nn`](../nn.html#module-torch.nn "torch.nn") 、 [`torch.autograd`](../autograd.html#module-torch.autograd "torch.autograd") 、 [`torch`](../torch.html#module-torch "torch") ，以及编写自定义 C++ 扩展。


## 扩展 [`torch.autograd`](../autograd.html#module-torch.autograd "torch.autograd")[¶](#extending-torch-autograd "永久链接到此标题")


 向 [`autograd`](../autograd.html#module-torch.autograd "torch.autograd") 添加操作需要实现一个新的 [`Function`](../autograd.html#torch.autograd.Function " torch.autograd.Function") 每个操作的子类。回想一下，函数是 [`autograd`](../autograd.html#module-torch.autograd "torch.autograd") 用于对操作历史记录和计算梯度进行编码的函数。


 本文档的第一部分重点介绍后向模式 AD，因为它是使用最广泛的功能。最后的一节讨论了前向模式 AD 的扩展。


### 何时使用 [¶](#when-to-use "此标题的永久链接")


 一般来说，如果您想在模型中执行不可微分或依赖于非 PyTorch 库(例如 NumPy)的计算，但仍希望您的操作与其他操作链接并使用 autograd 引擎，请实现自定义函数。


 在某些情况下，自定义函数也可用于提高性能和内存使用率：如果您使用 [C++ 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html) 实现前向和后向传递，您可以将它们包装在 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 中以与 autogradengine 交互。如果您想减少为向后传递保存的缓冲区数量，可以使用自定义函数将操作组合在一起。


### 何时不使用 [¶](#when-not-to-use "永久链接到此标题")


 如果您已经可以根据 PyTorch 的内置操作编写函数，则其后向图(很可能)已经能够由 autograd 记录。在这种情况下，您不需要自己实现向后函数。考虑使用普通的 Python 函数。


 如果您需要维护状态，即可训练参数，您应该(也)使用自定义模块。有关扩展 [`torch.nn`](../nn.html#module-torch.nn "torch.nn") 的更多信息，请参阅下面的部分。


 如果您想在向后传递过程中改变梯度或执行副作用，请考虑注册一个[tensor](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch.Tensor.register_hook) 或 [模块](https://pytorch.org/docs/stable/notes/modules.html#module-hooks) 钩子。


### 如何使用 [¶](#how-to-use "此标题的永久链接")


 采取以下步骤： 1．子类 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 并实现 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") ,(可选) `setup_context()` 和 [`backward()`](../generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward "torch.autograd.Function.backward") 方法.2.对 ctx 参数调用正确的方法。3.声明你的函数是否支持[double backward](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html).4.使用 gradcheck 验证您的渐变是否正确。


**步骤1：**子类化 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 后，您需要定义 3 个方法：



* [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 是执行该操作的代码。它可以接受任意数量的参数，如果您指定默认值，其中一些参数是可选的。这里接受所有类型的 Python 对象。跟踪历史记录的“Tensor”参数(即使用“requires_grad=True”)将在调用之前转换为不跟踪历史记录的参数，并且它们的使用将在图中注册。请注意，此逻辑不会遍历列表/字典/任何其他数据结构，并且只会考虑作为调用的直接参数的tensor。您可以返回单个 `Tensor` 输出，或者返回一个 [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.12)") 常量(如果存在)是多个输出。另外，请参阅 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 的文档来查找只能从 [`forward()` 调用的有用方法的描述](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward").
* `setup_context()` (可选)。人们可以编写一个“组合”[`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward")，它接受`ctx` 对象或(从 PyTorch 2.0 开始)单独的 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward")，不接受 `ctx` 和发生 `ctx` 修改的 `setup_context()` 方法。 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 应该有计算，而 `setup_context()` 应该只负责 `ctx` 修改(并且没有任何计算)。一般来说单独的 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 和 `setup_context()` 是更接近PyTorch本机操作的工作方式，因此更适合与各种PyTorch子系统组合。有关更多详细信息，请参阅[组合或单独的forward()和setup_context()](#combining-forward-context)。
* [`backward()`](../generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward "torch.autograd.Function.backward") (或 `vjp()` ) 定义梯度公式。它将给出“Tensor”参数与输出一样多，每个参数代表梯度 w.r.t。那个输出。重要的是切勿就地修改这些内容。它应该返回与输入一样多的tensor，每个tensor都包含梯度。其相应的输入。如果您的输入不需要梯度(“needs_input_grad”是一个布尔元组，指示每个输入是否需要梯度计算)，或者是非“Tensor”对象，则可以返回“python:None”。另外，如果您有 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 的可选参数，您可以返回梯度比输入更多，只要它们都是 [`None`](https://docs.python.org/3/library/constants.html#None "(in Python v3.12)") 。


**步骤 2：** 您有责任正确使用 `ctx` 中的函数，以确保新的 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.函数")与 autograd 引擎一起正常工作。



* [`save_for_backward()`](../generated/torch.autograd.function.FunctionCtx.save_for_backward.html#torch.autograd.function.FunctionCtx.save_for_backward "torch.autograd.function.FunctionCtx.save_for_backward" ) 必须用于保存要在向后传递中使用的任何tensor。非tensor应直接存储在 ctx 上。如果既不是输入也不是输出的tensor被保存为后向，您的 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 可能不支持双后向(请参阅步骤 3)。
* [`mark_dirty()`](../generated/torch.autograd.function.FunctionCtx.mark_dirty.html#torch.autograd.function.FunctionCtx.mark_dirty "torch.autograd.function.FunctionCtx.mark_dirty") 必须是用于标记由前向函数就地修改的任何输入。
* [`mark_non_Differentiable()`](../generated/torch.autograd.function.FunctionCtx.mark_non_Differentiable.html#torch.autograd.function.FunctionCtx.mark_non_Differentiable "torch.autograd.function.FunctionCtx.mark_non_Differentiable") 必须用于告诉引擎输出是否不可微分。默认情况下，所有可微分类型的输出tensor都将设置为需要梯度。不可微分类型(即整数类型)的tensor永远不会被标记为需要梯度。
* [`set_materialize_grads()`](../generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html#torch.autograd.function.FunctionCtx.set_materialize_grads "torch.autograd.function.FunctionCtx.set_materialize_grads") 可用于告诉 autograd 引擎在输出不依赖于输入的情况下优化梯度计算，方法是不具体化给予向后函数的梯度tensor。也就是说，如果设置为 False，Python 中的 None 对象或 C++ 中的“未定义tensor”(x.define() 为 False 的tensor x)在向后调用之前不会转换为用零填充的tensor，因此您的代码将需要像处理充满零的tensor一样处理此类对象。此设置的默认值为 True。


**步骤 3：** 如果你的 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 不支持双向后，你应该通过用 ` 向后装饰来显式声明这一点一次_可微分()` 。使用此装饰器，尝试通过函数执行双向后操作将产生错误。有关双向后操作的更多信息，请参阅我们的双向后教程。


**步骤4：** 建议您使用 [`torch.autograd.gradcheck()`](../generated/torch.autograd.gradcheck.html#torch.autograd.gradcheck "torch.autograd.gradcheck")通过使用后向函数计算雅可比矩阵并将值逐元素与使用有限差分数值计算的雅可比矩阵进行比较，检查后向函数是否正确计算前向的梯度。


### 示例 [¶](#example "此标题的永久链接")


 您可以在下面找到“Linear”函数的代码以及附加注释：


```
# Inherit from Function
class LinearFunction(Function):

    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

```


 现在，为了更轻松地使用这些自定义操作，我们建议对它们使用别名或将它们包装在函数中。包装在函数中让我们支持默认参数和关键字参数：


```
# Option 1: alias
linear = LinearFunction.apply

# Option 2: wrap in a function, to support default args and keyword args.
def linear(input, weight, bias=None):
    return LinearFunction.apply(input, weight, bias)

```


 在这里，我们给出了由非tensor参数参数化的函数的另一个示例：


```
class MulConstant(Function):
    @staticmethod
    def forward(tensor, constant):
        return tensor * constant

    @staticmethod
    def setup_context(ctx, inputs, output):
        # ctx is a context object that can be used to stash information
        # for backward computation
        tensor, constant = inputs
        ctx.constant = constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None

```


 在这里，我们通过调用 set_materialize_grads(False) 来优化上面的示例：


```
class MulConstant(Function):
    @staticmethod
    def forward(tensor, constant):
        return tensor * constant

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, constant = inputs
        ctx.set_materialize_grads(False)
        ctx.constant = constant

    @staticmethod
    def backward(ctx, grad_output):
        # Here we must handle None grad_output tensor. In this case we
        # can skip unnecessary computations and just return None.
        if grad_output is None:
            return None, None

        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None

```


 如果您需要在 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 中计算任何“中间”tensor要保存，它们必须作为输出返回，或者组合 `forward` 和 `setup_context()` (请参阅[组合或单独的forward() 和 setup_context()](#combining-forward-context) )请注意，这意味着如果您希望渐变流过这些中间值，则需要为它们定义渐变公式(另请参阅[双向后教程](https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html) )：


```
class MyCube(torch.autograd.Function):
    @staticmethod
    def forward(x):
        # We wish to save dx for backward. In order to do so, it must
        # be returned as an output.
        dx = 3 * x ** 2
        result = x ** 3
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
        # gradients, we must add the gradient contribution of `dx`,
        # which is grad_dx * 6 * x.
        result = grad_output * dx + grad_dx * 6 * x
        return result

# Wrap MyCube in a function so that it is clearer what the output is
def my_cube(x):
    result, dx = MyCube.apply(x)
    return result

```


!!! note "笔记"

    `backward` 的输入，即 `grad_output` ，也可以是跟踪历史的tensor。因此，如果使用可微分操作实现“向后”(例如，调用另一个自定义 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") )，则高阶导数将起作用在这种情况下，用 `save_for_backward` 保存的tensor也可以在向后使用，并且有梯度回流，但保存在 `ctx` 中的tensor不会有梯度回流。如果你需要梯度对于保存在 `ctx` 中的 Tensor 的流回，您应该将其作为自定义 `Function` 的输出，并使用 `save_for_backward` 保存它。


 您可能想检查您实现的后向方法是否实际计算了函数的导数。通过使用小的有限差分与数值近似进行比较是可能的：


```
from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
print(test)

```


 有关有限差分梯度比较的更多详细信息，请参阅[数值梯度检查](../autograd.html#grad-check)。如果您的函数用于高阶导数(区分向后传递)，您可以使用“gradgradcheck”函数从同一包检查高阶导数。


### 组合或单独的 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 和 `setup_context()`[¶](#combined-or-separate-forward-and-setup-context "此标题的永久链接")


 定义 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 的主要方法有两种。任何一个：



* 定义一个将前向计算逻辑与`setup_context()`
* (从 PyTorch 2.0 开始)定义一个单独的 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch. autograd.Function.forward") 和 `setup_context()`


 我们推荐第二个选项(单独的 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 和 `setup _context()` )因为这更接近 PyTorch 本机操作的实现方式，并且它由 [`torch.func`](../func.api.html#module-torch.func "torch.func") 转换组成。但是，我们计划未来支持这两种方法；结合 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward ") 与 `setup_context()` ：带来更大的灵活性，因为您可以保存中间体而不将它们作为输出返回。


 请参阅上一节了解如何使用单独的 [`forward()`](https://pytorch.org/docs/stable/generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 和 `setup_context()` 定义 [`Function`](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function  "torch.autograd.Function")。


 以下是如何结合使用 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 和`setup_context()`来定义 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 的示例：


```
class LinearFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(ctx, input, weight, bias=None):
        # The forward pass can use ctx.
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

```


### 转发模式 AD [¶](#forward-mode-ad "永久链接到此标题")


 覆盖正向模式 AD 公式具有非常相似的 API，但有一些不同的微妙之处。您可以实现 [`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") 函数。


 它将被给予与输入一样多的“Tensor”参数，每个参数代表梯度 w.r.t。该输入。它应该返回与输出一样多的tensor，每个tensor都包含梯度。其相应的输出。 [`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") 将在之后调用[`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 方法，在 `apply()` 之前返回。


[`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") 与 [`forward()`](../generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward "torch.autograd.Function.backward") 函数：



* 您可以使用 ctx 传递来自 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward ") 到 [`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") 函数。如果该状态将[`backward()`](../generated/torch.autograd.Function.backward.html#torch.autograd.Function.backward "torch.autograd.Function.backward") 不需要，您可以显式释放通过在 [`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function") 函数。
* [`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") 的实现必须是向后可微的，或者显式检查给定的前向模式梯度中没有一个设置了 `requires_grad`。
* [`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") 函数必须与 [`forward()`](../generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 。例如，如果第 i 个输入被就地修改，则第 i 个梯度必须就地更新。类似地，如果第 j 个输出是第 k 个输入的视图。那么返回的第 j 个输出梯度必须是给定的第 k 个输入梯度的视图。
* 因为用户无法指定需要计算哪个梯度，所以 [`jvp()`](../generated/torch.autograd.Function.jvp.html#torch.autograd.Function.jvp "torch.autograd.Function.jvp") 函数应该始终计算所有输出的梯度。
* 前向模式梯度确实遵循 [`set_materialize_grads()`](../generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html#torch.autograd.function.FunctionCtx.set_materialize_grads "torch.autograd.function.FunctionCtx.set_materialize_grads") 你可以得到禁用此功能时，无输入渐变。


### [`torch.func`](../func.api.html#module-torch.func "torch.func") 转换和/或 [`torch.vmap()`](../generated/torch.vmap.html#torch.vmap "torch.vmap")[¶](#torch-func-transforms-and-or-torch-vmap "此标题的永久链接")


 有关详细信息，请参阅[使用 autograd.Function 扩展 torch.func](extending.func.html#func-autograd-function)。


## 扩展 [`torch.nn`](../nn.html#module-torch.nn "torch.nn")[¶](#extending-torch-nn "永久链接到此标题")


[`nn`](../nn.html#module-torch.nn "torch.nn") 导出两种接口 
- 模块及其功能版本。您可以以两种方式扩展它，但我们建议对所有类型的层使用模块，以保存任何参数或缓冲区，并建议使用函数形式的无参数操作，如激活函数、池化等。


 上面的部分已经完全介绍了添加操作的功能版本。


### 添加一个 [`Module`](../generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")[¶](#adding-a-module "永久链接到这个标题")


 由于 [`nn`](../nn.html#module-torch.nn "torch.nn") 大量利用 [`autograd`](../autograd.html#module-torch.autograd "torch.autograd" ) ，添加新的 [`Module`](../generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") 需要实现 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 执行操作并可以计算梯度。从现在开始，假设我们想要实现一个“Linear”模块，并且我们已经实现了上面列表中的函数。添加此功能只需很少的代码。现在，有两个功能需要实现：



* `__init__` ( *可选* ) - 接受内核大小、特征数量等参数并初始化参数和缓冲区。
* [`forward()`](../generated/torch.nn.Module.html#torch.nn.Module.forward "torch.nn.Module.forward") - 实例化一个 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function")并使用它来执行操作。它与上面所示的功能包装非常相似。


 这是“Linear”模块的实现方式：


```
class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

```


## 扩展 [`torch`](../torch.html#module-torch "torch") Python API [¶](#extending-torch-python-api "永久链接到此标题")


 您可以通过使用与 “Tensor” 匹配的方法定义自定义类来创建模拟 Tensor 的自定义类型。 但是，如果您希望能够将这些类型传递给顶级 torch 命名空间中接受 [Tensor](https://pytorch.org/docs/stable/torch.html#module-torch) 操作数的 [`torch.add()`](../generated/torch.add.html#torch.add "torch.add") 等函数，该怎么办？


 如果您的自定义 Python 类型定义了一个名为 `__torch_function__` 的方法，当您的自定义类的实例传递给以下函数时，PyTorch 将调用您的 `__torch_function__` 实现[`torch`](../torch.html#module-torch "torch") 命名空间。这使得可以为您的 `__torch_function__` 实现的 [`torch`](../torch.html#module-torch "torch") 命名空间中的任何函数定义自定义实现可以调用，允许您的用户将您的自定义类型与他们已经为“Tensor”编写的现有 PyTorch 工作流程结合使用。这适用于与“Tensor”无关的“duck”类型以及用户定义的“Tensor”子类。


### 使用类似 `Tensor` 类型扩展 [`torch`](../torch.html#module-torch "torch") [¶](#extending-torch-with-a-tensor-like-type "此标题的永久链接")


!!! note "笔记"

    此功能受到 NumPy `__array_function__` 协议的启发。请参阅 [NumPy 文档](https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch) 和 [NEP-0018](https://numpy.org/neps/nep-0018-array-function-protocol.html)了解更多详细信息。


 为了具体说明这一点，让我们从一个简单的示例开始，说明 API 调度机制。我们将创建一个表示 2D 标量tensor的自定义类型，由阶数“N”和沿对角线条目的值“value”进行参数化：


```
class ScalarTensor(object):
   def __init__(self, N, value):
       self._N = N
       self._value = value

   def __repr__(self):
       return "ScalarTensor(N={}, value={})".format(self._N, self._value)

   def tensor(self):
       return self._value * torch.eye(self._N)

```


 设计的第一次迭代并不是很有用。 `ScalarTensor` 的主要功能是提供比基本tensor类更紧凑的标量tensor字符串表示形式：


```
>>> d = ScalarTensor(5, 2)
>>> d
ScalarTensor(N=5, value=2)
>>> d.tensor()
tensor([[2., 0., 0., 0., 0.],
 [0., 2., 0., 0., 0.],
 [0., 0., 2., 0., 0.],
 [0., 0., 0., 2., 0.],
 [0., 0., 0., 0., 2.]])

```


 如果我们尝试将此对象与 [`torch`](../torch.html#module-torch "torch") API 一起使用，我们将遇到问题：


```
>>> import torch
>>> torch.mean(d)
TypeError: mean(): argument 'input' (position 1) must be Tensor, not ScalarTensor

```


 在`ScalarTensor`中添加`__torch_function__`实现使得上述操作能够成功。让我们重新实现我们的实现，这次添加一个 `__torch_function__` 实现：


```
HANDLED_FUNCTIONS = {}
class ScalarTensor(object):
    def __init__(self, N, value):
        self._N = N
        self._value = value

    def __repr__(self):
        return "ScalarTensor(N={}, value={})".format(self._N, self._value)

    def tensor(self):
        return self._value * torch.eye(self._N)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONSfunc

```


 `__torch_function__` 方法有四个参数：`func`，对要重写的 torch API 函数的引用，types，实现 `__torch_function__` 的类似 `Tensor` 的类型列表，`args`，传递给函数的参数元组，以及 `kwargs`， 传递给函数的关键字参数的字典。 它使用名为 `HANDLED_FUNCTIONS` 的全局调度表来存储自定义实现。 该字典的键是 `torch` 命名空间中的函数，值是 `ScalarTensor` 的实现。


!!! note "笔记"

    使用全局调度表不是 `__torch_function__` API 的强制部分，它只是用于构建覆盖实现的有用设计模式。


 当我们向它传递一个“ScalarTensor”时，这个类定义不足以使“torch.mean”做正确的事情——我们还需要为“ScalarTensor”操作数定义“torch.mean”的实现，并将该实现添加到“ HANDLED_FUNCTIONS` 调度表字典。一种方法是定义一个装饰器：


```
import functools
def implements(torch_function):
 """Register a torch function override for ScalarTensor"""
    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

```


 这可以应用于我们的覆盖的实现：


```
@implements(torch.mean)
def mean(input):
    return float(input._value) / input._N

```


 通过此更改，我们现在可以将 `torch.mean` 与 `ScalarTensor` 一起使用：


```
>>> d = ScalarTensor(5, 2)
>>> torch.mean(d)
0.4

```


 当然，“torch.mean”是最简单的重写函数示例，因为它只需要一个操作数。我们可以使用相同的机制来重写需要多个操作数的函数，其中任何一个都可能是定义 `__torch_function__` 的tensor或类tensor，例如 [`torch.add ()`](../generated/torch.add.html#torch.add "torch.add") :


```
def ensure_tensor(data):
    if isinstance(data, ScalarTensor):
        return data.tensor()
    return torch.as_tensor(data)

@implements(torch.add)
def add(input, other):
   try:
       if input._N == other._N:
           return ScalarTensor(input._N, input._value + other._value)
       else:
           raise ValueError("Shape mismatch!")
   except AttributeError:
       return torch.add(ensure_tensor(input), ensure_tensor(other))

```


 当两个操作数都是“ScalarTensor”实例时，此版本有一个快速路径，当两个操作数不是“ScalarTensor”时，该版本还有一个较慢的路径，该路径会降级为将数据转换为tensor。当任一操作数是“ScalarTensor”或常规“Tensor”时，这使得重写函数正确：


```
>>> s = ScalarTensor(2, 2)
>>> torch.add(s, s)
ScalarTensor(N=2, value=4)
>>> t = torch.tensor([[1, 1,], [1, 1]])
>>> torch.add(s, t)
tensor([[3., 1.],
 [1., 3.]])

```


 请注意，我们的 add 实现不采用 alpha 或 out 作为关键字参数，例如 [`torch.add()`](../generated/torch.add.html#torch.add "torch.add" ) 做：


```
>>> torch.add(s, s, alpha=2)
TypeError: add() got an unexpected keyword argument 'alpha'

```


 为了速度和灵活性，`__torch_function__` 调度机制不会检查覆盖函数的签名是否与 [`torch`](../torch.html#模块火炬“火炬”)API。对于某些应用程序，忽略可选参数是可以的，但为了确保与“Tensor”完全兼容，torch API 函数的用户实现应注意精确模拟被覆盖函数的 API。


 [`torch`](../torch.html#module-torch "torch") API 中没有显式覆盖的函数将从 `__torch_function__` 返回 `NotImplemented` 。如果所有定义了 `__torch_function__` 的操作数都返回 `NotImplemented` ，PyTorch 将引发 `TypeError` 。这意味着大多数时候，当传递此类类型的实例时，没有显式覆盖类型的操作将引发“TypeError”：


```
>>> torch.mul(s, 3)
TypeError: no implementation found for 'torch.mul' on types that
implement __torch_function__: [ScalarTensor]

```


 实际上，这意味着如果您想使用 `__torch_function__` 实现来实现覆盖，您将需要显式实现完整的 [`torch`](../torch.html #module-torch "torch") API 或您关心的用例的 API 的整个子集。这可能是一个艰巨的任务，因为完整的 [`torch`](../torch.html#module-torch "torch") API 非常广泛。


 另一种选择是对于未处理的操作不返回“NotImplemented”，而是在没有覆盖时将“Tensor”传递给原始的 [`torch`](../torch.html#module-torch "torch") 函数可用的。例如，如果我们将 `ScalarTensor` 的 `__torch_function__` 的实现更改为以下之一：


```
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
        args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
        return func(*args, **kwargs)
    return HANDLED_FUNCTIONSfunc

```


 然后 [`torch.mul()`](../generated/torch.mul.html#torch.mul "torch.mul") 将正常工作，尽管Return type始终是 `Tensor` 而不是 `ScalarTensor ` ，即使两个操作数都是 `ScalarTensor` 实例：


```
>>> s = ScalarTensor(2, 2)
>>> torch.mul(s, s)
tensor([[4., 0.],
 [0., 4.]])

```


 另请参阅下面的“MetadataTensor”示例，了解此模式的另一种变体，但始终返回“MetadataTensor”以通过 [`torch`](../torch.html#module-torch "torch") API 中的操作传播元数据。


 `__torch_function__` 协议旨在完全覆盖 API，部分覆盖可能会导致不良结果，特别是某些函数会引发 `TypeError` 。对于子类尤其如此，其中 torch.add 、 torch.Tensor.__add__ 和 torch.Tensor.add 的所有三个都必须被覆盖，即使它们返回完全相同的结果。如果不这样做也可能导致无限递归。如果需要实现 `torch.Tensor` 子类中的函数，则必须在其实现中使用 `super().__torch_function__` 。


### 子类化 `torch.Tensor`[¶](#subclassing-torch-tensor "永久链接到此标题")


 从版本 1.7.0 开始，应用于“torch.Tensor”子类的“torch.Tensor”上的方法和公共“torch.*”命名空间中的函数将返回子类实例，而不是“torch.Tensor”实例：


```
>>> class SubTensor(torch.Tensor):
...     pass
>>> type(torch.add(SubTensor([0]), SubTensor([1]))).__name__
'SubTensor'
>>> type(torch.add(SubTensor([0]), torch.tensor([1]))).__name__
'SubTensor'

```


 如果存在多个子类，则默认选择层次结构中最低的一个。如果没有唯一的方法来确定这种情况，则会引发“TypeError”：


```
>>> type(torch.add(SubTensor2([0]), SubTensor([1]))).__name__
'SubTensor2'
>>> type(torch.add(SubTensor2([0]), torch.tensor([1]))).__name__
'SubTensor2'
>>> torch.add(SubTensor([0]), OtherSubTensor([1]))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: no implementation found for 'torch.add' on types that implement __torch_function__: [SubTensor, OtherSubTensor]

```


 如果希望对所有tensor方法进行全局覆盖，可以使用 `__torch_function__` 。这是一个记录所有函数/方法调用的示例：


```
class LoggingTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # NOTE: Logging calls Tensor.__repr__, so we can't log __repr__ without infinite recursion
        if func is not torch.Tensor.__repr__:
            logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

```


 但是，如果希望重写 Tensor 子类上的方法，则可以通过直接重写该方法(通过为子类定义它)或使用 `__torch_function__` 和与 `func` 匹配。


 在 `__torch_function__` 中应该小心，因为子类总是调用 `super().__torch_function__(func,...)` 而不是直接调用 `func` ，和1.7.0版本之前的情况一样。如果不这样做，可能会导致 `func` 递归回 `__torch_function__` ，从而导致无限递归。


### 使用 `Tensor` 包装类型扩展 [`torch`](../torch.html#module-torch "torch") [¶](#extending-torch-with-a-tensor-wrapper-type "此标题的永久链接")


 另一个有用的例子是包装 `Tensor` 的类型，无论是作为属性还是通过子类化。下面我们实现了这种类型的一个特殊情况，一个“MetadataTensor”，它将元数据字典附加到通过 [`torch`](../torch.html#module-torch "torch") 传播的“Tensor”运营。由于这是完整 [`torch`](../torch.html#module-torch "torch") API 的通用包装，因此我们不需要单独实现每个覆盖，因此我们可以制作 `__torch _function__` 实现对于允许哪些操作更加宽松：


```
class MetadataTensor(object):
    def __init__(self, data, metadata=None, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self._metadata = metadata

    def __repr__(self):
        return "Metadata:
{}

data:
{}".format(self._metadata, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
        args = [getattr(a, '_t', a) for a in args]
        assert len(metadatas) > 0
        ret = func(*args, **kwargs)
        return MetadataTensor(ret, metadata=metadatas[0])

```


 这个简单的实现不一定适用于 [`torch`](../torch.html#module-torch "torch") API 中的每个函数，但它足以捕获最常见的操作：


```
>>> metadata = {'owner': 'Ministry of Silly Walks'}
>>> m = MetadataTensor([[1, 2], [3, 4]], metadata=metadata)
>>> t = torch.tensor([[1, 2], [1, 2]])
>>> torch.add(t, m)
Metadata:
{'owner': 'Ministry of Silly Walks'}

data:
tensor([[2, 4],
 [4, 6]])
>>> torch.mul(t, m)
Metadata:
{'owner': 'Ministry of Silly Walks'}

data:
tensor([[1, 4],
 [3, 8]])

```


### 对定义 `__torch_function__` 的多种类型进行操作[¶](#operations-on-multiple-types-that-define-torch-function "Permalink to this header")


 可以将 torch API 与多个不同类型一起使用，每个类型都有一个 `__torch_function__` 实现，但必须特别小心。在这种情况下，规则是：



* 调度操作为每个操作数收集 `__torch_function__` 的所有不同实现，并按顺序调用它们：子类在超类之前，否则在运算符表达式中从左到右。
* 如果除 ` 之外的任何值返回 NotImplemented`，该值作为结果返回。实现可以通过返回 `NotImplemented` 来表明它们没有实现操作。*如果所有 `__torch_function__` 实现都返回 `NotImplemented` ，PyTorch 会引发 `TypeError` 。


### 测试 PyTorch API 覆盖的覆盖率 [¶](#testing-coverage-of-overrides-for-the-pytorch-api "Permalink to this header")


 实现 `__torch_function__` 的一个麻烦的方面是，如果某些操作有覆盖，而其他操作没有覆盖，那么用户充其量会看到不一致的体验，或者最坏的情况是在使用函数时会看到运行时引发的错误没有覆盖。为了简化此过程，PyTorch 提供了面向开发人员的 API，以确保完全支持 `__torch_function__` 覆盖。此 API 是私有的，将来可能会在没有警告的情况下进行更改。


 首先，要获取所有可重写函数的列表，请使用 `torch.overrides._get_overridable_functions` 。这会返回一个字典，其键是“PyTorch”Python API 中的命名空间，其值是该命名空间中可以覆盖的函数列表。例如，让我们打印 `torch.nn.function` 中可以被覆盖的前 5 个函数的名称：


```
>>> from torch.overrides import get_overridable_functions
>>> func_dict = get_overridable_functions()
>>> nn_funcs = func_dict[torch.nn.functional]
>>> print([f.__name__ for f in nn_funcs[:5])
['adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
 'adaptive_max_pool1d', 'adaptive_max_pool1d_with_indices']

```


 这个函数列表使得迭代所有可重写函数成为可能，但实际上，如果不费力地手动复制每个测试的每个函数的签名，这还不足以为所有这些函数编写测试。为了简化此过程，“torch.overrides._get_testing_overrides”函数返回一个字典，将“PyTorch”API 中的可重写函数映射到与原始函数具有相同签名但无条件返回 -1 的虚拟 lambda 函数。这些函数与“inspect”一起使用来分析原始“PyTorch”函数的函数签名最有用：


```
>>> import inspect
>>> from torch.overrides import get_testing_overrides
>>> override_dict = get_testing_overrides()
>>> dummy_add = override_dict[torch.add]
>>> inspect.signature(dummy_add)
<Signature (input, other, out=None)>

```


 最后， `torch.overrides.get_ignored_functions` 返回一个明确不能被 `__torch_function__` 覆盖的函数元组。此列表可用于确认“get_overridable_functions”返回的字典中不存在的函数无法被覆盖。


## 扩展 [`torch`](../torch.html#module-torch "torch") 原生 API [¶](#extending-torch-native-api "永久链接到此标题")


 虽然 `__torch_function__` 允许人们有效地扩展 PyTorch 的纯 Python 组件的行为，但它不允许人们扩展用 C++ 实现的 PyTorch 部分。 为此，Tensor 子类还可以定义 `__torch_dispatch__` ，它将能够覆盖 C++ 级别的行为。


 为了有效地使用此功能，了解 PyTorch 的本机部分是如何实现的非常重要。 最重要的组件是我们所说的“调度程序”(最好的描述可以在这篇[博客文章](http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)，尽管它有点过时了)。 正如其名称所暗示的，它负责为特定的函数调用调用正确的后端函数。 例如，当调用 torch.add(a, b) 时，调度程序将检查两个参数，找出哪个“功能”(autograd、autocast、功能化等)和哪个“后端”(CPU、CUDA、MPS 等) 应该用于此特定调用并最终调用所有正确的内核。 内核所做的一个非常常见的事情是“重新调度”。 例如，当使用 autocast 在 GPU 上运行神经网络时，第一个调用将是 autocast 内核，它将处理任何潜在的 autocast 逻辑并向下重新调度。 下一个功能将是 autograd，它将正确创建 autograd 图，然后重新调度。 最后，我们到达 CUDA 的后端内核，它将启动正确的 CUDA 内核并返回最终结果。 在退出时，autograd 会将图形附加到输出，最后，autocast 将有机会在退出时进行所需的任何更新。


 调度程序的一种配置是调用所有这些功能和后端键的顺序。 最新列表及其顺序可以在 DispatchKey 枚举内的 DispatchKey.h 中找到。 为了扩展 torch 的目的，本次讨论的重要顺序子集是：`vmap -> Autocast -> Autograd -> ZeroTensor -> Neg/Conj -> Functionize -> Python -> Backends`。 就本次讨论而言，最重要的关键是 Python，因为定义了 `__torch_dispatch__` 方法的每个 Tensor 子类都会调用此功能。 从那里调用用户定义的方法，并且可以任意覆盖行为。 从那里，再次调用提供的函数将执行“重新调度”。

 
此实现的一些重要含义是： 
 
- 此代码在“所有功能之下”运行。 因此，它只负责生成每个tensor的输出值，就像常规后端一样(并且可以并且应该忽略所有高级功能，例如 autograd、autocast 等)。 
- 如果任何高级功能在不重新分派的情况下实现给定函数，则它将永远不会到达 Python 键，因此 `__torch_dispatch__` 回调将永远不会被触发。 这种情况尤其发生在 CompositeImplicitAutograd 函数中，这些函数在 Autograd 级别进行评估而无需重新分派。 这是因为 CompositeImplicitAutograd 函数通过隐式调用其他本机操作来指定其 autograd 公式，因此在 Autograd 级别，该函数被分解为其本机操作，并对这些操作进行评估。 
- 回调 Python 以及包装结果时，将使用与常规 PyTorch Python/C++ 绑定相同的转换。 特别是，某些对象无法用 Python 表示，需要特殊处理(例如，未定义的tensor变为 None)。 
- 我们的本机函数被延迟填充为 torch.ops.{namespace}.{func_name}.{overload_name} 作为可调用的 Python 对象，以便能够从 Python 轻松地与它们交互。 赋予 `__torch_dispatch__ `的 func 对象始终是此命名空间中的一个条目。 该命名空间可用于直接调用本机操作并绕过常用的 Python API 和绑定代码。


 与 `__torch_function__` 能够插入所有 torch 的 Python API 和 Tensor 方法类似， `__torch_dispatch__` 能够拦截对 aten 本机 API 的所有调用。 请注意，tensor上的所有方法在进入调度程序之前都会转换为函数调用，因此将在此处显示为函数调用：torch.add(a, 2) 和 a + 2 将导致完全相同的 aten 调用。 大多数这些函数都在 native_functions.yaml 中定义，它指定了这些函数的属性及其后端实现。 然后，它们的实现以及指定的功能将通过 codegen 自动注册。 一些更奇特的函数或特性也在 C++ 代码库或用户定义的 C++ 扩展中的其他位置注册。


还可以使用 torch.library 添加新的本机函数。 此 Python 功能允许定义和/或添加新的实现到本机函数。 这可用于添加缺少的内核、替换现有内核或定义全新的本机函数。


您可以在 [subclass Zoo](https://github.com/albanD/subclass_zoo) 存储库中找到许多基于 `__torch_dispatch__` 的子类示例。


## 使用模式扩展所有 [`torch`](../torch.html#module-torch "torch") API [¶](#extending-all-torch-api-with-modes "永久链接到此标题")


 TODO 问：不接受tensor输入的函数怎么样？


 TODO 模式概念介绍


 TODO 日志记录模式示例


## 编写自定义 C++ 扩展 [¶](#writing-custom-c-extensions "永久链接到此标题")


 有关详细说明和示例，请参阅此 [PyTorch 教程](https://pytorch.org/tutorials/advanced/cpp_extension.html)。


 文档可在 [torch.utils.cpp_extension](../cpp_extension.html) 获取。