# 扩展 PyTorch

> 译者：[@那伊抹微笑](https://github.com/wangyangting)
> 
> 校对者：[@Twinkle](https://github.com/kemingzeng)

在本文中, 我们将介绍如何扩展 [`torch.nn`](../nn.html#module-torch.nn "torch.nn"), [`torch.autograd`](../autograd.html#module-torch.autograd "torch.autograd") 模块, 并且使用我们的 C 库来编写自定义的 C 扩展工具.

## 扩展 [`torch.autograd`](../autograd.html#module-torch.autograd "torch.autograd") 模块

将操作添加到 [`autograd`](../autograd.html#module-torch.autograd "torch.autograd") 模块需要为每一个操作实现一个新的 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 类的子类. 回想一下, [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 函数是 [`autograd`](../autograd.html#module-torch.autograd "torch.autograd") 模块用来计算结果和梯度, 并对操作历史进行编码的. 每一个新的函数需要你来实现两个方法:

*   [`forward()`](../autograd.html#torch.autograd.Function.forward "torch.autograd.Function.forward") - 进行操作的代码. 如果您指定默认值, 则可以根据需要使用任意数量的参数, 其中一些参数是可选的. 参数可接收各种类型的 Python 对象. [`Variable`](../autograd.html#torch.autograd.Variable "torch.autograd.Variable") 参数在被调用之前将被转换为 `Tensor` 对象, 并且它们的使用情况将会被注册到 graph (图) 中. 请注意, 这个逻辑不会遍历 lists, dicts, 和任何其它的数据结构, 只会考虑被调用为直接参数的变量. 如果有多个输出, 则可以考虑返回单个的 `Tensor` 类格式的输出, 或者 `Tensor` 类的 [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple) 类格式输出. 此外, 请参阅 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 类的文档来查找只能从 [`forward()`](../autograd.html#torch.autograd.Function.forward "torch.autograd.Function.forward") 调用的有用方法的描述.
*   [`backward()`](../autograd.html#torch.autograd.Function.backward "torch.autograd.Function.backward") - 计算梯度的公式. 它将被赋予与输出一样多的 [`Variable`](../autograd.html#torch.autograd.Variable "torch.autograd.Variable") 参数, 其中的每一个表示对应梯度的输出. 它应该返回与输入一样多的 [`Variable`](../autograd.html#torch.autograd.Variable "torch.autograd.Variable"), 其中的每一个表示都包含其相应输入的梯度. 如果输入不需要计算梯度 (请参阅 `needs_input_grad` 属性), 或者是非 [`Variable`](../autograd.html#torch.autograd.Variable "torch.autograd.Variable") 对象, 则可返回 `None` 类. 此外, 如果你在 `forward()` 方法中有可选的参数, 则可以返回比输入更多的梯度, 只要它们都是 [`None`](https://docs.python.org/3/library/constants.html#None) 类型即可.

下面你可以找到来自 [`torch.nn`](../nn.html#module-torch.nn "torch.nn") 模块的 `Linear` 函数代码, 以及注解

```py
# 继承自 Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_variables
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
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

```

现在, 为了更方便地使用这些自定义操作, 我们推荐使用 `apply` 方法

```py
linear = LinearFunction.apply

```

在这里, 我们给出了一个由非变量参数参数化的函数的例子

```py
class MulConstant(Function):
    @staticmethod
    def forward(ctx, tensor, constant):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.constant = constant
        return tensor * constant

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.constant, None

```

你可能想要检测你刚刚实现的 `backward` 方法是否正确的计算了梯度. 你可以使用小而有限的微分进行数值估计

```py
from torch.autograd import gradcheck

# gradchek takes a tuple of tensor as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(30,20).double(), requires_grad=True),)
test = gradcheck(Linear.apply, input, eps=1e-6, atol=1e-4)
print(test)

```

## 扩展 [`torch.nn`](../nn.html#module-torch.nn "torch.nn") 模块

[`nn`](../nn.html#module-torch.nn "torch.nn") 模块有两种类型的接口 - modules 和 their functional versions. 你可以用两种方法扩展它, 但是我们推荐使用各种层的模块, 用来存放任何 parameters(参数) 或者 buffers(缓冲), 并且推荐使用一个函数形式的无参数操作, 比如激活函数, 池化等等.

添加操作的函数版本已经在上面的章节中完整的介绍了.

### 添加 [`Module`](../nn.html#torch.nn.Module "torch.nn.Module") 类

由于 [`nn`](../nn.html#module-torch.nn "torch.nn") 模块大量的利用了 [`autograd`](../autograd.html#module-torch.autograd "torch.autograd") 模块, 添加一个新的 [`Module`](../nn.html#torch.nn.Module "torch.nn.Module") 类需要实现一个 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 类, 它会执行对应的操作并且计算梯度. 从现在开始, 假设我们想要实现一个 `Linear` 模块, 并且我们具有如上所列实现的功能. 有很少的代码需要添加这个. 现在有两个函数需要实现:

*   `__init__` (optional) - 接收诸如 kernel sizes (核大小) , numbers of features (特征数量) 等参数, 并初始化 parameters(参数) 和 buffers(缓冲区).
*   [`forward()`](../nn.html#torch.nn.Module.forward "torch.nn.Module.forward") - 实例化一个 [`Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function") 类, 并且用于执行操作. 这与上面的 functional wrapper (函数的包装) 非常相似.

这就是 `Linear` 模块的实现方式

```py
class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

```

## 编写自定义的 C 扩展

现在你可以在 [GitHub](https://github.com/pytorch/extension-ffi) 中找到一些例子.