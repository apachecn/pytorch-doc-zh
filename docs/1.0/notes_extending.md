# 扩展PyTorch

> 译者：[PEGASUS1993](https://github.com/PEGASUS1993)

本章中，将要介绍使用我们的C库如何扩展`torch.nn`，`torch.autograd`和编写自定义的`C`扩展工具。

## 扩展torch.autograd

添加操作`autograd`需要`Function`为每个操作实现一个新的子类。回想一下，`Function`使用`autograd`来计算结果和梯度，并对操作历史进行编码。每个新功能都需要您实现两种方法：

*   `forward()` - 执行操作的代码。如果您指定了默认值，则可以根据需求使用任意参数，其中一些参数可选。这里支持各种`Python`对象。`Variable`参数在调用之前会被转换`Tensor`，并且它们的使用情况将在`graph`中注册。请注意，此逻辑不会遍历`lists`/`dicts`/和其他任何数据的结构，并且只考虑被直接调用的`Variables`参数。如果有多个输出你可以返回单个`Tensor`或`Tensor`格式的元组。另外，请参阅`Function`文档查找只能被`forward()`调用的有用方法的说明。

*   `backward()` - 计算梯度的公式. 它将被赋予与输出一样多的`Variable`参数, 其中的每一个表示对应梯度的输出. 它应该返回与输入一样多的`Variable`, 其中的每一个表示都包含其相应输入的梯度. 如果输入不需要计算梯度 (请参阅`needs_input_grad`属性),或者是非`Variable`对象,则可返回`None`类.此外,如果你在`forward()`方法中有可选的参数,则可以返回比输入更多的梯度,只要它们都是`None`类型即可.

你可以从下面的代码看到`torch.nn`模块的`Linear`函数, 以及注解

```py
# Inherit from Function
class Linear(Function):

    # bias is an optional argument
    def forward(self, input, weight, bias=None):
        self.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = self.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if self.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if self.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias
```

现在，为了更方便使用这些自定义操作，推荐使用`apply`方法：

```py
linear = LinearFunction.apply
```

我们下面给出一个由非变量参数进行参数化的函数的例子:

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

* 注意
向后输入，即grad_output，也可以是跟踪历史的张量。因此，如果使用可微运算来实现向后运算(例如，调用另一个自定义函数），则更高阶导数将起作用。

你可能想检测你刚刚实现的`backward`方法是否正确的计算了梯度。你可以使用小的有限差分法(`Finite Difference`)进行数值估计。

```py
from torch.autograd import gradcheck

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (Variable(torch.randn(20,20).double(), requires_grad=True), Variable(torch.randn(30,20).double(), requires_grad=True),)
test = gradcheck(Linear.apply, input, eps=1e-6, atol=1e-4)
print(test)
```

有关有限差分梯度比较的更多详细信息，请参见[数值梯度检查](../autograd.html#grad-check)。

## 扩展 torch.nn

`nn`模块包含两种接口 - `modules`和他们的功能版本。你可以用两种方法扩展它,但是我们建议，在扩展`layer`的时候使用`modules`， 因为`modules`保存着参数和`buffer`。如果使用无参数操作的话，那么建议使用激活函数，池化等函数。

在上面的章节中,添加操作的功能版本已经介绍过了。

#### 增加一个`Module`。

由于`nn`大量使用`autograd`。所以， 添加一个新的[Module](https://pytorch.org/docs/master/nn.html#torch.nn.Module)类需要实现一个`Function`类, 它会执行对应的操作并且计算梯度。我们只需要很少的代码就可以实现上面`Linear`模块的功能。现在，我们需要实现两个函数：

*   `__init__ (optional)` - 接收`kernel sizes`内核大小，特征数量等参数，并初始化`parameters`参数和`buffers`缓冲区。
*   `forward()` - 实例化`Function`并使用它来执行操作。它与上面显示的`functional wrapper`非常相似。

下面是实现`Linear`模块的方式：

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
        # nn.Parameters require gradients by default.
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

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
```

### 编写自定义的C++扩展  

有关详细说明和示例，请参阅此[PyTorch教程](https://pytorch.org/tutorials/advanced/cpp_extension.html)。
文档可在[torch.utils.cpp_extension](../cpp_extension.html).获得。
### 编写自定义的C扩展

可用示例可以在[这个Github](https://github.com/pytorch/extension-ffi)仓库里面查看参考。
