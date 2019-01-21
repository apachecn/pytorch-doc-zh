# 扩展PyTorch
本篇文章中包含如何扩展 `torch.nn`, `torch.autograd`和 使用我们的 `C 库`编写自定义的`C`扩展。


## 扩展 torch.autograd
如果你想要添加一个新的 `Operation` 到`autograd`的话，你的`Operation`需要继承 `class Function`。`autograd`使用`Function`计算结果和梯度，同时编码 `operation`的历史。每个新的 `operation(function)` 都需要实现三个方法：

- `__init__ (optional)` - 如果你的`operation`包含非`Variable`参数，那么就将其作为`__init__`的参数传入到`operation`中。例如：`AddConstant Function`加一个常数，`Transpose Function`需要指定哪两个维度需要交换。如果你的`operation`不需要额外的参数，你可以忽略`__init__`。

- `forward()` - 在里面写执行此`operation`的代码。可以有任意数量的参数。如果你对某些参数指定了默认值，则这些参数是可传可不传的。记住：`forward()`的参数只能是`Variable`。函数的返回值既可以是 `Variable`也可以是`Variables`的`tuple`。同时，请参考 `Function`[function]的 `doc`，查阅有哪些 方法是只能在`forward`中调用的。
- `backward()` - 梯度计算公式。 参数的个数和`forward`返回值的个数一样，每个参数代表传回到此`operation`的梯度. `backward()`的返回值的个数应该和此`operation`输入的个数一样，每个返回值对应了输入值的梯度。如果`operation`的输入不需要梯度，或者不可导，你可以返回`None`。 如果`forward()`存在可选参数，你可以返回比输入更多的梯度，只是返回的是`None`。

下面是 `Linear` 的实现代码：

```python
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
现在，为了可以更简单的使用自定义的`operation`，我们建议将其用一个简单的 `helper function` 包装起来。 functions:

```python
def linear(input, weight, bias=None):
    # First braces create a Function object. Any arguments given here
    # will be passed to __init__. Second braces will invoke the __call__
    # operator, that will then use forward() to compute the result and
    # return it.
    return Linear()(input, weight, bias)
```

你可能想知道你刚刚实现的 `backward`方法是否正确的计算了梯度。你可以使用 小的有限的差分进行数值估计。

```python
from torch.autograd import gradcheck

# gradchek takes a tuple of tensor as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
input = (Variable(torch.randn(20,20).double(), requires_grad=True),)
test = gradcheck.gradcheck(Linear(), input, eps=1e-6, atol=1e-4)
print(test)
```

## 扩展 torch.nn

`nn` 包含两种接口 - `modules`和他们的`functional`版本。通过这两个接口，你都可以扩展`nn`。但是我们建议，在扩展`layer`的时候，使用`modules`， 因为`modules`保存着参数和`buffer`。如果不需要参数的话，那么建议使用`functional`(激活函数，pooling，这些都不需要参数)。

增加一个`operation`的 `functional`版本已经在上面一节介绍完毕。

增加一个模块(`module`)。
由于`nn`重度使用`autograd`。所以，添加一个新`module`需要实现一个 用来执行 计算 和 计算梯度 的`Function`。从现在开始，假定我们想要实现一个`Linear module`，记得之前我们已经实现了一个`Linear Funciton`。 只需要很少的代码就可以完成这个工作。 现在，我们需要实现两个方法：

- `__init__ (optional)` - 输入参数，例如`kernel sizes`, `numbers of features`, 等等。同时初始化 `parameters`和`buffers`。

- `forward()` - 实例化一个执行`operation`的`Function`，使用它执行`operation`。和`functional wrapper(上面实现的那个简单的wrapper)`十分类似。

`Linear module`实现代码:
```python
class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
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
        self.weight = nn.Parameter(torch.Tensor(input_features, output_features))
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
        return Linear()(input, self.weight, self.bias)
        #注意这个Linear是之前实现过的Linear
```
## 编写自定义`C`扩展

Coming soon. For now you can find an example at [GitHub](https://github.com/pytorch/extension-ffi).
