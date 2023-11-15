


 使用自定义函数的双重向后
 [¶](#double-backward-with-custom-functions "永久链接到此标题")
===============================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/custom_function_double_backward_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>



有时通过向后图向后运行两次很有用，例如计算高阶梯度。然而，需要理解 autograd 并注意支持双重向后。支持单次向后执行的函数不一定
能够支持双次向后执行。在本教程中，我们将展示如何
编写支持双向后的自定义自动分级函数，并
指出一些需要注意的事项。




 在将自定义自动分级函数编写为向后两次时，
 重要的是要了解自定义函数中执行的操作
 何时被自动分级记录、何时’t，以及最重要的是，如何
 \ n save_for_backward
 
 适用于所有这些。




 自定义函数以两种方式隐式影响分级模式:



* 在转发期间，autograd 不会记录在转发函数中执行的任何
操作的任何图表。当forward
完成时，自定义函数的backward函数
变成
 
 grad_fn
 
每个forward’s输出
* 在backward期间，autograd记录计算图如果指定了 create_graph，
则用于计算向后传递



 接下来，要了解 
 
 save_for_backward
 
 如何与上述交互，
我们可以探索几个示例：





 保存输入
 [¶](# saving-the-inputs "固定链接到此标题")
---------------------------------------------------------------------------



 考虑这个简单的平方函数。它保存一个输入张量以供向后使用。当 autograd
 能够记录向后传递中的操作时，双重向后自动工作，因此，当我们保存向后传递的输入时，通常
无需担心，因为
如果输入是任何函数，则输入应该具有 grad_fn张量
需要梯度。这使得梯度能够正确传播。






```
import torch

class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Because we are saving one of the inputs use `save_for_backward`
        # Save non-tensors and non-inputs/non-outputs directly on ctx
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_out):
        # A function support double backward automatically if autograd
        # is able to record the computations performed in backward
        x, = ctx.saved_tensors
        return grad_out * 2 * x

# Use double precision because finite differencing method magnifies errors
x = torch.rand(3, 3, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Square.apply, x)
# Use gradcheck to verify second-order derivatives
torch.autograd.gradgradcheck(Square.apply, x)

```




 我们可以使用 torchviz 来可视化图表以了解其工作原理






```
import torchviz

x = torch.tensor(1., requires_grad=True).clone()
out = Square.apply(x)
grad_x, = torch.autograd.grad(out, x, create_graph=True)
torchviz.make_dot((grad_x, x, out), {"grad_x": grad_x, "x": x, "out": out})

```




 我们可以看到 x 的梯度本身就是 x 的函数 (dout/dx = 2x)
并且该函数的图形已正确构建



[![https://user-images.githubusercontent.com/13428986/126559699-e04f3cb1-aaf2-4a9a-a83d-b8767d04fbd9.png](https://user-images.githubusercontent.com/13428986/126559699-e04f3cb1- aaf2-4a9a-a83d-b8767d04fbd9.png)](https://user-images.githubusercontent.com/13428986/126559699-e04f3cb1-aaf2-4a9a-a83d-b8767d04fbd9.png)



 保存输出
 [¶](# saving-the-outputs "固定链接到此标题")
---------------------------------------------------------------------------



 与上一个示例略有不同的是保存输出\而不是输入。机制相似，因为输出也
与 grad_fn 相关联。






```
class Exp(torch.autograd.Function):
    # Simple case where everything goes well
    @staticmethod
    def forward(ctx, x):
        # This time we save the output
        result = torch.exp(x)
        # Note that we should use `save_for_backward` here when
        # the tensor saved is an ouptut (or an input).
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        result, = ctx.saved_tensors
        return result * grad_out

x = torch.tensor(1., requires_grad=True, dtype=torch.double).clone()
# Validate our gradients using gradcheck
torch.autograd.gradcheck(Exp.apply, x)
torch.autograd.gradgradcheck(Exp.apply, x)

```




 使用 torchviz 可视化图表：






```
out = Exp.apply(x)
grad_x, = torch.autograd.grad(out, x, create_graph=True)
torchviz.make_dot((grad_x, x, out), {"grad_x": grad_x, "x": x, "out": out})

```



[![https://user-images.githubusercontent.com/13428986/126559780-d141f2ba-1ee8-4c33-b4eb-c9877b27a954.png](https://user-images.githubusercontent.com/13428986/126559780- d141f2ba-1ee8-4c33-b4eb-c9877b27a954.png)](https://user-images.githubusercontent.com/13428986/126559780-d141f2ba-1ee8-4c33-b4eb-c9877b27a954.png)



 保存中间结果
 [¶](# saving-intermediate-results "永久链接到此标题")
--------------------------------------------------------------------------------------------



 一个更棘手的情况是当我们需要保存中间结果时。
我们通过实现来演示这种情况：




 \[sinh(x) := \frac{e^x - e^{-x}}{2}

\]
 

 由于 sinh 的导数是 cosh，重用
 
 exp(x)
 
 和
 
 exp(-x)
 
 可能会很有用，这两个中间结果在向后计算中
。




 不过，中间结果不应该直接保存并在后向中使用。
因为前向是在无梯度模式下执行的，如果前向传递的中间结果
用于计算后向传递中的梯度
则前向传递的后向图梯度不包括计算中间结果的操作。这会导致不正确的渐变。






```
class Sinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        expx = torch.exp(x)
        expnegx = torch.exp(-x)
        ctx.save_for_backward(expx, expnegx)
        # In order to be able to save the intermediate results, a trick is to
        # include them as our outputs, so that the backward graph is constructed
        return (expx - expnegx) / 2, expx, expnegx

    @staticmethod
    def backward(ctx, grad_out, _grad_out_exp, _grad_out_negexp):
        expx, expnegx = ctx.saved_tensors
        grad_input = grad_out * (expx + expnegx) / 2
        # We cannot skip accumulating these even though we won't use the outputs
        # directly. They will be used later in the second backward.
        grad_input += _grad_out_exp * expx
        grad_input -= _grad_out_negexp * expnegx
        return grad_input

def sinh(x):
    # Create a wrapper that only returns the first output
    return Sinh.apply(x)[0]

x = torch.rand(3, 3, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(sinh, x)
torch.autograd.gradgradcheck(sinh, x)

```




 使用 torchviz 可视化图表：






```
out = sinh(x)
grad_x, = torch.autograd.grad(out.sum(), x, create_graph=True)
torchviz.make_dot((grad_x, x, out), params={"grad_x": grad_x, "x": x, "out": out})

```



[![https://user-images.githubusercontent.com/13428986/126560494-e48eba62-be84-4b29-8c90-a7f6f40b1438.png](https://user-images.githubusercontent.com/13428986/126560494- e48eba62-be84-4b29-8c90-a7f6f40b1438.png)](https://user-images.githubusercontent.com/13428986/126560494-e48eba62-be84-4b29-8c90-a7f6f40b1438.png)



 保存中间结果：不应该做什么
 [¶](# saving-intermediate-results-what-not-to-do "永久链接到此标题")
-------------------------------------------------------------------------------------------------------------------------------



 现在我们展示当我们不’t 也返回中间
结果作为输出时会发生什么：
 
 grad_x
 
 甚至不会有一个向后图
因为它纯粹是一个函数
 
 exp
 
 和
 
 expnegx
 
 ，不需要 grad。






```
class SinhBad(torch.autograd.Function):
    # This is an example of what NOT to do!
    @staticmethod
    def forward(ctx, x):
        expx = torch.exp(x)
        expnegx = torch.exp(-x)
        ctx.expx = expx
        ctx.expnegx = expnegx
        return (expx - expnegx) / 2

    @staticmethod
    def backward(ctx, grad_out):
        expx = ctx.expx
        expnegx = ctx.expnegx
        grad_input = grad_out * (expx + expnegx) / 2
        return grad_input

```




 使用 torchviz 可视化图表。请注意
 
 grad_x
 
 不是
图表的一部分！






```
out = SinhBad.apply(x)
grad_x, = torch.autograd.grad(out.sum(), x, create_graph=True)
torchviz.make_dot((grad_x, x, out), params={"grad_x": grad_x, "x": x, "out": out})

```



[![https://user-images.githubusercontent.com/13428986/126565889-13992f01-55bc-411a-8aee-05b721fe064a.png](https://user-images.githubusercontent.com/13428986/126565889- 13992f01-55bc-411a-8aee-05b721fe064a.png)](https://user-images.githubusercontent.com/13428986/126565889-13992f01-55bc-411a-8aee-05b721fe064a.png)



 当不跟踪向后时
 [¶](#when-backward-is-not-tracked "固定链接到此标题")
------------------------------------------------------------------------------------------------



 最后，让 ’s 考虑一个示例，
autograd 可能根本无法跟踪函数向后的梯度。
我们可以想象cube_backward 是一个可能需要
的函数非 PyTorch 库，例如 SciPy 或 NumPy，或编写为
C++ 扩展。此处演示的解决方法是创建另一个
自定义函数 CubeBackward，您还可以在其中手动指定
cube_backward 的后向！






```
def cube_forward(x):
    return x**3

def cube_backward(grad_out, x):
    return grad_out * 3 * x**2

def cube_backward_backward(grad_out, sav_grad_out, x):
    return grad_out * sav_grad_out * 6 * x

def cube_backward_backward_grad_out(grad_out, x):
    return grad_out * 3 * x**2

class Cube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return cube_forward(x)

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return CubeBackward.apply(grad_out, x)

class CubeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, x):
        ctx.save_for_backward(x, grad_out)
        return cube_backward(grad_out, x)

    @staticmethod
    def backward(ctx, grad_out):
        x, sav_grad_out = ctx.saved_tensors
        dx = cube_backward_backward(grad_out, sav_grad_out, x)
        dgrad_out = cube_backward_backward_grad_out(grad_out, x)
        return dgrad_out, dx

x = torch.tensor(2., requires_grad=True, dtype=torch.double)

torch.autograd.gradcheck(Cube.apply, x)
torch.autograd.gradgradcheck(Cube.apply, x)

```




 使用 torchviz 可视化图表：






```
out = Cube.apply(x)
grad_x, = torch.autograd.grad(out, x, create_graph=True)
torchviz.make_dot((grad_x, x, out), params={"grad_x": grad_x, "x": x, "out": out})

```



[![https://user-images.githubusercontent.com/13428986/126559935-74526b4d-d419-4983-b1f0-a6ee99428531.png](https://user-images.githubusercontent.com/13428986/126559935- 74526b4d-d419-4983-b1f0-a6ee99428531.png)](https://user-images.githubusercontent.com/13428986/126559935-74526b4d-d419-4983-b1f0-a6ee99428531.png)

 总而言之，是否双向后适用于您的自定义函数
简单地取决于向后传递是否可以通过自动梯度进行跟踪。
在前两个示例中，我们展示了双向后
开箱即用的情况。通过第三个和第四个示例，我们
演示了能够跟踪后向函数的技术，而
否则则无法跟踪后向函数。









