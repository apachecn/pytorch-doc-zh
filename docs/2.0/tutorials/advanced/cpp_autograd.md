


# C++ 前端中的 Autograd [¶](#autograd-in-c-frontend "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/cpp_autograd>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/cpp_autograd.html>




 `autograd`
 包对于在 PyTorch 中构建高度灵活和动态的神经网络至关重要。 PyTorch Python 前端中的大多数 autograd API 在 C++ 前端中也可用，允许将 autograd 代码从 Python 轻松转换为 C++。




 在本教程中，探索在 PyTorch C++ 前端中执行 autograd 的几个示例。
请注意，本教程假设您已经对 Python 前端中的
autograd 有基本的了解。如果’不是这种情况，请先阅读
 [Autograd：自动微分](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) 
.





## 基本自动分级操作 [¶](#basic-autograd-operations "永久链接到此标题")




 (改编自
 [本教程](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#autograd-automatic-differiation) 
 )




 创建一个tensor并设置
 `torch::requires_grad()`
 用它来跟踪计算






```
auto x = torch::ones({2, 2}, torch::requires_grad());
std::cout << x << std::endl;

```




 输出：






```
1 1
1 1
[ CPUFloatType{2,2} ]

```




 进行tensor运算:






```
auto y = x + 2;
std::cout << y << std::endl;

```




 输出：






```
 3 3
 3 3
[ CPUFloatType{2,2} ]

```




`y`
 是作为操作的结果而创建的，因此它具有
 `grad_fn`
 。






```
std::cout << y.grad_fn()->name() << std::endl;

```




 输出：






```
AddBackward1

```




 对
 `y` 执行更多操作






```
auto z = y * y * 3;
auto out = z.mean();

std::cout << z << std::endl;
std::cout << z.grad_fn()->name() << std::endl;
std::cout << out << std::endl;
std::cout << out.grad_fn()->name() << std::endl;

```




 输出：






```
 27 27
 27 27
[ CPUFloatType{2,2} ]
MulBackward1
27
[ CPUFloatType{} ]
MeanBackward0

```




`.requires_grad_(
 

...
 

 )`
 更改现有tensor’s
 `requires_grad `
 就地标记。






```
auto a = torch::randn({2, 2});
a = ((a * 3) / (a - 1));
std::cout << a.requires_grad() << std::endl;

a.requires_grad_(true);
std::cout << a.requires_grad() << std::endl;

auto b = (a * a).sum();
std::cout << b.grad_fn()->name() << std::endl;

```




 输出：






```
false
true
SumBackward0

```




 现在让’s 反向传播。因为
 `out`
 包含单个标量，
 `out.backward()`
 相当于
 `out.backward(torch::tensor(1.))`
 。






```
out.backward();

```




 打印梯度 d(out)/dx






```
std::cout << x.grad() << std::endl;

```




 输出：






```
 4.5000 4.5000
 4.5000 4.5000
[ CPUFloatType{2,2} ]

```




 你应该得到一个矩阵 
 `4.5`
 。有关如何得出此值的说明，
请参阅
 [本教程中的相应部分](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#gradients) 
 。




 现在让’s 看一下向量雅可比积的示例：






```
x = torch::randn(3, torch::requires_grad());

y = x * 2;
while (y.norm().item<double>() < 1000) {
 y = y * 2;
}

std::cout << y << std::endl;
std::cout << y.grad_fn()->name() << std::endl;

```




 输出：






```
-1021.4020
 314.6695
 -613.4944
[ CPUFloatType{3} ]
MulBackward1

```




 如果我们想要向量雅可比积，请将向量传递给
 `backward`
 作为参数：






```
auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
y.backward(v);

std::cout << x.grad() << std::endl;

```




 输出：






```
 102.4000
 1024.0000
 0.1024
[ CPUFloatType{3} ]

```




 您还可以通过将
 `torch::NoGradGuard`
 放入代码块中来阻止 autograd 跟踪需要梯度的tensor的历史







```
std::cout << x.requires_grad() << std::endl;
std::cout << x.pow(2).requires_grad() << std::endl;

{
 torch::NoGradGuard no_grad;
 std::cout << x.pow(2).requires_grad() << std::endl;
}

```




 输出：






```
true
true
false

```




 或者使用
 `.detach()`
 获取具有相同内容但
不需要梯度的新tensor：






```
std::cout << x.requires_grad() << std::endl;
y = x.detach();
std::cout << y.requires_grad() << std::endl;
std::cout << x.eq(y).all().item<bool>() << std::endl;

```




 输出：






```
true
false
true

```




 有关 C++ tensor自动求导 API 的更多信息，例如
 `grad`
 /
 `requires_grad`
 /
 `is_leaf`
 /
 `backward`\ n /
 `detach`
 /
 `detach_`
 /
 `register_hook`
 /
 `retain_grad`
 ,
请参阅
 [相应的 C++ API 文档](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html) 
.





## 在 C++ 中计算高阶梯度 [¶](#computing-higher-order-gradients-in-c "固定链接到此标题")
 -



 高阶梯度的应用之一是计算梯度惩罚。
让’s 使用使用它的示例
 `torch::autograd::grad`
 :






```
#include <torch/torch.h>

auto model = torch::nn::Linear(4, 3);

auto input = torch::randn({3, 4}).requires_grad_(true);
auto output = model(input);

// Calculate loss
auto target = torch::randn({3, 3});
auto loss = torch::nn::MSELoss()(output, target);

// Use norm of gradients as penalty
auto grad_output = torch::ones_like(output);
auto gradient = torch::autograd::grad({output}, {input}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true)[0];
auto gradient_penalty = torch::pow((gradient.norm(2, /*dim=*/1) - 1), 2).mean();

// Add gradient penalty to loss
auto combined_loss = loss + gradient_penalty;
combined_loss.backward();

std::cout << input.grad() << std::endl;

```




 输出：






```
-0.1042 -0.0638 0.0103 0.0723
-0.2543 -0.1222 0.0071 0.0814
-0.1683 -0.1052 0.0355 0.1024
[ CPUFloatType{3,4} ]

```




 请参阅文档
 `torch::autograd::backward`
 (
 [link](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1afa9b5d4329085df4b6b3d4b4be48914b.html) 
 )
and 
 `torch::autograd::grad`
 (
 [link](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1e03c42b14b40c306f9eb947ef842d9c.html) 
 )
有关如何使用它们的更多信息。\ n





## 在 C++ 中使用自定义 autograd 函数 [¶](#using-custom-autograd-function-in-c "永久链接到此标题")




 (改编自
 [本教程](https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd) 
 )




 将新的基本操作添加到
 `torch::autograd`
 需要为每个操作实现一个新
 
 `torch::autograd::Function`
 子类。
 `torch::autograd::Function `
 是 `torch::autograd`
 用于计算结果和梯度并对操作历史记录进行编码的内容。每个
新函数都需要您实现 2 个方法：
 `forward`
 和
 `backward`
 ，并且
请参阅
 [此链接](https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html) 
 了解详细要求。




 您可以在下面找到来自
 `torch::nn`
 的
 `Linear`
 函数的代码
 :






```
#include <torch/torch.h>

using namespace torch::autograd;

// Inherit from Function
class LinearFunction : public Function<LinearFunction> {
 public:
 // Note that both forward and backward are static functions

 // bias is an optional argument
 static torch::Tensor forward(
 AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
 ctx->save_for_backward({input, weight, bias});
 auto output = input.mm(weight.t());
 if (bias.defined()) {
 output += bias.unsqueeze(0).expand_as(output);
 }
 return output;
 }

 static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
 auto saved = ctx->get_saved_variables();
 auto input = saved[0];
 auto weight = saved[1];
 auto bias = saved[2];

 auto grad_output = grad_outputs[0];
 auto grad_input = grad_output.mm(weight);
 auto grad_weight = grad_output.t().mm(input);
 auto grad_bias = torch::Tensor();
 if (bias.defined()) {
 grad_bias = grad_output.sum(0);
 }

 return {grad_input, grad_weight, grad_bias};
 }
};

```




 然后，我们可以通过以下方式使用 
 `LinearFunction`
:






```
auto x = torch::randn({2, 3}).requires_grad_();
auto weight = torch::randn({4, 3}).requires_grad_();
auto y = LinearFunction::apply(x, weight);
y.sum().backward();

std::cout << x.grad() << std::endl;
std::cout << weight.grad() << std::endl;

```




 输出：






```
 0.5314 1.2807 1.4864
 0.5314 1.2807 1.4864
[ CPUFloatType{2,3} ]
 3.7608 0.9101 0.0073
 3.7608 0.9101 0.0073
 3.7608 0.9101 0.0073
 3.7608 0.9101 0.0073
[ CPUFloatType{4,3} ]

```




 在这里，我们给出一个由非tensor参数参数化的函数的附加示例：






```
#include <torch/torch.h>

using namespace torch::autograd;

class MulConstant : public Function<MulConstant> {
 public:
 static torch::Tensor forward(AutogradContext *ctx, torch::Tensor tensor, double constant) {
 // ctx is a context object that can be used to stash information
 // for backward computation
 ctx->saved_data["constant"] = constant;
 return tensor * constant;
 }

 static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
 // We return as many input gradients as there were arguments.
 // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
 return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
 }
};

```




 然后，我们可以通过以下方式使用 
 `MulConstant`
:






```
auto x = torch::randn({2}).requires_grad_();
auto y = MulConstant::apply(x, 5.5);
y.sum().backward();

std::cout << x.grad() << std::endl;

```




 输出：






```
 5.5000
 5.5000
[ CPUFloatType{2} ]

```




 有关
 `torch::autograd::Function`
 的更多信息，请参阅
 [其文档](https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html) 
 。 





## 将 autograd 代码从 Python 翻译为 C++ [¶](#translate-autograd-code-from-python-to-c "固定链接到此标题")




 在较高层面上，在 C++ 中使用 autograd 的最简单方法是首先在 Python 中
运行 autograd 代码，然后使用下表将 autograd 代码从 Python 转换为
C++:








| 
 Python
 | 
 C++
 |
| --- | --- |
| 
`torch.autograd.backward`
 | 
`torch::autograd::backward`
 (
 [链接](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1afa9b5d4329085df4b6b3d4b4be48914b.html)
 )
 |
| 
`torch.autograd.grad`
 | 
`torch::autograd::grad`
 (
 [链接](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1autograd_1a1e03c42b14b40c306f9eb947ef842d9c.html)
 )
 |
| 
`torch.Tensor.detach`
 | 
`torch::Tensor::detach`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor6detachEv) 
 )
 |
| 
`torch.Tensor.detach_`
 | 
`torch::Tensor::detach_`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7detach_Ev) 
 )
 |
| 
`torch.Tensor.backward`
 | 
`torch::Tensor::backward`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8backwardERK6Tensorbb)
 )
 |
| 
`torch.Tensor.register_hook`
 | 
`torch::Tensor::register_hook`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4I0ENK2at6Tensor13register_hookE18hook_return_void_tI1TERR1T)
 )
 |
| 
`torch.Tensor.requires_grad`
 | 
`torch::Tensor::requires_grad_`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor14requires_grad_Eb) 
 )
 | 
| 
`torch.Tensor.retain_grad`
 | 
`torch::Tensor::retain_grad`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor11retain_gradEv) 
 )
 |
| 
`torch.Tensor.grad`
 | 
`torch::Tensor::grad`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4gradEv) 
 )
 |
| 
`torch.Tensor.grad_fn`
 | 
`torch::Tensor::grad_fn`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7grad_fnEv) 
 )
 |
| 
`torch.Tensor.set_data`
 | 
`torch::Tensor::set_data`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor8set_dataERK6Tensor) 
 )
 |
| 
`torch.Tensor.data`
 | 
`torch::Tensor::data`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor4dataEv) 
 )
 |
| 
`torch.Tensor.output_nr`
 | 
`torch::Tensor::output_nr`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor9output_nrEv) 
 )
 |
| 
`torch.Tensor.is_leaf`
 | 
`torch::Tensor::is_leaf`
 (
 [链接](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html#_CPPv4NK2at6Tensor7is_leafEv) 
 )
 |



 翻译后，大部分 Python autograd 代码应该只能在 C++ 中运行。
如果’ 不是这种情况，请在 [GitHub issues](https://github.com) 提交错误报告/pytorch/pytorch/issues) 
 我们将尽快修复它。





## 结论 [¶](#conclusion "此标题的永久链接")




 您现在应该对 PyTorch’s C++ autograd API 有一个很好的概述。
您可以找到本说明中显示的代码示例
 [此处](https://github.com/pytorch/examples/树/master/cpp/autograd) 
 。与往常一样，如果您遇到任何
问题或有疑问，可以使用我们的
 [论坛](https://discuss.pytorch.org/) 
 或
 [GitHub 问题](https://github.com/pytorch/pytorch/issues) 
 取得联系。









