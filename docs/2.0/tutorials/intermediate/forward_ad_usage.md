# 正向模式自动微分（测试版） [¶](#forward-mode-automatic-differiation-beta "固定链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/forward_ad_usage>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>




 本教程演示如何使用前向模式 AD 来计算
方向导数（或等效的雅可比向量积）。




 下面的教程使用一些仅在版本 >= 1.11
（或夜间构建）中可用的 API。




 另请注意，转发模式 AD 目前处于测试阶段。 API
可能会发生变化，并且操作员覆盖范围仍不完整。





## 基本用法 [¶](#basic-usage "永久链接到此标题")




 与反向模式 AD 不同，正向模式 AD 在正向传递的同时急切地
计算梯度。我们可以使用前向模式 AD 通过像以前一样执行前向传递来计算方向导数，
除非我们首先将我们的输入与表示方向导数方向的另一个张量相关联（或者等效地，
 `v`\雅可比向量积中的 n）。当输入（我们称为 \xe2\x80\x9cprimal\xe2\x80\x9d）与 \xe2\x80\x9cdirection\xe2\x80\x9d 张量（我们称为 \xe2\x80\x9ctangent\xe2）相关联时\x80\x9d，
生成的新张量对象被称为\xe2\x80\x9cdual 张量\xe2\x80\x9d，用于连接
对偶数[0]。




 执行前向传递时，如果任何输入张量是对偶张量，
将执行额外计算来传播函数的此“灵敏度”。






```
import torch
import torch.autograd.forward_ad as fwAD

primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# All forward AD computation must be performed in the context of
# a ``dual_level`` context. All dual tensors created in such a context
# will have their tangents destroyed upon exit. This is to ensure that
# if the output or intermediate results of this computation are reused
# in a future forward AD computation, their tangents (which are associated
# with this computation) won't be confused with tangents from the later
# computation.
with fwAD.dual_level():
    # To create a dual tensor we associate a tensor, which we call the
    # primal with another tensor of the same size, which we call the tangent.
    # If the layout of the tangent is different from that of the primal,
    # The values of the tangent are copied into a new tensor with the same
    # metadata as the primal. Otherwise, the tangent itself is used as-is.
    #
    # It is also important to note that the dual tensor created by
    # ``make_dual`` is a view of the primal.
    dual_input = fwAD.make_dual(primal, tangent)
    assert fwAD.unpack_dual(dual_input).tangent is tangent

    # To demonstrate the case where the copy of the tangent happens,
    # we pass in a tangent with a layout different from that of the primal
    dual_input_alt = fwAD.make_dual(primal, tangent.T)
    assert fwAD.unpack_dual(dual_input_alt).tangent is not tangent

    # Tensors that do not have an associated tangent are automatically
    # considered to have a zero-filled tangent of the same shape.
    plain_tensor = torch.randn(10, 10)
    dual_output = fn(dual_input, plain_tensor)

    # Unpacking the dual returns a ``namedtuple`` with ``primal`` and ``tangent``
    # as attributes
    jvp = fwAD.unpack_dual(dual_output).tangent

assert fwAD.unpack_dual(dual_output).tangent is None

```





## 与模块一起使用 [¶](#usage-with-modules "永久链接到此标题")




 要将
 `nn.Module`
 与前向 AD 一起使用，请在执行前向传递之前用双张量替换
模型的参数。在撰写本文时，无法创建双张量
 [`](#id1)
 nn.Parameter`s。作为解决方法，必须将双张量
注册为模块的非参数属性。






```
import torch.nn as nn

model = nn.Linear(5, 5)
input = torch.randn(16, 5)

params = {name: p for name, p in model.named_parameters()}
tangents = {name: torch.rand_like(p) for name, p in params.items()}

with fwAD.dual_level():
    for name, p in params.items():
        delattr(model, name)
        setattr(model, name, fwAD.make_dual(p, tangents[name]))

    out = model(input)
    jvp = fwAD.unpack_dual(out).tangent

```





## 使用功能模块 API（测试版） [¶](#using-the-function-module-api-beta "永久链接到此标题")




 将 `nn.Module` 与转发 AD 结合使用的另一种方法是利用
功能模块 API（也称为无状态模块 API）。






```
from torch.func import functional_call

# We need a fresh module because the functional call requires the
# the model to have parameters registered.
model = nn.Linear(5, 5)

dual_params = {}
with fwAD.dual_level():
    for name, p in params.items():
        # Using the same ``tangents`` from the above section
        dual_params[name] = fwAD.make_dual(p, tangents[name])
    out = functional_call(model, dual_params, input)
    jvp2 = fwAD.unpack_dual(out).tangent

# Check our results
assert torch.allclose(jvp, jvp2)

```





## 自定义 autograd 函数 [¶](#custom-autograd-function "永久链接到此标题")




 自定义函数还支持转发模式 AD。要创建支持转发模式 AD 的自定义函数，请注册
 `jvp()`
 静态方法。自定义函数可以
但不强制支持前向
和后向AD。有关详细信息，请参阅
 [文档](https://pytorch.org/docs/master/notes/extending.html#forward-mode-ad)。






```
class Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, foo):
        result = torch.exp(foo)
        # Tensors stored in ``ctx`` can be used in the subsequent forward grad
        # computation.
        ctx.result = result
        return result

    @staticmethod
    def jvp(ctx, gI):
        gO = gI * ctx.result
        # If the tensor stored in`` ctx`` will not also be used in the backward pass,
        # one can manually free it using ``del``
        del ctx.result
        return gO

fn = Fn.apply

primal = torch.randn(10, 10, dtype=torch.double, requires_grad=True)
tangent = torch.randn(10, 10)

with fwAD.dual_level():
    dual_input = fwAD.make_dual(primal, tangent)
    dual_output = fn(dual_input)
    jvp = fwAD.unpack_dual(dual_output).tangent

# It is important to use ``autograd.gradcheck`` to verify that your
# custom autograd Function computes the gradients correctly. By default,
# ``gradcheck`` only checks the backward-mode (reverse-mode) AD gradients. Specify
# ``check_forward_ad=True`` to also check forward grads. If you did not
# implement the backward formula for your function, you can also tell ``gradcheck``
# to skip the tests that require backward-mode AD by specifying
# ``check_backward_ad=False``, ``check_undefined_grad=False``, and
# ``check_batched_grad=False``.
torch.autograd.gradcheck(Fn.apply, (primal,), check_forward_ad=True,
                         check_backward_ad=False, check_undefined_grad=False,
                         check_batched_grad=False)

```






```
True

```





## 功能 API（测试版） [¶](#function-api-beta "此标题的永久链接")




 我们还在 functorch
 中提供了更高级别的函数 API，
用于计算雅可比向量积，您可能会发现它更易于使用
，具体取决于您的用例。




 函数式 API 的好处是’ 不需要了解\或使用较低级别的双张量 API，并且您可以使用
其他
 [functorch 变换（如 vmap）来组合它](https://pytorch.org/functorch/stable/notebooks/jacobians_hessians.html) 
 ;
缺点是它提供的控制较少。




 请注意，本教程的其余部分将需要 functorch
(
 <https://github.com/pytorch/functorch>
 ) 才能运行。请在指定的链接中查找安装\说明。






```
import functorch as ft

primal0 = torch.randn(10, 10)
tangent0 = torch.randn(10, 10)
primal1 = torch.randn(10, 10)
tangent1 = torch.randn(10, 10)

def fn(x, y):
    return x ** 2 + y ** 2

# Here is a basic example to compute the JVP of the above function.
# The ``jvp(func, primals, tangents)`` returns ``func(*primals)`` as well as the
# computed Jacobian-vector product (JVP). Each primal must be associated with a tangent of the same shape.
primal_out, tangent_out = ft.jvp(fn, (primal0, primal1), (tangent0, tangent1))

# ``functorch.jvp`` requires every primal to be associated with a tangent.
# If we only want to associate certain inputs to `fn` with tangents,
# then we'll need to create a new function that captures inputs without tangents:
primal = torch.randn(10, 10)
tangent = torch.randn(10, 10)
y = torch.randn(10, 10)

import functools
new_fn = functools.partial(fn, y=y)
primal_out, tangent_out = ft.jvp(new_fn, (primal,), (tangent,))

```






```
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/deprecated.py:77: UserWarning:

We've integrated functorch into PyTorch. As the final step of the integration, functorch.jvp is deprecated as of PyTorch 2.0 and will be deleted in a future version of PyTorch >= 2.3. Please use torch.func.jvp instead; see the PyTorch 2.0 release notes and/or the torch.func migration guide for more details https://pytorch.org/docs/master/func.migrating.html

```





## 将功能 API 与模块结合使用 [¶](#using-the-function-api-with-modules "永久链接到此标题")




 要使用
 `nn.Module` 和
 `functorch.jvp`
 来计算
相对于模型参数的雅可比向量积，我们需要重新表述
 `nn.Module`\ n 作为接受模型参数和模块输入的函数。






```
model = nn.Linear(5, 5)
input = torch.randn(16, 5)
tangents = tuple([torch.rand_like(p) for p in model.parameters()])

# Given a ``torch.nn.Module``, ``ft.make_functional_with_buffers`` extracts the state
# (``params`` and buffers) and returns a functional version of the model that
# can be invoked like a function.
# That is, the returned ``func`` can be invoked like
# ``func(params, buffers, input)``.
# ``ft.make_functional_with_buffers`` is analogous to the ``nn.Modules`` stateless API
# that you saw previously and we're working on consolidating the two.
func, params, buffers = ft.make_functional_with_buffers(model)

# Because ``jvp`` requires every input to be associated with a tangent, we need to
# create a new function that, when given the parameters, produces the output
def func_params_only(params):
    return func(params, buffers, input)

model_output, jvp_out = ft.jvp(func_params_only, (params,), (tangents,))

```






```
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/deprecated.py:104: UserWarning:

We've integrated functorch into PyTorch. As the final step of the integration, functorch.make_functional_with_buffers is deprecated as of PyTorch 2.0 and will be deleted in a future version of PyTorch >= 2.3. Please use torch.func.functional_call instead; see the PyTorch 2.0 release notes and/or the torch.func migration guide for more details https://pytorch.org/docs/master/func.migrating.html

/opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/_functorch/deprecated.py:77: UserWarning:

We've integrated functorch into PyTorch. As the final step of the integration, functorch.jvp is deprecated as of PyTorch 2.0 and will be deleted in a future version of PyTorch >= 2.3. Please use torch.func.jvp instead; see the PyTorch 2.0 release notes and/or the torch.func migration guide for more details https://pytorch.org/docs/master/func.migrating.html

```




 [0]
 <https://en.wikipedia.org/wiki/Dual_number>




**脚本总运行时间:** 
 ( 0 分 0.146 秒)
