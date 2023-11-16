


 没有10



 单击
 [此处](#sphx-glr-download-intermediate-jacobians-hessians-py)
 下载完整的示例代码








 Jacobian、Hessians、hvp、vhp 等：组合函数变换
 [¶](#jacobians-hessians-hvp-vhp-and-more-composition-function-transforms "此标题的永久链接")
==============================================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/jacobians_hessians>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/jacobians_hessians.html>




 计算雅可比矩阵或粗麻布矩阵在许多非传统
深度学习模型中非常有用。使用 PyTorch’s 常规自动比较 API
(
 `Tensor.backward()`
 ,
 `torch.autograd.grad`
 有效地计算这些数量是很困难（或烦人）的。 ）。 PyTorch’s
 [JAX 启发](https://github.com/google/jax) 
[函数转换 API](https://pytorch.org/docs/master/func.html ) 
 提供了高效计算各种高阶自动微分量的方法。





 注意




 本教程需要 PyTorch 2.0.0 或更高版本。






 计算雅可比行列式
 [¶](#computing-the-jacobian "此标题的固定链接")
-------------------------------------------------------------------------------------------------





```
import torch
import torch.nn.functional as F
from functools import partial
_ = torch.manual_seed(0)

```




 让’s 从我们’d 想要计算其雅可比的函数开始。
这是一个具有非线性激活的简单线性函数。






```
def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()

```




 让’s 添加一些虚拟数据：权重、偏差和特征向量 x。






```
D = 16
weight = torch.randn(D, D)
bias = torch.randn(D)
x = torch.randn(D)  # feature vector

```




 让’s 将
 `predict`
 视为一个函数，将输入
 `x`
 从
 
 \(R^D \到 R^ D\)
 
.
PyTorch Autograd 计算矢量雅可比积。为了计算这个
 
 \(R^D \to R^D\)
 
函数的完整
雅可比行列式，我们必须使用
逐行计算它每次都有不同的单位向量。






```
def compute_jac(xp):
    jacobian_rows = [torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
                     for vec in unit_vectors]
    return torch.stack(jacobian_rows)

xp = x.clone().requires_grad_()
unit_vectors = torch.eye(D)

jacobian = compute_jac(xp)

print(jacobian.shape)
print(jacobian[0])  # show first row

```






```
torch.Size([16, 16])
tensor([-0.5956, -0.6096, -0.1326, -0.2295,  0.4490,  0.3661, -0.1672, -1.1190,
         0.1705, -0.6683,  0.1851,  0.1630,  0.0634,  0.6547,  0.5908, -0.1308])

```


我们可以使用 PyTorch’s
 `torch.vmap`
 函数转换来摆脱 for 循环并对计算进行矢量化，而不是逐行计算雅可比。我们可以’t直接将
 `vmap`
应用到
 `torch.autograd.grad`
；相反，PyTorch提供了
 `torch.func.vjp`
转换来组成与
 `torch.vmap`
 :






```
from torch.func import vmap, vjp

_, vjp_fn = vjp(partial(predict, weight, bias), x)

ft_jacobian, = vmap(vjp_fn)(unit_vectors)

# let's confirm both methods compute the same result
assert torch.allclose(ft_jacobian, jacobian)

```




 在后面的教程中，反向模式 AD 和
 `vmap`
 的组合将为我们提供
每样本梯度。
在本教程中，组合反向模式 AD 和
 `vmap`\ n 为我们提供雅可比
计算！
`vmap`
 和自动微分变换的各种组合
可以为我们提供不同的
有趣的量。




 PyTorch 提供
 `torch.func.jacrev`
 作为一个便利函数，执行
`vmap-vjp`
 组合来计算雅可比矩阵。
 `jacrev`
 接受
 `argnums `
 参数说明我们想要计算雅可比行列式
的参数。






```
from torch.func import jacrev

ft_jacobian = jacrev(predict, argnums=2)(weight, bias, x)

# Confirm by running the following:
assert torch.allclose(ft_jacobian, jacobian)

```




 让’s 比较计算雅可比的两种方法的性能。
函数变换版本要快得多（并且
输出越多，速度就越快）。




 一般来说，我们希望通过
 `vmap`
 进行矢量化可以帮助消除开销
并更好地利用硬件。




`vmap`
 通过将外循环下推到函数’s
原始操作中来实现这一魔法，以获得更好的性能。




 让’s 创建一个快速函数来评估性能并处理
微秒和毫秒测量:






```
def get_perf(first, first_descriptor, second, second_descriptor):
 """takes torch.benchmark objects and compares delta of second vs first."""
    faster = second.times[0]
    slower = first.times[0]
    gain = (slower-faster)/slower
    if gain < 0: gain *=-1
    final_gain = gain*100
    print(f" Performance delta: {final_gain:.4f} percent improvement with {second_descriptor} ")

```




 然后运行性能比较：






```
from torch.utils.benchmark import Timer

without_vmap = Timer(stmt="compute_jac(xp)", globals=globals())
with_vmap = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

no_vmap_timer = without_vmap.timeit(500)
with_vmap_timer = with_vmap.timeit(500)

print(no_vmap_timer)
print(with_vmap_timer)

```






```
<torch.utils.benchmark.utils.common.Measurement object at 0x7f93c815c190>
compute_jac(xp)
  1.99 ms
  1 measurement, 500 runs , 1 thread
<torch.utils.benchmark.utils.common.Measurement object at 0x7f93c7fbbf10>
jacrev(predict, argnums=2)(weight, bias, x)
  715.98 us
  1 measurement, 500 runs , 1 thread

```




 让’s 使用我们的
 `get_perf`
 函数对上述内容进行相对性能比较：






```
get_perf(no_vmap_timer, "without vmap",  with_vmap_timer, "vmap")

```






```
Performance delta: 64.0151 percent improvement with vmap

```




 此外，’s 很容易翻转问题并说我们想要
计算模型参数的雅可比行列式（权重、偏差）而不是输入






```
# note the change in input via ``argnums`` parameters of 0,1 to map to weight and bias
ft_jac_weight, ft_jac_bias = jacrev(predict, argnums=(0, 1))(weight, bias, x)

```






 反向模式雅可比行列式 (
 `jacrev`
 ) 与正向模式雅可比行列式 (
 `jacfwd`
 )
 [¶](#reverse-mode-jacobian-jacrev-vs-forward -mode-jacobian-jacfwd "此标题的永久链接")
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



 我们提供两个 API 来计算雅可比矩阵：
 `jacrev`
 和
 `jacfwd`
 :



* `jacrev`
 使用反向模式 AD。正如您在上面看到的，它是我们的
 `vjp`
 和
 `vmap`
 变换的组合。
* `jacfwd`
 使用前向模式 AD。它是作为我们的
 `jvp`
 和
 `vmap`
 变换的组合来实现的。



`jacfwd`
 和
 `jacrev`
 可以相互替换，但
它们具有不同的性能特征。




 作为一般经验法则，如果您’ 计算
 
 \(R^N \to R^M\)
 
 函数的雅可比，并且有输出多于输入（例如，
 
 \(M > N\)
 
 ），那么
 `jacfwd`
 是首选，否则使用
 `jacrev`
 。此规则也有例外，
但对此的非严格论证如下：




 在反向模式 AD 中，我们逐行计算雅可比矩阵，而在正向模式 AD（计算雅可比向量积）中，我们逐列计算
。雅可比矩阵有 M 行和 N 列，因此如果它
更高或更宽，我们可能更喜欢处理更少
行或列的方法。






```
from torch.func import jacrev, jacfwd

```




 首先，让’s 进行基准测试，输入多于输出：






```
Din = 32
Dout = 2048
weight = torch.randn(Dout, Din)

bias = torch.randn(Dout)
x = torch.randn(Din)

# remember the general rule about taller vs wider... here we have a taller matrix:
print(weight.shape)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

jacfwd_timing = using_fwd.timeit(500)
jacrev_timing = using_bwd.timeit(500)

print(f'jacfwd time: {jacfwd_timing}')
print(f'jacrev time: {jacrev_timing}')

```






```
torch.Size([2048, 32])
jacfwd time: <torch.utils.benchmark.utils.common.Measurement object at 0x7f93c7f8f5b0>
jacfwd(predict, argnums=2)(weight, bias, x)
  1.27 ms
  1 measurement, 500 runs , 1 thread
jacrev time: <torch.utils.benchmark.utils.common.Measurement object at 0x7f93caddfb50>
jacrev(predict, argnums=2)(weight, bias, x)
  10.48 ms
  1 measurement, 500 runs , 1 thread

```




 然后进行相对基准测试：






```
get_perf(jacfwd_timing, "jacfwd", jacrev_timing, "jacrev", );

```






```
Performance delta: 727.8659 percent improvement with jacrev

```




 现在相反 - 输出 (M) 多于输入 (N)：






```
Din = 2048
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

using_fwd = Timer(stmt="jacfwd(predict, argnums=2)(weight, bias, x)", globals=globals())
using_bwd = Timer(stmt="jacrev(predict, argnums=2)(weight, bias, x)", globals=globals())

jacfwd_timing = using_fwd.timeit(500)
jacrev_timing = using_bwd.timeit(500)

print(f'jacfwd time: {jacfwd_timing}')
print(f'jacrev time: {jacrev_timing}')

```






```
jacfwd time: <torch.utils.benchmark.utils.common.Measurement object at 0x7f93cade1270>
jacfwd(predict, argnums=2)(weight, bias, x)
  6.29 ms
  1 measurement, 500 runs , 1 thread
jacrev time: <torch.utils.benchmark.utils.common.Measurement object at 0x7f93caddff70>
jacrev(predict, argnums=2)(weight, bias, x)
  841.44 us
  1 measurement, 500 runs , 1 thread

```




 以及相对性能比较：






```
get_perf(jacrev_timing, "jacrev", jacfwd_timing, "jacfwd")

```






```
Performance delta: 647.4179 percent improvement with jacfwd

```






 使用 functorch.hessian 进行 Hessian 计算
 [¶](#hessian-computation-with-functorch-hessian "永久链接到此标题")
-----------------------------------------------------------------------------------------------------------------------------



 我们提供了一个方便的 API 来计算 hessians：
 `torch.func.hessiani`
 。
Hessians 是雅可比矩阵的雅可比矩阵（或者
偏导数的偏导数，也称为二阶）。




 这表明我们可以编写 functorch 雅可比变换来
计算 Hessian 矩阵。
事实上，
 `hessian(f)`
 就是简单的
 
 `jacfwd(jacrev(f))`
.




 注意：为了提高性能：根据您的模型，您可能还需要
使用
 `jacfwd(jacfwd(f))`
 或
 `jacrev(jacrev(f))`
 来计算hessians
利用上面关于更宽与更高矩阵的经验法则。






```
from torch.func import hessian

# lets reduce the size in order not to overwhelm Colab. Hessians require
# significant memory:
Din = 512
Dout = 32
weight = torch.randn(Dout, Din)
bias = torch.randn(Dout)
x = torch.randn(Din)

hess_api = hessian(predict, argnums=2)(weight, bias, x)
hess_fwdfwd = jacfwd(jacfwd(predict, argnums=2), argnums=2)(weight, bias, x)
hess_revrev = jacrev(jacrev(predict, argnums=2), argnums=2)(weight, bias, x)

```




 让’s 验证无论使用 hessian API 还是
使用
 `jacfwd(jacfwd())`
 都有相同的结果。






```
torch.allclose(hess_api, hess_fwdfwd)

```






```
True

```






 批量雅可比矩阵和批量海森矩阵
 [¶](#batch-jacobian-and-batch-hessian "永久链接到此标题")
--------------------------------------------------------------------------------------------------------------------



 在上面的示例中，我们’ 一直在使用单个特征向量。
在某些情况下，您可能需要相对于一批输入获取一批输出的雅可比行列式。也就是说，给定一批
shape
 `(B,
 

 N)`
 的输入和一个从
 
 \(R^N \to R^M \)
 
 ，我们想要
a 形状为
 `(B,
 

 M,
 

 N)`
 的雅可比行列式。




 最简单的方法是使用
 `vmap`
 :






```
batch_size = 64
Din = 31
Dout = 33

weight = torch.randn(Dout, Din)
print(f"weight shape = {weight.shape}")

bias = torch.randn(Dout)

x = torch.randn(batch_size, Din)

compute_batch_jacobian = vmap(jacrev(predict, argnums=2), in_dims=(None, None, 0))
batch_jacobian0 = compute_batch_jacobian(weight, bias, x)

```






```
weight shape = torch.Size([33, 31])

```




 如果您有一个从 (B, N) -> (B, M) 开始的函数，并且
确定每个输入都会产生独立的输出，那么’
有时也可以在不使用
 `vmap`
 的情况下通过对输出求和
然后计算该函数的雅可比行列式来完成此操作：






```
def predict_with_output_summed(weight, bias, x):
    return predict(weight, bias, x).sum(0)

batch_jacobian1 = jacrev(predict_with_output_summed, argnums=2)(weight, bias, x).movedim(1, 0)
assert torch.allclose(batch_jacobian0, batch_jacobian1)

```




 如果您有一个从
 
 \(R^N \to R^M\)
 
 但输入
经过批处理的函数，则可以编写
 `vmap` 
 使用
 `jacrev`
 计算批量雅可比矩阵:




 最后，批量粗麻布可以类似地计算。 ’ 通过使用
 `vmap`
 批量处理粗麻布计算来思考它们是最容易的，但在某些
情况下，求和技巧也有效。






```
compute_batch_hessian = vmap(hessian(predict, argnums=2), in_dims=(None, None, 0))

batch_hess = compute_batch_hessian(weight, bias, x)
batch_hess.shape

```






```
torch.Size([64, 33, 31, 31])

```






 计算 Hessian 向量积
 [¶](#computing-hessian-vector-products "固定链接到此标题")
---------------------------------------------------------------------------------------------------------------------



 计算 Hessian 向量积 (hvp) 的简单方法是具体化
完整的 Hessian 矩阵并与向量执行点积。我们可以做得更好：事实证明我们不需要具体化完整的 Hessian 矩阵来做到这一点。我们’ll
通过两种（许多）不同的策略来计算 Hessian 向量积：
- 用反向模式 AD 组合反向模式 AD
- 用正向模式 AD 组合反向模式 AD 




 将反向模式 AD 与正向模式 AD 组合（而不是反向模式
与反向模式）通常是计算 a
hvp 的更有效的内存方式，因为正向模式 AD 不需要’构建 Autograd 图并
保存向后的中间值:






```
from torch.func import jvp, grad, vjp

def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]

```




 这里’ 是一些示例用法。






```
def f(x):
  return x.sin().sum()

x = torch.randn(2048)
tangent = torch.randn(2048)

result = hvp(f, (x,), (tangent,))

```




 如果 PyTorch 正向 AD 无法覆盖您的操作，那么我们可以
将反向模式 AD 与反向模式 AD 组合起来：






```
def hvp_revrev(f, primals, tangents):
  _, vjp_fn = vjp(grad(f), *primals)
  return vjp_fn(*tangents)

result_hvp_revrev = hvp_revrev(f, (x,), (tangent,))
assert torch.allclose(result, result_hvp_revrev[0])

```




**脚本的总运行时间:** 
 ( 0 分 11.911 秒)






[`下载
 

 Python
 

 源
 

 代码:
 

 jacobians_hessians.py`](../_downloads/089b69a49b6eb4080d35c4b983b939a5/jacobians_hessians.py ）






[`下载
 

 Jupyter
 

 笔记本:
 

 jacobians_hessians.ipynb`](../_downloads/748f25c58a5ac0f57235c618e51c869b/jacobians_hessians.ipynb)






[Sphinx-Gallery 生成的图库](https://sphinx-gallery.github.io)









