


 没有10



 单击
 [此处](#sphx-glr-download-intermediate-parametrizations-py)
 下载完整的示例代码








 参数化教程
 [¶](#parametrizations-tutorial "永久链接到此标题")
====================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/parametrizations>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/parametrizations.html>




**作者** 
 :
 [Mario Lezcano](https://github.com/lezcano)




 正则化深度学习模型是一项令人惊讶的挑战性任务。
由于要优化的函数的复杂性，惩罚方法等经典技术在应用于非深度模型时通常会出现不足。
当使用不良函数时，这尤其成问题-条件模型。
这些示例包括在长序列和 GAN 上训练的 RNN。近年来，人们提出了许多技术来规范这些模型并提高其收敛性。在循环模型上，
有人建议控制循环核的奇异值，以使 RNN 得到良好的调节。例如，这可以通过使
循环内核
 [正交](https://en.wikipedia.org/wiki/Orthogonal_matrix) 
 来实现。
正则化循环模型的另一种方法是通过
\xe2 \x80\x9c
 [权重归一化](https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html)
\xe2\x80\x9d。
这种方法建议解耦从学习其范数来学习参数。为此，将参数除以其
 [Frobenius 范数](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm) 
 并学习编码其范数的单独参数。
类似的正则化建议以 
\xe2\x80\x9c
 [光谱归一化](https://pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html) 的名义用于 GAN 
 \xe2 \x80\x9d。此方法通过将网络参数除以[谱范数](https://en.wikipedia.org/wiki/Matrix_norm#Special_cases)
而不是 Frobenius 范数来控制网络的 Lipschitz 常数。 




 所有这些方法都有一个共同的模式：它们在使用参数之前都会以适当的方式对其进行转换。在第一种情况下，他们通过使用将矩阵映射到正交矩阵的函数来使其正交。在权重
和谱归一化的情况下，它们将原始参数除以其范数。




 更一般地说，所有这些示例都使用函数在参数上添加额外的结构。
换句话说，它们使用函数来约束参数。




 在本教程中，您将学习如何实现和使用此模式来
对您的模型施加约束。这样做就像编写自己的
 `nn.Module`
 一样简单。




 要求：
 `torch>=1.9.0`





 手动实现参数化
 [¶](#implementing-parametrizations-by-hand "永久链接到此标题")
------------------------------------------------------------------------------------------------------------------



 假设我们想要一个具有对称权重的方形线性层，即
其权重
 `X`
 使得
 `X
 

 =
 

 X\ xe1\xb5\x80`
 。一种方法是将矩阵的上三角部分复制到其下三角部分






```
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

def symmetric(X):
    return X.triu() + X.triu(1).transpose(-1, -2)

X = torch.rand(3, 3)
A = symmetric(X)
assert torch.allclose(A, A.T)  # A is symmetric
print(A)                       # Quick visual check

```




 然后我们可以使用这个想法来实现具有对称权重的线性层






```
class LinearSymmetric(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(n_features, n_features))

    def forward(self, x):
        A = symmetric(self.weight)
        return x @ A

```




 该图层可以用作常规线性图层






```
layer = LinearSymmetric(3)
out = layer(torch.rand(8, 3))

```




 此实现虽然正确且独立，但存在许多问题：



1.它重新实现了图层。我们必须将线性层实现为
 `x
 

 @
 

 A`
 。对于线性层来说这并不是什么大问题，但想象一下必须重新实现 CNN 或 Transformerxe2x80xa6n2。它不会将层和参数化分开。如果参数化
更困难，我们将不得不为我们想要使用它的每一层重写其代码
。
3.每次我们使用该层时，它都会重新计算参数化。如果我们在前向传递过程中多次使用该层（想象一下 RNN 的循环内核），则每次调用该层时都会计算相同的“A”。





 参数化简介
 [¶](#introduction-to-parametrizations "此标题的永久链接")
--------------------------------------------------------------------------------------------------------------------



 参数化可以解决所有这些问题以及其他问题。




 让’s 首先使用
 `torch.nn.utils.parametrize` 重新实现上面的代码
 。
我们唯一要做的就是将参数化编写为常规
 ` nn.模块`






```
class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

```




 这就是我们需要做的。一旦我们有了这个，我们就可以通过执行以下操作将任何常规层转换为
对称层






```
layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Symmetric())

```




 现在，线性层的矩阵是对称的






```
A = layer.weight
assert torch.allclose(A, A.T)  # A is symmetric
print(A)                       # Quick visual check

```




 我们可以对任何其他层做同样的事情。例如，我们可以创建一个具有
 [skew-对称](https://en.wikipedia.org/wiki/Skew-symmetry_matrix)
 内核的 CNN。
我们使用类似的参数化，复制上三角部分符号
反转到下三角部分






```
class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)


cnn = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3)
parametrize.register_parametrization(cnn, "weight", Skew())
# Print a few kernels
print(cnn.weight[0, 1])
print(cnn.weight[2, 2])

```






 检查参数化模块
 [¶](#inspecting-a-parametrized-module "固定链接到此标题")
----------------------------------------------------------------------------------------------------------------------



 当模块参数化时，我们发现模块在三个方面发生了变化：



1. `model.weight`
 现在是一个属性
2.它有一个新
 `module.parametrizations`
 属性
3。非参数化权重已移至
 `module.parametrizations.weight.original`









 参数化后
 `weight`
 ，
 `layer.weight`
 被转换为
 [Python 属性](https://docs.python.org/3/library/functions. html#property) 
.
每次我们请求
 `layer.weight`
 时，此属性都会计算
 `parametrization(weight)`
 就像我们在上面的
 `LinearSymmetric`
 实现中所做的那样.




 注册的参数化存储在模块内的
 `parametrizations`
 属性下。






```
layer = nn.Linear(3, 3)
print(f"Unparametrized:{layer}")
parametrize.register_parametrization(layer, "weight", Symmetric())
print(f"Parametrized:{layer}")

```




 这个
 `parametrizations`
 属性是一个
 `nn.ModuleDict`
 ，并且可以这样访问






```
print(layer.parametrizations)
print(layer.parametrizations.weight)

```




 此
 `nn.ModuleDict`
 的每个元素都是一个
 `ParametrizationList`
 ，其行为类似于
 `nn.Sequential`
 。该列表将允许我们在一个权重上连接参数化。
由于这是一个列表，我们可以访问索引它的参数化。这里’s
我们的
 `对称`
 参数化所在






```
print(layer.parametrizations.weight[0])

```




 我们注意到的另一件事是，如果我们打印参数，我们会看到
参数
 `weight`
 已被移动






```
print(dict(layer.named_parameters()))

```




 它现在位于
 `layer.parametrizations.weight.original` 下






```
print(layer.parametrizations.weight.original)

```




 除了这三个小差异之外，参数化的作用与我们的手动实现完全相同






```
symmetric = Symmetric()
weight_orig = layer.parametrizations.weight.original
print(torch.dist(layer.weight, symmetric(weight_orig)))

```






 参数化是一等公民
 [¶](#parametrizations-are-first-class-citizens "永久链接到此标题")
---------------------------------------------------------------------------------------------------------------------------



 由于
 `layer.parametrizations`
 是
 `nn.ModuleList`
 ，这意味着参数化
 已正确注册为原始模块的子模块。因此，在模块中注册参数的相同规则适用于注册参数化。例如，如果参数化具有参数，则在调用
 `model
 

 时，这些参数将从 CPU
 移动到 CUDA =
 

 model.cuda()`
.






 缓存参数化的值
 [¶](#caching-the-value-of-a-parametrization "永久链接到此标题")
--------------------------------------------------------------------------------------------------------------------



 参数化通过上下文管理器附带内置缓存系统
 `parametrize.cached()`






```
class NoisyParametrization(nn.Module):
    def forward(self, X):
        print("Computing the Parametrization")
        return X

layer = nn.Linear(4, 4)
parametrize.register_parametrization(layer, "weight", NoisyParametrization())
print("Here, layer.weight is recomputed every time we call it")
foo = layer.weight + layer.weight.T
bar = layer.weight.sum()
with parametrize.cached():
    print("Here, it is computed just the first time layer.weight is called")
    foo = layer.weight + layer.weight.T
    bar = layer.weight.sum()

```






 连接参数化
 [¶](#concatenating-parametrizations "永久链接到此标题")
----------------------------------------------------------------------------------------------------



 连接两个参数化就像将它们注册在同一个张量上一样简单。
我们可以使用它从更简单的参数化创建更复杂的参数化。例如，
 [凯莱映射](https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map)
 将斜对称矩阵映射到正行列式的正交矩阵。我们可以
连接
 `Skew`
 和实现凯莱图的参数化，以获得具有
正交权重的层






```
class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):
        # (I + X)(I - X)^{-1}
        return torch.linalg.solve(self.Id - X, self.Id + X)

layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
parametrize.register_parametrization(layer, "weight", CayleyMap(3))
X = layer.weight
print(torch.dist(X.T @ X, torch.eye(3)))  # X is orthogonal

```




 这也可以用于修剪参数化模块，或重用参数化。例如，
矩阵指数将对称矩阵映射到对称正定 (SPD) 矩阵
但是矩阵指数还将斜对称矩阵映射到正交矩阵。
利用这两个事实，我们可以重用之前的参数化我们的优势






```
class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", MatrixExponential())
X = layer_orthogonal.weight
print(torch.dist(X.T @ X, torch.eye(3)))         # X is orthogonal

layer_spd = nn.Linear(3, 3)
parametrize.register_parametrization(layer_spd, "weight", Symmetric())
parametrize.register_parametrization(layer_spd, "weight", MatrixExponential())
X = layer_spd.weight
print(torch.dist(X, X.T))                        # X is symmetric
print((torch.linalg.eigvalsh(X) > 0.).all())  # X is positive definite

```






 初始化参数化
 [¶](#initializing-parametrizations "永久链接到此标题")
--------------------------------------------------------------------------------------------------



 参数化带有一种初始化它们的机制。如果我们实现一个带有签名的方法
 `right_inverse`






```
def right_inverse(self, X: Tensor) -> Tensor

```




 分配给参数化张量时将使用它。




 让’s 升级
 `Skew` 类的实现以支持此






```
class Skew(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)

    def right_inverse(self, A):
        # We assume that A is skew-symmetric
        # We take the upper-triangular elements, as these are those used in the forward
        return A.triu(1)

```




 我们现在可以初始化一个用
 `Skew` 参数化的层






```
layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
X = torch.rand(3, 3)
X = X - X.T                             # X is now skew-symmetric
layer.weight = X                        # Initialize layer.weight to be X
print(torch.dist(layer.weight, X))      # layer.weight == X

```




 当我们连接参数化时，
 `right_inverse`
 按预期工作。
要看到这一点，让 ’s 升级 Cayley 参数化以也支持初始化






```
class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):
        # Assume X skew-symmetric
        # (I + X)(I - X)^{-1}
        return torch.linalg.solve(self.Id - X, self.Id + X)

    def right_inverse(self, A):
        # Assume A orthogonal
        # See https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map
        # (X - I)(X + I)^{-1}
        return torch.linalg.solve(X + self.Id, self.Id - X)

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", CayleyMap(3))
# Sample an orthogonal matrix with positive determinant
X = torch.empty(3, 3)
nn.init.orthogonal_(X)
if X.det() < 0.:
    X[0].neg_()
layer_orthogonal.weight = X
print(torch.dist(layer_orthogonal.weight, X))  # layer_orthogonal.weight == X

```




 这个初始化步骤可以更简洁地写为






```
layer_orthogonal.weight = nn.init.orthogonal_(layer_orthogonal.weight)

```




 这个方法的名字来自于我们经常期望

 `forward(right_inverse(X))
 

 ==
 

 X`\名词这是一种直接的重写方法，
用值初始化后的转发应该返回值
 `X`
 。
实际中并没有强烈执行此约束。事实上，有时放松这种关系可能是有益的。例如，考虑以下随机修剪方法的
实现:






```
class PruningParametrization(nn.Module):
    def __init__(self, X, p_drop=0.2):
        super().__init__()
        # sample zeros with probability p_drop
        mask = torch.full_like(X, 1.0 - p_drop)
        self.mask = torch.bernoulli(mask)

    def forward(self, X):
        return X * self.mask

    def right_inverse(self, A):
        return A

```




 在这种情况下，对于每个矩阵 A
 `forward(right_inverse(A))
 

 ==
 

 A`
 是不正确的。\仅当矩阵
 `A`
 在与掩码相同的位置具有零时，这才是正确的。
即使如此，如果我们将张量分配给修剪后的参数，也不会感到惊讶
张量将是，事实上，已修剪






```
layer = nn.Linear(3, 4)
X = torch.rand_like(layer.weight)
print(f"Initialization matrix:{X}")
parametrize.register_parametrization(layer, "weight", PruningParametrization(layer.weight))
layer.weight = X
print(f"Initialized weight:{layer.weight}")

```






 正在删除参数化
 [¶](#removing-parametrizations "永久链接到此标题")
------------------------------------------------------------------------------------------



 我们可以使用
 `parametrize.remove_parametrizations()` 从模块中的参数或缓冲区中删除所有参数化







```
layer = nn.Linear(3, 3)
print("Before:")
print(layer)
print(layer.weight)
parametrize.register_parametrization(layer, "weight", Skew())
print("Parametrized:")
print(layer)
print(layer.weight)
parametrize.remove_parametrizations(layer, "weight")
print("After. Weight has skew-symmetric values but it is unconstrained:")
print(layer)
print(layer.weight)

```




 删除参数化时，我们可以选择通过设置标志 
 `leave\ 来保留原始参数（即 `layer.parametriations.weight.original` 中的参数），而不是其参数化版本。 \_参数化=False`






```
layer = nn.Linear(3, 3)
print("Before:")
print(layer)
print(layer.weight)
parametrize.register_parametrization(layer, "weight", Skew())
print("Parametrized:")
print(layer)
print(layer.weight)
parametrize.remove_parametrizations(layer, "weight", leave_parametrized=False)
print("After. Same as Before:")
print(layer)
print(layer.weight)

```




**脚本的总运行时间:** 
 ( 0 分 0.000 秒)






[`下载
 

 Python
 

 源
 

 代码:
 

 parametrizations.py`](../_downloads/621174a140b9f76910c50ed4afb0e621/parametrizations.py)






[`下载
 

 Jupyter
 

 笔记本:
 

 parametrizations.ipynb`](../_downloads/c9153ca254003481aecc7a760a7b046f/parametrizations.ipynb)






[Sphinx-Gallery 生成的图库](https://sphinx-gallery.github.io)









