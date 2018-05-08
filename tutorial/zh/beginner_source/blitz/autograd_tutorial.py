# -*- coding: utf-8 -*-
"""
自动求导: 自动微分
===================================

PyTorch 中所有神经网络的核心是 ``autograd`` 自动求导包. 
我们先来简单介绍一下, 然后我们会去训练我们的第一个神经网络.


``autograd`` 自动求导包针对张量上的所有操作都提供了自动微分操作. 
这是一个逐个运行的框架, 这意味着您的反向传播是由您的代码如何运行来
定义的, 每个单一的迭代都可以不一样.

让我们用一些更简单的术语与例子来了解这些套路.

Tensor (张量)
--------

``torch.Tensor`` 是包的核心类. 如果你将其属性 ``.requires_grad`` 
设置为 ``True``, 它就开始追踪上面的所有操作. 当你完成了计算, 你可以调用 
``.backward()`` 使所有梯度自动计算. 这个张量的梯度将会被计入
 ``.grad`` 属性中.

要停止追踪一个张量, 你可以调用 ``.detach()`` 来从计算历史中分离它, 防止未来的梯度
计算被追踪.

要阻止追踪历史 (占用内存), 你也可以用 ``with torch.no_grad():`` 包装代码块. 
这在验证一个模型时尤其有帮助, 因为此时模型有 `requires_grad=True` 的可训练参数, 
但我们不需要梯度计算. 

还有一个针对自动求导实现来说非常重要的类 - ``Function``.

``Tensor`` 和 ``Function`` 是相互联系的, 并且它们构建了一个非循环的图, 
编码了一个完整的计算历史信息. 每一个 Tensor (变量) 都有一个 ``.grad_fn`` 
属性,  它引用了一个已经创建了 ``Tensor`` 的 ``Function`` ( 除了用户创建
的  ``Tensor`` 之外 - 它们的 ``grad_fn is None`` ).

如果你想计算导数, 你可以在 ``Tensor`` 上调用 ``.backward()`` 方法. 
如果 ``Tensor`` 是标量的形式 (例如, 它包含一个元素数据), 你不必指定任何参数给 ``backward()``,
但是, 如果它有更多的元素. 你需要去指定一个 ``gradient`` 参数, 该参数是一个匹配 shape (形状) 的张量.
"""

import torch

###############################################################
# 创建一个张量并设置 requires_grad=True 以在上面追踪计算
x = torch.ones(2, 2, requires_grad=True)
print(x)

###############################################################
# 张量的操作:
y = x + 2
print(y)

###############################################################
# ``y`` 是一个操作的结果, 所以它有 ``grad_fn``.
print(y.grad_fn)

###############################################################
# 在 y 上做更多的操作
z = y * y * 3
out = z.mean()

print(z, out)

################################################################
# ``.requires_grad_( ... )`` 在空间内操作改变了已有张量的 
# ``requires_grad`` 标记, 其输入标记缺省时默认为 ``True``.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

###############################################################
# 梯度
# ---------
# 我们现在后向传播
# 因为 ``out`` 包含单个标量, ``out.backward()`` 等同于 
# ``out.backward(torch.tensor(1))``.

out.backward()

###############################################################
# 输出梯度值 d(out)/dx
#

print(x.grad)

###############################################################
# 你应该得到一个 ``4.5`` 的矩阵. 让我们推导出 ``out``
# *Variable* “:math:`o`”.
# 我们有 :math:`o = \frac{1}{4}\sum_i z_i`,
# :math:`z_i = 3(x_i+2)^2` 和 :math:`z_i\bigr\rvert_{x_i=1} = 27`.
# 因此,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)`, 所以
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5`.

###############################################################
# 你可以使用自动求导来做很多有趣的事情

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

###############################################################
#
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)

###############################################################
# 你也可以在 ``with torch.no_grad():`` 包裹块中写代码, 来停止梯度在 
# requires_grad=True 的张量上自动追踪计算. 
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)

###############################################################
# **稍候阅读:**
#
# ``autograd`` 和 ``Function`` 的文档请参阅
# http://pytorch.apachecn.org/cn/docs/0.4.0/autograd.html
