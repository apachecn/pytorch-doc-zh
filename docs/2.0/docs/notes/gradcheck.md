# Gradcheck 机制 [¶](#gradcheck-mechanics "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/gradcheck>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/gradcheck.html>


 本说明概述了 [`gradcheck()`](../generated/torch.autograd.gradcheck.html#torch.autograd.gradcheck "torch.autograd.gradcheck") 和 [`gradgradcheck()`] (../generated/torch.autograd.gradgradcheck.html#torch.autograd.gradgradcheck "torch.autograd.gradgradcheck") 函数有效。


 它将涵盖实数和复数值函数以及高阶导数的前向和后向模式 AD。本注释还涵盖 gradcheck 的默认行为以及“fast_mode=True”参数为的情况通过(以下简称快速毕业检查)。



* [符号和背景信息](#notations-and-background-information)
* [默认向后模式梯度检查行为](#default-backward-mode-gradcheck-behavior)



+ [实数到实数函数](#real-to-real-functions) 
+ [复数到实数函数](#complex-to-real-functions) 
+ [具有复数输出的函数](#functions-with -complex-outputs)
* [快速向后模式梯度检查](#fast-backward-mode-gradcheck)



+ [实数到实数函数的快速梯度检查](#fast-gradcheck-for-real-to-real-functions) 
+ [复数到实数函数的快速梯度检查](#fast-gradcheck-for-complex
- to-real-functions) 
+ [具有复杂输出的函数的快速 gradcheck](#fast-gradcheck-for-functions-with-complex-outputs)
* [Gradgradcheck 实现](#gradgradcheck-implementation)


## [符号和背景信息](#id2) [¶](#notations-and-background-information "此标题的固定链接")


 在本说明中，我们将使用以下约定：


1.x


 X


 X




 ,
 


 y


 y


 y




 ,
 


 A


 A


 A




 ,
 


 乙


 乙


 乙




 ,
 


 v


 v


 v




 ,
 


 你


 你


 你




 ,
 


 你


 你的


 你


 和


 你我


 用户界面


 你我


 是实值向量并且


 z


 z


 z


 是一个复值向量，可以用两个实值向量重写为


 z = a 
+ i b


 z = a 
+ i b


 z



 =
 


 A



 +
 


 IB


.2.氮


 氮


 氮


 和


 中号


 中号


 中号


 是两个整数，我们将分别用于输入和输出空间的维度。3。 F  ：


 RN


 →
 


 R M


 f: \mathcal{R}^N 	o \mathcal{R}^M


 F



 :
 


 右


 氮




 →
 


 右


 中号


 是我们的基本实数到实数函数，使得


 y = f ( x )


 y = f(x)


 y



 =
 


 f(x)


.4. G  ：


 CN


 →
 


 R M


 g: \mathcal{C}^N 	o \mathcal{R}^M


 G



 :
 


 C


 氮




 →
 


 右


 中号


 是我们的基本复实函数，使得


 y = g ( z )


 y = g(z)


 y



 =
 


 克(z)


 。


 对于简单的真实情况，我们写为


 Jf


 J_f


 J


 F




 ​
 


 雅可比矩阵与


 F


 F


 F


 尺寸的


 中号×中号


 M \乘N


 中号



 ×
 


 氮


.该矩阵包含所有偏导数，使得位置处的条目


 ( i , j )


 (i,j)


 (  我  ，


 j)


 包含


 ∂
 


 义




 ∂
 


 xj


 rac{\部分 y_i}{\部分 x_j}


 ∂
 


 X


 j




 ​
 




 ∂
 


 y


 我




 ​
 



 ​
 


.后向模式 AD 然后针对给定向量进行计算


 v


 v


 v


 尺寸的


 中号


 中号


 中号


 ， 数量


 电压


 Jf


 v^T J_f


 v


 时间


 J


 F




 ​
 


.另一方面，前向模式AD正在计算，对于给定的向量


 你


 你


 你


 尺寸的


 氮


 氮


 氮


 ， 数量


 Jf


 你


 J_fu


 J


 F




 ​
 


 你


 。


 对于包含复杂值的函数，情况要复杂得多。我们在这里仅提供要点，完整的描述可以在 [复数 Autograd](autograd.html#complex-autograd-doc) 中找到。


 满足复可微分性(柯西-黎曼方程)的约束对于所有实值损失函数来说限制太大，因此我们选择使用 Wirtinger 微积分。在 Wirtinger 微积分的基本设置中，链式法则需要访问 Wirtinger 导数(称为


 瓦


 瓦


 瓦


 如下)和共轭维廷格导数(称为


 连续波


 连续波


 连续波


 如下)。两者


 瓦


 瓦


 瓦


 和


 连续波


 连续波


 连续波


 需要传播，因为一般来说，尽管有它们的名称，但其中一个不是另一个的复共轭。


 为了避免必须传播两个值，对于后向模式 AD，我们始终假设正在计算其导数的函数是实值函数或者是更大实值函数的一部分。这个假设意味着我们在向后传递过程中计算的所有中间梯度也与实值函数相关联。实际上，在进行优化时，这个假设不是限制性的，因为此类问题需要实值目标(因为没有自然排序复数)。


 在此假设下，使用


 瓦


 瓦


 瓦


 和


 连续波


 连续波


 连续波


 定义，我们可以证明


 W=C


 W*


 W = CW^\*


 瓦



 =
 


 C


 瓦


 *


 (我们用


 *


 \*
 


 *


 在这里表示复杂的共轭)，因此实际上只有两个值之一需要“通过图形向后”，因为另一个值可以很容易地恢复。为了简化内部计算，PyTorch 使用


 2 
* CW


 2 * 连续波


 2


 *


 连续波


 当用户要求梯度时，它会向后返回并返回值。与实际情况类似，当输出实际上是


 R M


 \mathcal{R}^M


 右


 中号


 ，后向模式 AD 不计算


 2 
* CW


 2 * 连续波


 2


 *


 连续波


 但只有


 电压


 ( 2 
* CW )


 v^T (2 * CW)


 v


 时间


 ( 2


 *


 CW)


 对于给定向量


 v ∈


 R M


 v \in \mathcal{R}^M


 v


 ε


 右


 中号


 。


 对于前向模式 AD，我们使用类似的逻辑，在这种情况下，假设该函数是一个较大函数的一部分，该函数的输入为


 右


 \mathcal{R}


 右


 。在这个假设下，我们可以做出类似的声明，即每个中间结果都对应于一个函数，该函数的输入为


 右


 \mathcal{R}


 右


 在这种情况下，使用


 瓦


 瓦


 瓦


 和


 连续波


 连续波


 连续波


 定义，我们可以证明


 W = CW


 W = 连续波


 瓦



 =
 




 C
 

 W
 


 对于中间函数。为了确保前向和后向模式在一维函数的基本情况下计算相同的量，前向模式还计算


 2 
* CW


 2 * 连续波


 2
 



 ∗
 


 连续波


.与真实情况类似，当输入实际上是


 RN


 \mathcal{R}^N



 R
 



 N
 


 ，前向模式AD不计算


 2 
* CW


 2 * 连续波


 2
 



 ∗
 


 连续波


 但只有


 ( 2 
* C W ) u


 (2 * CW) u


 ( 2



 ∗
 


 CW) u


 对于给定向量


 u ∈


 RN


 u \in \mathcal{R}^N


 u
 



 ∈
 


 R
 



 N
 


.
 


## [默认向后模式 gradcheck 行为](#id3) [¶](#default-backward-mode-gradcheck-behavior "永久链接到此标题")


### [实数到实数函数](#id4) [¶](#real-to-real-functions "此标题的永久链接")


 测试一个功能


 F  ：


 RN


 →
 


 R M


 , x → y


 f: \mathcal{R}^N 	o \mathcal{R}^M, x 	o y


 f
 



 :
 


 R
 



 N
 




 →
 


 R
 



 M
 


 ,
 



 x
 



 →
 




 y
 


 ，我们重建完整的雅可比矩阵


 Jf



 J\_f
 



 J
 



 f
 




 ​
 


 尺寸的


 中号×中号


 M \乘N


 M
 



 ×
 




 N
 


 有两种方式：解析方式和数值方式。解析版本使用我们的后向模式 AD，而数值版本使用有限差分。然后将两个重建的雅可比矩阵按元素进行比较以确保相等。


#### 默认实数输入数值评估 [¶](#default-real-input-numerical-evaluation "永久链接到此标题")


 如果我们考虑一维函数的基本情况(


 N = M = 1


 N = M = 1


 N
 



 =
 




 M
 



 =
 




 1
 


 )，然后我们可以使用[维基百科文章](https://en.wikipedia.org/wiki/Finite_difference)中的基本有限差分公式。我们使用“中心差”来获得更好的数值属性：


 ∂y


 ∂x



 ≈
 


 f ( x 
+ e p s ) − f ( x − e p s )


 2 
* e p s


 rac{\partial y}{\partial x} pprox rac{f(x 
+ eps) 
- f(x 
- eps)}{2 * eps}


 ∂x


 ∂y




 ​
 



 ≈
 



 2
 



 ∗
 


 eps


 f ( x



 +
 


 eps)



 −
 


 f ( x



 −
 


 eps)




 ​
 


 这个公式很容易推广到多个输出(


 中号 
> 1


 中号\gt 1


 M
 



 >
 




 1
 


 ) 有了


 ∂y


 ∂x


 rac{\偏y}{\偏x}


 ∂x


 ∂y


 ​
 


 是大小为的列向量


 米×1


 中号\乘以1


 M
 



 ×
 




 1
 




 like
 


 f ( x 
+ e p s )


 f(x 
+ 每股收益)


 f ( x



 +
 


 eps)


 在这种情况下，上面的公式可以按原样重新使用，并仅用用户函数的两次评估来近似完整的雅可比矩阵(即


 f ( x 
+ e p s )


 f(x 
+ 每股收益)


 f ( x



 +
 


 eps)




 and
 


 f ( x − e p s )


 f(x 
- 每股收益)


 f ( x



 −
 


 eps)




 ).
 


 处理具有多个输入的情况的计算成本更高(


 N 
> 1


 N \gt 1


 N
 



 >
 




 1
 


 )。在这种情况下，我们一个接一个地循环所有输入并应用


 eps


 eps
 


 eps


 每个元素的扰动



 x
 


 x
 


 x
 


 一个接一个地。这使我们能够重建


 Jf



 J\_f
 



 J
 



 f
 




 ​
 


 矩阵逐列。


#### 默认实际输入分析评估 [¶](#default-real-input-analytical-evaluation "永久链接到此标题")


 对于分析评估，我们使用如上所述的事实，即后向模式 AD 计算


 电压


 Jf


 v^T J_f



 v
 



 T
 



 J
 



 f
 




 ​
 


 对于具有单个输出的函数，我们只需使用


 v = 1


 v = 1
 


 v
 



 =
 




 1
 


 通过一次向后传递恢复完整的雅可比矩阵。


 对于具有多个输出的函数，我们采用 for 循环，它迭代输出，其中每个输出



 v
 


 v
 


 v
 


 是一个单独的向量，对应于每个输出一个接一个。这允许重建


 Jf



 J\_f
 



 J
 



 f
 




 ​
 


 矩阵逐行。


### [复数到实数函数](#id5) [¶](#complex-to-real-functions "永久链接到此标题")


 测试一个功能


 G  ：


 CN


 →
 


 R M


 , z → y


 g: \mathcal{C}^N 	o \mathcal{R}^M, z 	o y


 g
 



 :
 


 C
 



 N
 




 →
 


 R
 



 M
 


 ,
 



 z
 



 →
 




 y
 




 with
 


 z = a 
+ i b


 z = a 
+ i b


 z
 



 =
 




 a
 



 +
 




 ib
 


 ，我们重建包含的(复值)矩阵


 2 
* CW


 2 * 连续波


 2
 



 ∗
 


 连续波




.
 


#### 默认复数输入数值评估 [¶](#default-complex-input-numerical-evaluation "永久链接到此标题")


 考虑基本情况，其中


 N = M = 1


 N = M = 1


 N
 



 =
 




 M
 



 =
 




 1
 


 第一的。我们从[本研究论文](https://arxiv.org/pdf/1701.00392.pdf)(第3章)得知：


 CW :=


 ∂y



 ∂
 


 z*




 =
 


 1 2


 ＊(


 ∂y


 ∂a


 
+ 我


 ∂y


 ∂b



 )
 


 CW := rac{\partial y}{\partial z^\*} = rac{1}{2} * (rac{\partial y}{\partial a} 
+ i rac{\partial y }{\部分b})


 连续波



 :=
 



 ∂
 


 z
 



 ∗
 


 ∂y




 ​
 



 =
 



 2
 




 1
 




 ​
 



 ∗
 




 (
 


 ∂a


 ∂y




 ​
 



 +
 




 i
 


 ∂b


 ∂y




 ​
 




 )
 


 注意


 ∂y


 ∂a


 rac{\偏 y}{\偏 a}


 ∂a


 ∂y


 ​
 




 and
 


 ∂y


 ∂b


 rac{\部分 y}{\部分 b}


 ∂b


 ∂y


 ​
 


 ，在上式中，是


 右 → 右


 \mathcal{R} \到 \mathcal{R}


 R
 



 →
 




 R
 


 为了对这些进行数值评估，我们使用上述针对实际情况的方法。这使我们能够计算


 连续波


 CW
 


 连续波


 矩阵，然后乘以



 2
 


 2
 


 2
 




.
 


 请注意，截至撰写本文时，代码以稍微复杂的方式计算该值：


```
# Code from https://github.com/pytorch/pytorch/blob/58eb23378f2a376565a66ac32c93a316c45b6131/torch/autograd/gradcheck.py#L99-L105
# Notation changes in this code block:
# s here is y above
# x, y here are a, b above

ds_dx = compute_gradient(eps)
ds_dy = compute_gradient(eps \* 1j)
# conjugate wirtinger derivative
conj_w_d = 0.5 \* (ds_dx + ds_dy \* 1j)
# wirtinger derivative
w_d = 0.5 \* (ds_dx - ds_dy \* 1j)
d[d_idx] = grad_out.conjugate() \* conj_w_d + grad_out \* w_d.conj()

# Since grad_out is always 1, and W and CW are complex conjugate of each other, the last line ends up computing exactly `conj_w_d + w_d.conj() = conj_w_d + conj_w_d = 2 \* conj_w_d`.

```


#### 默认复杂输入分析评估 [¶](#default-complex-input-analytical-evaluation "永久链接到此标题")


 由于后向模式 AD 的计算量恰好是


 连续波


 CW
 


 连续波


 已经导数了，我们只需使用与此处的实数到实数情况相同的技巧，并在存在多个实数输出时逐行重建矩阵。


### [具有复杂输出的函数](#id6) [¶](#functions-with-complex-outputs "永久链接到此标题")


 在这种情况下，用户提供的函数不遵循 autograd 的假设，即我们计算后向 AD 的函数是实值。这意味着直接在此函数上使用 autograd 没有明确定义。为了解决这个问题，我们将替换函数的测试


 H  ：


 PN


 →
 


 厘米


 h: \mathcal{P}^N 	o \mathcal{C}^M


 h
 



 :
 


 P
 



 N
 




 →
 


 C
 



 M
 


 (在哪里



 P
 


 \数学{P}


 P
 


 可以是



 R
 


 \mathcal{R}


 R
 




 or
 



 C
 


 \数学{C}


 C
 


 )，有两个功能：


 小时


 hr
 


 小时




 and
 


 你好


 hi
 


 hi
 


 这样：


 小时 (q)


 : = 实数 ( f ( q ) )


 我 (q)


 : = i m a g ( f ( q ) )


 egin{对齐} hr(q) &:= real(f(q)) \ hi(q) &:= imag(f(q))\end{对齐}


 小时 (q)


 你好(q)




 ​
 




 :=
 


 真实 l ( f ( q ))


 :=
 


 ima g ( f ( q ))




 ​
 


 where
 


 q ∈ P


 q \in \mathcal{P}


 q
 



 ∈
 




 P
 


.然后我们对两者进行基本的分级检查


 小时


 hr
 


 小时




 and
 


 你好


 hi
 


 hi
 


 使用上述真实到真实或复杂到真实的情况，具体取决于



 P
 


 \数学{P}


 P
 




.
 


 请注意，截至撰写本文时，代码并未显式创建这些函数，而是执行链式法则


 真实的


 real
 


 我们是一个l




 or
 


 我是一个g


 imag
 


 伊玛格


 通过传递手动功能


 毕业_out


 	ext{毕业_out}


 毕业_out


 不同函数的参数。当


 毕业率= 1


 	ext{grad\_out} = 1


 毕业_out


 = 


 1 


 ，那么我们正在考虑


 h  r 


 hr 


 h  r 


.When 


 梯度_out = 1 j


 	ext{grad\_out} = 1j


 毕业_out


 = 


 1  j 


 ，那么我们正在考虑


 h  i 


 hi 


 hi 


. 


## [快退模式 gradcheck](#id7) [¶](#fast-backward-mode-gradcheck "永久链接到此标题")


 虽然上面的 gradcheck 公式很好，可以确保正确性和可调试性，但它非常慢，因为它重建了完整的雅可比矩阵。本节介绍了一种以更快的方式执行 gradcheck 而不影响其正确性的方法。可调试性可以是当我们检测到错误时，通过添加特殊逻辑来恢复。在这种情况下，我们可以运行重建完整矩阵的默认版本，以向用户提供完整的详细信息。


 这里的高级策略是找到一个标量，该标量可以通过数值和分析方法有效计算，并且能够很好地表示由慢速梯度检查计算出的完整矩阵，以确保它能够捕获雅可比行列式中的任何差异。


### [实数到实数函数的快速 gradcheck](#id8) [¶](#fast-gradcheck-for-real-to-real-functions "永久链接到此标题")


 我们要在这里计算的标量是


 v  T 


 J  f 


 u 


 v^T J_fu


 v 


 T 


 J 


 f 


 ​ 


 u 


 对于给定的随机向量


 v  ∈ 


 R  M 


 v \in \mathcal{R}^M


 v
 



 ∈
 


 R
 



 M
 


 和一个随机单位范数向量



 u
 

 ∈
 


 R
 

 N
 


 u \in \mathcal{R}^N


 u
 



 ∈
 


 R
 



 N
 


.
 


 对于数值评估，我们可以有效地计算


 J
 

 f
 


 u
 

 ≈
 


 f ( x 
+ u 
* e p s ) − f ( x − u 
* e p s )


 2 
* e p s



.
 


 J_fu pprox rac{f(x 
+ u * eps) 
- f(x 
- u * eps)}{2 * eps}。



 J
 



 f
 




 ​
 


 u
 



 ≈
 



 2
 



 ∗
 


 eps


 f ( x



 +
 



 u
 



 ∗
 


 eps)



 −
 


 f ( x



 −
 



 u
 



 ∗
 


 eps)




 ​
 




.
 


 然后我们执行该向量和之间的点积



 v
 


 v
 


 v
 


 获得利息的标量值。


 对于解析版本，我们可以使用后向模式 AD 来计算




 v
 

 T
 



 J
 

 f
 


 v^T J_f



 v
 



 T
 



 J
 



 f
 




 ​
 


 直接地。然后我们执行点积



 u
 


 u
 


 u
 


 以获得期望值。


### [复数到实数函数的快速 gradcheck](#id9) [¶](#fast-gradcheck-for-complex-to-real-functions "永久链接到此标题")


 与真实情况类似，我们想要对整个矩阵进行约简。但是


 2 
* CW


 2 * 连续波


 2
 



 ∗
 




 C
 

 W
 


 矩阵是复值，因此在这种情况下，我们将与复标量进行比较。


 由于我们在数值情况下可以有效计算的内容存在一些限制，并且为了将数值计算的数量保持在最低限度，我们计算以下(尽管令人惊讶的)标量值：


 s : = 2 *


 v
 

 T
 


 (real(CW)ur+i*imag(CW)ui)


 s := 2 * v^T (real(CW) ur 
+ i * imag(CW) ui)


 s
 



 :=
 




 2
 



 ∗
 


 v
 



 T
 


 ( 真实 ( CW ) 你



 +
 




 i
 



 ∗
 


 i 中有 g ( CW )



 where
 



 v
 

 ∈
 


 R
 

 M
 


 v \in \mathcal{R}^M


 v
 



 ∈
 


 R
 



 M
 


 ,
 


 你 ε


 R
 

 N
 


 ur \in \mathcal{R}^N


 u
 

 r
 



 ∈
 


 R
 



 N
 


 and
 


 u i ∈


 R
 

 N
 


 ui \in \mathcal{R}^N


 u
 

 i
 



 ∈
 


 R
 



 N
 


.
 


#### 快速复数输入数值评估 [¶](#fast-complex-input-numerical-evaluation "永久链接到此标题")


 我们首先考虑如何计算



 s
 


 s
 


 s
 


 用数值方法。为此，请记住我们正在考虑



 g
 

 :
 


 C
 

 N
 


 →
 


 R
 

 M
 


 , z → y


 g: \mathcal{C}^N 	o \mathcal{R}^M, z 	o y


 g
 



 :
 


 C
 



 N
 




 →
 


 R
 



 M
 


 ,
 



 z
 



 →
 




 y
 




 with
 


 z = a 
+ i b


 z = a 
+ i b


 z
 



 =
 




 a
 



 +
 




 ib
 


 ， 然后


 CW =


 1
 

 2
 


 ∗
 

 (
 



 ∂
 

 y
 



 ∂
 

 a
 



 +
 

 i
 



 ∂
 

 y
 



 ∂
 

 b
 



 )
 


 CW = rac{1}{2} * (rac{\partial y}{\partial a} 
+ i rac{\partial y}{\partial b})


 C
 

 W
 



 =
 




 2
 



 1
 


 ​
 



 ∗
 




 (
 




 ∂
 

 a
 



 ∂
 

 y
 


 ​
 



 +
 




 i
 




 ∂
 

 b
 



 ∂
 

 y
 


 ​
 




 )
 


 ，我们将其重写如下：




 s
 


 = 2 *


 v
 

 T
 


 (real(CW)ur+i*imag(CW)ui)


 = 2 *


 v
 

 T
 


 (
 


 1
 

 2
 


 ∗
 



 ∂
 

 y
 



 ∂
 

 a
 


 你+我*


 1
 

 2
 


 ∗
 



 ∂
 

 y
 



 ∂
 

 b
 


 你我)


 =
 


 v
 

 T
 


 (
 



 ∂
 

 y
 



 ∂
 

 a
 


 你+我*



 ∂
 

 y
 



 ∂
 

 b
 


 你我)


 =
 


 v
 

 T
 


 (
 

 (
 



 ∂
 

 y
 



 ∂
 

 a
 


 ur)+i*(



 ∂
 

 y
 



 ∂
 

 b
 


 你我))


 egin{aligned} s &= 2 * v^T (real(CW) ur 
+ i * imag(CW) ui) \ &= 2 * v^T (rac{1}{2} \ 
* rac{\partial y}{\partial a} ur 
+ i * rac{1}{2} * rac{\partial y}{\partial b} ui) \ &= v^T ( rac{\partial y}{\partial a} ur 
+ i * rac{\partial y}{\partial b} ui) \ &= v^T ((rac{\partial y}{\partial a} ur) 
+ i * (rac{\partial y}{\partial b} ui))\end{对齐}



 s
 




 ​
 




 =
 



 2
 



 ∗
 




 v
 



 T
 


 ( 真实 ( CW ) 你



 +
 



 i
 



 ∗
 


 i 中有 g ( CW )


 =
 



 2
 



 ∗
 




 v
 



 T
 


 (
 



 2
 




 1
 




 ​
 



 ∗
 


 ∂
 

 a
 




 ∂
 

 y
 




 ​
 




 u
 

 r
 



 +
 



 i
 



 ∗
 


 2
 




 1
 




 ​
 



 ∗
 


 ∂
 

 b
 




 ∂
 

 y
 




 ​
 


 你我)


 =
 




 v
 



 T
 


 (
 



 ∂
 

 a
 




 ∂
 

 y
 




 ​
 




 u
 

 r
 



 +
 



 i
 



 ∗
 


 ∂
 

 b
 




 ∂
 

 y
 




 ​
 


 你我)


 =
 




 v
 



 T
 


 ((
 



 ∂
 

 a
 




 ∂
 

 y
 




 ​
 


 你)



 +
 



 i
 



 ∗
 



 (
 



 ∂
 

 b
 




 ∂
 

 y
 




 ​
 


 你我))




 ​
 


 在这个公式中，我们可以看到


 ∂
 

 y
 



 ∂
 

 a
 



 u
 

 r
 


 rac{\partial y}{\partial a} ur


 ∂
 

 a
 



 ∂
 

 y
 


 ​
 




 u
 

 r
 




 and
 


 ∂
 

 y
 



 ∂
 

 b
 



 u
 

 i
 


 rac{\partial y}{\partial b} ui


 ∂
 

 b
 



 ∂
 

 y
 


 ​
 




 u
 

 i
 


 可以用与实数到实数情况的快速版本相同的方式进行评估。计算出这些实数值量后，我们可以重建右侧的复向量，并与实数值量进行点积



 v
 


 v
 


 v
 


 向量。


#### 快速复杂输入分析评估 [¶](#fast-complex-input-analytical-evaluation "永久链接到此标题")


 对于分析情况，事情更简单，我们将公式重写为：




 s
 


 = 2 *


 v
 

 T
 


 (real(CW)ur+i*imag(CW)ui)


 =
 


 v
 

 T
 


 真实 ( 2 
* C W ) ur 
+ i *


 v
 

 T
 


 i m a g ( 2 ∗ C W ) u i )


 = 实数 (


 v
 

 T
 


 ( 2 
* C W ) ) ur 
+ i 
* i m a g (


 v
 

 T
 


 ( 2 
* C W ) ) u i


 egin{aligned} s &= 2 * v^T (real(CW) ur 
+ i * imag(CW) ui) \ &= v^T real(2 * CW) ur 
+ i * v ^T imag(2 * CW) ui) \ &= real(v^T (2 * CW)) ur 
+ i * imag(v^T (2 * CW)) ui\end{对齐}



 s
 




 ​
 




 =
 



 2
 



 ∗
 




 v
 



 T
 


 ( 真实 ( CW ) 你



 +
 



 i
 



 ∗
 


 i 中有 g ( CW )


 =
 




 v
 



 T
 


 真实的 (



 ∗
 


 CW)你



 +
 



 i
 



 ∗
 




 v
 



 T
 


 我是g (2



 ∗
 


 CW ) u i )


 =
 


 真实的 (


 v
 



 T
 


 (
 

 2
 



 ∗
 


 CW )) 你



 +
 



 i
 



 ∗
 


 我是g(


 v
 



 T
 


 (
 

 2
 



 ∗
 


 CW )) u 我




 ​
 


 因此，我们可以利用后向模式 AD 为我们提供了一种有效的计算方法




 v
 

 T
 


 ( 2 
* CW )


 v^T (2 * CW)



 v
 



 T
 


 (
 

 2
 



 ∗
 


 CW)


 然后执行实部的点积



 u
 

 r
 


 ur
 


 u
 

 r
 


 和虚部



 u
 

 i
 


 ui
 


 u
 

 i
 


 在重建最终的复标量之前



 s
 


 s
 


 s
 




.
 


#### 为什么不使用复杂的



 u
 


 u
 


 u
 


[¶](#why-not-use-a-complex-u"此标题的永久链接")


 此时，您可能想知道为什么我们不选择复杂的



 u
 


 u
 


 u
 


 并刚刚执行了减少



 2
 

 ∗
 


 v
 

 T
 


 C
 

 W
 


 u
 

 ′
 


 2 * v^T CW u'


 2
 



 ∗
 


 v
 



 T
 


 C
 

 W
 


 u
 




 ′
 


 为了深入探讨这一点，在本段中，我们将使用复杂版本



 u
 


 u
 


 u
 




 noted
 




 u
 

 ′
 


 =
 

 u
 


 r
 

 ′
 


 
+ 我你


 i
 

 ′
 


 u' = ur' 
+ i ui'



 u
 




 ′
 


 =
 




 u
 


 r
 




 ′
 


 +
 




 i
 

 u
 


 i
 




 ′
 


.使用如此复杂的




 u
 

 ′
 



 u'
 



 u
 




 ′
 


 ，问题是在进行数值评估时，我们需要计算：


 2 
* CW


 u
 

 ′
 




 =
 

 (
 



 ∂
 

 y
 



 ∂
 

 a
 



 +
 

 i
 



 ∂
 

 y
 



 ∂
 

 b
 


 )(你


 r
 

 ′
 


 
+ 我你


 i
 

 ′
 


 )
 


 =
 



 ∂
 

 y
 



 ∂
 

 a
 



 u
 


 r
 

 ′
 


 +
 

 i
 



 ∂
 

 y
 



 ∂
 

 a
 



 u
 


 i
 

 ′
 


 +
 

 i
 



 ∂
 

 y
 



 ∂
 

 b
 



 u
 


 r
 

 ′
 


 −
 



 ∂
 

 y
 



 ∂
 

 b
 



 u
 


 i
 

 ′
 


 egin{对齐} 2\*CW u' &= (rac{\partial y}{\partial a} 
+ i rac{\partial y}{\partial b})(ur' 
+ i ui') \ \ &= rac{\partial y}{\partial a} ur' 
+ i rac{\partial y}{\partial a} ui' 
+ i rac{\partial y}{\partial b} ur' 
- rac{\partial y}{\partial b} ui'\end{对齐}



 2
 



 ∗
 



 C
 

 W
 


 u
 




 ′
 



 ​
 




 =
 



 (
 



 ∂
 

 a
 




 ∂
 

 y
 




 ​
 



 +
 



 i
 



 ∂
 

 b
 




 ∂
 

 y
 




 ​
 


 )(你


 r
 




 ′
 


 +
 



 i
 

 u
 


 i
 




 ′
 



 )
 


 =
 


 ∂
 

 a
 




 ∂
 

 y
 




 ​
 




 u
 


 r
 




 ′
 


 +
 



 i
 



 ∂
 

 a
 




 ∂
 

 y
 




 ​
 




 u
 


 i
 




 ′
 


 +
 



 i
 



 ∂
 

 b
 




 ∂
 

 y
 




 ​
 




 u
 


 r
 




 ′
 


 −
 


 ∂
 

 b
 




 ∂
 

 y
 




 ​
 




 u
 


 i
 




 ′
 



 ​
 


 这将需要对实数到实数有限差分进行四次评估(与上面提出的方法相比是两倍)。由于这种方法没有更多的自由度(相同数量的实值变量)，并且我们尝试获得最快的结果这里可能的评估，我们使用上面的其他公式。


### [具有复杂输出的函数的快速 gradcheck](#id10) [¶](#fast-gradcheck-for-functions-with-complex-outputs "永久链接到此标题")


 就像在慢速情况下一样，我们考虑两个实值函数，并对每个函数使用上面的适当规则。


## [Gradgradcheck 实现](#id11) [¶](#gradgradcheck-implementation "此标题的永久链接")


 PyTorch 还提供了一个验证二阶梯度的实用程序。这里的目标是确保向后实现也可以正确微分并计算出正确的东西。


 该功能是通过考虑功能来实现的


 F : x , v →


 v
 

 T
 



 J
 

 f
 


 F: x, v 	o v^T J_f


 F
 



 :
 




 x
 

 ,
 



 v
 



 →
 


 v
 



 T
 



 J
 



 f
 




 ​
 


 并在此函数上使用上面定义的 gradcheck。请注意



 v
 


 v
 


 v
 


 在这种情况下只是一个与以下类型相同的随机向量


 f(x)


 f(x)
 


 f(x)




.
 


 gradcheck 的快速版本是通过在同一函数上使用 gradcheck 的快速版本来实现的



 F
 


 F
 


 F
 




.