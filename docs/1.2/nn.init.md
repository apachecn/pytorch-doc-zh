# torch.nn.init

`torch.nn.init.``calculate_gain`( _nonlinearity_ , _param=None_
)[[source]](_modules/torch/nn/init.html#calculate_gain)

    

返回推荐增益值对于给定的非线性函数。该值如下：

非线性

|

获得  
  
---|---  
  
线性/身份

|

1  1  1  
  
CONV {1,2,3} d

|

111  
  
乙状结肠

|

111  
  
正切

|

5  3  \压裂{5} { 3}  3  5  
  
RELU

|

2  \ SQRT {2}  2  
  
漏RELU

|

2  1  \+  negative_slope  2  \ SQRT {\压裂{2} {1个+ \文本{负\ _slope } ^ 2}}  1  \+
negative_slope  2  2  [HTG10 2]  
  
Parameters

    

  * **非线性** \- 非线性函数（ nn.functional 名称）

  * **PARAM** \- 为对非线性函数可选参数

例子

    
    
    >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    

`torch.nn.init.``uniform_`( _tensor_ , _a=0.0_ , _b=1.0_
)[[source]](_modules/torch/nn/init.html#uniform_)

    

填充输入张量与值从均匀分布 绘制U  （ 一 ， b  ） \ mathcal {U】（A，b） U  （ 一 ， b  ） 。

Parameters

    

  * **张量** \- n维 torch.Tensor 

  * **一** \- 下界的均匀分布的

  * **B** \- 上界的均匀分布的

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.uniform_(w)
    

`torch.nn.init.``normal_`( _tensor_ , _mean=0.0_ , _std=1.0_
)[[source]](_modules/torch/nn/init.html#normal_)

    

填充与值的输入张量从正常的分布中抽取〔HTG0]  N  （ 意味着 ， STD  2  ） \ mathcal {N}（\文本{意味着}，\文本{STD}
^ 2） N  （ 意味着 ， STD  2  ） 。

Parameters

    

  * **tensor** – an n-dimensional torch.Tensor

  * **意味着** \- 正常分布的平均值

  * **STD** \- 正态分布的标准偏差

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.normal_(w)
    

`torch.nn.init.``constant_`( _tensor_ , _val_
)[[source]](_modules/torch/nn/init.html#constant_)

    

填充输入张量与值 VAL  \文本{VAL}  VAL  。

Parameters

    

  * **tensor** – an n-dimensional torch.Tensor

  * **VAL** \- 的值来填充与张力

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.constant_(w, 0.3)
    

`torch.nn.init.``ones_`( _tensor_
)[[source]](_modules/torch/nn/init.html#ones_)

    

填充与标量值 1 输入张量。

Parameters

    

**tensor** – an n-dimensional torch.Tensor

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.ones_(w)
    

`torch.nn.init.``zeros_`( _tensor_
)[[source]](_modules/torch/nn/init.html#zeros_)

    

填充与标量值 0 输入张量。

Parameters

    

**tensor** – an n-dimensional torch.Tensor

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.zeros_(w)
    

`torch.nn.init.``eye_`( _tensor_
)[[source]](_modules/torch/nn/init.html#eye_)

    

填充与单位矩阵的2维输入张量。保留的输入在线性层，其中一样多的输入被保留尽可能的身份。

Parameters

    

**张量** \- 2维 torch.Tensor

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.eye_(w)
    

`torch.nn.init.``dirac_`( _tensor_
)[[source]](_modules/torch/nn/init.html#dirac_)

    

填充{3，4，5}维输入张量与狄拉克δ函数。保留的输入在卷积层，其中尽可能多的输入通道被保留尽可能的身份。

Parameters

    

**张量** \- 一个{3,4，5}维 torch.Tensor

Examples

    
    
    >>> w = torch.empty(3, 16, 5, 5)
    >>> nn.init.dirac_(w)
    

`torch.nn.init.``xavier_uniform_`( _tensor_ , _gain=1.0_
)[[source]](_modules/torch/nn/init.html#xavier_uniform_)

    

填充输入张量与根据在理解训练深前馈神经网络的难度所描述的方法的值 - Glorot，X. &安培; Bengio，Y
。（2010），使用的均匀分布。将得到的张量将已经从采样值 U  （ \-  一个 ， 一 ） \ mathcal【U}（ - A，A） U  （ \-
一 ， 一 ） 其中

a=gain×6fan_in+fan_outa = \text{gain} \times \sqrt{\frac{6}{\text{fan\\_in} +
\text{fan\\_out}}} a=gain×fan_in+fan_out6​​

又称Glorot初始化。

Parameters

    

  * **tensor** – an n-dimensional torch.Tensor

  * **获得** \- 任选的比例因子

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    

`torch.nn.init.``xavier_normal_`( _tensor_ , _gain=1.0_
)[[source]](_modules/torch/nn/init.html#xavier_normal_)

    

填充输入张量与根据在理解训练深前馈神经网络的难度所描述的方法的值 - Glorot，X. &安培; Bengio，Y
。（2010），使用正态分布。将得到的张量将已经从采样值 N  （ 0  STD  2  ） \ mathcal {N}（0，\文本{性病} ^ 2） N
（ 0  ， STD  2  ） 其中

std=gain×2fan_in+fan_out\text{std} = \text{gain} \times
\sqrt{\frac{2}{\text{fan\\_in} + \text{fan\\_out}}} std=gain×fan_in+fan_out2​​

Also known as Glorot initialization.

Parameters

    

  * **tensor** – an n-dimensional torch.Tensor

  * **gain** – an optional scaling factor

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.xavier_normal_(w)
    

`torch.nn.init.``kaiming_uniform_`( _tensor_ , _a=0_ , _mode='fan_in'_ ,
_nonlinearity='leaky_relu'_
)[[source]](_modules/torch/nn/init.html#kaiming_uniform_)

    

填充输入张量与根据在深钻研整流器所描述的方法的值：对ImageNet分类超越人类水平的性能 - 赫，K。等人。
（2015），使用的均匀分布。将得到的张量将已经从采样值 U  （ \-  结合 ， 结合 ） \ mathcal【U}（ -
\文本{结合}，\文本{结合}）  U  （ \-  结合 ， 结合 ） 其中

bound=6(1+a2)×fan_in\text{bound} = \sqrt{\frac{6}{(1 + a^2) \times
\text{fan\\_in}}} bound=(1+a2)×fan_in6​​

也被称为他的初始化。

Parameters

    

  * **tensor** – an n-dimensional torch.Tensor

  * **一** \- （默认为0 RELU）该层之后使用的整流器的负斜率

  * **模式** \- 为`'fan_in' `（默认）或`'fan_out' `。选择`“fan_in”`保留在直传的权重的方差的大小。选择`'fan_out' `保留在向后传量值。

  * **非线性** \- 非线性函数（ nn.functional 名称），建议只使用与`'RELU' `或`'leaky_relu' `（默认）。

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    

`torch.nn.init.``kaiming_normal_`( _tensor_ , _a=0_ , _mode='fan_in'_ ,
_nonlinearity='leaky_relu'_
)[[source]](_modules/torch/nn/init.html#kaiming_normal_)

    

填充输入张量与根据在深钻研整流器所描述的方法的值：对ImageNet分类超越人类水平的性能 - 赫，K。等人。
（2015），使用正态分布。将得到的张量将已经从采样值 N  （ 0  STD  2  ） \ mathcal {N}（0，\文本{性病} ^ 2） N
（ 0  ， STD  2  ） 其中

std=2(1+a2)×fan_in\text{std} = \sqrt{\frac{2}{(1 + a^2) \times
\text{fan\\_in}}} std=(1+a2)×fan_in2​​

Also known as He initialization.

Parameters

    

  * **tensor** – an n-dimensional torch.Tensor

  * **a** – the negative slope of the rectifier used after this layer (0 for ReLU by default)

  * **mode** – either `'fan_in'`(default) or `'fan_out'`. Choosing `'fan_in'`preserves the magnitude of the variance of the weights in the forward pass. Choosing `'fan_out'`preserves the magnitudes in the backwards pass.

  * **nonlinearity** – the non-linear function (nn.functional name), recommended to use only with `'relu'`or `'leaky_relu'`(default).

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    

`torch.nn.init.``orthogonal_`( _tensor_ , _gain=1_
)[[source]](_modules/torch/nn/init.html#orthogonal_)

    

填充输入张量具有（半）正交矩阵，如在的精确解说明在深的线性神经网络学习的非线性动力学 - 萨克斯，A。等。
（2013年）。输入张量必须至少有2个维度，以及用于具有多于2个维度张量后尺寸变平。

Parameters

    

  * **张量** \- n维 torch.Tensor ，其中 n的 ≥ 2  n的\ GEQ 2  n的 ≥ 2 

  * **获得** \- 任选的比例因子

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.orthogonal_(w)
    

`torch.nn.init.``sparse_`( _tensor_ , _sparsity_ , _std=0.01_
)[[source]](_modules/torch/nn/init.html#sparse_)

    

填充2D输入张量作为稀疏矩阵，其中的非零元素将从正态分布 被吸入N  （ 0  ， 0.01  ） \ mathcal {N} （0，0.01） N  （
0  ， 0  。 0  1  ） ，经由自由Hessian矩阵的优化在深学习描述 - 马丁，J。（2010）。

Parameters

    

  * **tensor** – an n-dimensional torch.Tensor

  * **稀疏** \- 元素中的每一列的级分被设置为零

  * **STD** \- 所使用的正态分布的标准偏差，以产生非零值

Examples

    
    
    >>> w = torch.empty(3, 5)
    >>> nn.init.sparse_(w, sparsity=0.1)
    

[Next ![](_static/images/chevron-right-orange.svg)](optim.html "torch.optim")
[![](_static/images/chevron-right-orange.svg) Previous](nn.functional.html
"torch.nn.functional")

* * *

©版权所有2019年，Torch 贡献者。