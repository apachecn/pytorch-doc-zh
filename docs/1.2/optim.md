# torch.optim

`torch.optim`是实施各种优化算法的软件包。最常用的方法已经被支持，并且接口足够一般情况下，让更尖端的也可以很容易地集成到未来。

## 如何使用一个优化

要使用 `torch.optim`你必须构造一个优化的对象，这将保持当前状态，并且会更新基于计算梯度的参数。

### 其构建

为了构建一个 `优化 `你必须给它包含的参数可迭代（都应该`可变 `S ）优化。然后，您可以指定特定的优化选项，如学习率，权衰减等。

注意

如果你需要移动的模型通过`到GPU .cuda（） `，请构建优化的前这样做。之后`的模型的参数.cuda（） `将与那些呼叫之前不同的对象。

在一般情况下，你应该确保优化的参数生活在一致的位置时，优化器的建造和使用。

例：

    
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)
    

### 每参数选项

`优化 `S还支持指定每个参数的选项。要做到这一点，而不是传递`可变 `S的迭代，通过在[ `可迭代DICT`
](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.7\)") S
。它们中的每将定义一个单独的参数组，并应包含一个`PARAMS`键，包含属于它的参数列表。其他键应该匹配优化器接受关键字参数，并且将作为该组优化选项。

Note

您还可以通过选项作为关键字参数。他们将作为默认设置，在不重写它们的组。如果你只是想改变一个选项，同时保持参数组之间是一致的所有其他人，这非常有用。

例如，这是非常有用的，当一个人想指定每层的学习率：

    
    
    optim.SGD([
                    {'params': model.base.parameters()},
                    {'params': model.classifier.parameters(), 'lr': 1e-3}
                ], lr=1e-2, momentum=0.9)
    

这意味着`model.base 的`参数将使用的`缺省学习速率 1E-2`，`model.classifier`的参数将使用的`1E-3
`，和`0.9  [动量HTG19学习速率]将用于所有参数。`

### 以优化步骤

所有优化实现 `步骤（） `的方法，即更新的参数。它可以以两种方式使用：

#### `optimizer.step（） `

这是最优化支持的一个简化版本。一旦梯度使用例如计算的功能可以被称为`向后（） `。

Example:

    
    
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    

#### `optimizer.step（闭合） `

一些优化算法，如共轭梯度和LBFGS需要的功能，多次重新评估，所以你必须在一个封闭，使他们能够重新计算模型通过。封闭应清除的梯度，计算损失，并将其返回。

Example:

    
    
    for input, target in dataset:
        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss
        optimizer.step(closure)
    

## 算法

_class_`torch.optim.``Optimizer`( _params_ , _defaults_
)[[source]](_modules/torch/optim/optimizer.html#Optimizer)

    

基类的所有优化。

警告

参数需要被指定为具有确定性的排序是运行之间的一致的集合。不满足这些属性的对象的例子是组和迭代过的字典值。

Parameters

    

  * **PARAMS** （ _可迭代_ ） - [ `torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") S的迭代或[ `字典 `](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.7\)")秒。指定了张量应该优化。

  * **默认** \- （字典）：含有的优化选项的默认值的字典（当使用时的参数组不指定它们）。

`add_param_group`( _param_group_
)[[source]](_modules/torch/optim/optimizer.html#Optimizer.add_param_group)

    

一组PARAM添加到 `优化 `S  param_groups 。

当微调预训练的网络作为冷冻层可以由可训练并添加到 `优化 `作为训练的进行这可以是有用的。

Parameters

    

  * **param_group** （[ _DICT_ ](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.7\)")） - 指定哪张量应与组一起被优化

  * **优化选项。** （ _具体_ ） - 

`load_state_dict`( _state_dict_
)[[source]](_modules/torch/optim/optimizer.html#Optimizer.load_state_dict)

    

加载优化状态。

Parameters

    

**state_dict** （[ _DICT_
](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.7\)")）
- 优化状态。应该从一个调用返回一个目的是 `state_dict（） `。

`state_dict`()[[source]](_modules/torch/optim/optimizer.html#Optimizer.state_dict)

    

返回作为[ `字典 `](https://docs.python.org/3/library/stdtypes.html#dict "\(in
Python v3.7\)")优化的状态。

它包含两个条目：

  * state - a dict holding current optimization state. Its content
    

优化类之间是不同的。

  * param_groups - 包含所有参数组的字典

`step`( _closure_
)[[source]](_modules/torch/optim/optimizer.html#Optimizer.step)

    

执行单一优化步骤（参数更新）。

Parameters

    

**闭合** （ _可调用_ ） - 即重新评估该模型，并返回损失的闭合件。可选的最优化。

`zero_grad`()[[source]](_modules/torch/optim/optimizer.html#Optimizer.zero_grad)

    

清除所有优化[ `torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") S的梯度。

_class_`torch.optim.``Adadelta`( _params_ , _lr=1.0_ , _rho=0.9_ , _eps=1e-06_
, _weight_decay=0_ )[[source]](_modules/torch/optim/adadelta.html#Adadelta)

    

实现Adadelta算法。

它已在[ ADADELTA被提出：一种自适应学习速率法[HTG1。](https://arxiv.org/abs/1212.5701)

Parameters

    

  * **PARAMS** （ _可迭代_ ） - 的参数可迭代优化或类型的字典定义的参数组

  * **RHO** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 系数用于计算平方梯度（缺省的运行平均值： 0.9）

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 术语添加到分母以提高数值稳定性（默认值：1E -6）

  * **LR** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 系数尺度增量之前它被施加到参数（缺省：1.0）

  * **weight_decay** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 重量衰变（L2罚分）（默认值：0）

`step`( _closure=None_
)[[source]](_modules/torch/optim/adadelta.html#Adadelta.step)

    

执行单一优化步骤。

Parameters

    

**闭合** （ _可调用_ _，_ _可选_ ） - 即重新评估该模型，并返回损失的闭合件。

_class_`torch.optim.``Adagrad`( _params_ , _lr=0.01_ , _lr_decay=0_ ,
_weight_decay=0_ , _initial_accumulator_value=0_
)[[source]](_modules/torch/optim/adagrad.html#Adagrad)

    

实现Adagrad算法。

它已在[HTG0自适应次梯度法在线学习和随机优化被提出。

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **LR** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 学习率（默认值：1E-2）

  * **lr_decay** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 学习速率衰变（默认值：0）

  * **weight_decay** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – weight decay (L2 penalty) (default: 0)

`step`( _closure=None_
)[[source]](_modules/torch/optim/adagrad.html#Adagrad.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

_class_`torch.optim.``Adam`( _params_ , _lr=0.001_ , _betas=(0.9_ , _0.999)_ ,
_eps=1e-08_ , _weight_decay=0_ , _amsgrad=False_
)[[source]](_modules/torch/optim/adam.html#Adam)

    

亚当实现算法。

它已经提出了[亚当：一种随机优化](https://arxiv.org/abs/1412.6980)方法。

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **LR** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 学习率（默认值：1E-3）

  * **贝塔** （ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") __ _，_ _可选_ ） - 用于计算梯度的运行平均值和其平方的系数（默认值：0.9（，0.999））

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 术语添加到分母以提高数值稳定性（默认值：1E -8）

  * **weight_decay** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – weight decay (L2 penalty) (default: 0)

  * **amsgrad** （ _布尔_ _，_ _可选_ ） - 是否使用该算法的AMSGrad变体从纸[在收敛亚当和超越](https://openreview.net/forum?id=ryQu7f-RZ)（默认值：false）

`step`( _closure=None_ )[[source]](_modules/torch/optim/adam.html#Adam.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

_class_`torch.optim.``AdamW`( _params_ , _lr=0.001_ , _betas=(0.9_ , _0.999)_
, _eps=1e-08_ , _weight_decay=0.01_ , _amsgrad=False_
)[[source]](_modules/torch/optim/adamw.html#AdamW)

    

实现AdamW算法。

原来亚当算法提出[亚当：一种随机优化](https://arxiv.org/abs/1412.6980)方法。该AdamW变种在[解耦权衰减正则建议[HTG3。](https://arxiv.org/abs/1711.05101)

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **lr** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – learning rate (default: 1e-3)

  * **betas** ( _Tuple_ _[_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _]_ _,_ _optional_ ) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))

  * **eps** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – term added to the denominator to improve numerical stability (default: 1e-8)

  * **weight_decay** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 重量衰减系数（默认值：1E-2）

  * **amsgrad** ( _boolean_ _,_ _optional_ ) – whether to use the AMSGrad variant of this algorithm from the paper [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) (default: False)

`step`( _closure=None_
)[[source]](_modules/torch/optim/adamw.html#AdamW.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

_class_`torch.optim.``SparseAdam`( _params_ , _lr=0.001_ , _betas=(0.9_ ,
_0.999)_ , _eps=1e-08_
)[[source]](_modules/torch/optim/sparse_adam.html#SparseAdam)

    

实现适用于稀疏张量亚当算法的懒惰版本。

在该变型中，只有在梯度显示时刻得到更新，并且只有所述梯度的那些部分会应用于参数。

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **lr** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – learning rate (default: 1e-3)

  * **betas** ( _Tuple_ _[_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_[ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _]_ _,_ _optional_ ) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))

  * **eps** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – term added to the denominator to improve numerical stability (default: 1e-8)

`step`( _closure=None_
)[[source]](_modules/torch/optim/sparse_adam.html#SparseAdam.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

_class_`torch.optim.``Adamax`( _params_ , _lr=0.002_ , _betas=(0.9_ , _0.999)_
, _eps=1e-08_ , _weight_decay=0_
)[[source]](_modules/torch/optim/adamax.html#Adamax)

    

实现Adamax算法（亚当基于无穷范数变体）。

It has been proposed in [Adam: A Method for Stochastic
Optimization](https://arxiv.org/abs/1412.6980).

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **LR** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 学习率（默认值：2E-3）

  * **贝塔** （ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") __ _，_ _可选_ ） - 用于计算梯度及正方形的运行平均值的系数

  * **eps** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – term added to the denominator to improve numerical stability (default: 1e-8)

  * **weight_decay** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – weight decay (L2 penalty) (default: 0)

`step`( _closure=None_
)[[source]](_modules/torch/optim/adamax.html#Adamax.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

_class_`torch.optim.``ASGD`( _params_ , _lr=0.01_ , _lambd=0.0001_ ,
_alpha=0.75_ , _t0=1000000.0_ , _weight_decay=0_
)[[source]](_modules/torch/optim/asgd.html#ASGD)

    

器具场均随机梯度下降。

它已在[随机逼近的加速度的平均值](http://dl.acm.org/citation.cfm?id=131098)被提出。

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **lr** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – learning rate (default: 1e-2)

  * **lambd** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 衰减项（默认值：1E-4）

  * **阿尔法** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 功率ETA更新（默认值：0.75）

  * **T0** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 点处开始平均（默认值：1E6）

  * **weight_decay** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – weight decay (L2 penalty) (default: 0)

`step`( _closure=None_ )[[source]](_modules/torch/optim/asgd.html#ASGD.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

_class_`torch.optim.``LBFGS`( _params_ , _lr=1_ , _max_iter=20_ ,
_max_eval=None_ , _tolerance_grad=1e-05_ , _tolerance_change=1e-09_ ,
_history_size=100_ , _line_search_fn=None_
)[[source]](_modules/torch/optim/lbfgs.html#LBFGS)

    

实现L-BFGS算法，很大程度上受到启发minFunc & LT ;
https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html & GT [; ] 。

Warning

这种优化不支持每个参数的选项和参数组（只能有一个）。

Warning

现在所有的参数都必须在单个设备上。这将在未来得到改善。

Note

这是一个非常内存密集型优化器（它需要额外的`param_bytes  *  （history_size  +  1）
`字节）。如果不装入内存尝试降低历史记录的大小，或者使用不同的算法。

Parameters

    

  * **LR** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 学习率（默认值：1）

  * **max_iter** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 每优化步骤迭代的最大数量（默认值：20）

  * **max_eval** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 每优化步骤功能评价的最大数量（默认值：max_iter * 1.25）。

  * （：1E-5默认）上一阶最优终止公差 - **tolerance_grad** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")）。

  * （：1E-9默认）上函数值/参数改变终止公差 - **tolerance_change** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")）。

  * **history_size** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 更新历史尺寸（默认值：100）。

  * **line_search_fn** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） - 要么“strong_wolfe”或无（默认：无）。

`step`( _closure_ )[[source]](_modules/torch/optim/lbfgs.html#LBFGS.step)

    

Performs a single optimization step.

Parameters

    

**闭合** （ _可调用_ ） - 即重新评估该模型，并返回损失的闭合件。

_class_`torch.optim.``RMSprop`( _params_ , _lr=0.01_ , _alpha=0.99_ ,
_eps=1e-08_ , _weight_decay=0_ , _momentum=0_ , _centered=False_
)[[source]](_modules/torch/optim/rmsprop.html#RMSprop)

    

实现RMSprop算法。

由G.韩丁在他的[HTG0当然提出。

居中版本首先出现在[生成的序列的递归神经网络](https://arxiv.org/pdf/1308.0850v5.pdf)。

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **lr** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – learning rate (default: 1e-2)

  * **动量** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 动量因子（默认值：0）

  * **阿尔法** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 平滑常数（默认值：0.99）

  * **eps** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – term added to the denominator to improve numerical stability (default: 1e-8)

  * **居中** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 如果`真 `，计算居中RMSProp，梯度是由它的方差的估计归一化

  * **weight_decay** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – weight decay (L2 penalty) (default: 0)

`step`( _closure=None_
)[[source]](_modules/torch/optim/rmsprop.html#RMSprop.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

_class_`torch.optim.``Rprop`( _params_ , _lr=0.01_ , _etas=(0.5_ , _1.2)_ ,
_step_sizes=(1e-06_ , _50)_
)[[source]](_modules/torch/optim/rprop.html#Rprop)

    

实现了弹性BP算法。

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **lr** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – learning rate (default: 1e-2)

  * **ETAS** （ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") __ _，_ _可选_ ） - 双（etaminus，etaplis），即是乘法增加和减少的因素（默认值：（0.5，1.2 ））

  * **step_sizes** （ _元组_ _[_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ [ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") __ _，_ _可选_ ） - 一对最小和最大允许的步长大小（默认值：（1E-6，50））

`step`( _closure=None_
)[[source]](_modules/torch/optim/rprop.html#Rprop.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

_class_`torch.optim.``SGD`( _params_ , _lr= <required parameter>_,
_momentum=0_ , _dampening=0_ , _weight_decay=0_ , _nesterov=False_
)[[source]](_modules/torch/optim/sgd.html#SGD)

    

实现随机梯度下降（任选地与动量）。

涅斯捷罗夫势头是基于[公式在初始化和动量的深度学习](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)的重要性。

Parameters

    

  * **params** ( _iterable_ ) – iterable of parameters to optimize or dicts defining parameter groups

  * **LR** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 学习率

  * **momentum** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – momentum factor (default: 0)

  * **weight_decay** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _,_ _optional_ ) – weight decay (L2 penalty) (default: 0)

  * **阻尼** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _，_ _可选_ ） - 润湿动量（默认值：0）

  * **涅斯捷罗夫** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)") _，_ _可选_ ） - 使涅斯捷罗夫势头（默认值：false）

例

    
    
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> optimizer.zero_grad()
    >>> loss_fn(model(input), target).backward()
    >>> optimizer.step()
    

Note

SGD与动量执行/涅斯捷罗夫巧妙地不同于Sutskever等。人。而在一些其他框架的实现。

考虑动量的特定情况下，更新可以写成

v=ρ∗v+gp=p−lr∗vv = \rho * v + g \\\ p = p - lr * v v=ρ∗v+gp=p−lr∗v

其中p，G，V和 ρ \哌 [分别HTG13]  ρ 表示的参数，梯度，速度和动量。

这是相对于Sutskever等。人。和其采用的形式的更新其他框架

v=ρ∗v+lr∗gp=p−vv = \rho * v + lr * g \\\ p = p - v v=ρ∗v+lr∗gp=p−v

该版本涅斯捷罗夫，类似修改。

`step`( _closure=None_ )[[source]](_modules/torch/optim/sgd.html#SGD.step)

    

Performs a single optimization step.

Parameters

    

**closure** ( _callable_ _,_ _optional_ ) – A closure that reevaluates the
model and returns the loss.

## 如何调整学习率

`torch.optim.lr_scheduler`提供了几种方法来调整基于历元的数目的学习速率。`
torch.optim.lr_scheduler.ReduceLROnPlateau`允许动态学习速率还原性基于一些验证测量。

学习速率调度后，应优化器的更新应用;例如，你应该写你的代码是这样的：

    
    
    >>> scheduler = ...
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()
    

Warning

到PyTorch 1.1.0之前，学习率调度，预计优化的更新之前被称为; 1.1.0在BC破的方式改变了这种行为。如果您使用的学习速率调度（调用`
scheduler.step（） `）优化的更新前（调用`optimizer.step（） `
），这将跳过学习税率表的第一价值。如果您无法升级到1.1.0 PyTorch后重现的结果，请检查您是否调用`scheduler.step（）
`在错误的时间。

_class_`torch.optim.lr_scheduler.``LambdaLR`( _optimizer_ , _lr_lambda_ ,
_last_epoch=-1_ )[[source]](_modules/torch/optim/lr_scheduler.html#LambdaLR)

    

设置每个参数组的学习速率的初始LR倍的给定功能。当last_epoch = -1，设置初始LR作为LR。

Parameters

    

  * **优化** （ _优化_ ） - 包裹优化器。

  * **lr_lambda** （ _函数_ _或_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 其计算给定的整数参数划时代一个乘法因子的函数，或这样的功能的列表，每一个组中optimizer.param_groups。

  * **last_epoch** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 最后历元的索引。缺省值：-1。

Example

    
    
    >>> # Assuming optimizer has two groups.
    >>> lambda1 = lambda epoch: epoch // 30
    >>> lambda2 = lambda epoch: 0.95 ** epoch
    >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()
    

`load_state_dict`( _state_dict_
)[[source]](_modules/torch/optim/lr_scheduler.html#LambdaLR.load_state_dict)

    

加载调度状态。

Parameters

    

**state_dict** （[ _DICT_
](https://docs.python.org/3/library/stdtypes.html#dict "\(in Python v3.7\)")）
- 调度器状态。应该从一个调用返回一个目的是 `state_dict（） `。

`state_dict`()[[source]](_modules/torch/optim/lr_scheduler.html#LambdaLR.state_dict)

    

返回作为[ `字典 `](https://docs.python.org/3/library/stdtypes.html#dict "\(in
Python v3.7\)")调度器的状态。

它包含了每一个变量自.__
dict__的项，这并不是优化。如果他们是可调用的对象，而不是他们是否是函数或lambda表达式学习率lambda函数只会被保存。

_class_`torch.optim.lr_scheduler.``StepLR`( _optimizer_ , _step_size_ ,
_gamma=0.1_ , _last_epoch=-1_
)[[source]](_modules/torch/optim/lr_scheduler.html#StepLR)

    

设置每个参数组，以通过γ每STEP_SIZE历元衰减初始LR学习率。当last_epoch = -1，设置初始LR作为LR。

Parameters

    

  * **optimizer** ( _Optimizer_) – Wrapped optimizer.

  * **STEP_SIZE** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 学习速率衰变的时期。

  * **伽马** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 学习速率衰变的乘法因子。默认值：0.1。

  * **last_epoch** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – The index of last epoch. Default: -1.

Example

    
    
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.05     if epoch < 30
    >>> # lr = 0.005    if 30 <= epoch < 60
    >>> # lr = 0.0005   if 60 <= epoch < 90
    >>> # ...
    >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()
    

_class_`torch.optim.lr_scheduler.``MultiStepLR`( _optimizer_ , _milestones_ ,
_gamma=0.1_ , _last_epoch=-1_
)[[source]](_modules/torch/optim/lr_scheduler.html#MultiStepLR)

    

每个参数组的学习率一旦历元的数目达到里程碑之一设置为通过γ衰减初始LR。当last_epoch = -1，设置初始LR作为LR。

Parameters

    

  * **optimizer** ( _Optimizer_) – Wrapped optimizer.

  * **里程碑** （[ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 历元索引列表。必须增加。

  * **gamma** ([ _float_](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")) – Multiplicative factor of learning rate decay. Default: 0.1.

  * **last_epoch** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – The index of last epoch. Default: -1.

Example

    
    
    >>> # Assuming optimizer uses lr = 0.05 for all groups
    >>> # lr = 0.05     if epoch < 30
    >>> # lr = 0.005    if 30 <= epoch < 80
    >>> # lr = 0.0005   if epoch >= 80
    >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
    >>> for epoch in range(100):
    >>>     train(...)
    >>>     validate(...)
    >>>     scheduler.step()
    

_class_`torch.optim.lr_scheduler.``ExponentialLR`( _optimizer_ , _gamma_ ,
_last_epoch=-1_
)[[source]](_modules/torch/optim/lr_scheduler.html#ExponentialLR)

    

每个参数组的学习率设置为通过γ每历元衰减初始LR。当last_epoch = -1，设置初始LR作为LR。

Parameters

    

  * **optimizer** ( _Optimizer_) – Wrapped optimizer.

  * **伽马** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 学习速率衰变的乘法因子。

  * **last_epoch** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – The index of last epoch. Default: -1.

_class_`torch.optim.lr_scheduler.``CosineAnnealingLR`( _optimizer_ , _T_max_ ,
_eta_min=0_ , _last_epoch=-1_
)[[source]](_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR)

    

使用余弦退火时间表，其中 η 米设定各参数组的学习率 一 × \ eta_ {MAX}  η M  一 × 被设定为初始LR和 T  C  U  R  T_
{ CUR}  [H TG95] T  C  U  [R  是因为在SGDR上次重启历元数：

ηt=ηmin+12(ηmax−ηmin)(1+cos⁡(TcurTmaxπ))\eta_t = \eta_{min} +
\frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{T_{cur}}{T_{max}}\pi))
ηt​=ηmin​+21​(ηmax​−ηmin​)(1+cos(Tmax​Tcur​​π))

当last_epoch = -1，设置初始LR作为LR。

它已经提出了[
SGDR：随机梯度下降以热烈的重新启动[HTG1。请注意，这仅实现SGDR的余弦退火的一部分，而不是重新启动。](https://arxiv.org/abs/1608.03983)

Parameters

    

  * **optimizer** ( _Optimizer_) – Wrapped optimizer.

  * **T_MAX** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 迭代的最大数量。

  * **eta_min** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 最小学习速率。默认值：0。

  * **last_epoch** ([ _int_](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")) – The index of last epoch. Default: -1.

_class_`torch.optim.lr_scheduler.``ReduceLROnPlateau`( _optimizer_ ,
_mode='min'_ , _factor=0.1_ , _patience=10_ , _verbose=False_ ,
_threshold=0.0001_ , _threshold_mode='rel'_ , _cooldown=0_ , _min_lr=0_ ,
_eps=1e-08_
)[[source]](_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau)

    

当指标已停止提高降低学习速率。模型通常受益于2-10一次学习停滞的一个因素减少学习率。该调度器读取指标量，并且如果没有改善被视作历元的一个“忍耐”号，学习率降低。

Parameters

    

  * **optimizer** ( _Optimizer_) – Wrapped optimizer.

  * **模式** （[ _STR_ [HTG5） - 酮的分钟HTG7]， MAX 。 ;在分钟HTG11]模式中，当监视的量已经停止下降LR将减少在 MAX 模式将监视时的数量已停止增加被减小。默认：“分钟”。](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")

  * **因子** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 因子，通过该学习率将降低。 new_lr = LR *因子。默认值：0.1。

  * 没有改善之后，学习率将降低历元数 - **耐性** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")）。例如，如果耐性= 2 ，然后我们将忽略所述第一2个时期没有改善，并且第三时期后只会降低LR如果损失仍然没有再提高。默认值：10。

  * **冗长** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `，打印一条消息到标准输出每个更新。默认值：`假 [HTG13。`

  * 用于测量新的最佳，仅着眼于显著变化阈值 - **阈** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")）。默认值：1E-4。

  * **threshold_mode** （[ _STR_ [HTG5） - 酮的 REL ， ABS 。在 REL 模式，dynamic_threshold =最好*（1个+阈值）在“最大”模式或最佳* - 在分钟HTG13]模式（1个阈值）。在 ABS 模式，dynamic_threshold =在最大最佳+阈模式或最佳 - 阈值分钟HTG19]模式。默认：“相对”。](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")

  * **冷却时间** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 历元LR之后恢复正常操作之前要等待的数量已经减少。默认值：0。

  * **min_lr** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 标量或标量的列表。的下界所有PARAM群体的学习速率或每个组分别。默认值：0。

  * **EPS** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 最小衰变应用于LR。如果新旧LR的差比EPS小，更新被忽略。默认值：1E-8。

Example

    
    
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
    >>> for epoch in range(10):
    >>>     train(...)
    >>>     val_loss = validate(...)
    >>>     # Note that step should be called after validate()
    >>>     scheduler.step(val_loss)
    

_class_`torch.optim.lr_scheduler.``CyclicLR`( _optimizer_ , _base_lr_ ,
_max_lr_ , _step_size_up=2000_ , _step_size_down=None_ , _mode='triangular'_ ,
_gamma=1.0_ , _scale_fn=None_ , _scale_mode='cycle'_ , _cycle_momentum=True_ ,
_base_momentum=0.8_ , _max_momentum=0.9_ , _last_epoch=-1_
)[[source]](_modules/torch/optim/lr_scheduler.html#CyclicLR)

    

设置根据周期性学习率政策（CLR）的各参数组的学习率。策略周期两个边界之间的学习率具有恒定频率，如在文献[周期性学习价格的训练神经网络](https://arxiv.org/abs/1506.01186)详述。两个边界之间的距离可以在每个迭代或每个周期的基础上进行缩放。

周期性学习率的政策变化，每批次后的学习率。 步骤应该被称为一个批次已被用于训练。

这个类有三个内置的政策，如纸提出：“三角”：

> 基本三角形的周期与/没有幅度缩放。

“triangular2”:

    

通过半每个周期扩展初始幅度A基本三角形的周期。

“exp_range”:

    

在每个循环迭代通过γ**（循环迭代）缩放初始幅度A循环。

此实现改编自GitHub库：[ bckenstler / CLR ](https://github.com/bckenstler/CLR)

Parameters

    

  * **optimizer** ( _Optimizer_) – Wrapped optimizer.

  * **base_lr** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 初始学习速率是低边界在循环的每个参数组。

  * **max_lr** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 在循环上学习率边界每个参数组。在功能上，它定义了周期振幅（max_lr - base_lr）。在任何周期的LR是base_lr和振幅的某些缩放的总和;因此max_lr实际上可能没有达到根据缩放功能。

  * **step_size_up** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 在增加周期的一半训练迭代次数。默认值：2000

  * **step_size_down** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 在一个周期的减小一半训练迭代次数。如果step_size_down是无，它被设置为step_size_up。默认值：无

  * **模式** （[ _STR_ [HTG5） - 酮{三角形，triangular2，exp_range}的。值对应于上述详细的政策。如果scale_fn不是无，则忽略此参数。默认：“三角”](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")

  * **伽马** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)")） - 常量在“exp_range”缩放功能：伽马**（循环迭代）缺省值：1.0

  * **scale_fn** （ _函数_ ） - 定义的缩放比例由单个参数lambda函数，定义的策略，其中0 & LT ; = scale_fn（X）& LT ; = 1对于所有的x & GT ; = 0。如果指定，则 '模式' 被忽略。默认值：无

  * **scale_mode** （[ _STR_ ](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） - {“循环”，“次迭代”}。定义是否scale_fn是（因为循环的开始训练迭代）上循环数或周期的迭代评估。默认：“周期”

  * **cycle_momentum** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 如果`真 `，动量进行逆循环到 'base_momentum' 和之间学习率'max_momentum'。默认值：true

  * **base_momentum** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 在循环下动量边界每个参数组。需要注意的是势头是负循环到学习速率;在一个周期的高峰期，气势“base_momentum”和学习率“max_lr”。默认值：0.8

  * **max_momentum** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in Python v3.7\)") _或_ [ _列表_ ](https://docs.python.org/3/library/stdtypes.html#list "\(in Python v3.7\)")） - 在周期上动量边界每个参数组。在功能上，它定义了周期振幅（max_momentum - base_momentum）。在任何周期的动量是max_momentum和振幅的一些结垢差;因此base_momentum实际上可能没有达到根据缩放功能。需要注意的是势头是负循环到学习速率;在一个周期的开始，气势“max_momentum”和学习率“base_lr”默认值：0.9

  * **last_epoch** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 最后一批的索引。恢复训练作业时使用此参数。由于步骤（）应当每批之后，而不是每个时期之后被调用，此数字表示计算，而不是计算总历元的数目的 _批次的总数。当last_epoch = -1，计划从一开始启动。默认值：-1_

Example

    
    
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    >>> scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    >>> data_loader = torch.utils.data.DataLoader(...)
    >>> for epoch in range(10):
    >>>     for batch in data_loader:
    >>>         train_batch(...)
    >>>         scheduler.step()
    

`get_lr`()[[source]](_modules/torch/optim/lr_scheduler.html#CyclicLR.get_lr)

    

在计算指数批学习率。该函数将 self.last_epoch 作为最后一批指标。

如果 self.cycle_momentum 是`真 `时，此功能有更新优化的势头的副作用。

[Next ![](_static/images/chevron-right-orange.svg)](autograd.html "Automatic
differentiation package - torch.autograd") [![](_static/images/chevron-right-
orange.svg) Previous](nn.init.html "torch.nn.init")

* * *

©版权所有2019年，Torch 贡献者。