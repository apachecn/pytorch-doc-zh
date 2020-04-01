# torch.optim

> 译者：[@于增源](https://github.com/ZengyuanYu)
> 
> 校对者：[@青梅往事](https://github.com/2556120684)

`torch.optim` is a package implementing various optimization algorithms. Most commonly used methods are already supported, and the interface is general enough, so that more sophisticated ones can be also easily integrated in the future.

## 如何使用 optimizer (优化器)

为了使用 `torch.optim` 你需要创建一个 optimizer 对象, 这个对象能够保持当前的状态以及依靠梯度计算 来完成参数更新.

### 构建

要构建一个 `Optimizer` 你需要一个可迭代的参数 (全部都应该是 [`Variable`](autograd.html#torch.autograd.Variable "torch.autograd.Variable")) 进行优化. 然后, 你能够设置优化器的参数选项, 例如学习率, 权重衰减等.

注解：

如果你需要通过 `.cuda()` 将模型移动到 GPU 上, 请在构建优化器之前来移动. 模型的参数在进行 `.cuda()` 之后将变成不同的对象,该对象与之前调用的参数不同.

通常来说, 在对优化器进行构建和调用的时候, 你应该要确保优化参数位于相同的 地点.

例子

```py
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)

```

### 为每个参数单独设置选项

`Optimizer` 也支持为每个参数单独设置选项. 若要这么做, 不要直接使用 `~torch.autograd.Variable` 的迭代, 而是使用 [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) 的迭代. 每一个 dict 都分别定义了一组参数, 并且应该要包含 `params` 键,这个键对应列表的参数. 其他的键应该与 optimizer 所接受的其他参数的关键字相匹配, 并且会被用于对这组参数的优化.

注解：

你仍然能够传递选项作为关键字参数.在未重写这些选项的组中, 它们会被用作默认值. 这非常适用于当你只想改动一个参数组的选项, 但其他参数组的选项不变的情况.

例如, 当我们想指定每一层的学习率时, 这是非常有用的:

```py
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)

```

这意味着 `model.base` 的参数将会使用 `1e-2` 的学习率,``model.classifier`` 的参数将会使用 `1e-3` 的学习率, 并且 `0.9` 的 momentum 将应用于所有参数.

### 进行单步优化

所有的优化器都实现了 `step()` 方法, 且更新到所有的参数. 它可以通过以下两种方式来使用:

#### `optimizer.step()`

这是大多数 optimizer 所支持的简化版本. 一旦使用 [`backward()`](autograd.html#torch.autograd.Variable.backward "torch.autograd.Variable.backward") 之类的函数计算出来梯度之后我们就可以调用这个函数了.

例子

```py
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

```

#### `optimizer.step(closure)`

一些优化算法例如 Conjugate Gradient 和 LBFGS 需要重复多次计算函数, 因此你需要传入一个闭包去允许它们重新计算你的模型. 这个闭包应当清空梯度, 计算损失, 然后返回.

例子

```py
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)

```

## 算法

```py
class torch.optim.Optimizer(params, defaults)
```

优化器的基类.

参数：

* `params (iterable)`: `Variable` 或 `dict` 的迭代, 指定了应该优化哪些参数. 
* `defaults (dict)`: 包含了优化选项默认值的字典(一个参数组没有指定的参数选项将会使用默认值).

```py
add_param_group(param_group)
```

增加一组参数到 `Optimizer` 的 `param_groups` 里面.

当微调一个预训练好的网络作为冻结层时是有用的, 它能够使用可训练的和可增加的参数到 `Optimizer` 作为一个训练预处理.

参数：`param_group (dict)` – 指定这一组中具有特殊优化选项的那些 Variables 能够被优化.


```py
load_state_dict(state_dict)
```

加载优化器状态.

参数：`state_dict (dict)` – 优化器状态. 是调用 `state_dict()` 时所返回的对象.


```py
state_dict()
```

以 [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) 的形式返回优化器的状态.

它包含两部分内容:

*   state - 一个包含当前优化状态的字典(dict）, 字典里的内容因优化器的不同而变换.
*   param_groups - 一个包含所有参数组的字典(dict）.

```py
step(closure)
```

进行单次优化(参数更新).

参数：`closure (callable)` – 一个重新评价模型并返回 loss 的闭包大多数优化器可选择.


```py
zero_grad()
```

Clears the gradients of all optimized `Variable` s.

```py
class torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```

实施 Adadelta 算法.

它在 [ADADELTA: 一种可调节学习率的方法](https://arxiv.org/abs/1212.5701) 中提出

Args: * params (iterable): 通过参数迭代去优化或者字典的形式定义参数组. * rho (float, 可选): 用来计算平均平方梯度的系数(默认值: 0.9) * eps (float, 可选): 增加分母来确保数值稳定性(默认值: 1e-6) * lr (float, 可选): 在将 delta 应用于参数之前对其进行系数的缩放(默认值: 1.0) * weight_decay (float, 可选): 权重衰减 (L2正则化) (默认值: 0)

```py
step(closure=None)
```

实行单步优化.

参数：`closure (callable, 可选)` – 重新评估模型并返回误差损失的闭包.


```py
class torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
```

实现 Adagrad 算法.

它在 [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://jmlr.org/papers/v12/duchi11a.html) 中被提出.

Args: * params (iterable): 迭代的优化参数或者以字典的形式定义参数组 * lr (float, 可选): 学习率 (默认值: 1e-2) * lr_decay (float, 可选): 学习率衰减 (默认值: 0) * weight_decay (float, 可选): 权重衰减 (L2正则化) (默认值: 0)

```py
step(closure=None)
```

进行单步优化.

参数：`closure (callable, 可选)` – 一个重新评价模型并返回误差的闭包.


```py
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```

实现 Adam 算法.

它在 [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) 中被提出.

Args: * params (iterable): 迭代的优化参数或者以字典的形式定义参数组. * lr (float, 可选): 学习率 (默认值: 1e-3) * betas (Tuple[float, float], 可选): 用来计算梯度和平方梯度的系数 (默认值: (0.9, 0.999)) * eps (float, 可选): 增加分母来确保数值稳定性 (默认值: 1e-8) * weight_decay (float, 可选): 权重衰减 (L2 正则化) (默认值: 0)

```py
step(closure=None)
```

进行单步优化.

参数：`closure (callable, 可选)` – 一个重新评价模型并返回误差的闭包.


```py
class torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)
```

实现上一版本 Adam 算法来适用于 sparse tensors.

在这个变化下,只将显示出来的梯度进行更新存储并且只将这部分梯度应用到参数中.

Args: * params (iterable): 待优化的迭代参数或者是定义了参数组的 dict * lr (float, 可选): 学习率 (default: 1e-3) * betas (Tuple[float, float], 可选): 用来计算梯度和平方梯度的系数 (默认值: (0.9, 0.999)) * eps (float, 可选): 增加分母来确保数值稳定性 (默认值: 1e-8)

```py
step(closure=None)
```

进行单步优化.

参数：`closure (callable, 可选)` – 一个重新评价模型并返回 loss 的闭包, 对于大多数参数来说是可选的.


```py
class torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```

实现 Adamax 算法 ( Adam 的一种基于无穷范数的变种).

它在 [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) 中被提出.

Args: * params (iterable): 迭代的优化参数或者以字典的形式定义参数组. * lr (float, 可选): 学习率 (默认值: 2e-3) * betas (Tuple[float, float], 可选): 用来计算梯度和平方梯度的系数 * eps (float, 可选): 增加分母来确保数值稳定性 (默认值: 1e-8) * weight_decay (float, 可选): 权重衰减 (L2 正则化) (默认值: 0)

```py
step(closure=None)
```

进行单步优化.

参数：`closure (callable, 可选)` – 一个重新评价模型并返回误差的闭包.


```py
class torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
```

实现平均随机梯度下降算法.

它在 [Acceleration of stochastic approximation by averaging](http://dl.acm.org/citation.cfm?id=131098) 中被提出

Args: * params (iterable): 迭代的优化参数或者以字典的形式定义参数组 * lr (float, 可选): 学习率 (默认值: 1e-2) * lambd (float, 可选): 衰减期 (默认值: 1e-4) * alpha (float, 可选): eta 更新的权重 (默认值: 0.75) * t0 (float, 可选): 指明在哪一次开始平均化 (默认值: 1e6) * weight_decay (float, 可选): 权重衰减 (L2 正则化) (默认值: 0)

```py
step(closure=None)
```

进行单步优化.

参数：`closure (callable, 可选)` – 一个重新评价模型并返回误差的闭包.


```py
class torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
```

实现 L-BFGS 算法.

警告：

这个 optimizer 不支持为每个参数单独设置选项以及不支持参数组(只能有一个）.

警告：

目前所有的参数不得不都在同一设备上. 这在将来会得到改进.

注解：

这是一个内存高度密集的 optimizer (它要求额外的 `param_bytes * (history_size + 1)` 个字节). 如果它不适应内存, 尝试减小历史规格, 或者使用不同的算法.

Args: * lr (float): 学习率 (默认值: 1) * max_iter (int): 每一步优化的最大迭代次数 (默认值: 20) * max_eval (int): 每一步优化的最大函数评估次数 (默认值: max_iter * 1.25). * tolerance_grad (float): 一阶最优的终止容忍度 (默认值: 1e-5). * tolerance_change (float): 在函数值/参数变化量上的终止容忍度 (默认值: 1e-9). * history_size (int): 更新历史尺寸 (默认值: 100).

```py
step(closure)
```

进行单步优化.

参数：`closure (callable)` – 一个重新评价模型并返回 loss 的闭包, 对于大多数参数来说是可选的.


```py
class torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

实现 RMSprop 算法.

由 G. Hinton 在此提出 [course](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

中心版本首次出现在 [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf).

算法: * params (iterable): 待优化的迭代参数或者是定义了参数组的 dict * lr (float, 可选): 学习率 (默认值: 1e-2) * momentum (float, 可选): 动量因子 (默认值: 0) * alpha (float, 可选): 平滑常量 (default: 0.99) * eps (float, 可选): 为了增加数值计算的稳定性而加到分母里的项 (默认值: 1e-8) * centered (bool, 可选) : 如果为 `True`, 计算 RMSProp 的中值, 并且用它的方差预测值对梯度进行归一化 * weight_decay (float, 可选): weight decay (L2 penalty) (default: 0)

```py
step(closure=None)
```

Performs a single optimization step.

参数：`closure (callable, 可选)` – A closure that reevaluates the model and returns the loss.


```py
class torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
```

实现弹性反向传播算法.

Args: * params (iterable): 待优化的迭代参数或者是定义了参数组的 dict * lr (float, 可选): 学习率 (默认值: 1e-2) * etas (Tuple[float, float], 可选): 一对 (etaminus, etaplis), t它们分别是乘法

> 的增加和减小的因子 (默认值: (0.5, 1.2))

*   `step_sizes (Tuple[float, float], 可选)`: 允许的一对最小和最大的步长 (默认值: (1e-6, 50))

```py
step(closure=None)
```

进行单步优化.

参数：`closure (callable, 可选)` – 一个重新评价模型并返回 loss 的闭包, 对于大多数参数来说是可选的.


```py
class torch.optim.SGD(params, lr=<object object>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

实现随机梯度下降算法 (momentum 可选）.

Nesterov 动量基于 [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf) 中的公式.

Args: * params (iterable): 待优化的迭代参数或者是定义了参数组的 dict * lr (float): 学习率 * momentum (float, 可选): 动量因子 (默认值: 0) * weight_decay (float, 可选): 权重衰减 (L2 正则化) (默认值: 0) * dampening (float, 可选): 动量的抑制因子 (默认值: 0) * nesterov (bool, 可选): 使用 Nesterov 动量 (默认值: False)

示例：

```py
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> optimizer.zero_grad()
>>> loss_fn(model(input), target).backward()
>>> optimizer.step()

```

注解：

带有动量 /Nesterov 的 SGD 的实现稍微不同于 Sutskever 等人以及其他框架中的实现. 考虑动量的具体情况, 更新可以写成

![\begin{split}v = \rho * v + g \\ p = p - lr * v\end{split}](img/tex-af7d83f04e92a8b320227914575d095e.gif)

其中 p, g, v 和 ![\rho](img/tex-d2606be4e0cd2c9a6179c8f2e3547a85.gif) 分别是参数、梯度、速度和动量.

这跟 Sutskever 等人以及其他框架的实现是相反的, 它们采用这样的更新.

![\begin{split}v = \rho * v + lr * g \\ p = p - v\end{split}](img/tex-b443a240c72b14e728ae05fe8f11db67.gif)

Nesterov 的版本也相应的被修改了.

```py
step(closure=None)
```

进行单步优化.

参数：`closure (callable, 可选)` – 一个重新评价模型并返回 loss 的闭包, 对于大多数参数来说是可选的.


## 如何调整学习率

| mod: | `torch.optim.lr_scheduler` 基于循环的次数提供了一些方法来调节学习率. |
| --- | --- |
| class: | `torch.optim.lr_scheduler.ReduceLROnPlateau` 基于验证测量结果来设置不同的学习率. |
| --- | --- |

```py
class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

将每个参数组的学习速率设置为给定函数的初始LR. 当 last_epoch=-1, 设置出事的 lr 作为 lr.

Args: * optimizer (Optimizer): 封装好的优化器. * lr_lambda (function or list): 计算给定整数参数历元的乘法因子的函数, 或者一系列的此类函数, 每组的一个都在 optimizer.param_groups 中. * last_epoch (int): 最后一个 epoch 的索引. 默认值: -1.

示例：

```py
>>> # Assuming optimizer has two groups.
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```py
class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

通过 gamma 在每一个 epoch 里面的 step_size 设置每个参数组的初始学习率衰减变量. 当 last_epoch=-1, 设置初始 lr 为 lr.

Args: * optimizer (Optimizer): 封装好的优化器. * step_size (int): 学习率衰减周期. * gamma (float): 学习率衰减的乘法因子. 默认值: 0.1. * last_epoch (int): 最后一个 epoch 的索引. 默认值: -1.

示例：

```py
>>> # Assuming optimizer uses lr = 0.5 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 60
>>> # lr = 0.0005   if 60 <= epoch < 90
>>> # ...
>>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```py
class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

一旦 epoch 的数量达到了一个临界点通过 gamma 在每一个 epoch 里面的 step_size 设置每个参数 组的初始学习率衰减变量.当 last_epoch=-1, 设置初始 lr 作为 lr.

Args: * optimizer (Optimizer): 封装好的优化器. * milestones (list): epoch 索引列表. 必须为递增的. * gamma (float): 学习率衰减的乘法因子.

> 默认值: 0.1.

*   `last_epoch (int)`: 最后一个 epoch 的索引. 默认值: -1.

示例：

```py
>>> # Assuming optimizer uses lr = 0.5 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 80
>>> # lr = 0.0005   if epoch >= 80
>>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)

```

```py
class torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

通过 gamma 在每一个 epoch 里面的 step_size 设置每个参数组的初始学习率衰减变量 . 当 last_epoch=-1, 设置初始 lr 作为 lr. :param optimizer: 封装好的优化器. :type optimizer: Optimizer :param gamma: 学习率衰减的乘法因子. :type gamma: float :param last_epoch: 最后一个 epoch 的索引. 默认值: -1. :type last_epoch: int

```py
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

当一个指标已经停止提升时减少学习率.模型通常受益于通过一次2-10的学习停止因素减少学习率 这个调度程序读取一个指标质量 以及看到 ‘patience’ 的数量在一个 epoch 里面如果没有提升, 这时学习率已经减小.

Args: * optimizer (Optimizer): 封装好的优化器.

> mode (str): `min`, `max` 其中一个. 在 `min` 模块下,当质量监测已经 停止下降时 lr 将被减少; 在 `max` 模块下 当质量监测已经停止上升时 lr 将 被减少. 默认值: ‘min’.

*   `factor (float)`: 哪个学习率将会被减少的影响因子 .
*   new_lr = lr * factor. 默认值: 0.1.
*   `patience (int)`: epoch 中没有改善的次数, 学习率将会降低. . 默认值: 10.
*   `verbose (bool)`: 若为 `True`, 每次更新打印信息到控制台输出. 默认值: `False`.
*   `threshold (float)`: 测量新的最佳阈值, 只关注有重大意义的改变. 默认值: 1e-4.
*   `threshold_mode (str)`: `rel`, `abs` 中的一个. 在 `rel` 模式下, dynamic_threshold = best * ( 1 + threshold ) 在 ‘max’ 模式下或者在 `min` 模式下 best * ( 1 - threshold ) . 在 `abs` 模式下, dynamic_threshold = best + threshold 在 `max` 模式下或者在 `min` 模式下 best - threshold . 默认值: ‘rel’.
*   `cooldown (int)`: lr 已经减少之后去等待最佳的正常操作之前的 epoch 数目. 默认值: 0.
*   `min_lr (float or list)`: 一个列表的标量.所有参数组或每个组的学习率下限. 默认值: 0.
*   `eps (float)`: lr 最小的衰减值适应于. 如果新 lr 和旧 lr 之间的差异小于 eps,更新可以忽略. 默认值: 1e-8.

示例：

```py
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = ReduceLROnPlateau(optimizer, 'min')
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)

```