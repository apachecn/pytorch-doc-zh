# 自动混合精度包 - torch.amp [¶](#automatic-mixed-precision-package-torch-amp "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/amp>
>
> 原始地址：<https://pytorch.org/docs/stable/amp.html>


[`torch.amp`](#module-torch.amp "torch.amp") 提供了混合精度的便捷方法，其中某些操作使用 `torch.float32` (`float`) 数据类型，其他操作使用较低精度的浮点数据类型(`lower_ precision_fp`)：`torch.float16`(`half`)或`torch.bfloat16`。一些操作，例如线性层和卷积，在“lower_ precision_fp”中要快得多。其他操作(例如归约)通常需要 `float32` 的动态范围。混合精度尝试将每个操作与其适当的数据类型相匹配。


 通常，数据类型为 `torch.float16` 的“自动混合精度训练”会同时使用 [`torch.autocast`](#torch.autocast "torch.autocast") 和 [`torch.cuda.amp.GradScaler`](#torch.cuda.amp.GradScaler "torch.cuda.amp.GradScaler")，如 [CUDA 自动混合精度示例](notes/amp_examples.html#amp-examples) 和 [CUDA 自动混合精度配方](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) 中所示。 但是，[`torch.autocast`](#torch.autocast "torch.autocast") 和 [`torch.cuda.amp.GradScaler`] (#torch.cuda.amp.GradScaler "torch.cuda.amp.GradScaler") 是模块化的，如果需要，可以单独使用。 如 [`torch.autocast`](#torch.autocast "torch.autocast")  的CPU示例部分所示，数据类型为 `torch.bfloat16` 的CPU上的 “自动混合精度训练/推理” 仅使用 [`torch.autocast`](#torch.autocast "torch.autocast") 。


 对于CUDA和CPU，也分别提供了API：



* `torch.autocast("cuda", args...)` 等价于 `torch.cuda.amp.autocast(args...)`.
* `torch.autocast("cpu", args...) ` 相当于 `torch.cpu.amp.autocast(args...)` 。对于CPU，目前仅支持较低精度的浮点数据类型“torch.bfloat16”。


[`torch.autocast`](#torch.autocast "torch.autocast") 和 [`torch.cpu.amp.autocast`](#torch.cpu.amp.autocast "torch.cpu.amp.autocast") 是1.10 版本中的新功能。



* [自动投射](#autocasting)
* [渐变缩放](#gradient-scaling)
* [自动投射操作参考](#autocast-op-reference)
    + [Op 资格](#op-eligibility) 
    + [CUDA Op 特定行为](#cuda-op-specific-behavior) 
        - [可以自动转换为 `float16` 的 CUDA Ops](#cuda-ops-that-c​​an-autocast-to-float16) 
        - [可以自动转换为 `float32` 的 CUDA 操作](#cuda-ops-that-c​​an-autocast-to-float32) 
        - [升级为最宽输入类型的 CUDA 操作](#cuda-ops-that-promote-to-the-widest-input-type) 
        - [更喜欢 `binary_cross_entropy_with_logits` 而不是 `binary_cross_entropy`](#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy) 
    + [CPU Op 特定行为](#cpu-op-specific-behavior) 
        - [可以自动转换为 `bfloat16` 的 CPU Ops](#cpu-ops-that-can-autocast-to-bfloat16) 
        - [可以自动转换为 `float32` 的 CPU Ops](#cpu-ops-that-c​​an-autocast-to-float32) 
        - [提升为最宽输入类型的 CPU Ops](#cpu-ops-that-promote-to-the-widest-input-type)


## [Autocasting](#id4) [¶](#autocasting "此标题的永久链接")


> *CLASS* `torch.autocast(device_type, dtype=None, enabled=True, cache_enabled=None)` [[source]](_modules/torch/amp/autocast_mode.html#autocast)[¶](#torch.autocast "此定义的永久链接")


 [`autocast`](#torch.autocast "torch.autocast") 的实例充当上下文管理器或装饰器，允许脚本区域以混合精度运行。


 在这些区域中，操作以 autocast 选择的操作特定数据类型运行，以提高性能，同时保持准确性。有关详细信息，请参阅 [Autocast Op 参考](#autocast-op-reference)。


 当进入启用自动转换的区域时，张量可以是任何类型。使用自动转换时，不应在模型或输入上调用“half()”或“bfloat16()”。


[`autocast`](#torch.autocast "torch.autocast") 应仅包装网络的前向传递，包括损失计算。不建议在自动转换下向后传递。向后操作的运行类型与自动转换用于相应前向操作的类型相同。


 CUDA 设备示例：


```
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    with torch.autocast(device_type="cuda"):
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    optimizer.step()

```


 请参阅 [CUDA 自动混合精度示例](notes/amp_examples.html#amp-examples)，了解在更复杂的场景(例如，梯度惩罚、多个模型/损失、自定义自动梯度函数)中的使用情况(以及梯度缩放)。


[`autocast`](#torch.autocast "torch.autocast") 也可以用作装饰器，例如，在模型的 `forward` 方法上：


```
class AutocastModel(nn.Module):
    ...
    @torch.autocast(device_type="cuda")
    def forward(self, input):
        ...

```


 在启用自动转换的区域中生成的浮点张量可能是“float16”。返回到禁用自动转换的区域后，将它们与不同数据类型的浮点张量一起使用可能会导致类型不匹配错误。如果是这样，则将自动转换区域中生成的张量转换回“float32”(或其他需要的数据类型)。如果自动转换区域中的张量已经是“float32”，则转换是无操作，并且会产生没有额外的开销。CUDA示例：


```
# Creates some tensors in default dtype (here assumed to be float32)
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")
c_float32 = torch.rand((8, 8), device="cuda")
d_float32 = torch.rand((8, 8), device="cuda")

with torch.autocast(device_type="cuda"):
    # torch.mm is on autocast's list of ops that should run in float16.
    # Inputs are float32, but the op runs in float16 and produces float16 output.
    # No manual casts are required.
    e_float16 = torch.mm(a_float32, b_float32)
    # Also handles mixed input types
    f_float16 = torch.mm(d_float32, e_float16)

# After exiting autocast, calls f_float16.float() to use with d_float32
g_float32 = torch.mm(d_float32, f_float16.float())

```


 CPU 训练示例：


```
# Creates model and optimizer in default precision
model = Net()
optimizer = optim.SGD(model.parameters(), ...)

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(input)
            loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

```


 CPU 推理示例：


```
# Creates model in default precision
model = Net().eval()

with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    for input in data:
        # Runs the forward pass with autocasting.
        output = model(input)

```


 使用 Jit Trace 的 CPU 推理示例：


```
class TestModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc1(x)

input_size = 2
num_classes = 2
model = TestModel(input_size, num_classes).eval()

# For now, we suggest to disable the Jit Autocast Pass,
# As the issue: https://github.com/pytorch/pytorch/issues/75956
torch._C._jit_set_autocast_mode(False)

with torch.cpu.amp.autocast(cache_enabled=False):
    model = torch.jit.trace(model, torch.randn(1, input_size))
model = torch.jit.freeze(model)
# Models Run
for _ in range(3):
    model(torch.randn(1, input_size))

```


 *在*启用自动施放的区域中，类型不匹配错误是一个错误；如果这是您观察到的情况，请提出问题。


`autocast(enabled=False)` 子区域可以嵌套在启用了 autocast 的区域中。本地禁用 autocast 可能很有用，例如，如果您想强制子区域在特定的 `dtype` 中运行。禁用自动转换可以让您显式控制执行类型。在子区域中，来自周围区域的输入在使用前应转换为“dtype”：


```
# Creates some tensors in default dtype (here assumed to be float32)
a_float32 = torch.rand((8, 8), device="cuda")
b_float32 = torch.rand((8, 8), device="cuda")
c_float32 = torch.rand((8, 8), device="cuda")
d_float32 = torch.rand((8, 8), device="cuda")

with torch.autocast(device_type="cuda"):
    e_float16 = torch.mm(a_float32, b_float32)
    with torch.autocast(device_type="cuda", enabled=False):
        # Calls e_float16.float() to ensure float32 execution
        # (necessary because e_float16 was created in an autocasted region)
        f_float32 = torch.mm(c_float32, e_float16.float())

    # No manual casts are required when re-entering the autocast-enabled region.
    # torch.mm again runs in float16 and produces float16 output, regardless of input types.
    g_float16 = torch.mm(d_float32, f_float32)

```


 自动转换状态是线程本地的。如果您希望在新线程中启用它，则必须在该线程中调用上下文管理器或装饰器。这会影响 [`torch.nn.DataParallel`](generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 和 [`torch.nn.parallel.DistributedDataParallel`](generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 当每个进程与多个 GPU 一起使用时(请参阅 [使用多个 GPU](notes/amp_examples.html#amp-multigpu) )。


 Parameters

* **device_type** ( [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.12)")*, *required* ) – 要使用的设备类型。 可能的值为：“cuda”、“cpu”、“xpu”和“hpu”。 该类型与 [`torch.device`](tensor_attributes.html#torch.device "torch.device") 的 type 属性相同。 因此，您可以使用 Tensor.device.type 获取张量的设备类型。
* **enabled** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.12)"), *optional* ) – 是否应在该区域启用自动广播。 默认值：`True`
* **dtype** ( *torch_dtype*, *optional* ) – 是否使用 torch.float16 或 torch.bfloat16。
* **cache_enabled** ( [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.12)"), *optional* ) – 是否应启用自动投射内的权重缓存。 默认值：`True`


> *CLASS* `torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True)` [[source]](_modules/torch/cuda/amp/autocast_mode.html#autocast)[¶](#torch.cuda.amp.autocast "此定义的永久链接")


 请参阅 [`torch.autocast`](#torch.autocast "torch.autocast") , `torch.cuda.amp.autocast(args...)` 相当于 `torch.autocast("cuda", args...)`

>> `torch.cuda.amp.custom_fwd(fwd=None, *, cast_inputs=None)` [[source]](_modules/torch/cuda/amp/autocast_mode.html#custom_fwd)[¶](#torch.cuda.amp.custom_fwd "此定义的永久链接")


 自定义 autograd 函数的 `forward` 方法的辅助装饰器( [`torch.autograd.Function`](autograd.html#torch.autograd.Function "torch.autograd.Function") 的子类)。有关更多详细信息，请参阅[示例页面](notes/amp_examples.html#amp-custom-examples)。


 Parameters


* **cast_inputs** ( [`torch.dtype`](tensor_attributes.html#torch.dtype "torch.dtype") 或 None，optional，default=None) – 如果不是 `None` ，则当 `forward` 运行时在启用自动转换的区域中，将传入的浮点 CUDA 张量转换为目标 dtype(非浮点张量不受影响)，然后在禁用自动转换的情况下执行 `forward`。如果 `None` ，则 `forward` 的内部操作以当前自动施放状态执行。


!!! note "笔记"

    如果在启用自动转换的区域之外调用修饰后的 `forward`，则 [`custom_fwd`](#torch.cuda.amp.custom_fwd "torch.cuda.amp.custom_fwd") 是无操作，并且 `cast_inputs` 没有效果。


>> `torch.cuda.amp.custom_bwd(bwd)` [[source]](_modules/torch/cuda/amp/autocast_mode.html#custom_bwd)[¶](#torch.cuda.amp.custom_bwd "此定义的永久链接")


 自定义 autograd 函数的向后方法的辅助装饰器([`torch.autograd.Function`](autograd.html#torch.autograd.Function "torch.autograd.Function") 的子类)。确保 `backward` 以相同的方式执行自动转换状态为 `forward`。有关更多详细信息，请参阅[示例页面](notes/amp_examples.html#amp-custom-examples)。


> *CLASS* `torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True)` [[source]](_modules/torch/cpu/amp/autocast_mode.html#autocast)[¶](#torch.cpu.amp.autocast "此定义的永久链接")


 请参阅 [`torch.autocast`](#torch.autocast "torch.autocast") 。 `torch.cpu.amp.autocast(args...)` 相当于 `torch.autocast("cpu", args...)`


## [渐变缩放](#id5) [¶](#gradient-scaling "此标题的永久链接")


 如果特定操作的前向传递具有“float16”输入，则该操作的后向传递将产生“float16”梯度。小幅度的梯度值可能无法在“float16”中表示。这些值将刷新为零(“下溢” )，因此相应参数的更新将会丢失。


 为了防止下溢，“梯度缩放”将网络的损失乘以比例因子，并对缩放后的损失调用向后传递。然后通过网络向后流动的梯度按相同的因子缩放。换句话说，梯度值具有较大的幅度，因此它们不会刷新为零。


 每个参数的梯度(`.grad` 属性)应该在优化器更新参数之前取消缩放，因此缩放因子不会干扰学习率。




!!! note "笔记"

    AMP/fp16 可能不适用于所有型号！例如，大多数 bf16 预训练模型无法在最大 65504 的 fp16 数值范围内运行，并且会导致梯度上溢而不是下溢。在这种情况下，比例因子可能会减小到 1 以下，以尝试将梯度带到可在 fp16 动态范围内表示的数字。虽然人们可能期望比例始终高于 1，但我们的 GradScaler 并不能保证保持性能。如果在使用 AMP/fp16 运行时在损失器梯度中遇到 NaN，请验证您的模型是否兼容。


> *CLASS* `torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)` [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler)[¶](#torch.cuda.amp.GradScaler "此定义的永久链接")


>> get_backoff_factor() [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.get_backoff_factor)[¶](#torch.cuda.amp.GradScaler.get_backoff_factor "此定义的永久链接")


 返回包含比例退避因子的 Python 浮点数。


>> get_growth_factor() [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.get_growth_factor)[¶](#torch.cuda.amp.GradScaler.get_growth_factor "此定义的永久链接")


 返回包含比例增长因子的 Python 浮点数。


>> get_growth_interval() [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.get_growth_interval)[¶](#torch.cuda.amp.GradScaler.get_growth_interval "此定义的永久链接")


 返回包含增长区间的 Python int。


>> get_scale() [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.get_scale)[¶](#torch.cuda.amp.GradScaler.get_scale "此定义的永久链接")


 返回包含当前比例的 Python 浮点数，如果禁用缩放，则返回 1.0。


!!! warning "警告"

    [`get_scale()`](#torch.cuda.amp.GradScaler.get_scale "torch.cuda.amp.GradScaler.get_scale") 会导致 CPU-GPU 同步。


>> is_enabled() [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.is_enabled)[¶](#torch.cuda.amp.GradScaler.is_enabled "此定义的永久链接")


 返回一个布尔值，指示该实例是否启用。


>> load_state_dict( *state_dict* ) [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.load_state_dict)[¶](#torch.cuda.amp.GradScaler.load_state_dict "此定义的永久链接")


 加载缩放器状态。如果禁用此实例，则 [`load_state_dict()`](#torch.cuda.amp.GradScaler.load_state_dict "torch.cuda.amp.GradScaler.load_state_dict") 是无操作。


 Parameters


* **state_dict** ( [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.12)") ) – 缩放器状态。应该是从调用 [`state_dict()`](#torch.cuda.amp.GradScaler.state_dict "torch.cuda.amp.GradScaler.state_dict") 返回的对象。




>> scale( *outputs* ) [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.scale)[¶](#torch.cuda.amp.GradScaler.scale "此定义的永久链接")


 将张量或张量列表乘以(“缩放”)缩放因子。


 返回缩放后的输出。如果未启用 [`GradScaler`](#torch.cuda.amp.GradScaler "torch.cuda.amp.GradScaler") 的此实例，则返回未修改的输出。


 Parameters

* *outputs** ( [*Tensor*](tensors.html#torch.Tensor "torch.Tensor") 或 Tensors的迭代) —— 按比例输出。


>> set_backoff_factor( *new_factor* ) [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.set_backoff_factor)[¶](#torch.cuda.amp.GradScaler.set_backoff_factor "此定义的永久链接")


 Parameters


* **new_scale** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.12)") ) – 用作新的缩放退避因子。


>> set_growth_factor( *new_factor* ) [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.set_growth_factor)[¶](#torch.cuda.amp.GradScaler.set_growth_factor "此定义的永久链接")


 Parameters


* **new_scale** ( [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.12)") ) – 用作新的规模增长因素。


>> set_growth_interval( *new_interval* ) [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.set_growth_interval)[¶](#torch.cuda.amp.GradScaler.set_growth_interval "此定义的永久链接")


 Parameters


* **new_interval** ( [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.12)") ) – 用作新的增长区间。


>> state_dict() [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.state_dict)[¶](#torch.cuda.amp.GradScaler.state_dict "此定义的永久链接")


 以 [`dict`](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.12)") 形式返回缩放器的状态。它包含五个条目：



* `"scale"` - 包含当前比例的 Python float
* `"growth_factor"` - 包含当前增长因子的 Python float
* `"backoff_factor"` - 包含当前退避因子的 Python float
* ` "growth_interval"` - 包含当前增长间隔的 Python int
* `"_growth_tracker"` - 包含最近连续未跳过步骤数的 Python int。


 如果未启用此实例，则返回一个空字典。




!!! note "笔记"

    如果您希望在特定迭代后检查缩放器的状态，应在 [`state_dict()`](#torch.cuda.amp.GradScaler.state_dict "torch.cuda.amp.GradScaler.state_dict") 之后调用 [`update()`](#torch.cuda.amp.GradScaler.update "torch.cuda.amp.GradScaler.update") 。



>> `step(optimizer, *args, **kwargs)` [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.step)[¶](#torch.cuda.amp.GradScaler.step "此定义的永久链接")


[`step()`](#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 执行以下两个操作：


1. 内部调用 `unscale_(optimizer)` (除非明确调用 [`unscale_()`](#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_")迭代早期的“优化器”)。作为 [`unscale_()`](#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 的一部分，检查梯度是否有 infs/NaNs.2。如果未找到 inf/NaN 梯度，则使用未缩放的梯度调用“optimizer.step()”。否则，将跳过“optimizer.step()”以避免损坏Parameters。


`*args` 和 `**kwargs` 被转发到 `optimizer.step()` 。


 返回 `optimizer.step(*args, **kwargs)` 的返回值。


 Parameters

* **optimizer** ( [*torch.optim.Optimizer*](optim.html#torch.optim.Optimizer "torch.optim.Optimizer") ) – 应用梯度的优化器。
* **args** – 任何参数。
* **kwargs** – 任何关键字参数。


!!! warning "警告"

     目前不支持闭包使用。


>> unscale_ ( *optimizer* ) [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.unscale_)[¶](#torch.cuda.amp.GradScaler.unscale_ "此定义的永久链接")


 将优化器的梯度张量除以比例因子(“取消缩放”)。


unscale_() 是可选的，适用于需要修改或检查向后传递和 step() 之间的梯度的情况。 如果未显式调用 unscale_()，则梯度将在 step() 期间自动取消缩放。

[`unscale_()`](#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 是可选的，适用于需要 [修改或检查](notes/amp_examples.html#working-with-unscaled-gradients) 向后传递 和 [`step()`](#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 之间如果未显式调用 [`unscale_()`](#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_")，则在 [`step() 期间将自动取消缩放渐变`](#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 。


 简单的示例，使用 [`unscale_()`](#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 启用未缩放渐变的裁剪：


```
...
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()

```


 Parameters


* **optimizer** ( [*torch.optim.Optimizer*](optim.html#torch.optim.Optimizer "torch.optim.Optimizer") ) – 拥有要取消缩放的渐变的优化器。



!!! note "笔记"

    [`unscale_()`](#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 不会引起 CPU-GPU 同步。


!!! warning "警告"

    [`unscale_()`](#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 每个优化器每个 [`step()`](#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 调用，并且仅在累积了该优化器指定参数的所有梯度之后。调用 [`unscale_()`](#torch. cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 对于给定优化器，在每个 [`step()`](#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 触发运行时错误。


!!! warning "警告"

    [`unscale_()`](#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 可能会将稀疏梯度取消缩放，替换 `.grad` 属性。


>> `update(new_scale=None)` [[source]](_modules/torch/cuda/amp/grad_scaler.html#GradScaler.update)[¶](#torch.cuda.amp.GradScaler.update "此定义的永久链接")


 更新比例因子。


 如果跳过任何优化器步骤，则比例将乘以“backoff_factor”以减少它。如果连续发生“growth_interval”未跳过的迭代，则将比例乘以“growth_factor”来增加它。


 传递 `new_scale` 手动设置新的比例值。 ( `new_scale` 不直接使用，它用于填充 GradScaler 的内部尺度张量。因此，如果 `new_scale` 是一个张量，以后对该张量的就地更改不会进一步影响 GradScaler 内部使用的尺度。)


 Parameters


* **new_scale** (float 或 `torch.cuda.FloatTensor` ，optional，default=None) – 新比例因子。


!!! warning "警告"

    [`update()`](#torch.cuda.amp.GradScaler.update "torch.cuda.amp.GradScaler.update") 只能在迭代结束时、在 `scaler.step(optimizer)` 之后调用已为本次迭代使用的所有优化器调用。


!!! warning "警告"

     出于性能原因，我们不检查比例因子值以避免同步，因此比例因子不保证高于 1。如果比例低于 1 和/或您在梯度或损失中看到 NaN，则可能存在问题。例如，由于动态范围不同，bf16 预训练模型通常与 AMP/fp16 不兼容。


## [Autocast Op 参考](#id6) [¶](#autocast-op-reference "此标题的永久链接")


### [Op 资格](#id7) [¶](#op-eligibility "永久链接到此标题")


 在“float64”或非浮点数据类型中运行的操作不符合条件，并且无论是否启用自动转换都将以这些类型运行。


 只有异地操作和张量方法才符合资格。在启用自动转换的区域中允许显式提供“out=...”张量的就地变体和调用，但不会经过自动转换。例如，启用自动投射的区域 `a.addmm(b, c)` 可以自动投射，但 `a.addmm_(b, c)` 和 `a.addmm(b, c, out=d)` 不能自动投射。最好性能和稳定性，更喜欢在启用自动施放的区域中进行异地操作。


 使用显式“dtype=...”参数调用的操作不符合条件，并且将产生尊重“dtype”参数的输出。


### [CUDA Op 特定行为](#id8) [¶](#cuda-op-specific-behavior "永久链接到此标题")


 以下列表描述了启用自动转换的区域中合格操作的行为。这些操作始终会经过自动转换，无论它们是作为 [`torch.nn.Module`]( generated/torch.nn.Module.html#torch 的一部分被调用).nn.Module "torch.nn.Module") ，作为函数，或作为 [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 方法。如果函数在多个命名空间中公开，则无论命名空间如何，它们都会经历自动转换。


 下面未列出的操作不会经过自动施放。它们按照输入定义的类型运行。但是，如果未列出的操作位于自动转换操作的下游，自动转换仍可能会更改其运行的类型。


 如果一个操作未列出，我们假设它在“float16”中数值稳定。如果您认为未列出的操作在“float16”中数值不稳定，请提出问题。


#### [可以自动转换为 `float16` 的 CUDA Ops](#id9)[¶](#cuda-ops-that-c​​an-autocast-to-float16 "此标题的永久链接")


`__matmul__` 、 `addbmm` 、 `addmm` 、 `addmv` 、 `addr` 、 `baddbmm` 、 `bmm` 、 `chain_matmul` 、 `multi_dot` 、 `conv1d` , `conv2d` , `conv3d` , `conv_transpose1d` , `conv_transpose2d` , `conv_transpose3d` , `GRUCell` , `linear` , `LSTMCell` , `matmul` , `mm` , `mv` 、`prelu`、`RNNCell`


#### [可以自动转换为 `float32` 的 CUDA Ops](#id10)[¶](#cuda-ops-that-c​​an-autocast-to-float32 "此标题的永久链接")


`__pow__` 、 `__rdiv__` 、 `__rpow__` 、 `__rtruediv__` 、 `acos` 、 `asin `、`binary_cross_entropy_with_logits`、`cosh`、`cosine_embedding_loss`、`cdist`、`cosine_similarity`、`cross_entropy`、`cumprod`、`cumsum`、 `dist` 、 `erfinv` 、 `exp` 、 `expm1` 、 `group_norm` 、 `hinge_embedding_loss` 、 `kl_div` 、 `l1_loss` 、 `layer_norm` 、 `log `、`log_softmax`、`log10`、`log1p`、`log2`、`margin_ranking_loss`、`mse_loss`、`multilabel_margin_loss`、`multi_margin_loss`、 `nll_loss` 、 `norm` 、 `normalize` 、 `pdist` 、 `poisson_nll_loss` 、 `pow` 、 `prod` 、 `reciprocal` 、 `rsqrt` 、 `sinh` 、 `smooth_l1 _loss` 、 `soft_margin_loss` 、 `softmax` 、 `softmin` 、 `softplus` 、 `sum` 、 `renorm` 、 `tan` 、 `triplet_margin_loss`


#### [提升到最宽输入类型的 CUDA Ops](#id11) [¶](#cuda-ops-that-promote-to-the-widest-input-type "永久链接到此标题")


 这些操作不需要特定的数据类型来保证稳定性，但需要多个输入并要求输入的数据类型匹配。如果所有输入都是 `float16` ，则操作在 `float16` 中运行。如果任何输入是 `float32` ，则 autocast 将所有输入转换为 `float32` 并运行 `float32` 中的操作。


`addcdiv`、`addcmul`、`atan2`、`bilinear`、`cross`、`dot`、`grid_sample`、`index_put`、`scatter_add`、`tensordot`


 此处未列出的某些操作(例如，像“add”这样的二进制操作)本身会提升输入，而无需自动转换的干预。如果输入是 `float16` 和 `float32` 的混合，这些操作在 `float32` 中运行并产生 `float32` 输出，无论是否启用了自动转换。


#### [首选 `binary_cross_entropy_with_logits` 而不是 `binary_cross_entropy`](#id12)[¶](#prefer-binary-cross-entropy-with-logits-over-binary -交叉熵"此标题的永久链接")


 [`torch.nn.function.binary_cross_entropy()`]( generated/torch.nn.function.binary_cross_entropy.html#torch.nn.function.binary_cross_entropy "torch.nn.function.binary_cross_entropy" 的向后传递)(和 [`torch.nn.BCELoss`]( generated/torch.nn.BCELoss.html#torch.nn.BCELoss "torch.nn.BCELoss") ，它包装它)可以产生无法在`float16` 。在启用自动转换的区域中，前向输入可能是“float16”，这意味着后向梯度必须可以用“float16”表示(将“float16”前向输入自动转换为“float32”没有帮助，因为该转换必须在向后反转)。因此，“binary_cross_entropy”和“BCELoss”在启用自动转换的区域中会引发错误。


 许多模型在二元交叉熵层之前使用 sigmoid 层。 在这种情况下，使用 [`torch.nn.function.binary_cross_entropy_with_logits()`](generated/torch.nn.functional.binary_cross_entropy_with_logits.html#torch.nn.functional.binary_cross_entropy_with_logits "torch.nn.function.binary_cross_entropy_with_logits") 或 [`torch.nn.BCEWithLogitsLoss`](生成/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss "torch.nn.BCEWithLogitsLoss") 组合两个层。 `binary_cross_entropy_with_logits` 和 `BCEWithLogits` 可以安全地自动转换。


### [CPU Op 特定行为](#id13) [¶](#cpu-op-specific-behavior "永久链接到此标题")


 以下列表描述了启用自动转换的区域中合格操作的行为。这些操作始终会经过自动转换，无论它们是作为 [`torch.nn.Module`]( generated/torch.nn.Module.html#torch 的一部分被调用).nn.Module "torch.nn.Module") ，作为函数，或作为 [`torch.Tensor`](tensors.html#torch.Tensor "torch.Tensor") 方法。如果函数在多个命名空间中公开，则无论命名空间如何，它们都会经历自动转换。


 下面未列出的操作不会经过自动施放。它们按照输入定义的类型运行。但是，如果未列出的操作位于自动转换操作的下游，自动转换仍可能会更改其运行的类型。


 如果一个操作未列出，我们假设它在“bfloat16”中数值稳定。如果您认为未列出的操作在“bfloat16”中数值不稳定，请提出问题。


#### [可以自动转换为 `bfloat16` 的 CPU Ops](#id14)[¶](#cpu-ops-that-c​​an-autocast-to-bfloat16 "此标题的永久链接")


`conv1d`、`conv2d`、`conv3d`、`bmm`、`mm`、`baddbmm`、`addmm`、`addbmm`、`linear`、`matmul`、`_convolution`


#### [可以自动转换为 `float32` 的 CPU Ops](#id15)[¶](#cpu-ops-that-c​​an-autocast-to-float32 "永久链接到此标题")


`conv_transpose1d` 、 `conv_transpose2d` 、 `conv_transpose3d` 、 `avg_pool3d` 、 `binary_cross_entropy` 、 `grid_sampler` 、 `grid_sampler_2d` 、 `_grid_sampler_2d_cpu_fallback`、`grid_sampler_3d`、`polar`、`prod`、`quantile`、`nanquantile`、`stft`、`cdist`、`trace`、`view_as_complex`、`cholesky`、`cholesky_inverse`、`cholesky_solve`、`inverse`、`lu_solve`、`orgqr`、`inverse`、`ormqr`、`pinverse`、`max_pool3d` 、`max_unpool2d`、`max_unpool3d`、`adaptive_avg_pool3d`、`reflection_pad1d`、`reflection_pad2d`、`replication_pad1d`、`replication_pad2d`、`replication_pad3d` 、`mse_loss`、`ctc_loss`、`kl_div`、`multilabel_margin_loss`、`fft_fft`、`fft_ifft`、`fft_fft2`、`fft_ifft2` 、 `fft_fftn` 、 `fft_ifftn` 、 `fft_rfft` 、 `fft_irfft` 、 `fft_rfft2` 、 `fft_irfft2` 、 `fft_rfftn` 、 `fft_irfftn` 、 ` fft_hfft`、`fft_ihfft`、`linalg_matrix_norm`、`linalg_cond`、`linalg_matrix_rank`、`linalg_solve`、`linalg_cholesky`、`linalg_svdvals` 、 `linalg_eigvals` 、 `linalg_eigvalsh` 、 `linalg_inv` 、 `linalg_householder_product` 、 `linalg_tensorinv` 、 `linalg_tensorsolve` 、 `fake_quantize_per_tensor_affine` 、 `eig` 、 `geqrf` 、 `lstsq` 、 `_lu_with_info` 、 `qr` 、 `solve` 、 `svd` 、 `symeig` 、 `triangular_solve` 、 `fractional_max_pool2d `、`fractional_max_pool3d`、`adaptive_max_pool3d`、`multilabel_margin_loss_forward`、`linalg_qr`、`linalg_cholesky_ex`、`linalg_svd`、`linalg_svd_eig` 、 `linalg_eigh` 、 `linalg_lstsq` 、 `linalg_inv_ex`


#### [提升到最宽输入类型的 CPU Ops](#id16) [¶](#cpu-ops-that-promote-to-the-widest-input-type "永久链接到此标题")


 这些操作不需要特定的数据类型来保证稳定性，但需要多个输入并要求输入的数据类型匹配。如果所有输入都是“bfloat16”，则操作在“bfloat16”中运行。如果任何输入是 `float32` ，则 autocast 将所有输入转换为 `float32` 并运行 `float32` 中的操作。


`cat` 、 `stack` 、 `index_copy`


 此处未列出的某些操作(例如，像“add”这样的二进制操作)本身会提升输入，而无需自动转换的干预。如果输入是“bfloat16”和“float32”的混合，则这些操作在“float32”中运行并产生“float32”输出，无论是否启用了自动转换。