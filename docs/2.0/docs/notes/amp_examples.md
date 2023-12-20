# CUDA 自动混合精度示例 [¶](#cuda-automatic-mixed-precision-examples "固定链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/amp_examples>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/amp_examples.html>


 通常，“自动混合精度训练”意味着使用 [`torch.autocast`](../amp.html#torch.autocast "torch.autocast") 和 [`torch.cuda.amp.GradScaler`](.. /amp.html#torch.cuda.amp.GradScaler“torch.cuda.amp.GradScaler”)在一起。


 [`torch.autocast`](../amp.html#torch.autocast "torch.autocast") 的实例可以对选定区域启用自动投射。自动投射会自动选择 GPU 操作的精度，以提高性能，同时保持准确性。


 [`torch.cuda.amp.GradScaler`](../amp.html#torch.cuda.amp.GradScaler "torch.cuda.amp.GradScaler") 的实例有助于方便地执行梯度缩放步骤。梯度缩放通过最大限度地减少梯度下溢来提高具有“float16”梯度的网络的收敛性，如[此处](../amp.html#gradient-scaling)所述。


[`torch.autocast`](../amp.html#torch.autocast "torch.autocast") 和 [`torch.cuda.amp.GradScaler`](../amp.html#torch.cuda.amp. GradScaler“torch.cuda.amp.GradScaler”)是模块化的。在下面的示例中，每个示例都按照其单独的文档建议使用。


 (此处的示例仅供参考。有关可运行的演练，请参阅[自动混合精度配方](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)。)



* [典型混合精度训练](#typical-mixed-precision-training)
* [使用未缩放梯度](#working-with-unscaled-gradients)
    + [渐变剪裁](#gradient-clipping)
* [使用缩放渐变](#working-with-scaled-gradients)
    + [梯度累积](#gradient-accumulation) 
    + [梯度惩罚](#gradient-penalty)
* [使用多个模型、损失和优化器](#working-with-multiple-models-losses-and-optimizers) 
* [使用多个 GPU](#working-with-multiple-gpus)
    + [单进程中的 DataParallel](#dataparallel-in-a-single-process) 
    + [DistributedDataParallel，每个进程一个 GPU](#distributeddataparallel-one-gpu-per-process) 
    + [DistributedDataParallel，每个进程多个 GPU](#distributeddataparallel-multiple-gpus-per-process)
* [Autocast 和自定义 Autograd 函数](#autocast-and-custom-autograd-functions)
    + [具有多个输入或可自动转换操作的函数](#functions-with-multiple-inputs-or-autocastable-ops) 
    + [需要特定`dtype`的函数](#functions-that-need-a-prefer-d-dtype )


## [典型混合精度训练](#id2) [¶](#typical-mixed-precision-training "此标题的永久链接")


```
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

```


## [使用未缩放的渐变](#id3) [¶](#working-with-unscaled-gradients "此标题的永久链接")


 由 `scaler.scale(loss).backward()` 产生的所有梯度都会被缩放。如果您希望修改或检查 `backward()` 和 `scaler.step(optimizer)` 之间参数的 `.grad` 属性，您应该首先取消缩放它们。例如，梯度裁剪操作一组梯度，使其全局范数(参见 [`torch.nn.utils.clip_grad_norm_()`](../generated/torch.nn.utils.clip_grad_norm_. html#torch.nn.utils.clip_grad_norm_ "torch.nn.utils.clip_grad_norm_") )或最大幅度(参见 [`torch.nn.utils.clip_grad_value_()`](../generated/torch.nn.utils.clip_grad_value_.html#torch.nn.utils.clip_grad_value_ "torch.nn.utils.clip_grad_value_") ) 是 <= 一些用户施加的阈值。如果您尝试在*不*取消缩放的情况下进行剪辑，则渐变的范数/最大幅度也会缩放，因此您请求的阈值(这意味着*未缩放*渐变的阈值)将无效。


`scaler.unscale_(optimizer)` 取消由 `optimizer` 分配的参数保存的梯度。如果您的模型或多个模型包含分配给另一个优化器的其他参数(例如 `optimizer2` )，您可以调用 `scaler.unscale _(optimizer2)` 也单独取消缩放这些参数的梯度。


### [渐变剪辑](#id4) [¶](#gradient-clipping "此标题的永久链接")


 在裁剪之前调用 `scaler.unscale_(optimizer)` 可以让你像往常一样裁剪未缩放的渐变：


```
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

```


`scaler` 记录了此优化器在本次迭代中已经调用了 `scaler.unscale_(optimizer)`，因此 `scaler.step(optimizer)` 知道在(内部)调用 `optimizer.step()` 之前不要冗余地取消缩放梯度。


!!! warning "警告"

    [`unscale_`](../amp.html#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 只能在每个优化器的每个 [`step`](../amp.html#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 调用中调用一次，并且仅在该优化器分配的参数的所有梯度都已累积之后。在每个 [`step`](../amp.html #torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 之间为给定优化器调用 [`unscale_`](../amp.html#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") 两次会触发 RuntimeErro


## [使用缩放渐变](#id5) [¶](#working-with-scaled-gradients "此标题的永久链接")


### [梯度累积](#id6) [¶](#gradient-accumulation "永久链接到此标题")


 梯度累积在大小为“batch_per_iter * iters_to_accumulate”的有效批次上添加梯度(“* num_procs”，如果是分布式的)。应针对有效批次校准刻度，这意味着 inf/NaN 检查，如果找到 inf/NaN 梯度则跳过步骤，并且刻度更新应以有效批次粒度进行。此外，梯度应保持缩放，比例因子应保持不变，同时累积给定有效批次的梯度。如果在累积完成之前未缩放梯度(或比例因子发生变化)，则下一个向后传递会将缩放的梯度添加到未缩放的梯度(或按不同因子缩放的梯度)，之后无法恢复累积的未缩放的梯度[`step`](../amp.html#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 必须适用。


 因此，如果您想要 [`unscale_`](../amp.html#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_") grads(例如，允许裁剪未缩放的grads)，在 [`step`](../amp.html#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") ，毕竟即将到来的 [`step`](../amp.html#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step")已累积。另外，仅在迭代结束时调用 [`update`](../amp.html#torch.cuda.amp.GradScaler.update "torch.cuda.amp.GradScaler.update")，其中您调用了 [`step`](../amp.html#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 以获得完整的有效批次：


```
scaler = GradScaler()

for epoch in epochs:
    for i, (input, target) in enumerate(data):
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)
            loss = loss / iters_to_accumulate

        # Accumulates scaled gradients.
        scaler.scale(loss).backward()

        if (i + 1) % iters_to_accumulate == 0:
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

```


### [梯度惩罚](#id7) [¶](#gradient-penalty "永久链接到此标题")


 梯度惩罚实现通常使用 [`torch.autograd.grad()`](../generated/torch.autograd.grad.html#torch.autograd.grad "torch.autograd.grad") 创建梯度，并将它们组合起来创建惩罚值，并将惩罚值添加到损失中。


 这是一个没有梯度缩放或自动转换的 L2 惩罚的普通示例：


```
for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)

        # Creates gradients
        grad_params = torch.autograd.grad(outputs=loss,
                                          inputs=model.parameters(),
                                          create_graph=True)

        # Computes the penalty term and adds it to the loss
        grad_norm = 0
        for grad in grad_params:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        loss = loss + grad_norm

        loss.backward()

        # clip gradients here, if desired

        optimizer.step()

```


 为了通过梯度缩放实现梯度惩罚，“输出”张量传递给 [`torch.autograd.grad()`](../generated/torch.autograd.grad.html#torch.autograd. grad“torch.autograd.grad”)应该缩放。因此，生成的梯度将被缩放，并且在组合以创建惩罚值之前应该取消缩放。


 此外，惩罚项计算是前向传递的一部分，因此应该位于 [`autocast`](../amp.html#torch.cuda.amp.autocast "torch.cuda.amp.autocast") 上下文中。


 以下是相同 L2 惩罚的情况：


```
scaler = GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output = model(input)
            loss = loss_fn(output, target)

        # Scales the loss for autograd.grad's backward pass, producing scaled_grad_params
        scaled_grad_params = torch.autograd.grad(outputs=scaler.scale(loss),
                                                 inputs=model.parameters(),
                                                 create_graph=True)

        # Creates unscaled grad_params before computing the penalty. scaled_grad_params are
        # not owned by any optimizer, so ordinary division is used instead of scaler.unscale_:
        inv_scale = 1./scaler.get_scale()
        grad_params = [p * inv_scale for p in scaled_grad_params]

        # Computes the penalty term and adds it to the loss
        with autocast(device_type='cuda', dtype=torch.float16):
            grad_norm = 0
            for grad in grad_params:
                grad_norm += grad.pow(2).sum()
            grad_norm = grad_norm.sqrt()
            loss = loss + grad_norm

        # Applies scaling to the backward call as usual.
        # Accumulates leaf gradients that are correctly scaled.
        scaler.scale(loss).backward()

        # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

        # step() and update() proceed as usual.
        scaler.step(optimizer)
        scaler.update()

```


## [使用多个模型、损失和优化器](#id8) [¶](#working-with-multiple-models-losses-and-optimizers "永久链接到此标题")


 如果您的网络有多个损失，则必须对每个损失调用 [`scaler.scale`](../amp.html#torch.cuda.amp.GradScaler.scale "torch.cuda.amp.GradScaler.scale")如果您的网络有多个优化器，您可以调用 [`scaler.unscale_`](../amp.html#torch.cuda.amp.GradScaler.unscale_ "torch.cuda.amp.GradScaler.unscale_")单独对其中任何一个进行调用，并且您必须对每个调用调用 [`scaler.step`](../amp.html#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step")单独。


 但是，在所有优化器使用此迭代之后， [`scaler.update`](../amp.html#torch.cuda.amp.GradScaler.update "torch.cuda.amp.GradScaler.update") 只应调用一次已采取步骤：


```
scaler = torch.cuda.amp.GradScaler()

for epoch in epochs:
    for input, target in data:
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            output0 = model0(input)
            output1 = model1(input)
            loss0 = loss_fn(2 * output0 + 3 * output1, target)
            loss1 = loss_fn(3 * output0 - 5 * output1, target)

        # (retain_graph here is unrelated to amp, it's present because in this
        # example, both backward() calls share some sections of graph.)
        scaler.scale(loss0).backward(retain_graph=True)
        scaler.scale(loss1).backward()

        # You can choose which optimizers receive explicit unscaling, if you
        # want to inspect or modify the gradients of the params they own.
        scaler.unscale_(optimizer0)

        scaler.step(optimizer0)
        scaler.step(optimizer1)

        scaler.update()

```


 每个优化器都会检查其 infs/NaN 的梯度，并独立决定是否跳过该步骤。这可能会导致一个优化器跳过该步骤，而另一个优化器则不会。由于跳跃很少发生(每几百次迭代)，这不应妨碍收敛。如果您在将梯度缩放添加到多重优化器模型后观察到收敛不良，请报告错误。


## [使用多个 GPU](#id9) [¶](#working-with-multiple-gpus "此标题的永久链接")


 此处描述的问题仅影响 [`autocast`](../amp.html#torch.cuda.amp.autocast "torch.cuda.amp.autocast") 。 [`GradScaler`](../amp.html#torch.cuda.amp.GradScaler "torch.cuda.amp.GradScaler") 的用法不变。


### [单个进程中的 DataParallel](#id10) [¶](#dataparallel-in-a-single-process "永久链接到此标题")


 即使 [`torch.nn.DataParallel`](../generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 生成线程以在每个设备上运行前向传递。自动施放状态会在每个状态中传播，并且以下内容将起作用：


```
model = MyModel()
dp_model = nn.DataParallel(model)

# Sets autocast in the main thread
with autocast(device_type='cuda', dtype=torch.float16):
    # dp_model's internal threads will autocast.
    output = dp_model(input)
    # loss_fn also autocast
    loss = loss_fn(output)

```


### [DistributedDataParallel，每个进程一个 GPU](#id11) [¶](#distributeddataparallel-one-gpu-per-process "永久链接到此标题")


[`torch.nn.parallel.DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 的文档建议使用一个 GPU每个进程以获得最佳性能。在这种情况下，“DistributedDataParallel”不会在内部生成线程，因此使用 [`autocast`](../amp.html#torch.cuda.amp.autocast "torch.cuda.amp.autocast") 和 [`GradScaler `](../amp.html#torch.cuda.amp.GradScaler "torch.cuda.amp.GradScaler") 不受影响。


### [DistributedDataParallel，每个进程多个 GPU](#id12) [¶](#distributeddataparallel-multiple-gpus-per-process "永久链接到此标题")


 这里 [`torch.nn.parallel.DistributedDataParallel`](../generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 可能会产生一个副线程在每个设备上运行前向传递，例如 [`torch.nn.DataParallel`](../generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 。 [修复方法是相同的](#amp-dataparallel)：将自动转换作为模型“forward”方法的一部分应用，以确保它在侧线程中启用。


## [Autocast 和自定义 Autograd 函数](#id13) [¶](#autocast-and-custom-autograd-functions "永久链接到此标题")


 如果您的网络使用[自定义 autograd 函数](extending.html#extending-autograd)([`torch.autograd.Function`](../autograd.html#torch.autograd.Function "torch.autograd.Function" 的子类) ) )，如果有任何功能，则需要更改 autocast 兼容性



* 接受多个浮点张量输入，
* 包装任何可自动转换的操作(请参阅 [自动转换操作参考](../amp.html#autocast-op-reference) )，或者
* 需要特定的“dtype”(例如，如果它包装了仅针对 `dtype` 编译的 [CUDA 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html)。


 在所有情况下，如果您要导入函数并且无法更改其定义，则安全的后备方法是在发生错误的任何使用点禁用自动转换并强制执行“float32”(或“dtype”)：


```
with autocast(device_type='cuda', dtype=torch.float16):
    ...
    with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        output = imported_function(input1.float(), input2.float())

```


 如果您是该函数的作者(或可以更改其定义)，更好的解决方案是使用 [`torch.cuda.amp.custom_fwd()`](../amp.html#torch.cuda.amp. custom_fwd "torch.cuda.amp.custom_fwd") 和 [`torch.cuda.amp.custom_bwd()`](../amp.html#torch.cuda.amp.custom_bwd "torch.cuda.amp.custom_bwd ") 装饰器，如下面的相关案例所示。


### [具有多个输入或可自动转换操作的函数](#id14) [¶](#functions-with-multiple-inputs-or-autocastable-ops "永久链接到此标题")


 应用 [`custom_fwd`](../amp.html#torch.cuda.amp.custom_fwd "torch.cuda.amp.custom_fwd") 和 [`custom_bwd`](../amp.html#torch.cuda.amp.custom_bwd "torch.cuda.amp.custom_bwd") (不带参数)分别为 `forward` 和 `backward`。这些确保“forward”以当前自动转换状态执行，“backward”以与“forward”相同的自动转换状态执行(这可以防止类型不匹配错误)：


```
class MyMM(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.mm(b)
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad.mm(b.t()), a.t().mm(grad)

```


 现在可以在任何地方调用“MyMM”，而无需禁用自动转换或手动转换输入：


```
mymm = MyMM.apply

with autocast(device_type='cuda', dtype=torch.float16):
    output = mymm(input1, input2)

```


### [需要特定 `dtype` 的函数](#id15)[¶](#functions-that-need-a-prefer-dtype "永久链接到此标题")


 考虑一个需要 `torch.float32` 输入的自定义函数。应用 [`custom_fwd(cast_inputs=torch.float32)`](../amp.html#torch.cuda.amp.custom_fwd "torch.cuda. amp.custom_fwd") 到 `forward` 和 [`custom_bwd`](../amp.html#torch.cuda.amp.custom_bwd "torch.cuda.amp.custom_bwd") (不带参数)到 `backward如果 `forward` 在启用自动转换的区域中运行，装饰器会将浮点 CUDA Tensorinputs 转换为 `float32` ，并在 `forward` 和 `backward` 期间本地禁用自动转换：


```
class MyFloat32Func(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input):
        ctx.save_for_backward(input)
        ...
        return fwd_output
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        ...

```


 现在可以在任何地方调用`MyFloat32Func`，无需手动禁用自动转换或转换输入：


```
func = MyFloat32Func.apply

with autocast(device_type='cuda', dtype=torch.float16):
    # func will run in float32, regardless of the surrounding autocast state
    output = func(input)

```