# torch.func [¶](#torch-func "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/func>
>
> 原始地址：<https://pytorch.org/docs/stable/func.html>


 torch.func，以前称为“functorch”，是 PyTorch 的[类似 JAX](https://github.com/google/jax) 可组合函数转换。




!!! note "笔记"

    该库目前处于[测试阶段](https://pytorch.org/blog/pytorch-feature-classification-changes/#beta)。这意味着这些功能通常可以工作(除非另有说明)并且我们(PyTorch团队)致力于推动这个图书馆的发展。但是，API 可能会根据用户反馈进行更改，并且我们无法全面覆盖 PyTorch 操作。


 如果您对希望涵盖的 API 或用例有建议，请打开 GitHub 问题或联系我们。我们很想听听您如何使用图书馆。


## 什么是可组合函数变换？ [¶](#what-are-composable-function-transforms"此标题的永久链接")



* “函数变换”是一个高阶函数，它接受数值函数并返回计算不同数量的新函数。
* [`torch.func`](func.api.html#module-torch.func "torch. func") 具有自动微分变换( `grad(f)` 返回一个计算“f”梯度的函数)，一个矢量化/批处理变换( `vmap(f)` 返回一个在批次上计算 `f` 的函数输入)，以及其他。*这些函数变换可以任意组合。例如，组合`vmap(grad(f))`会计算一个称为“每个样本梯度”的量，而 PyTorch 目前无法有效计算该量。


## 为什么可组合函数会发生变换？ [¶](#why-composable-function-transforms"永久链接到此标题")


 如今，PyTorch 中有许多用例很难实现：



* 计算每个样本的梯度(或其他每个样本的量)
* 在单台机器上运行模型集合
* 在 MAML 内循环中高效地将任务批量组合在一起
* 高效计算雅克比矩阵和海森矩阵
* 高效计算批量雅克比矩阵和海森矩阵


 编写 [`vmap()`]( generated/torch.func.vmap.html#torch.func.vmap "torch.func.vmap") 、 [`grad()`](generated/torch.func.grad.html#torch.func.grad "torch.func.grad") 和 [`vjp()`](generated/torch.func.vjp.html#torch.func.vjp "torch.func.vjp") 转换允许我们来表达上述内容，而不需要为每个设计单独的子系统。这种可组合函数转换的想法来自 [JAX 框架](https://github.com/google/jax) 。


## 阅读更多内容 [¶](#read-more "此标题的永久链接")



* [torch.func 旋风之旅](func.whirlwind_tour.html) 
    + [什么是 torch.func？](func.whirlwind_tour.html#what-is-torch-func) 
    + [为什么可组合函数会转换？](func.Whirlwind_tour.html#why-composable-function-transforms) 
    + [什么是变换？](func.whirlwind_tour.html#what-are-the-transforms)
* [torch.func API 参考](func.api.html) 
    + [函数转换](func.api.html#function-transforms) 
    + [使用 torch.nn.Modules 的实用程序](func.api.html#utilities-for-working-with-torch-nn-modules)
* [UX 限制](func.ux_limitations.html) 
    + [一般限制](func.ux_limitations.html#general-limitations) 
    + [torch.autograd API](func.ux_limitations.html#torch-autograd-apis) 
    + [vmap限制](func.ux_limitations.html#vmap-limitations) 
    + [随机性](func.ux_limitations.html#randomness)
* [从 functorch 迁移到 torch.func](func.migration.html) 
    + [函数转换](func.migration.html#function-transforms) 
    + [NN 模块实用程序](func.migration.html#nn-module-utilities) 
    + [functorch.compile](func.migration.html#functorch-compile)