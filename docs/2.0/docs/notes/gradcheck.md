> 翻译任务

* 目前该页面无人翻译，期待你的加入
* 翻译奖励: https://github.com/orgs/apachecn/discussions/243
* 任务认领: https://github.com/apachecn/pytorch-doc-zh/discussions/583


"公式 - 改得头大，暂时放弃。。"


# Gradcheck 机制 [¶](#gradcheck-mechanics "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/gradcheck>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/gradcheck.html>


 本说明概述了 [`gradcheck()`](../generated/torch.autograd.gradcheck.html#torch.autograd.gradcheck "torch.autograd.gradcheck") 和 [`gradgradcheck()`](../generated/torch.autograd.gradgradcheck.html#torch.autograd.gradgradcheck "torch.autograd.gradgradcheck") 函数有效。


 它将涵盖实数和复数值函数以及高阶导数的前向和后向模式 AD。本注释还涵盖 gradcheck 的默认行为以及“fast_mode=True”参数为的情况通过(以下简称快速毕业检查)。



* [符号和背景信息](#notations-and-background-information)
* [默认向后模式梯度检查行为](#default-backward-mode-gradcheck-behavior)
	+ [实数到实数函数](#real-to-real-functions) 
	+ [复数到实数函数](#complex-to-real-functions) 
	+ [具有复数输出的函数](#functions-with-complex-outputs)
* [快速向后模式梯度检查](#fast-backward-mode-gradcheck)
	+ [实数到实数函数的快速梯度检查](#fast-gradcheck-for-real-to-real-functions) 
	+ [复数到实数函数的快速梯度检查](#fast-gradcheck-for-complex-to-real-functions) 
	+ [具有复杂输出的函数的快速 gradcheck](#fast-gradcheck-for-functions-with-complex-outputs)
* [Gradgradcheck 实现](#gradgradcheck-implementation)


## [符号和背景信息](#id2) [¶](#notations-and-background-information "此标题的固定链接")


 在本说明中，我们将使用以下约定：

。。。