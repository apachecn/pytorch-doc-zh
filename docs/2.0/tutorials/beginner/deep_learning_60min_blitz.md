


 使用 PyTorch 进行深度学习：60 分钟闪电战
 [¶](#deep-learning-with-pytorch-a-60-month-blitz "永久链接到此标题")
=============================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/deep_learning_60min_blitz>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>




**作者** 
 :
 [Soumith Chintala](http://soumith.ch)









 什么是 PyTorch？
 [¶](#what-is-pytorch "此标题的永久链接")
----------------------------------------------------------------------



 PyTorch 是一个基于 Python 的科学计算包，有两个广泛的用途：



* NumPy 的替代品，可利用 GPU 和其他加速器的功能。
* 一个自动微分库，可用于实现神经网络。





 本教程的目标：
 [¶](#goal-of-this-tutorial "永久链接到此标题")
------------------------------------------------------------------------------------


* 深入了解 PyTorch’s 张量库和神经网络。
* 训练小型神经网络来对图像进行分类



 要运行下面的教程，请确保您拥有
 [torch](https://github.com/pytorch/pytorch) 
 ,
 [torchvision](https://github.com/pytorch/Vision) 
 、
和
 [matplotlib](https://github.com/matplotlib/matplotlib) 
 已安装软件包。















 张量
 

 在本教程中，您将学习 PyTorch 张量的基础知识。










 代码















 torch.autograd 简要介绍
 

 了解 autograd。










 代码















 神经网络
 

 本教程演示如何在 PyTorch 中训练神经网络。










 代码















 训练分类器
 

 了解如何使用
CIFAR10 数据集在 PyTorch 中训练图像分类器。










 代码















