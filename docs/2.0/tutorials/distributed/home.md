


# 分布式和并行训练教程 [¶](#distributed-and-parallel-training-tutorials "固定链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/distributed/home>
>
> 原始地址：<https://pytorch.org/tutorials/distributed/home.html>




 分布式训练是一种模型训练范例，
涉及将训练工作负载分散到多个工作节点，从而
显着提高训练速度和模型准确性。虽然分布式训练
可用于任何类型的 ML 模型训练，但
将其用于大型模型和计算要求较高的任务(例如深度学习)
最为有利。




 您可以通过多种方法在 PyTorch 中执行分布式训练，每种方法在某些用例中都有其优势：



* [分布式数据并行 (DDP)](#learn-ddp)
* [完全分片数据并行 (FSDP)](#learn-fsdp)
* [远程过程调用 (RPC) 分布式训练](#learn-rpc) 
* [自定义扩展](#custom-extensions)



 在 [分布式概述](../beginner/dist_overview.html) 中了解有关这些选项的更多信息
.





## 学习 DDP [¶](#learn-ddp "此标题的永久链接")













 DDP 介绍视频教程
 

 有关如何开始使用
 
 DistributedDataParallel
 
 并进一步了解更复杂主题的分步视频系列










 代码










 视频















 分布式数据并行入门
 

 本教程简要介绍 PyTorch
DistributedData Parallel。










 代码















 使用连接上下文管理器进行不均匀输入的分布式训练



 本教程介绍连接上下文管理器并
演示它’ 与 DistributedData Parallel 的使用。










 代码














## 学习 FSDP [¶](#learn-fsdp "永久链接到此标题")













 FSDP 入门
 

 本教程演示如何使用 FSDP 在 MNIST 数据集上
执行分布式训练。










 代码















 FSDP Advanced
 

 在本教程中，您将学习如何使用 FSDP 微调 HuggingFace (HF) T5
模型以进行文本摘要。










 代码














## 学习 RPC [¶](#learn-rpc "永久链接到此标题")













 分布式 RPC 框架入门
 

 本教程演示如何开始使用基于 RPC 的分布式
训练。










 代码















 使用分布式 RPC 框架实现参数服务器
 

 本教程将引导您完成使用 PyTorch’s 分布式 RPC 框架
实现参数服务器的简单示例。










 代码















 使用异步执行实现批量 RPC 处理
 

 在本教程中，您将使用 @rpc.functions.async_execution 装饰器
构建批处理 RPC 应用程序。










 代码



















 将分布式 DataParallel 与分布式 RPC 框架相结合
 

 在本教程中，您将学习如何将分布式数据并行性与分布式模型并行性相结合。










 代码














## 自定义扩展 [¶](#custom-extensions "永久链接到此标题")













 使用 Cpp 扩展自定义进程组后端
 

 在本教程中，您将学习实现自定义
 
 ProcessGroup
 
 后端并使用 
cpp 扩展将其插入 PyTorch 分布式包。










 代码















