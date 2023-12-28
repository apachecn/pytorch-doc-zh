

**简介** 
 ||
 [什么是 DDP](ddp_series_theory.html) 
 ||
 [单节点多 GPU 训练](ddp_series_multigpu.html) 
 ||\ n [容错](ddp_series_fault_tolerance.html) 
 ||
 [多节点训练](../intermediate/ddp_series_multinode.html) 
 ||
 [minGPT 训练](../intermediate/ddp_series_minGPT. html)





# PyTorch 中的分布式数据并行 - 视频教程 [¶](#distributed-data-parallel-in-pytorch-video-tutorials "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/ddp_series_intro>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/ddp_series_intro.html>




 作者：
 [Suraj Subramanian](https://github.com/suraj813)




 请观看下面或 [youtube](https://www.youtube.com/watch/-K3bZYHYHEA) 上的视频
 。








 本系列视频教程将引导您通过 DDP 在 PyTorch 中完成
分布式训练。




 本系列从简单的非分布式训练作业开始，到
在集群中的多台机器上部署训练作业结束。
在此过程中，您还将了解
 [torchrun](https://pytorch.org/docs/stable/elastic/run.html)
 用于
容错分布式训练。




 本教程假设您基本熟悉 PyTorch 中的模型训练。





## 运行代码 [¶](#running-the-code "固定链接到此标题")
 -



 您将需要多个 CUDA GPU 来运行教程代码。通常，
这可以在具有多个 GPU 的云实例上完成(教程
使用具有 4 个 GPU 的 Amazon EC2 P3 实例)。




 教程代码托管在此
 [github 存储库](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series) 
 。
克隆存储库并按照说明进行操作!





## 教程部分 [¶](#tutorial-sections "此标题的永久链接")



0. 简介(本页)
1. [什么是DDP？](ddp_series_theory.html) 
 轻轻介绍DDP 的底层原理
2. [单节点多GPU训练](ddp_series_multigpu.html)
在单台机器上使用多个GPU
训练模型
3. [容错分布式训练](ddp_series_fault_tolerance.html) 
 使用 torchrun 让分布式训练作业更加稳健
4。 [多节点训练](../intermediate/ddp_series_multinode.html)
在多台机器上使用
多个 GPU 训练模型
5. [使用 DDP 训练 GPT 模型](../intermediate/ddp_series_minGPT.html) 
 “Real-world”
训练示例
 [minGPT](https://github.com/karpathy/minGPT) 
 具有 DDP 的模型








