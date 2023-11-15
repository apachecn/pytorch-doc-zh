


 没有10



 单击
 [此处](#sphx-glr-download-beginner-onnx-intro-onnx-py)
 下载完整的示例代码





**ONNX 简介** 
 ||
 [将 PyTorch 模型导出到 ONNX](export_simple_model_to_onnx_tutorial.html) 
 ||
 [扩展 ONNX 注册表](onnx_registry_tutorial.html)





 ONNX 简介
 [¶](#introduction-to-onnx "此标题的永久链接")
=============================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/onnx/intro_onnx>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/onnx/intro_onnx.html>




 作者：
 [Thiago Crepaldi](https://github.com/thiagocrepaldi)
 ,




[开放神经网络交换 (ONNX)](https://onnx.ai/)
 是一种用于表示机器学习模型的开放标准
格式。 
 `torch.onnx`
 模块提供 API 来
从本机 PyTorch
 捕获计算图
 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "(在 PyTorch v2.1 中)")
 模型并
 转换
 为
 [ONNX 图表](https://github.com/onnx/onnx/blob/main/docs/IR.md) 
.




 导出的模型可以由许多
 [支持 ONNX 的运行时](https://onnx.ai/supported-tools.html#deployModel) 中的任何一个使用，
包括 Microsoft’s 
 [ONNX 运行时](https://www.onnxruntime.ai)
.





 注意




 目前，ONNX 导出器 API 有两种风格，
但本教程将重点介绍
 `torch.onnx.dynamo_export`
 。





 TorchDynamo 引擎用于挂钩 Python’s 框架评估 API 并动态地将其
字节码重写为
 [FX 图表](https://pytorch.org/docs/stable/fx.html) 
.
最终生成的 FX 图在最终转换为 
 [ONNX 图](https://github.com/onnx/onnx/blob/main/docs/IR.md) 之前经过打磨 
 n.




 这种方法的主要优点是
 [FX 图表](https://pytorch.org/docs/stable/fx.html) 
 使用字节码分析捕获，
保留了模型的动态特性而不是使用传统的静态跟踪技术。





 依赖项
 [¶](#dependencies "永久链接到此标题")
-------------------------------------------------------------------------



 需要 PyTorch 2.1.0 或更高版本。




 ONNX 导出器依赖于额外的 Python 包:




> 
> 
> * [ONNX](https://onnx.ai) 
> 标准库
> * [ONNX Script](https://onnxscript.ai) 
> 启用的库开发人员以富有表现力且简单的方式使用 Python 子集来编写 ONNX 运算符、函数和模型。
> 
> 
> 
>



 它们可以通过
 [pip](https://pypi.org/project/pip/)安装
 :






```
pip install --upgrade onnx onnxscript

```




 要验证安装，请运行以下命令:






```
import torch
print(torch.__version__)

import onnxscript
print(onnxscript.__version__)

from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now

import onnxruntime
print(onnxruntime.__version__)

```




 每个
 
 导入
 
 必须成功且没有任何错误，并且必须打印出库版本。






 进一步阅读
 [¶](#further-reading "此标题的永久链接")
-----------------------------------------------------------------------------------



 下面的列表涉及从基本示例到高级场景的教程，
不一定按照列出的顺序。
您可以直接跳到您感兴趣的特定主题，或者
耐心地阅读所有内容让他们了解有关 ONNX 导出器的所有信息。





 1.
 [将 PyTorch 模型导出到 ONNX](export_simple_model_to_onnx_tutorial.html)


 2.
 [扩展 ONNX 注册表](onnx_registry_tutorial.html)





**脚本的总运行时间:** 
 ( 0 分 0.000 秒)






[`下载
 

 Python
 

 源
 

 代码:
 

 intro_onnx.py`](../../_downloads/ea6986634c1fca7a6c0eaddbfd7f799c/简介_onnx.py)






[`下载
 

 Jupyter
 

 笔记本:
 

 intro_onnx.ipynb`](../../_downloads/33f8140bedc02273a55c752fe79058e5/intro_onnx.ipynb)






[Sphinx-Gallery 生成的图库](https://sphinx-gallery.github.io)









