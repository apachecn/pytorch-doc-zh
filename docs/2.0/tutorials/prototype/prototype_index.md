# Pytorch Prototype Recipes

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/prototype/prototype_index>
>
> 原始地址：<https://pytorch.org/tutorials/prototype/prototype_index.html>

Prototype 功能不作为 PyPI 或 Conda 等二进制发行版的一部分提供（除了可能在运行时标志后面）。为了测试这些功能，我们建议根据功能从 master 构建或使用 <pytorch.org> 上提供的 nightlywheel 进行构建。

承诺程度：我们承诺仅收集有关这些功能的高带宽反馈。根据这些反馈和社区成员之间潜在的进一步参与，我们作为一个社区将决定是否要提高承诺水平或快速失败。

## Debugging

### PyTorch 数字套件教程

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何使用 PyTorch Numeric Suite 支持量化调试工作](https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html)


## FX

###  FX 图形模式量化用户指南

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解 FX 图形模式量化](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)


### FX 图形模式训练后动态量化

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何基于 torch.fx 在图形模式下进行训练后动态量化](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)


### FX 图形模式训练后静态量化

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何基于 torch.fx 在图形模式下进行训练后静态量化](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)


## MaskedTensor

### Masked Tensor 概述

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解Masked Tensor，指定和未指定值的真实来源](https://pytorch.org/tutorials/prototype/maskedtensor_overview.html)

### Masked Tensor 稀疏性

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何在 MaskedTensor 中利用稀疏布局（例如 COO 和 CSR）](https://pytorch.org/tutorials/prototype/maskedtensor_sparsity.html)

### Masked Tensor 高级语义

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[详细了解 Masked Tensor 的高级语义（与 NumPy 的 MaskedArray 的简化和比较）](https://pytorch.org/tutorials/prototype/maskedtensor_advanced_semantics.html)

### Masked Tensor：简化 Adagrad 稀疏语义

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[查看有关 Masked Tensor 如何实现稀疏语义并提供更清晰的开发体验的展示](https://pytorch.org/tutorials/prototype/maskedtensor_adagrad.html)


## Mobile

### 在 PyTorch 中使用 iOS GPU

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/ios.png" width=20% />

[了解如何在 iOS GPU 上运行模型](https://pytorch.org/tutorials/prototype/ios_gpu_workflow.html)


### 将 MobileNetV2 转换为 NNAPI

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/android.png" width=20% />

[了解如何准备计算机视觉模型以使用 Android 的神经网络 API (NNAPI)](https://pytorch.org/tutorials/prototype/nnapi_mobilenetv2.html)


### PyTorch Vulkan 后端用户工作流程

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/android.png" width=20% />

[了解如何在移动 GPU 上使用 Vulkan 后端](https://pytorch.org/tutorials/prototype/vulkan_workflow.html)


### 基于跟踪的选择性构建 Android 和 iOS Mobile Interpreter

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/mobile.png" width=20% />

[了解如何通过基于跟踪的选择性构建来优化 Mobile Interpreter 大小](https://pytorch.org/tutorials/prototype/tracing_based_selective_build.html)


### 将 Mobilenetv2 转换为 Core ML

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/ios.png" width=20% />

[了解如何准备计算机视觉模型以使用 PyTorch Core ML 移动后端](https://pytorch.org/tutorials/prototype/ios_coreml_workflow.html)


## Model Optimization

### 电感器 Cpp 包装教程

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[使用 Inductor Cpp Wrapper 加速您的模型](https://pytorch.org/tutorials/prototype/inductor_cpp_wrapper_tutorial.html)


## Model Optimization

### (prototype) 利用半结构化 (2:4) 稀疏性加速 BERT

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[将 BERT 修剪为 2:4 稀疏并加速推理](https://pytorch.org/tutorials/prototype/prototype/semi_structured_sparse.html)


## Modules

### 在 PyTorch 1.10 中跳过模块参数初始化

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[描述在 PyTorch 1.10 中的模块构建期间跳过参数初始化，避免浪费计算](https://pytorch.org/tutorials/prototype/skip_param_init.html)


## NestedTensor

### Nested Tensor

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解 Nested Tensor，这是批处理异构长度数据的新方法](https://pytorch.org/tutorials/prototype/nestedtensor.html)


## Quantization

### FX 图形模式量化用户指南

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解 FX 图形模式量化](https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide.html)

### FX 图形模式训练后动态量化

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何基于 torch.fx 在图形模式下进行训练后动态量化](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)

### FX 图形模式训练后静态量化

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何基于 torch.fx 在图形模式下进行训练后静态量化](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)

### BERT 上的图模式动态量化

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/graph-mode-dynamic-bert.png" width=20% />

[了解如何在 BERT 模型上使用图模式量化进行训练后动态量化](https://pytorch.org/tutorials/prototype/graph_mode_dynamic_bert_tutorial.html)

### PyTorch 数字套件教程

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何使用 PyTorch Numeric Suite 支持量化调试工作](https://pytorch.org/tutorials/prototype/numeric_suite_tutorial.html)

### 如何为 PyTorch 2 导出量化编写量化器

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何实现 PT2 导出量化的量化器](https://pytorch.org/tutorials/prototype/pt2e_quantizer.html)

### PyTorch 2 导出训练后量化

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何在 PyTorch 2 Export 中使用训练后量化](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)

### PyTorch 2 导出量化感知训练

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解如何在 PyTorch 2 Export 中使用量化感知训练](https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html)


## Text

### BERT 上的图模式动态量化

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/graph-mode-dynamic-bert.png" width=20% />

[了解如何在 BERT 模型上使用图模式量化进行训练后动态量化](https://pytorch.org/tutorials/prototype/graph_mode_dynamic_bert_tutorial.html)


## TorchScript

###  TorchScript 中的模型 Freezing

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[Freezing 是将 Pytorch 模块参数和属性值内联到 TorchScript 内部表示的过程](https://pytorch.org/tutorials/prototype/torchscript_freezing.html)


## vmap

### 使用 torch.vmap

<img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=20% />

[了解 torch.vmap，这是一个用于 PyTorch 操作的自动矢量化器](https://pytorch.org/tutorials/prototype/vmap_recipe.html)