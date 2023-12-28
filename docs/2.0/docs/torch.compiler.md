# torch.compiler [¶](#torch-compiler "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/torch.compiler>
>
> 原始地址：<https://pytorch.org/docs/stable/torch.compiler.html>


`torch.compiler` 是一个命名空间，通过它可以显示一些内部编译器方法以供用户使用。该命名空间中的主要功能和特性是“torch.compile”。


torch.compile 是 PyTorch 2.x 中引入的 PyTorch 函数，旨在解决 PyTorch 中准确捕获图形的问题，最终使软件工程师能够更快地运行他们的 PyTorch 程序。 `torch.compile` 是用 Python 编写的，它标志着 PyTorch 从 C++ 到 Python 的过渡。


`torch.compile` 利用以下底层技术：



* **TorchDynamo (torch._dynamo)** 是一个内部 API，它使用称为框架评估 API 的 CPython 功能来安全地捕获 PyTorch 图形。PyTorch 用户在外部可用的方法通过 `torch.compiler` 命名空间显示。
* **TorchInductor** 是默认的“torch.compile”深度学习编译器，可为多个加速器和后端生成快速代码。您需要使用后端编译器来通过“torch.compile”实现加速。对于 NVIDIA 和 AMD GPU，它利用 OpenAI Triton 作为关键构建块。
* **AOT Autograd** 不仅捕获用户级代码，还捕获反向传播，从而“提前”捕获反向传播。这使得使用 TorchInductor 可以加速前向和后向传递。




!!! note "笔记"

    在某些情况下，术语“torch.compile”、“TorchDynamo”、“torch.compiler”在本文档中可能可以互换使用。


 如上所述，为了更快地运行工作流程，通过 TorchDynamo 的“torch.compile”需要一个后端，将捕获的图形转换为快速机器代码。不同的后端会带来不同的优化增益。默认的后端称为 TorchInductor，也称为 *inductor* ，TorchDynamo 有一个由我们的合作伙伴开发的支持的后端列表，可以通过运行 `torch.compiler.list_backends( )` 每个都有其可选的依赖项。


 一些最常用的后端包括：


**训练和推理后端**


| 	 Backend	  | 	 Description	  |
| --- | --- |
| 	`torch.compile(m,	 		 backend="inductor")`	 | 	 Uses the TorchInductor backend.	 [Read more](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) 	 |
| 	`torch.compile(m,	 		 backend="cudagraphs")`	 | 	 CUDA graphs with AOT Autograd.	 [Read more](https://github.com/pytorch/torchdynamo/pull/757) 	 |
| 	`torch.compile(m,	 		 backend="ipex")`	 | 	 Uses IPEX on CPU.	 [Read more](https://github.com/intel/intel-extension-for-pytorch) 	 |
| 	`torch.compile(m,	 		 backend="onnxrt")`	 | 	 Uses ONNX Runtime for training on CPU/GPU.	 [Read more](onnx_dynamo_onnxruntime_backend.html)	 |


**仅推理后端**


| 	 Backend	  | 	 Description	  |
| --- | --- |
| 	`torch.compile(m,	 		 backend="tensorrt")`	 | 	 Uses ONNX Runtime to run TensorRT for inference optimizations.	 [Read more](https://github.com/onnx/onnx-tensorrt) 	 |
| 	`torch.compile(m,	 		 backend="ipex")`	 | 	 Uses IPEX for inference on CPU.	 [Read more](https://github.com/intel/intel-extension-for-pytorch) 	 |
| 	`torch.compile(m,	 		 backend="tvm")`	 | 	 Uses Apache TVM for inference optimizations.	 [Read more](https://tvm.apache.org/) 	 |


## 阅读更多内容 [¶](#read-more "此标题的永久链接")


 PyTorch 用户入门



* [入门](torch.compiler_get_started.html)
* [torch.compiler API 参考](torch.compiler_api.html)
* [PyTorch 2.0 性能仪表板](torch.compiler_performance_dashboard.html)
* [用于细粒度跟踪的 TorchDynamo API ](torch.compiler_fine_grain_apis.html)
* [TorchInductor GPU 分析](torch.compiler_inductor_profiling.html)
* [通过分析了解 torch.compile 性能](torch.compiler_profiling_torch_compile.html)
* [常见问题](torch.compiler_faq.html )
* [PyTorch 2.0 故障排除](torch.compiler_troubleshooting.html)


 PyTorch 开发人员深入探讨



* [TorchDynamo 深入探究](torch.compiler_deepdive.html)
* [Guards 概述](torch.compiler_guards_overview.html)
* [动态形状](torch.compiler_dynamic_shapes.html)
* [PyTorch 2.0 NNModule 支持](torch.compiler_nn_module.html )
* [后端最佳实践](torch.compiler_best_practices_for_backends.html)
* [CUDAGraph 树](torch.compiler_cudagraph_trees.html)
* [假tensor](torch.compiler_fake_tensor.html)


 PyTorch 后端供应商指南



* [自定义后端](torch.compiler_custom_backends.html)
* [在 ATen IR 上编写图形转换](torch.compiler_transformations.html)
* [IR](torch.compiler_ir.html)