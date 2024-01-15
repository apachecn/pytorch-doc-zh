# PyTorch Recipes

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/recipes/recipes_index>
>
> 原始地址：<https://pytorch.org/tutorials/recipes/recipes_index.html>

Recipes 是关于如何使用特定 PyTorch 功能的简短、可操作的示例，与我们的完整教程不同。

=== "Basics"

    <div class="grid cards" markdown>

    -   __在 PyTorch 中加载数据__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/loading-data.PNG" width=40% />

        [了解如何使用 PyTorch 包为您的模型准备和加载通用数据集](https://pytorch.org/tutorials/recipes/recipes/loading_data_recipe.html)

    -   __定义神经网络__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/defining-a-network.PNG" width=40% />

        [了解如何使用 PyTorch 的 torch.nn 包为 MNIST 数据集创建和定义神经网络](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)

    -   __PyTorch 中的 state_dict 是什么__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/what-is-a-state-dict.PNG" width=40% />

        [了解如何使用 state_dict 对象和 Python 字典从 PyTorch 保存或加载模型](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)

    -   __在 PyTorch 中保存和加载模型以进行推理__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/saving-and-loading-models-for-inference.PNG" width=40% />

        [了解在 PyTorch 中保存和加载模型进行推理的两种方法 - 通过 state\_dict 和通过整个模型](https://pytorch.org/tutorials/recipes/recipes/    saving_and_loading_models_for_inference.html)

    -   __在 PyTorch 中保存和加载常规检查点__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/saving-and-loading-general-checkpoint.PNG" width=40% />

        [保存和加载用于推理或恢复训练的通用检查点模型有助于从上次停下的地方继续。在本节中，探索如何保存和加载多个检查点](https://pytorch.org/tutorials/recipes/recipes/ saving_and_loading_a_general_checkpoint.html)

    -   __使用 PyTorch 在一个文件中保存和加载多个模型__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/saving-multiple-models.PNG" width=40% />

        [在本节中，了解保存和加载多个模型如何有助于重用您之前训练过的模型](https://pytorch.org/tutorials/recipes/recipes/saving_multiple_models_in_one_file.html)


    -   __使用 PyTorch 中不同模型的参数热启动模型__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/warmstarting-models.PNG" width=40% />

        [了解通过部分加载模型或加载部分模型来热启动训练过程如何帮助您的模型比从头开始训练更快地收敛](https://pytorch.org/tutorials/recipes/recipes/ warmstarting_model_using_parameters_from_a_different_model.html)


    -   __在 PyTorch 中跨设备保存和加载模型__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/saving-and-loading-models-across-devices.PNG" width=40% />

        [了解如何使用 PyTorch 相对简单地跨设备(CPU 和 GPU)保存和加载模型](https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html)


    -   __在 PyTorch 中将梯度归零__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/zeroing-out-gradients.PNG" width=40% />

        [了解何时应该将梯度归零以及这样做如何有助于提高模型的准确性](https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)


    -   __PyTorch 基准测试__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [了解如何使用 PyTorch 的基准测试模块来测量和比较代码的性能](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)


    -   __PyTorch 基准测试(快速入门)__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [了解如何测量代码片段运行时间并收集指令](https://pytorch.org/tutorials/recipes/recipes/timer_quick_start.html)


    -   __PyTorch Profiler__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [了解如何使用 PyTorch 的分析器来测量操作员的时间和内存消耗](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

    -   __PyTorch Profiler 具有仪器和跟踪技术 API (ITT API) 支持__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [了解如何将 PyTorch 的分析器与仪器和跟踪技术 API (ITT API) 结合使用，以可视化英特尔® VTune™ Profiler GUI 中的操作员标签](https://pytorch.org/tutorials/recipes/ profile_with_itt.html)

    -   __Torch 编译 IPEX 后端__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [了解如何使用 torch.compile IPEX 后端](https://pytorch.org/tutorials/recipes/torch_compile_backend_ipex.html)

    -   __在 PyTorch 中推理形状__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [了解如何使用元设备来推理模型中的形状](https://pytorch.org/tutorials/recipes/recipes/reasoning_about_shapes.html)

    -   __从检查点加载 nn.Module 的提示__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [了解从检查点加载 nn.Module 的技巧](https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)

    </div>


=== "Captum"

    <div class="grid cards" markdown>

    -   __使用 Captum 进行模型解释__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/model-interpretability-using-captum.png" width=40% />

        [了解如何使用 Captum 将图像分类器的预测归因于其相应的图像特征，并可视化归因结果](https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html)

    </div>


=== "Distributed Training"

    <div class="grid cards" markdown>

    -   __使用 ZeroRedundancyOptimizer 的分片优化器状态__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [如何使用 ZeroRedundancyOptimizer 来减少内存消耗](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html)

    -   __使用 TensorPipe RPC 进行直接设备间通信__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [如何使用 RPC 进行 GPU 到 GPU 的直接通信](https://pytorch.org/tutorials/recipes/cuda_rpc.html)

    -   __支持 TorchScript 的分布式优化器__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [如何启用对分布式优化器的 TorchScript 支持](https://pytorch.org/tutorials/recipes/distributed_optim_torchscript.html)

    -   __分布式检查点 (DCP) 入门__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/Getting-Started-with-DCP.png" width=40% />

        [了解如何使用分布式检查点包检查分布式模型](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)

    </div>


=== "Interpretability"

    <div class="grid cards" markdown>

    -   __使用 Captum 进行模型解释__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/model-interpretability-using-captum.png" width=40% />

        [了解如何使用 Captum 将图像分类器的预测归因于其相应的图像特征，并可视化归因结果](https://pytorch.org/tutorials/recipes/recipes/Captum_Recipe.html)

    </div>


=== "Mobile"

    <div class="grid cards" markdown>

    -   __PyTorch Mobile 性能秘诀__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/mobile.png" width=40% />

        [在移动设备(Android 和 iOS)上使用 PyTorch 的性能优化方法列表](https://pytorch.org/tutorials/recipes/mobile_perf.html)

    -   __制作使用 PyTorch Android 预构建库的 Android 本机应用程序__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/android.png" width=40% />

        [了解如何从头开始制作使用 LibTorch C++ API 并使用 TorchScript 模型和自定义 C++ 运算符的 Android 应用程序](https://pytorch.org/tutorials/recipes/    android_native_app_with_custom_op.html)

    -   __Fuse Modules Recipe__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/mobile.png" width=40% />

        [了解如何将 PyTorch 模块列表融合为单个模块，以在量化之前减小模型大小](https://pytorch.org/tutorials/recipes/fuse.html)

    -   __Mobile Recipes 的量化__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/mobile.png" width=40% />

        [了解如何减小模型大小并使其运行速度更快，同时又不会损失太多准确性](https://pytorch.org/tutorials/recipes/quantization.html)

    -   __针对移动设备编写脚本并进行优化__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/mobile.png" width=40% />

        [了解如何将模型转换为 TorchScipt 并(可选)针对移动应用程序对其进行优化](https://pytorch.org/tutorials/recipes/script_optimized.html)

    -   __iOS Recipes 的模型准备__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/ios.png" width=40% />

        [了解如何在 iOS 项目中添加模型并使用适用于 iOS 的 PyTorch pod](https://pytorch.org/tutorials/recipes/model_preparation_ios.html)

    -   __Android Recipes 的模型准备__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/android.png" width=40% />

        [了解如何在 Android 项目中添加模型并使用适用于 Android 的 PyTorch 库](https://pytorch.org/tutorials/recipes/model_preparation_android.html)

    -   __Android 和 iOS 中的 Mobile Interpreter 工作流程__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/mobile.png" width=40% />

        [了解如何在 iOS 和 Andriod 设备上使用 Mobile Interpreter](https://pytorch.org/tutorials/recipes/mobile_interpreter.html)

    </div>


=== "Model Optimization"

    <div class="grid cards" markdown>

    -   __动态量化__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/using-dynamic-post-training-quantization.png" width=40% />

        [将动态量化应用于简单的 LSTM 模型](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)

    -   __PyTorch Mobile 性能秘诀__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/mobile.png" width=40% />

        [在移动设备(Android 和 iOS)上使用 PyTorch 的性能优化方法列表](https://pytorch.org/tutorials/recipes/mobile_perf.html)


    -   __自动混合精度__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/amp.png" width=40% />

        [使用 torch.cuda.amp 缩短运行时间并节省 NVIDIA GPU 上的内存](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)


    -   __性能调优指南__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [实现最佳性能的技巧](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

    -   __利用英特尔® 高级矩阵扩展__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [了解如何利用英特尔® 高级矩阵扩展](https://pytorch.org/tutorials/recipes/amx.html)

    -   __适用于 PyTorch 的英特尔® 扩展__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [推出适用于 PyTorch* 的英特尔® 扩展](https://pytorch.org/tutorials/recipes/intel_extension_for_pytorch.html)

    -   __适用于 PyTorch 的英特尔® 神经压缩器__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [使用英特尔® 神经压缩器对 PyTorch 进行易于使用的量化](https://pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html)

    </div>


=== "Production"

    <div class="grid cards" markdown>

    -   __使用 Flask 进行部署__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/using-flask-create-restful-api.png" width=40% />

        [了解如何使用 Flask(一种轻量级 Web 服务器)从经过训练的 PyTorch 模型快速设置 Web API](https://pytorch.org/tutorials/recipes/deployment_with_flask.html)

    -   __分析基于 PyTorch RPC 的工作负载__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profile.png" width=40% />

        [如何使用 PyTorch 分析器分析基于 RPC 的工作负载](https://pytorch.org/tutorials/recipes/distributed_rpc_profiling.html)

    -   __将 PyTorch Stable Diffusion 模型部署为 Vertex AI 端点__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/generic-pytorch-logo.png" width=40% />

        [了解如何使用 TorchServe 在 Vertex AI 中部署模型](https://pytorch.org/tutorials/recipes/torchserve_vertexai_tutorial.html)

    </div>


=== "Quantization"

    <div class="grid cards" markdown>

    -   __动态量化__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/using-dynamic-post-training-quantization.png" width=40% />

        [将动态量化应用于简单的 LSTM 模型](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)

    -   __Mobile Recipes 的量化__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/mobile.png" width=40% />

        [了解如何减小模型大小并使其运行速度更快，同时又不会损失太多准确性](https://pytorch.org/tutorials/recipes/quantization.html)

    -   __适用于 PyTorch 的英特尔® 神经压缩器__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [使用英特尔® 神经压缩器对 PyTorch 进行易于使用的量化](https://pytorch.org/tutorials/recipes/intel_neural_compressor_for_pytorch.html)

    </div>


=== "TensorBoard"

    <div class="grid cards" markdown>

    -   __如何将 TensorBoard 与 PyTorch 结合使用__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/tensorboard_scalars.png" width=40% />

        [了解 TensorBoard 与 PyTorch 的基本用法，以及如何在 TensorBoard UI 中可视化数据](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

    </div>


=== "Text"

    <div class="grid cards" markdown>

    -   __动态量化__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/using-dynamic-post-training-quantization.png" width=40% />

        [将动态量化应用于简单的 LSTM 模型](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html)

    </div>


=== "TorchScript"

    <div class="grid cards" markdown>

    -   __用于部署的 TorchScript__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/torchscript_overview.png" width=40% />

        [了解如何以 TorchScript 格式导出经过训练的模型以及如何以 C++ 加载 TorchScript 模型并进行推理](https://pytorch.org/tutorials/recipes/torchscript_inference.html)

    -   __使用 Flask 进行部署__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/using-flask-create-restful-api.png" width=40% />

        [了解如何使用 Flask(一种轻量级 Web 服务器)从经过训练的 PyTorch 模型快速设置 Web API](https://pytorch.org/tutorials/recipes/deployment_with_flask.html)

    -   __支持 TorchScript 的分布式优化器__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/cropped/profiler.png" width=40% />

        [如何启用对分布式优化器的 TorchScript 支持](https://pytorch.org/tutorials/recipes/distributed_optim_torchscript.html)

    </div>


=== "Visualization"

    <div class="grid cards" markdown>

    -   __如何将 TensorBoard 与 PyTorch 结合使用__

        <img src="https://pytorch.org/tutorials/_static/img/thumbnails/tensorboard_scalars.png" width=40% />

        [了解 TensorBoard 与 PyTorch 的基本用法，以及如何在 TensorBoard UI 中可视化数据](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)

    </div>
