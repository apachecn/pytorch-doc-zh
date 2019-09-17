# <center>PyTorch 1.0 中文文档 & 教程</center>

![](../../docs/img/logo.svg)

<center>PyTorch 是一个针对深度学习, 并且使用 GPU 和 CPU 来优化的 tensor library (张量库)</center>
<br/>
<table>
  <tr align="center">
    <td><a title="Pytorch 1.0 中文版本" href="https://pytorch.apachecn.org/docs/1.0/" target="_blank"><font size="5">1.0 中文版本</font></a></td>
    <td><a title="Pytorch 最新 英文教程" href="https://pytorch.org/tutorials/" target="_blank"><font size="5">最新 英文教程</font></a></td>
    <td><a title="Pytorch 最新 英文文档" href="https://pytorch.org/docs/master/" target="_blank"><font size="5">最新 英文文档</font></a></td>
  </tr>
  <tr align="center">
    <td><a title="Pytorch 0.4 中文版本" href="https://pytorch.apachecn.org/docs/0.4/" target="_blank"><font size="5">0.4 中文版本</font></a></td>
    <td><a title="Pytorch 0.3 中文版本" href="https://pytorch.apachecn.org/docs/0.3/" target="_blank"><font size="5">0.3 中文版本</font></a></td>
    <td><a title="Pytorch 0.2 中文版本" href="https://pytorch.apachecn.org/docs/0.2/" target="_blank"><font size="5">0.2 中文版本</font></a></td>
  </tr>
</table>
<br/>

> 欢迎任何人参与和完善：一个人可以走的很快，但是一群人却可以走的更远。

+ [在线阅读](http://pytorch.apachecn.org)
+ [ApacheCN 学习资源](http://www.apachecn.org/)
+ [PyTorch 中文翻译组 | ApacheCN 713436582](http://shang.qq.com/wpa/qunwpa?idkey=349eb1bbaeeff1cf20408899cbe75669132ef145ff5ee6599f78a77dd144c367)

## 目录结构

* [Introduction](README.md)
* 中文教程
    * [起步](tut_getting_started.md)
        * [PyTorch 深度学习: 60 分钟极速入门](deep_learning_60min_blitz.md)
            * [什么是 PyTorch？](blitz_tensor_tutorial.md)
            * [Autograd：自动求导](blitz_autograd_tutorial.md)
            * [神经网络](blitz_neural_networks_tutorial.md)
            * [训练分类器](blitz_cifar10_tutorial.md)
            * [可选：数据并行处理](blitz_data_parallel_tutorial.md)
        * [数据加载和处理教程](data_loading_tutorial.md)
        * [用例子学习 PyTorch](pytorch_with_examples.md)
        * [迁移学习教程](transfer_learning_tutorial.md)
        * [混合前端的 seq2seq 模型部署](deploy_seq2seq_hybrid_frontend_tutorial.md)
        * [Saving and Loading Models](saving_loading_models.md)
        * [What is torch.nn really?](nn_tutorial.md)
    * [图像](tut_image.md)
        * [Torchvision 模型微调](finetuning_torchvision_models_tutorial.md)
        * [空间变换器网络教程](spatial_transformer_tutorial.md)
        * [使用 PyTorch 进行图像风格转换](neural_style_tutorial.md)
        * [对抗性示例生成](fgsm_tutorial.md)
        * [使用 ONNX 将模型从 PyTorch 传输到 Caffe2 和移动端](super_resolution_with_caffe2.md)
    * [文本](tut_text.md)
        * [聊天机器人教程](chatbot_tutorial.md)
        * [使用字符级别特征的 RNN 网络生成姓氏](char_rnn_generation_tutorial.md)
        * [使用字符级别特征的 RNN 网络进行姓氏分类](char_rnn_classification_tutorial.md)
        * [Deep Learning for NLP with Pytorch](deep_learning_nlp_tutorial.md)
            * [PyTorch 介绍](nlp_pytorch_tutorial.md)
            * [使用 PyTorch 进行深度学习](nlp_deep_learning_tutorial.md)
            * [Word Embeddings: Encoding Lexical Semantics](nlp_word_embeddings_tutorial.md)
            * [序列模型和 LSTM 网络](nlp_sequence_models_tutorial.md)
            * [Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](nlp_advanced_tutorial.md)
        * [基于注意力机制的 seq2seq 神经网络翻译](seq2seq_translation_tutorial.md)
    * [生成](tut_generative.md)
        * [DCGAN Tutorial](dcgan_faces_tutorial.md)
    * [强化学习](tut_reinforcement_learning.md)
        * [Reinforcement Learning (DQN) Tutorial](reinforcement_q_learning.md)
    * [扩展 PyTorch](tut_extending_pytorch.md)
        * [用 numpy 和 scipy 创建扩展](numpy_extensions_tutorial.md)
        * [Custom C++   and CUDA Extensions](cpp_extension.md)
        * [Extending TorchScript with Custom C++   Operators](torch_script_custom_ops.md)
    * [生产性使用](tut_production_usage.md)
        * [Writing Distributed Applications with PyTorch](dist_tuto.md)
        * [使用 Amazon AWS 进行分布式训练](aws_distributed_training_tutorial.md)
        * [ONNX 现场演示教程](ONNXLive.md)
        * [在 C++ 中加载 PYTORCH 模型](cpp_export.md)
    * [其它语言中的 PyTorch](tut_other_language.md)
        * [使用 PyTorch C++ 前端](cpp_frontend.md)
* 中文文档
    * [注解](docs_notes.md)
        * [自动求导机制](notes_autograd.md)
        * [广播语义](notes_broadcasting.md)
        * [CUDA 语义](notes_cuda.md)
        * [Extending PyTorch](notes_extending.md)
        * [Frequently Asked Questions](notes_faq.md)
        * [Multiprocessing best practices](notes_multiprocessing.md)
        * [Reproducibility](notes_randomness.md)
        * [Serialization semantics](notes_serialization.md)
        * [Windows FAQ](notes_windows.md)
    * [包参考](docs_package_ref.md)
        * [torch](torch.md)
            * [Tensors](torch_tensors.md)
            * [Random sampling](torch_random_sampling.md)
            * [Serialization, Parallelism, Utilities](torch_serialization_parallelism_utilities.md)
            * [Math operations](torch_math_operations.md)
                * [Pointwise Ops](torch_math_operations_pointwise_ops.md)
                * [Reduction Ops](torch_math_operations_reduction_ops.md)
                * [Comparison Ops](torch_math_operations_comparison_ops.md)
                * [Spectral Ops](torch_math_operations_spectral_ops.md)
                * [Other Operations](torch_math_operations_other_ops.md)
                * [BLAS and LAPACK Operations](torch_math_operations_blas_lapack_ops.md)
        * [torch.Tensor](tensors.md)
        * [Tensor Attributes](tensor_attributes.md)
        * [数据类型信息](type_info.md)
        * [torch.sparse](sparse.md)
        * [torch.cuda](cuda.md)
        * [torch.Storage](storage.md)
        * [torch.nn](nn.md)
        * [torch.nn.functional](nn_functional.md)
        * [torch.nn.init](nn_init.md)
        * [torch.optim](optim.md)
        * [Automatic differentiation package - torch.autograd](autograd.md)
        * [Distributed communication package - torch.distributed](distributed.md)
        * [Probability distributions - torch.distributions](distributions.md)
        * [Torch Script](jit.md)
        * [多进程包 - torch.multiprocessing](multiprocessing.md)
        * [torch.utils.bottleneck](bottleneck.md)
        * [torch.utils.checkpoint](checkpoint.md)
        * [torch.utils.cpp_extension](docs_cpp_extension.md)
        * [torch.utils.data](data.md)
        * [torch.utils.dlpack](dlpack.md)
        * [torch.hub](hub.md)
        * [torch.utils.model_zoo](model_zoo.md)
        * [torch.onnx](onnx.md)
        * [Distributed communication package (deprecated) - torch.distributed.deprecated](distributed_deprecated.md)
    * [torchvision 参考](docs_torchvision_ref.md)
        * [torchvision.datasets](torchvision_datasets.md)
        * [torchvision.models](torchvision_models.md)
        * [torchvision.transforms](torchvision_transforms.md)
        * [torchvision.utils](torchvision_utils.md)

