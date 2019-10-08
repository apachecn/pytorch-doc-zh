# <center>PyTorch 1.2 中文文档 & 教程</center>

![](../../docs/img/logo.svg)

<center>PyTorch 是一个针对深度学习, 并且使用 GPU 和 CPU 来优化的 tensor library (张量库)</center>
<br/>
<table>
  <tr align="center">
    <td colspan="3"><a title="Pytorch 1.2 中文版本" href="https://pytorch.apachecn.org/docs/1.2/" target="_blank"><font size="5">正在校验: 1.2 中文版本</font></a></td>
  </tr>
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
    * 入门
        * [PyTorch 深度学习: 60 分钟极速入门](beginner/deep_learning_60min_blitz.html)
        * [数据加载和处理教程](beginner/data_loading_tutorial.html)
        * [用例子学习 PyTorch](beginner/pytorch_with_examples.html)
        * [迁移学习教程](beginner/transfer_learning_tutorial.html)
        * [部署与TorchScript一个Seq2Seq模型](beginner/deploy_seq2seq_hybrid_frontend_tutorial.html)
        * [可视化模型，数据，和与训练TensorBoard](intermediate/tensorboard_tutorial.html)
        * [保存和加载模型](beginner/saving_loading_models.html)
        * [torch.nn 到底是什么？](beginner/nn_tutorial.html)
    * 图片
        * [TorchVision对象检测教程细化和微调](intermediate/torchvision_tutorial.html)
        * [微调Torchvision模型](beginner/finetuning_torchvision_models_tutorial.html)
        * [空间变压器网络教程](intermediate/spatial_transformer_tutorial.html)
        * [使用PyTorch进行神经网络传递](advanced/neural_style_tutorial.html)
        * [对抗性示例生成](beginner/fgsm_tutorial.html)
        * [DCGAN教程](beginner/dcgan_faces_tutorial.html)
    * 音频
        * [torchaudio教程](beginner/audio_preprocessing_tutorial.html)
    * 文本
        * [NLP从头：判断名称与字符级RNN](intermediate/char_rnn_classification_tutorial.html)
        * [NLP从头：生成名称与字符级RNN](intermediate/char_rnn_generation_tutorial.html)
        * [NLP从无到有：用序列到序列网络和翻译注意](intermediate/seq2seq_translation_tutorial.html)
        * [文本分类与TorchText ](beginner/text_sentiment_ngrams_tutorial.html)
        * [语言翻译与TorchText ](beginner/torchtext_translation_tutorial.html)
        * [序列到序列与nn.Transformer和TorchText建模](beginner/transformer_tutorial.html)
    * 强化学习
        * [强化学习（DQN）教程](intermediate/reinforcement_q_learning.html)
    * 在生产部署PyTorch模型
        * [1.部署PyTorch在Python经由REST API从Flask](intermediate/flask_rest_api_tutorial.html)
        * [2.介绍TorchScript](beginner/Intro_to_TorchScript_tutorial.html)
        * [3.装载++一个TorchScript模型在C ](advanced/cpp_export.html)
        * [4.（可选）从导出到PyTorch一个ONNX模型并使用ONNX运行时运行它](advanced/super_resolution_with_onnxruntime.html)
    * 并行和分布式训练
        * [1.型号并行最佳实践](intermediate/model_parallel_tutorial.html)
        * [2.入门分布式数据并行](intermediate/ddp_tutorial.html)
        * [3. PyTorch编写分布式应用](intermediate/dist_tuto.html)
        * [4.（高级）PyTorch 1.0分布式训练与Amazon AWS](beginner/aws_distributed_training_tutorial.html) 
    * 扩展PyTorch
        * [使用自定义 C++ 扩展算TorchScript ](advanced/torch_script_custom_ops.html)
        * [创建扩展使用numpy的和SciPy的](advanced/numpy_extensions_tutorial.html)
        * [自定义 C++ 和CUDA扩展](advanced/cpp_extension.html)
    * PyTorch在其他语言
        * [使用PyTorch C++ 前端](advanced/cpp_frontend.html)
* 中文文档
    * 注解
        * [自动求导机制](notes/autograd.html)
        * [广播语义](notes/broadcasting.html)
        * [CPU线程和TorchScript推理](notes/cpu_threading_torchscript_inference.html)
        * [CUDA语义](notes/cuda.html)
        * [扩展PyTorch](notes/extending.html)
        * [常见问题](notes/faq.html)
        * [对于大规模部署的特点](notes/large_scale_deployments.html)
        * [多处理最佳实践](notes/multiprocessing.html)
        * [重复性](notes/randomness.html)
        * [序列化语义](notes/serialization.html)
        * [Windows 常见问题](notes/windows.html)
    * 社区
        * [PyTorch贡献说明书](community/contribution_guide.html)
        * [PyTorch治理](community/governance.html)
        * [PyTorch治理 | 感兴趣的人](community/persons_of_interest.html)
    * 封装参考文献
        * [torch](torch.html)
        * [torch.Tensor](tensors.html)
        * [Tensor Attributes](tensor_attributes.html)
        * [Type Info](type_info.html)
        * [torch.sparse](sparse.html)
        * [torch.cuda](cuda.html)
        * [torch.Storage](storage.html)
        * [torch.nn](nn.html)
        * [torch.nn.functional](nn.functional.html)
        * [torch.nn.init](nn.init.html)
        * [torch.optim](optim.html)
        * [torch.autograd](autograd.html)
        * [torch.distributed](distributed.html)
        * [torch.distributions](distributions.html)
        * [torch.hub](hub.html)
        * [torch.jit](jit.html)
        * [torch.multiprocessing](multiprocessing.html)
        * [torch.random](random.html)
        * [torch.utils.bottleneck](bottleneck.html)
        * [torch.utils.checkpoint](checkpoint.html)
        * [torch.utils.cpp_extension](cpp_extension.html)
        * [torch.utils.data](data.html)
        * [torch.utils.dlpack](dlpack.html)
        * [torch.utils.model_zoo](model_zoo.html)
        * [torch.utils.tensorboard](tensorboard.html)
        * [torch.onnx](onnx.html)
        * [torch.__ config__](__config__.html)
    * torchvision Reference
        * [torchvision](torchvision/index.html)
    * torchaudio Reference
        * [torchaudio](https://pytorch.org/audio)
    * torchtext Reference
        * [torchtext](https://pytorch.org/text)
