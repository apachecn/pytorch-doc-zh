# 贡献指南

> 请您勇敢地去翻译和改进翻译。虽然我们追求卓越，但我们并不要求您做到十全十美，因此请不要担心因为翻译上犯错——在大部分情况下，我们的服务器已经记录所有的翻译，因此您不必担心会因为您的失误遭到无法挽回的破坏。（改编自维基百科）

可能有用的链接：

+   [1.0 英文文档](https://pytorch.org/docs/)
+   [1.0 英文教程](https://pytorch.org/tutorials/)
+   [0.3 中文教程 & 文档](https://pytorch.apachecn.org/docs/0.3/)
+   [0.4 中文文档](https://pytorch.apachecn.org/docs/0.4/)

负责人：

+   [飞龙](https://github.com/wizardforcel)：562826179
+   [片刻](https://github.com/jiangzhonglian)：529815144
+   [咸鱼](https://github.com/)：1034616238

## 章节列表

+   [Getting Started](docs/1.0/tut_getting_started.md)
    +   [Deep Learning with PyTorch: A 60 Minute Blitz](docs/1.0/deep_learning_60min_blitz.md)
        + [What is PyTorch?](docs/1.0/blitz_tensor_tutorial.md)
        + [Autograd: Automatic Differentiation](docs/1.0/blitz_autograd_tutorial.md)
        + [Neural Networks](docs/1.0/blitz_neural_networks_tutorial.md)
        + [Training a Classifier](docs/1.0/blitz_cifar10_tutorial.md)
        + [Optional: Data Parallelism](docs/1.0/blitz_data_parallel_tutorial.md)
    +   [Data Loading and Processing Tutorial](docs/1.0/data_loading_tutorial.md)
    +   [Learning PyTorch with Examples](docs/1.0/pytorch_with_examples.md)
    +   [Transfer Learning Tutorial](docs/1.0/transfer_learning_tutorial.md)
    +   [Deploying a Seq2Seq Model with the Hybrid Frontend](docs/1.0/deploy_seq2seq_hybrid_frontend_tutorial.md)
    +   [Saving and Loading Models](docs/1.0/saving_loading_models.md)
    +   [What is &lt;cite&gt;torch.nn&lt;/cite&gt; _really_?](docs/1.0/nn_tutorial.md)
+   [Image](docs/1.0/tut_image.md)
    +   [Finetuning Torchvision Models](docs/1.0/finetuning_torchvision_models_tutorial.md)
    +   [Spatial Transformer Networks Tutorial](docs/1.0/spatial_transformer_tutorial.md)
    +   [Neural Transfer Using PyTorch](docs/1.0/neural_style_tutorial.md)
    +   [Adversarial Example Generation](docs/1.0/fgsm_tutorial.md)
    +   [Transfering a Model from PyTorch to Caffe2 and Mobile using ONNX](docs/1.0/super_resolution_with_caffe2.md)
+   [Text](docs/1.0/tut_text.md)
    +   [Chatbot Tutorial](docs/1.0/chatbot_tutorial.md)
    +   [Generating Names with a Character-Level RNN](docs/1.0/char_rnn_generation_tutorial.md)
    +   [Classifying Names with a Character-Level RNN](docs/1.0/char_rnn_classification_tutorial.md)
    +   [Deep Learning for NLP with Pytorch](docs/1.0/deep_learning_nlp_tutorial.md)
        + [Introduction to PyTorch](docs/1.0/nlp_pytorch_tutorial.md)
        + [Deep Learning with PyTorch](docs/1.0/nlp_deep_learning_tutorial.md)
        + [Word Embeddings: Encoding Lexical Semantics](docs/1.0/nlp_word_embeddings_tutorial.md)
        + [Sequence Models and Long-Short Term Memory Networks](docs/1.0/nlp_sequence_models_tutorial.md)
        + [Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](docs/1.0/nlp_advanced_tutorial.md)
    +   [Translation with a Sequence to Sequence Network and Attention](docs/1.0/seq2seq_translation_tutorial.md)
+   [Generative](docs/1.0/tut_generative.md)
    +   [DCGAN Tutorial](docs/1.0/dcgan_faces_tutorial.md)
+   [Reinforcement Learning](docs/1.0/tut_reinforcement_learning.md)
    +   [Reinforcement Learning (DQN) Tutorial](docs/1.0/reinforcement_q_learning.md)
+   [Extending PyTorch](docs/1.0/tut_extending_pytorch.md)
    +   [Creating Extensions Using numpy and scipy](docs/1.0/numpy_extensions_tutorial.md)
    +   [Custom C++   and CUDA Extensions](docs/1.0/cpp_extension.md)
    +   [Extending TorchScript with Custom C++   Operators](docs/1.0/torch_script_custom_ops.md)
+   [Production Usage](docs/1.0/tut_production_usage.md)
    +   [Writing Distributed Applications with PyTorch](docs/1.0/dist_tuto.md)
    +   [PyTorch 1.0 Distributed Trainer with Amazon AWS](docs/1.0/aws_distributed_training_tutorial.md)
    +   [ONNX Live Tutorial](docs/1.0/ONNXLive.md)
    +   [Loading a PyTorch Model in C++](docs/1.0/cpp_export.md)
+   [PyTorch in Other Languages](docs/1.0/tut_other_language.md)
    +   [Using the PyTorch C++   Frontend](docs/1.0/cpp_frontend.md)

## 流程

### 一、认领

首先查看[整体进度](https://github.com/apachecn/pytorch-doc-zh/issues/274)，确认没有人认领了你想认领的章节。
 
然后回复 ISSUE，注明“章节 + QQ 号”（一定要留 QQ）。

### 二、翻译

可以合理利用翻译引擎（例如[谷歌](https://translate.google.cn/)），但一定要把它变得可读！

可以参照之前版本的中文文档，如果有用的话。

如果遇到格式问题，请随手把它改正。

### 三、提交

+   `fork` Github 项目
+   将译文放在`docs/1.0`文件夹下
+   `push`
+   `pull request`

请见 [Github 入门指南](https://github.com/apachecn/kaggle/blob/dev/docs/GitHub)。
