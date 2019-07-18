# 在深度学习和 NLP 中使用 Pytorch

> 译者 [bruce1408](https://github.com/bruce1408)
>
> 校对者：[FontTian](https://github.com/fonttian)

**作者**: [Robert Guthrie](https://github.com/rguthrie3/DeepLearningForNLPInPytorch)

本文带您进入pytorch框架进行深度学习编程的核心思想。Pytorch的很多概念(比如计算图抽象和自动求导)并非它所独有的,和其他深度学习框架相关。

我写这篇教程是专门针对那些从未用任何深度学习框架(例如：Tensorflow, Theano, Keras, Dynet)编写代码而从事NLP领域的人。我假设你已经知道NLP领域要解决的核心问题：词性标注、语言模型等等。我也认为你通过[AI](http://aima.cs.berkeley.edu/)这本书中所讲的知识熟悉了神经网络达到了入门的级别。通常这些课程都会介绍反向传播算法和前馈神经网络，并指出它们是线性组合和非线性组合构成的链。本文在假设你已经有了这些知识的情况下，教你如何开始写深度学习代码。


注意这篇文章主要关于_models_，而不是数据。对于所有的模型，我只创建一些数据维度较小的测试示例以便你可以看到权重在训练过程中如何变化。如果你想要尝试一些真实数据，您有能力删除本示例中的模型并重新训练他们。

![https://pytorch.org/tutorials/_images/sphx_glr_pytorch_tutorial_thumb.png](img/ad16fc851a032d82abda756c6d96f5a6.jpg)

[Introduction to PyTorch](nlp/pytorch_tutorial.html#sphx-glr-beginner-nlp-pytorch-tutorial-py)

![https://pytorch.org/tutorials/_images/sphx_glr_deep_learning_tutorial_thumb.png](img/988692ea092d586b1d9352c724893e4f.jpg)

[Deep Learning with PyTorch](nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py)

![https://pytorch.org/tutorials/_images/sphx_glr_word_embeddings_tutorial_thumb.png](img/791b9958674d0128b50db5c03f618ef6.jpg)

[Word Embeddings: Encoding Lexical Semantics](nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py)

![https://pytorch.org/tutorials/_images/sphx_glr_sequence_models_tutorial_thumb.png](img/130f884f42a8ea020b8c3f40045eeb8b.jpg)

[Sequence Models and Long-Short Term Memory Networks](nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py)

![https://pytorch.org/tutorials/_images/sphx_glr_advanced_tutorial_thumb.png](img/3a74e6a54afb73f4fb8d7353ab2e9914.jpg)

[Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py)

