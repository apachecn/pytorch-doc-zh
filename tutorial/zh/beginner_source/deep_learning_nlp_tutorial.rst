针对NLP的Pytorch深度学习
**********************************
**作者**: `Robert Guthrie <https://github.com/rguthrie3/DeepLearningForNLPInPytorch>`_

本教程将带你浏览基于Pytorch深度学习编程的核心思想.其中很多思想(例如计算图形抽象化以及自动求导)
并不是Pytorch特有的,他们和任何深度学习工具包都是相关的.

本教程针对那些从未在任何深度学习框架下编写过代码的人(例如TensorFlow,Theano, Keras, Dynet),并
专注于NLP.它提出了应用知识中NLP的核心问题:词性标注,语言建模等.它同样提出了在AI入门级别熟悉神经
网络(例如Russel和Norvig的书).通常情况下, 这些课程包括了基于前馈神经网络的基本的反向传播算法,
并使你了解到它们是线性和非线性组成的链条.本教程目的使你开始编写深度学习代码并给你首要必备的知识.

提示一下, 这仅关乎于 *模型* , 并非是数据.针对所有模型,我仅仅提出了一些低纬度的例子
以便于你可以观察当训练时权重的变化.如果你有一些真实数据去尝试,可以将本教程中模型
复制下并将数据应用到模型上.



.. toctree::
    :hidden:

    /beginner/nlp/pytorch_tutorial
    /beginner/nlp/deep_learning_tutorial
    /beginner/nlp/word_embeddings_tutorial
    /beginner/nlp/sequence_models_tutorial
    /beginner/nlp/advanced_tutorial


.. galleryitem:: /beginner/nlp/pytorch_tutorial.py
    :intro: 所有的深度学习都是在张量上计算的,其中张量是一个可以被超过二维索引的矩阵的一般化.

.. galleryitem:: /beginner/nlp/deep_learning_tutorial.py
    :intro: 深度学习以巧妙的方式将 non-linearities 和 linearities 组合在一起. non-linearities 的引入允许

.. galleryitem:: /beginner/nlp/word_embeddings_tutorial.py
    :intro: 单词嵌入是真实数字的密集向量,在你的词汇表中每一个单词都是. 在 NLP 中, 通常情况下, 您的特性就是

.. galleryitem:: /beginner/nlp/sequence_models_tutorial.py
    :intro: 之前我们已经学过了许多的前馈网络. 所谓前馈网络, 就是网络中不会保存状态. 然而有时这并不是我们想要的效果. 

.. galleryitem:: /beginner/nlp/advanced_tutorial.py
    :intro: 动态 VS 静态深度学习工具集. Pytorch 是一个动态神经网络工具包. 


.. raw:: html

    <div style='clear:both'></div>
