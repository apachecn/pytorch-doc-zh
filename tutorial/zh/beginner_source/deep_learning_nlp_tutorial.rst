针对NLP的Pytorch深度学习
**********************************
**作者**: `Robert Guthrie <https://github.com/rguthrie3/DeepLearningForNLPInPytorch>`_

本教程将带你浏览基于Pytorch深度学习编程的核心思想.其中很多思想(例如计算图形抽象化以及自动求导)
并不是Pytorch特有的,他们和任何深度学习工具包都是相关的.

本教程针对那些从未在任何深度学习框架下编写过代码的人(例如TensorFlow,Theano, Keras, Dynet),并
专注于NLP.它提出了应用知识中NLP的核心问题:词性标注,语言建模等.它同样提出了在AI入门级别熟悉神经
网络(例如Russel和Norvig的书).通常情况下, 这些课程包括了基于前馈神经网络的基本的反向传播算法,
并使你了解到它们是线性和非线性组成的链条.本教程目的使你开始编写深度学习代码并给你首要必备的知识.

提示一下，这仅关乎于 *模型* , 并非是数据.针对所有模型,我仅仅提出了一些低纬度的例子
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
    :intro: All of deep learning is computations on tensors, which are generalizations of a matrix that can be 

.. galleryitem:: /beginner/nlp/deep_learning_tutorial.py
    :intro: Deep learning consists of composing linearities with non-linearities in clever ways. The introduction of non-linearities allows

.. galleryitem:: /beginner/nlp/word_embeddings_tutorial.py
    :intro: Word embeddings are dense vectors of real numbers, one per word in your vocabulary. In NLP, it is almost always the case that your features are

.. galleryitem:: /beginner/nlp/sequence_models_tutorial.py
    :intro: At this point, we have seen various feed-forward networks. That is, there is no state maintained by the network at all. 

.. galleryitem:: /beginner/nlp/advanced_tutorial.py
    :intro: Dyanmic versus Static Deep Learning Toolkits. Pytorch is a *dynamic* neural network kit. 


.. raw:: html

    <div style='clear:both'></div>
