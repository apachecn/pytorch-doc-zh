欢迎阅读 PyTorch 中文教程
============================

要开始学习 PyTorch, 可以从我们的初学者教程开始.
一般从 :doc:`60 分钟极速入门教程 </beginner/deep_learning_60min_blitz>` 开始, 它可以让你快速的了解 PyTorch.
如果你喜欢通过例子来学习, 你会喜欢上这个
:doc:`/beginner/pytorch_with_examples` 教程.

如果您想通过 IPython / Jupyter 交互式地完成该教程,
每个教程都有一个 Jupyter Notebook 和 Python 源代码的下载链接.

我们还提供了很多高质量的例子, 涵盖图像分类, 无监督学习, 强化学习, 机器翻译以及许多其它的应用场景, 请访问 https://github.com/pytorch/examples/

你可以在 http://pytorch.apachecn.org/cn/docs/0.3.0 或通过内置的帮助找到 PyTorch API 和神经网络层的参考文档.
如果您希望该教程部分有所改进, 请打开一个 github issue 附上你的反馈: https://github.com/apachecn/pytorch

初学者教程
------------------

.. customgalleryitem::
   :figure: /_static/img/thumbnails/pytorch-logo-flat.png
   :tooltip: Understand PyTorch’s Tensor library and neural networks at a high level.
   :description: :doc:`/beginner/deep_learning_60min_blitz`

.. customgalleryitem::
   :tooltip: Understand similarities and differences between torch and pytorch.
   :figure: /_static/img/thumbnails/torch-logo.png
   :description: :doc:`/beginner/former_torchies_tutorial`

.. customgalleryitem::
   :tooltip: This tutorial introduces the fundamental concepts of PyTorch through self-contained examples.
   :figure: /_static/img/thumbnails/examples.png
   :description: :doc:`/beginner/pytorch_with_examples`

.. galleryitem:: beginner/transfer_learning_tutorial.py

.. galleryitem:: beginner/data_loading_tutorial.py

.. customgalleryitem::
    :tooltip: I am writing this tutorial to focus specifically on NLP for people who have never written code in any deep learning framework
    :figure: /_static/img/thumbnails/babel.jpg
    :description: :doc:`/beginner/deep_learning_nlp_tutorial`

.. raw:: html

    <div style='clear:both'></div>


.. toctree::
   :maxdepth: 2
   :hidden:
   :includehidden:
   :caption: 初学者教程

   beginner/deep_learning_60min_blitz
   beginner/former_torchies_tutorial
   beginner/pytorch_with_examples
   beginner/transfer_learning_tutorial
   beginner/data_loading_tutorial
   beginner/deep_learning_nlp_tutorial

中级教程
----------------------

.. galleryitem:: intermediate/char_rnn_classification_tutorial.py

.. galleryitem:: intermediate/char_rnn_generation_tutorial.py
  :figure: _static/img/char_rnn_generation.png

.. galleryitem:: intermediate/seq2seq_translation_tutorial.py
  :figure: _static/img/seq2seq_flat.png

.. galleryitem:: intermediate/reinforcement_q_learning.py
    :figure: _static/img/cartpole.gif

.. customgalleryitem::
   :tooltip: Writing Distributed Applications with PyTorch.
   :description: :doc:`/intermediate/dist_tuto`
   :figure: _static/img/distributed/DistPyTorch.jpg


.. galleryitem:: intermediate/spatial_transformer_tutorial.py


.. raw:: html

    <div style='clear:both'></div>

.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 中级教程

   intermediate/char_rnn_classification_tutorial
   intermediate/char_rnn_generation_tutorial
   intermediate/seq2seq_translation_tutorial
   intermediate/reinforcement_q_learning
   intermediate/dist_tuto
   intermediate/spatial_transformer_tutorial


高级教程
------------------

.. galleryitem:: advanced/neural_style_tutorial.py
    :intro: This tutorial explains how to implement the Neural-Style algorithm developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.

.. galleryitem:: advanced/numpy_extensions_tutorial.py

.. galleryitem:: advanced/super_resolution_with_caffe2.py

.. customgalleryitem::
   :tooltip: Implement custom extensions in C.
   :description: :doc:`/advanced/c_extension`


.. raw:: html

    <div style='clear:both'></div>


.. toctree::
   :maxdepth: 2
   :includehidden:
   :hidden:
   :caption: 高级教程

   advanced/neural_style_tutorial
   advanced/numpy_extensions_tutorial
   advanced/super_resolution_with_caffe2
   advanced/c_extension

.. toctree::
   :maxdepth: 1
   :caption: 项目相关

   项目贡献者 <project-contributors>
   组织学习交流群 <apachecn-learning-group>