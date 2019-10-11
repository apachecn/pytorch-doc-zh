# 利用 TorchScript 部署 Seq2Seq 模型

> **作者：** [Matthew Inkawhich](https://github.com/MatthewInkawhich)
> 
> 译者：[Foxerlee](https://github.com/FoxerLee)
>
> 校验：[Foxerlee](https://github.com/FoxerLee)

本教程已经更新以适配 pyTorch 1.2 版本。

本教程将逐步介绍使用 TorchScript API 将 sequence-to-sequence 模型转换为TorchScript 的过程。我们将转换的模型是[聊天机器人教程](https://pytorch.apachecn.org/docs/1.0/chatbot_tutorial.html)的 Chatbot 模型。您可以将本教程视为聊天机器人教程的“第 2 部分”，并部署自己的预训练模型，也可以从本文档开始使用我们提供的预训练模型。如果您选择使用我们提供的预训练模型，您也可以参考原始的聊天机器人教程，以获取有关数据预处理，模型理论和定义，以及模型训练的详细信息。

## 什么是 TorchScript？

在基于深度学习的项目的研究和开发阶段，能够与**及时**、命令行的界面（例如PyTorch的界面）进行交互是非常有利的。这使用户能够使用熟悉、惯用的 Python 编写 Python 的数据结构，控制流操作，print 语句和调试方法。尽管及时的界面对于研究和实验应用程序是一种有益的工具，但是当需要在生产环境中部署模型时，**基于图形**的模型表现将会更加适用。延迟的图形表示意味着可以进行无序执行等优化，并具有针对高度优化的硬件体系结构的能力。此外，基于图的表示形式还可以导出框架无关的模型。 PyTorch 提供了将及时模式代码增量转换为 TorchScript 的机制。TorchScript 是 Python 的静态可分析和可优化的子集，Torch 使用它以不依赖于 Python 而运行深度学习程序。

在 `torch.jit` 模块中可以找到将及时模式的 PyTorch 程序转换为 TorchScript 的 API。该模块中两种将及时模式模型转换为 TorchScript 图形表示形式的核心方式分别为：`tracing`--`追踪`和 `scripting`--`脚本`。`torch.jit.trace` 函数接受一个模块或函数以及一组示例的输入。然后通过输入的函数或模块运行输入示例，同时跟跟踪遇到的计算步骤，最后输出一个可以展示跟踪流程的基于图的函数。对于不涉及依赖数据的控制流的简单模块和功能（例如标准卷积神经网络），`tracing`--`追踪`非常有用。然而，如果一个有数据依赖的if语句和循环的函数被跟踪，则只记录示例输入沿执行路径调用的操作。换句话说，控制流本身并没有被捕获。为了转换包含依赖于数据的控制流的模块和功能，TorchScript 提供了 `scripting`--`脚本`机制。 `torch.jit.script` 函数/修饰器接受一个模块或函数，不需要示例输入。之后 `scripting`--`脚本` 显式化地将模型或函数转换为 TorchScript，包括所有控制流。使用脚本化的需要注意的一点是，它只支持 Python 的一个受限子集。因此您可能需要重写代码以使其与 TorchScript 语法兼容。

有关所有支持的功能的详细信息，请参阅[TorchScript 语言参考](https://pytorch.apachecn.org/docs/1.2/jit.html)。 为了提供最大的灵活性，您还可以将 `tracing`--`追踪`和  `scripting`--`脚本`模式混合在一起使用而表现整个程序，这种方式可以通过增量的形式实现。

![https://pytorch.org/tutorials/_images/pytorch_workflow.png](img/pytorch_workflow.png)

## 致谢

本教程的灵感来自以下来源：

  1. 袁阿贵吴pytorch - 聊天机器人实现：[ https://github.com/ywk991112/pytorch-chatbot ](https://github.com/ywk991112/pytorch-chatbot)
  2. 肖恩·罗伯逊的实际-pytorch seq2seq翻译例如：[ https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation ](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation)
  3. FloydHub的康奈尔电影语料库预处理代码：[ https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus ](https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus)

## 准备环境

首先，我们将导入所需的模块，并设置一些常量。如果您在使用自己的模型规划，是确保MAX_LENGTH 常数设置正确`
[HTG1。作为提醒，该恒定的训练和最大长度输出，该模型能够产生的过程中定义的最大允许句子长度。`

    
    
    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
    from __future__ import unicode_literals
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import re
    import os
    import unicodedata
    import numpy as np
    
    device = torch.device("cpu")
    
    
    MAX_LENGTH = 10  # Maximum sentence length
    
    # Default word tokens
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    

## 模型概述

如所提到的，我们使用的模型是[序列到序列](https://arxiv.org/abs/1409.3215)（seq2seq）模型。这种类型的模型中的情况下使用时，我们的输入是一个可变长度的序列，而我们的输出也不一定是输入的一一对一映射的可变长度的序列。甲seq2seq模型由该协同工作，二期复发神经网络（RNNs）组成：
**编码** 和a **解码器** 。

![model](img/seq2seq_ts.png)

图像源：[ https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/
](https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/)

### 编码器

通过输入句子一个令牌（例如字）编码器RNN迭代的时间，在每个时间步骤输出一个“输出”向量和“隐藏状态”载体。然后，将隐藏状态矢量被传递到下一个时间步长，而输出矢量被记录。该编码器将其转换看见在序列中的每个点为一组在高维空间中的点，其中解码器将使用以产生用于给定任务一个有意义的输出的情况下。

### 解码器

解码器RNN在令牌通过令牌方式产生响应句。它采用了编码器的上下文载体，以及内部隐藏的状态，以产生序列中的下一个单词。直到它输出 _EOS_token_
，表示句末它将继续产生字。我们使用[注意机制](https://arxiv.org/abs/1409.0473)在我们的解码器，以帮助它“注意”到输入的某些部分产生输出的时候。对于我们的模型，我们实现[陈德良等人。
](https://arxiv.org/abs/1508.04025)的‘全球关注’模块，并把它作为我们的解码模式的子模块。

## 数据处理

虽然我们的模型概念上的令牌序列的处理，在现实中，他们对付像所有的机器学习模型做数字。在这种情况下，模型中的词汇，这是训练之前建立的每一个字，被映射到一个整数索引。我们使用`
的Voc`对象包含的映射从字索引，以及在所述词汇字的总数。我们运行模型之前，我们将在稍后加载对象。

此外，为了让我们能够运行的评估，我们必须为我们处理字符串输入的工具。的`normalizeString
`函数的所有字符转换的字符串为小写，并删除所有非字母字符。的`indexesFromSentence`函数接受词的句子，并返回字索引的对应序列。

    
    
    class Voc:
        def __init__(self, name):
            self.name = name
            self.trimmed = False
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD
    
        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)
    
        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1
    
        # Remove words below a certain count threshold
        def trim(self, min_count):
            if self.trimmed:
                return
            self.trimmed = True
            keep_words = []
            for k, v in self.word2count.items():
                if v >= min_count:
                    keep_words.append(k)
    
            print('keep_words {} / {} = {:.4f}'.format(
                len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
            ))
            # Reinitialize dictionaries
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3 # Count default tokens
            for word in keep_words:
                self.addWord(word)
    
    
    # Lowercase and remove non-letter characters
    def normalizeString(s):
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
    
    
    # Takes string sentence, returns sentence of word indexes
    def indexesFromSentence(voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    

## 定义编码器

我们以实现我们的编码器的RNN的`torch.nn.GRU
`模块，我们一次仅进一批句子（字嵌入物的载体）和它在内部遍历句子一个令牌计算隐藏状态。我们初始化这个模块是双向的，这意味着我们有两个独立的灰鹤：一个按照时间顺序的序列进行迭代，并以相反的顺序另一种迭代。我们最终退掉这两丹顶鹤输出的总和。由于我们的模型是用配料的训练，我们的`
EncoderRNN`模型`转发 `函数需要填充输入批次。批量可变长度的句子，我们允许最多 _MAX_LENGTH_
在一个句子中的令牌，并在一批具有比 _减去所有句子MAX_LENGTH_ 令牌在我们的专用年底补齐 _PAD_token_ 令牌。要使用与PyTorch
RNN模块填充批次，我们必须缠上`torch.nn.utils.rnn.pack_padded_sequence`和`torch.nn直传通话。
utils.rnn.pad_packed_sequence`数据转换。请注意，`向前 `功能还需要一个`input_lengths
`列表，其中包含的每个句子的在批处理的长度。此输入由`torch.nn.utils.rnn.pack_padded_sequence
`功能时的填充使用。

### TorchScript备注：

由于编码器的`转发 `功能不包含任何数据有关的控制流程，我们将使用 **追踪**
将其转换为脚本模式。当跟踪模块，我们可以把模块定义原样。我们运行评估之前，我们将初始化所有车型对这一文件的末尾。

    
    
    class EncoderRNN(nn.Module):
        def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
            super(EncoderRNN, self).__init__()
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.embedding = embedding
    
            # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
            #   because our input size is a word embedding with number of features == hidden_size
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
    
        def forward(self, input_seq, input_lengths, hidden=None):
            # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
            # Convert word indexes to embeddings
            embedded = self.embedding(input_seq)
            # Pack padded batch of sequences for RNN module
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
            # Forward pass through GRU
            outputs, hidden = self.gru(packed, hidden)
            # Unpack padding
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
            # Sum bidirectional GRU outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
            # Return output and final hidden state
            return outputs, hidden
    

## 定义解码器的注意模块

接下来，我们将定义我们的注意力模块（`经办人
`）。请注意，此模块将被用来作为我们的解码器模型的子模块。陈德良等人。综合考虑各种“得分函数”，其取当前解码器输出RNN和整个编码器的输出，并返回注意“能量”。这种关注能量张量的大小与编码器输出相同，并且两个最终相乘，产生一个加权的张量，其最大的值表示查询句子的最重要的部分在解码的特定时间步长。

    
    
    # Luong attention layer
    class Attn(nn.Module):
        def __init__(self, method, hidden_size):
            super(Attn, self).__init__()
            self.method = method
            if self.method not in ['dot', 'general', 'concat']:
                raise ValueError(self.method, "is not an appropriate attention method.")
            self.hidden_size = hidden_size
            if self.method == 'general':
                self.attn = nn.Linear(self.hidden_size, hidden_size)
            elif self.method == 'concat':
                self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
                self.v = nn.Parameter(torch.FloatTensor(hidden_size))
    
        def dot_score(self, hidden, encoder_output):
            return torch.sum(hidden * encoder_output, dim=2)
    
        def general_score(self, hidden, encoder_output):
            energy = self.attn(encoder_output)
            return torch.sum(hidden * energy, dim=2)
    
        def concat_score(self, hidden, encoder_output):
            energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
            return torch.sum(self.v * energy, dim=2)
    
        def forward(self, hidden, encoder_outputs):
            # Calculate the attention weights (energies) based on the given method
            if self.method == 'general':
                attn_energies = self.general_score(hidden, encoder_outputs)
            elif self.method == 'concat':
                attn_energies = self.concat_score(hidden, encoder_outputs)
            elif self.method == 'dot':
                attn_energies = self.dot_score(hidden, encoder_outputs)
    
            # Transpose max_length and batch_size dimensions
            attn_energies = attn_energies.t()
    
            # Return the softmax normalized probability scores (with added dimension)
            return F.softmax(attn_energies, dim=1).unsqueeze(1)
    

## 定义解码器

类似于`EncoderRNN`，我们用我们的解码器的RNN的`torch.nn.GRU
`模块。然而这一次，我们使用了单向GRU。需要注意的是不同的编码器，我们将饲料解码器RNN一个词在一个时间是很重要的。我们通过获取当前单词的嵌入和应用[降](https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.Dropout)启动。接下来，我们转发的嵌入和最后的隐藏状态的GRU和获取当前GRU输出和隐藏状态。然后，我们用我们的`
经办人 `模块作为一个层，以获得关注的权重，这是我们通过编码器的输出，以获得我们的出席编码器输出繁殖。我们使用这个出席编码器输出作为我们的`背景
`张量，它代表的加权和指出哪些编码器的输出的部分要注意。从这里，我们使用线性层和SOFTMAX正常化选择在输出序列中的下一个单词。

    
    
    # TorchScript Notes:
    # ~~~~~~~~~~~~~~~~~~~~~~
    #
    # Similarly to the ``EncoderRNN``, this module does not contain any
    # data-dependent control flow. Therefore, we can once again use
    # **tracing** to convert this model to TorchScript after it
    # is initialized and its parameters are loaded.
    #
    
    class LuongAttnDecoderRNN(nn.Module):
        def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
            super(LuongAttnDecoderRNN, self).__init__()
    
            # Keep for reference
            self.attn_model = attn_model
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers
            self.dropout = dropout
    
            # Define layers
            self.embedding = embedding
            self.embedding_dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
            self.concat = nn.Linear(hidden_size * 2, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)
    
            self.attn = Attn(attn_model, hidden_size)
    
        def forward(self, input_step, last_hidden, encoder_outputs):
            # Note: we run this one step (word) at a time
            # Get embedding of current input word
            embedded = self.embedding(input_step)
            embedded = self.embedding_dropout(embedded)
            # Forward through unidirectional GRU
            rnn_output, hidden = self.gru(embedded, last_hidden)
            # Calculate attention weights from the current GRU output
            attn_weights = self.attn(rnn_output, encoder_outputs)
            # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            # Concatenate weighted context vector and GRU output using Luong eq. 5
            rnn_output = rnn_output.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((rnn_output, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            # Predict next word using Luong eq. 6
            output = self.out(concat_output)
            output = F.softmax(output, dim=1)
            # Return output and final hidden state
            return output, hidden
    

## 定义评价

### 贪婪搜索解码器

正如在聊天机器人教程中，我们使用了`GreedySearchDecoder
`模块以便于实际解码处理。该模块具有经训练的编码器和解码器模型作为属性，并驱动编码输入句子（字索引的矢量），并且迭代一次进行解码的输出响应序列中的一个字（字索引）的过程。

编码输入序列是直接的：简单地转发在整个序列张量及其相应的长度向量到`编码 `。要注意，此模块一次仅与一个输入序列涉及这一点很重要， **NOT**
序列的批次。因此，当恒定 **1**
用于声明张量的尺寸，这对应于1批量大小为解码给定解码器的输出，必须反复地向前运行通过我们的解码器模型，其输出SOFTMAX分数对应于每个字是所述解码序列以正确的下一个单词的概率。我们初始化`
decoder_input`一种含有 _SOS_token_ 的张量。每个后通过`解码器 `，我们 _贪婪_
具有最高SOFTMAX可能性的单词追加到`decoded_words`名单。我们也用这个词作为`decoder_input
`为下一次迭代。解码过程终止于：如果所述`decoded_words`列表已经达到的 _MAX_LENGTH_ 的长度，或者如果所预测的单词是
_EOS_token_ 。

### TorchScript备注：

的`向前 `该模块的方法，在一个解码输出序列中的一个字时涉及迭代过的 \（[0，最大值\ _length）的范围\）时间。正因为如此，我们应该用
**脚本** 这个模块转换为TorchScript。不像我们的编码器和解码器模型，我们可以追踪，我们必须为了没有错误初始化对象到`
GreedySearchDecoder
`模块的一些必要的修改。换句话说，我们必须确保我们的模块附着在TorchScript机制的规则，不使用的Python的子集TorchScript包含之外的任何语言功能。

为了获得可能需要一些操作的想法，我们将在从聊天机器人教程`GreedySearchDecoder
`执行和实施，我们在下面的电池使用之间的差异列表。请注意，行以红色突出显示从原来实行删除，线条突出显示为绿色线是新的。

![diff](img/diff.png)

#### 变更：

  * 新增`decoder_n_layers`在构造函数的参数
    * 这种变化从这样一个事实，我们通过给此模块的编码器和解码器模型将是`TracedModule`（未`模块 [HTG7子]）。因此，我们不能与`decoder.n_layers`访问层的解码器的数量。相反，我们考虑这一点，并在模块施工过程中通过此值。`
  * 保存好新的属性为常数
    * 在最初的实现，我们是自由的，我们的`GreedySearchDecoder`的`转发 `方法使用变量从周围的（全球）范围。然而，现在我们正在使用的脚本，我们没有这样的自由，与脚本的假设是，我们不一定能坚持到Python对象，出口时尤其如此。一个简单的解决方法是将这些值从全球范围的属性在构造函数中的模块存储，并把它们添加到一个名为`一个特殊列表__constants__`，使他们可以使用在`向前 `方法构建图时作为文字值。这种用法的一个例子是在新行19，在那里，而不是使用`装置 `和`SOS_token`全局值，我们使用我们的恒定属性`self._device`和`self._SOS_token`。
  * 执行类型的`向前 `方法参数
    * 默认情况下，在TorchScript函数的所有参数都假定为张量。如果我们需要通过不同类型的参数，我们可以使用函数类型注释如[ PEP 3107 ](https://www.python.org/dev/peps/pep-3107/)引入。此外，也可以使用声明MyPy风格类型注释不同类型的参数（见[ DOC ](https://pytorch.org/docs/master/jit.html#types)）。
  * 的`更改初始化decoder_input`
    * 在最初的实现，我们初始化我们的`decoder_input`与`torch.LongTensor（[SOS_token]） `张量。脚本时，我们是不允许的字面这样初始化张量。相反，我们可以初始化我们有一个明确的Torch 函数张量，如`torch.ones  [HTG11。在这种情况下，我们可以很容易地通过由存储在我们的SOS_token值乘以1复制标量`decoder_input`张量的常数`self._SOS_token`。`

    
    
    class GreedySearchDecoder(nn.Module):
        def __init__(self, encoder, decoder, decoder_n_layers):
            super(GreedySearchDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self._device = device
            self._SOS_token = SOS_token
            self._decoder_n_layers = decoder_n_layers
    
        __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']
    
        def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
            # Forward input through encoder model
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
            # Prepare encoder's final hidden layer to be first hidden input to the decoder
            decoder_hidden = encoder_hidden[:self._decoder_n_layers]
            # Initialize decoder input with SOS_token
            decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
            # Initialize tensors to append decoded words to
            all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
            all_scores = torch.zeros([0], device=self._device)
            # Iteratively decode one word token at a time
            for _ in range(max_length):
                # Forward pass through decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # Obtain most likely word token and its softmax score
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                # Record token and score
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, decoder_scores), dim=0)
                # Prepare current token to be next decoder input (add a dimension)
                decoder_input = torch.unsqueeze(decoder_input, 0)
            # Return collections of word tokens and scores
            return all_tokens, all_scores
    

### 评价输入

接下来，我们定义用于评价输入某些功能。的`评价 `函数将归一化的字符串的句子，它处理到其对应的字索引的张量（以1批量大小），并通过这个张量的`
GreedySearchDecoder`实例调用`搜索者
`来处理的编码/解码处理。搜索器返回输出字索引向量和对应于SOFTMAX分数为每个解码字令牌的得分张量。最后的步骤是使用`voc.index2word
`到每个字索引转换回它的字符串表示。

我们还定义了用于评价输入句子两种功能。的`evaluateInput
`函数提示的输入的用户，并且评估它。它会继续下去，直到用户输入“Q”或“退出”，要求另一输入。

的`evaluateExample`功能简单地采用一个字符串输入句子作为一个参数，它归一化，计算它，并打印的响应。

    
    
    def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words
    
    
    # Evaluate inputs from user input (stdin)
    def evaluateInput(searcher, voc):
        input_sentence = ''
        while(1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit': break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = evaluate(searcher, voc, input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))
    
            except KeyError:
                print("Error: Encountered unknown word.")
    
    # Normalize input sentence and call evaluate()
    def evaluateExample(sentence, searcher, voc):
        print("> " + sentence)
        # Normalize sentence
        input_sentence = normalizeString(sentence)
        # Evaluate sentence
        output_words = evaluate(searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        print('Bot:', ' '.join(output_words))
    

## 加载预训练参数

好了，它的时间来加载我们的模型！

### 使用托管模式

要加载托管模式：

  1. 下载模型[此处[HTG1。](https://download.pytorch.org/models/tutorials/4000_checkpoint.tar)
  2. 在`loadFilename`变量设置为路径下载的检查点文件。
  3. 离开`检查点 =  torch.load（loadFilename） `线路未注释，因为托管模型上CPU训练。

### 使用您自己的模型

加载您自己的预先训练模式：

  1. 在`loadFilename`变量设置为路径，要加载检查点文件。请注意，如果您是用于保存从聊天机器人教程模型中的约定，这可能涉及更改`MODEL_NAME`，`encoder_n_layers`，`decoder_n_layers`，`hidden_​​size`和`checkpoint_iter`（因为这些值在模型中使用路径）。
  2. 如果你培养了CPU的型号，请确保您正在使用`检查点 =  torch.load（loadFilename）HTG6] [HTG7打开检查站]线。如果你训练了GPU的模式，在CPU上运行本教程中，取消注释`检查点 =  torch.load（loadFilename， map_location = torch.device（ 'CPU'）） `线。`

### TorchScript备注：

请注意，我们初始化和负载参数到我们的编码器和解码器模组如常。如果您使用的跟踪模式（ torch.jit.trace
）为你的模型的某些部分，则必须调用。要（设备）设置模式和设备选项.eval（）来设置漏失层，以测试模式 **之前** 跟踪模型。  TracedModule
对象不继承`至 `或`EVAL
`的方法。由于在本教程中，我们仅使用脚本，而不是跟踪，我们只需要做到这一点，我们之前做评价（这是与我们在急切模式通常做的）。

    
    
    save_dir = os.path.join("data", "save")
    corpus_name = "cornell movie-dialogs corpus"
    
    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64
    
    # If you're loading your own model
    # Set checkpoint to load from
    checkpoint_iter = 4000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                             '{}_checkpoint.tar'.format(checkpoint_iter))
    
    # If you're loading the hosted model
    loadFilename = 'data/4000_checkpoint.tar'
    
    # Load model
    # Force CPU device options (to match tensors in this tutorial)
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc = Voc(corpus_name)
    voc.__dict__ = checkpoint['voc_dict']
    
    
    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    # Load trained model params
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()
    print('Models built and ready to go!')
    

日期：

    
    
    Building encoder and decoder ...
    Models built and ready to go!
    

## 转换模型TorchScript

### 编码器

正如前面提到的，到编码器模型TorchScript转换，我们使用
**脚本[HTG1。编码器模型接受一个输入序列和相应的长度张量。因此，我们创建一个示例输入序列张量`test_seq
`，这是适当的尺寸（MAX_LENGTH，1），包含在适当范围内 \（[0，VOC号码。 NUM \
_words）\），并且是适当的类型（int64类型）的。我们还创建了一个`test_seq_length`标量，它真实地包含对应于有多少话是在`
test_seq`的值。下一个步骤是使用`torch.jit.trace
`函数来追踪模型。请注意，我们传递的第一个参数是我们要跟踪的模块，第二是参数模块的`转发 `方法的元组。**

### 解码器

我们进行追踪解码器，因为我们没有编码器相同的过程。请注意，我们称之为前一组随机输入到traced_encoder得到我们需要的解码器的输出。这不是必须的，因为我们也可以简单地制作正确的形状，类型和值范围的张量。因为在我们的例子中，我们没有对张量的值，任何约束，因为我们没有可能在故障超出范围的输入的任何操作此方法是可行的。

### GreedySearchDecoder

回想一下，我们照本宣科我们的搜索模块由于数据相关控制流的存在。在脚本的情况下，我们做必要的语言更改以确保落实与TorchScript规定。我们初始化脚本搜索，我们将初始化一个未脚本变种同样的方式。

    
    
    ### Compile the whole greedy search model to TorchScript model
    # Create artificial inputs
    test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words).to(device)
    test_seq_length = torch.LongTensor([test_seq.size()[0]]).to(device)
    # Trace the model
    traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))
    
    ### Convert decoder model
    # Create and generate artificial inputs
    test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
    test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
    test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
    # Trace the model
    traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))
    
    ### Initialize searcher module by wrapping ``torch.jit.script``call
    scripted_searcher = torch.jit.script(GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers))
    

## 打印图形

现在，我们的模型在TorchScript形式，我们可以打印的每一个曲线图，以确保我们适当地捕获的计算图表。由于TorchScript让我们递归编译整个模型的层次结构和内联`
编码器 `和`解码器 `图到一个图，我们只需要打印 scripted_searcher 图

    
    
    print('scripted_searcher graph:\n', scripted_searcher.graph)
    

Out:

    
    
    scripted_searcher graph:
     graph(%self : ClassType<GreedySearchDecoder>,
          %input_seq.1 : Tensor,
          %input_length.1 : Tensor,
          %max_length.1 : int):
      %23 : bool? = prim::Constant()
      %21 : int? = prim::Constant()
      %161 : int = prim::Constant[value=9223372036854775807](), scope: EncoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:310:0
      %156 : float = prim::Constant[value=0](), scope: EncoderRNN # /opt/conda/lib/python3.6/site-packages/torch/nn/utils/rnn.py:322:0
      %146 : float = prim::Constant[value=0.1](), scope: EncoderRNN/GRU[gru] # /opt/conda/lib/python3.6/site-packages/torch/nn/modules/rnn.py:682:0
      %145 : int = prim::Constant[value=2](), scope: EncoderRNN/GRU[gru] # /opt/conda/lib/python3.6/site-packages/torch/nn/modules/rnn.py:682:0
      %144 : bool = prim::Constant[value=1](), scope: EncoderRNN/GRU[gru] # /opt/conda/lib/python3.6/site-packages/torch/nn/modules/rnn.py:682:0
      %138 : int = prim::Constant[value=6](), scope: EncoderRNN/GRU[gru] # /opt/conda/lib/python3.6/site-packages/torch/nn/modules/rnn.py:691:0
      %136 : int = prim::Constant[value=500](), scope: EncoderRNN/GRU[gru] # /opt/conda/lib/python3.6/site-packages/torch/nn/modules/rnn.py:691:0
      %127 : int = prim::Constant[value=4](), scope: EncoderRNN # /opt/conda/lib/python3.6/site-packages/torch/nn/utils/rnn.py:265:0
      %126 : Device = prim::Constant[value="cpu"](), scope: EncoderRNN # /opt/conda/lib/python3.6/site-packages/torch/nn/utils/rnn.py:265:0
      %123 : bool = prim::Constant[value=0](), scope: EncoderRNN/Embedding[embedding] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1467:0
      %122 : int = prim::Constant[value=-1](), scope: EncoderRNN/Embedding[embedding] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1467:0
      %12 : int = prim::Constant[value=0]() # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:560:26
      %14 : int = prim::Constant[value=1]() # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:560:26
      %4 : ClassType<EncoderRNN> = prim::GetAttr[name="encoder"](%self)
      %103 : ClassType<Embedding> = prim::GetAttr[name="embedding"](%4)
      %weight.3 : Tensor = prim::GetAttr[name="weight"](%103)
      %105 : ClassType<GRU> = prim::GetAttr[name="gru"](%4)
      %106 : Tensor = prim::GetAttr[name="weight_ih_l0"](%105)
      %107 : Tensor = prim::GetAttr[name="weight_hh_l0"](%105)
      %108 : Tensor = prim::GetAttr[name="bias_ih_l0"](%105)
      %109 : Tensor = prim::GetAttr[name="bias_hh_l0"](%105)
      %110 : Tensor = prim::GetAttr[name="weight_ih_l0_reverse"](%105)
      %111 : Tensor = prim::GetAttr[name="weight_hh_l0_reverse"](%105)
      %112 : Tensor = prim::GetAttr[name="bias_ih_l0_reverse"](%105)
      %113 : Tensor = prim::GetAttr[name="bias_hh_l0_reverse"](%105)
      %114 : Tensor = prim::GetAttr[name="weight_ih_l1"](%105)
      %115 : Tensor = prim::GetAttr[name="weight_hh_l1"](%105)
      %116 : Tensor = prim::GetAttr[name="bias_ih_l1"](%105)
      %117 : Tensor = prim::GetAttr[name="bias_hh_l1"](%105)
      %118 : Tensor = prim::GetAttr[name="weight_ih_l1_reverse"](%105)
      %119 : Tensor = prim::GetAttr[name="weight_hh_l1_reverse"](%105)
      %120 : Tensor = prim::GetAttr[name="bias_ih_l1_reverse"](%105)
      %121 : Tensor = prim::GetAttr[name="bias_hh_l1_reverse"](%105)
      %input.7 : Float(10, 1, 500) = aten::embedding(%weight.3, %input_seq.1, %122, %123, %123), scope: EncoderRNN/Embedding[embedding] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1467:0
      %lengths : Long(1) = aten::to(%input_length.1, %126, %127, %123, %123), scope: EncoderRNN # /opt/conda/lib/python3.6/site-packages/torch/nn/utils/rnn.py:265:0
      %input.1 : Float(10, 500), %batch_sizes : Long(10) = aten::_pack_padded_sequence(%input.7, %lengths, %123), scope: EncoderRNN # /opt/conda/lib/python3.6/site-packages/torch/nn/utils/rnn.py:275:0
      %137 : int[] = prim::ListConstruct(%127, %14, %136), scope: EncoderRNN/GRU[gru]
      %hx : Float(4, 1, 500) = aten::zeros(%137, %138, %12, %126, %123), scope: EncoderRNN/GRU[gru] # /opt/conda/lib/python3.6/site-packages/torch/nn/modules/rnn.py:691:0
      %143 : Tensor[] = prim::ListConstruct(%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121), scope: EncoderRNN/GRU[gru]
      %149 : Float(10, 1000), %150 : Float(4, 1, 500) = aten::gru(%input.1, %batch_sizes, %hx, %143, %144, %145, %146, %123, %144), scope: EncoderRNN/GRU[gru] # /opt/conda/lib/python3.6/site-packages/torch/nn/modules/rnn.py:682:0
      %152 : int = aten::size(%batch_sizes, %12), scope: EncoderRNN # /opt/conda/lib/python3.6/site-packages/torch/nn/utils/rnn.py:313:0
      %max_seq_length : Long() = prim::NumToTensor(%152), scope: EncoderRNN
      %154 : int = aten::Int(%max_seq_length), scope: EncoderRNN
      %outputs : Float(10, 1, 1000), %158 : Long(1) = aten::_pad_packed_sequence(%149, %batch_sizes, %123, %156, %154), scope: EncoderRNN # /opt/conda/lib/python3.6/site-packages/torch/nn/utils/rnn.py:322:0
      %163 : Float(10, 1, 1000) = aten::slice(%outputs, %12, %12, %161, %14), scope: EncoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:310:0
      %168 : Float(10, 1, 1000) = aten::slice(%163, %14, %12, %161, %14), scope: EncoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:310:0
      %173 : Float(10, 1!, 500) = aten::slice(%168, %145, %12, %136, %14), scope: EncoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:310:0
      %178 : Float(10, 1, 1000) = aten::slice(%outputs, %12, %12, %161, %14), scope: EncoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:310:0
      %183 : Float(10, 1, 1000) = aten::slice(%178, %14, %12, %161, %14), scope: EncoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:310:0
      %188 : Float(10, 1!, 500) = aten::slice(%183, %145, %136, %161, %14), scope: EncoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:310:0
      %190 : Float(10, 1, 500) = aten::add(%173, %188, %14), scope: EncoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:310:0
      %decoder_hidden.1 : Tensor = aten::slice(%150, %12, %12, %145, %14) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:560:26
      %19 : int[] = prim::ListConstruct(%14, %14)
      %24 : Tensor = aten::ones(%19, %127, %21, %126, %23) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:562:25
      %decoder_input.1 : Tensor = aten::mul(%24, %14) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:562:25
      %27 : int[] = prim::ListConstruct(%12)
      %all_tokens.1 : Tensor = aten::zeros(%27, %127, %21, %126, %23) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:564:22
      %35 : int[] = prim::ListConstruct(%12)
      %all_scores.1 : Tensor = aten::zeros(%35, %21, %21, %126, %23) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:565:22
      %all_tokens : Tensor, %all_scores : Tensor, %decoder_hidden : Tensor, %decoder_input : Tensor = prim::Loop(%max_length.1, %144, %all_tokens.1, %all_scores.1, %decoder_hidden.1, %decoder_input.1) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:567:9
        block0(%48 : int, %all_tokens.6 : Tensor, %all_scores.6 : Tensor, %decoder_hidden.5 : Tensor, %decoder_input.9 : Tensor):
          %49 : ClassType<LuongAttnDecoderRNN> = prim::GetAttr[name="decoder"](%self)
          %192 : ClassType<Embedding> = prim::GetAttr[name="embedding"](%49)
          %weight.1 : Tensor = prim::GetAttr[name="weight"](%192)
          %194 : ClassType<GRU> = prim::GetAttr[name="gru"](%49)
          %195 : Tensor = prim::GetAttr[name="weight_ih_l0"](%194)
          %196 : Tensor = prim::GetAttr[name="weight_hh_l0"](%194)
          %197 : Tensor = prim::GetAttr[name="bias_ih_l0"](%194)
          %198 : Tensor = prim::GetAttr[name="bias_hh_l0"](%194)
          %199 : Tensor = prim::GetAttr[name="weight_ih_l1"](%194)
          %200 : Tensor = prim::GetAttr[name="weight_hh_l1"](%194)
          %201 : Tensor = prim::GetAttr[name="bias_ih_l1"](%194)
          %202 : Tensor = prim::GetAttr[name="bias_hh_l1"](%194)
          %203 : ClassType<Linear> = prim::GetAttr[name="concat"](%49)
          %weight.2 : Tensor = prim::GetAttr[name="weight"](%203)
          %bias.1 : Tensor = prim::GetAttr[name="bias"](%203)
          %206 : ClassType<Linear> = prim::GetAttr[name="out"](%49)
          %weight : Tensor = prim::GetAttr[name="weight"](%206)
          %bias : Tensor = prim::GetAttr[name="bias"](%206)
          %input.2 : Float(1, 1, 500) = aten::embedding(%weight.1, %decoder_input.9, %122, %123, %123), scope: LuongAttnDecoderRNN/Embedding[embedding] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1467:0
          %input.3 : Float(1, 1, 500) = aten::dropout(%input.2, %146, %123), scope: LuongAttnDecoderRNN/Dropout[embedding_dropout] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:806:0
          %216 : Tensor[] = prim::ListConstruct(%195, %196, %197, %198, %199, %200, %201, %202), scope: LuongAttnDecoderRNN/GRU[gru]
          %hidden : Float(1, 1, 500), %224 : Float(2, 1, 500) = aten::gru(%input.3, %decoder_hidden.5, %216, %144, %145, %146, %123, %123, %123), scope: LuongAttnDecoderRNN/GRU[gru] # /opt/conda/lib/python3.6/site-packages/torch/nn/modules/rnn.py:679:0
          %225 : Float(10, 1, 500) = aten::mul(%hidden, %190), scope: LuongAttnDecoderRNN/Attn[attn] # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:344:0
          %227 : int[] = prim::ListConstruct(%145), scope: LuongAttnDecoderRNN/Attn[attn]
          %attn_energies : Float(10, 1) = aten::sum(%225, %227, %123, %21), scope: LuongAttnDecoderRNN/Attn[attn] # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:344:0
          %input.4 : Float(1!, 10) = aten::t(%attn_energies), scope: LuongAttnDecoderRNN/Attn[attn] # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:364:0
          %234 : Float(1, 10) = aten::softmax(%input.4, %14, %21), scope: LuongAttnDecoderRNN/Attn[attn] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1230:0
          %attn_weights : Float(1, 1, 10) = aten::unsqueeze(%234, %14), scope: LuongAttnDecoderRNN/Attn[attn] # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:367:0
          %239 : Float(1!, 10, 500) = aten::transpose(%190, %12, %14), scope: LuongAttnDecoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:428:0
          %context.1 : Float(1, 1, 500) = aten::bmm(%attn_weights, %239), scope: LuongAttnDecoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:428:0
          %rnn_output : Float(1, 500) = aten::squeeze(%hidden, %12), scope: LuongAttnDecoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:430:0
          %context : Float(1, 500) = aten::squeeze(%context.1, %14), scope: LuongAttnDecoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:431:0
          %245 : Tensor[] = prim::ListConstruct(%rnn_output, %context), scope: LuongAttnDecoderRNN
          %input.5 : Float(1, 1000) = aten::cat(%245, %14), scope: LuongAttnDecoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:432:0
          %248 : Float(1000!, 500!) = aten::t(%weight.2), scope: LuongAttnDecoderRNN/Linear[concat] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1369:0
          %251 : Float(1, 500) = aten::addmm(%bias.1, %input.5, %248, %14, %14), scope: LuongAttnDecoderRNN/Linear[concat] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1369:0
          %input.6 : Float(1, 500) = aten::tanh(%251), scope: LuongAttnDecoderRNN # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:433:0
          %253 : Float(500!, 7826!) = aten::t(%weight), scope: LuongAttnDecoderRNN/Linear[out] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1369:0
          %input : Float(1, 7826) = aten::addmm(%bias, %input.6, %253, %14, %14), scope: LuongAttnDecoderRNN/Linear[out] # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1369:0
          %259 : Float(1, 7826) = aten::softmax(%input, %14, %21), scope: LuongAttnDecoderRNN # /opt/conda/lib/python3.6/site-packages/torch/nn/functional.py:1230:0
          %decoder_scores.1 : Tensor, %decoder_input.3 : Tensor = aten::max(%259, %14, %123) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:571:45
          %66 : Tensor[] = prim::ListConstruct(%all_tokens.6, %decoder_input.3)
          %all_tokens.3 : Tensor = aten::cat(%66, %12) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:573:26
          %72 : Tensor[] = prim::ListConstruct(%all_scores.6, %decoder_scores.1)
          %all_scores.3 : Tensor = aten::cat(%72, %12) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:574:26
          %decoder_input.7 : Tensor = aten::unsqueeze(%decoder_input.3, %12) # /var/lib/jenkins/workspace/beginner_source/deploy_seq2seq_hybrid_frontend_tutorial.py:576:29
          -> (%144, %all_tokens.3, %all_scores.3, %224, %decoder_input.7)
      %80 : (Tensor, Tensor) = prim::TupleConstruct(%all_tokens, %all_scores)
      return (%80)
    

## 运行评价

最后，我们将运行使用TorchScript模型的聊天机器人模型的评价。如果正确地转换，该机型的行为，正是因为他们会在他们的渴望模式表示。

默认情况下，我们评估了几个常见的查询语句。如果你想与机器人聊天自己，取消对`evaluateInput`行，并给它一个旋转。

    
    
    # Use appropriate device
    scripted_searcher.to(device)
    # Set dropout layers to eval mode
    scripted_searcher.eval()
    
    # Evaluate examples
    sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
    for s in sentences:
        evaluateExample(s, scripted_searcher, voc)
    
    # Evaluate your input
    #evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)
    

Out:

    
    
    > hello
    Bot: hello .
    > what's up?
    Bot: i m going to get my car .
    > who are you?
    Bot: i m the owner .
    > where am I?
    Bot: in the house .
    > where are you from?
    Bot: south america .
    

## 保存模型

现在，我们已经成功地转换我们的模型TorchScript，我们将连载它在非Python的部署环境中使用。要做到这一点，我们可以简单地保存我们的`
scripted_searcher
`模块，因为这是运行推论反对聊天机器人模型面向用户的界面。当保存脚本模块，使用script_module.save（PATH），而不是torch.save（模型，PATH）。

    
    
    scripted_searcher.save("scripted_chatbot.pth")
    

**脚本的总运行时间：** （0分钟0.862秒）

[`Download Python source code:
deploy_seq2seq_hybrid_frontend_tutorial.py`](../_downloads/6dce8206a711b28b2b916bfd7de16bbc/deploy_seq2seq_hybrid_frontend_tutorial.py)

[`Download Jupyter notebook:
deploy_seq2seq_hybrid_frontend_tutorial.ipynb`](../_downloads/e7870c00b625a9c7c808f8fa7a88fcab/deploy_seq2seq_hybrid_frontend_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../intermediate/tensorboard_tutorial.html "Visualizing Models,
Data, and Training with TensorBoard") [![](../_static/images/chevron-right-
orange.svg) Previous](transfer_learning_tutorial.html "Transfer Learning
Tutorial")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * 部署与TorchScript一个Seq2Seq模型
    * 什么是TorchScript？ 
    * 致谢
    * 准备环境
    * 模型概述
      * 编码
      * 解码器
    * 数据处理
    * 定义编码器
      * TorchScript备注：
    * 定义解码器的注意模块
    * 定义解码器
    * 定义评价
      * 贪婪搜索解码器
      * TorchScript备注：
        * 变更：
      * 评价输入
    * 载入预训练的参数
      * 使用托管模式
      * 使用自己的模型
      * TorchScript备注：
    * 转换模型TorchScript 
      * 编码
      * 解码器
      * GreedySearchDecoder 
    * 打印图形
    * 运行评价
    * 保存模型

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



