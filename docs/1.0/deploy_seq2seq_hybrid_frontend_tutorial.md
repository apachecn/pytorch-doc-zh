# 混合前端的seq2seq模型部署

> 译者：[cangyunye](https://github.com/cangyunye)
>
> 校对者：[FontTian](https://github.com/fonttian)

**作者:** [Matthew Inkawhich](https://github.com/MatthewInkawhich)

本教程将介绍如何是`seq2seq`模型转换为PyTorch可用的前端混合Torch脚本。 我们要转换的模型是来自于聊天机器人教程 [Chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html). 你可以把这个教程当做Chatbot tutorial的第二篇章,并且部署你的预训练模型，或者你也可以依据本文使用我们采取的预训练模型。就后者而言，你可以从原始的Chatbot tutorial参考更详细的数据预处理，模型理论和定义以及模型训练。

## 什么是混合前端(Hybrid Frontend）?

在一个基于深度学习项目的研发阶段, 使用像PyTorch这样**即时**`eager`、命令式的界面进行交互能带来很大便利。 这使用户能够在使用Python数据结构、控制流操作、打印语句和调试实用程序时通过熟悉的、惯用的Python脚本编写。尽管即时性界面对于研究和试验应用程序是一个有用的工具，但是对于生产环境中部署模型时，使用**基于图形**`graph-based`的模型表示将更加适用的。 一个延迟的图型展示意味着可以优化，比如无序执行操作，以及针对高度优化的硬件架构的能力。 此外，基于图形的表示支持框架无关的模型导出。PyTorch提供了将即时模式的代码增量转换为Torch脚本的机制，Torch脚本是一个在Python中的静态可分析和可优化的子集，Torch使用它来在Python运行时独立进行深度学习。

在Torch中的`torch.jit`模块可以找到将即时模式的PyTorch程序转换为Torch脚本的API。 这个模块有两个核心模式用于将即时模式模型转换为Torch脚本图形表示: **跟踪**`tracing` 以及 **脚本化**`scripting`。`torch.jit.trace` 函数接受一个模块或者一个函数和一组示例的输入，然后通过函数或模块运行输入示例，同时跟跟踪遇到的计算步骤，然后输出一个可以展示跟踪流程的基于图形的函数。**跟踪**`Tracing`对于不涉及依赖于数据的控制流的直接的模块和函数非常有用，就比如标准的卷积神经网络。然而，如果一个有数据依赖的if语句和循环的函数被跟踪，则只记录示例输入沿执行路径调用的操作。换句话说，控制流本身并没有被捕获。要将带有数据依赖控制流的模块和函数进行转化，已提供了一个脚本化机制。脚本显式地将模块或函数代码转换为Torch脚本，包括所有可能的控制流路径。 如需使用脚本模式`script mode`， 要确定继承了 `torch.jit.ScriptModule`基本类 (取代`torch.nn.Module`) 并且增加 `torch.jit.script` 装饰器到你的Python函数或者 `torch.jit.script_method` 装饰器到你的模块方法。使用脚本化的一个警告是，它只支持Python的一个受限子集。要获取与支持的特性相关的所有详细信息，请参考 Torch Script [language reference](https://pytorch.org/docs/master/jit.html)。为了达到最大的灵活性，可以组合Torch脚本的模式来表示整个程序，并且可以增量地应用这些技术。

[![workflow](https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/eb1caa84cb095a30117f2a78a3aa69e4.jpg)](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/eb1caa84cb095a30117f2a78a3aa69e4.jpg)

## 致谢

本篇教程灵感来自如下资源：

1. Yuan-Kuei Wu’s pytorch-chatbot implementation: <https://github.com/ywk991112/pytorch-chatbot>
2. Sean Robertson’s practical-pytorch seq2seq-translation example: <https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation>
3. FloydHub’s Cornell Movie Corpus preprocessing code: <https://github.com/floydhub/textutil-preprocess-cornell-movie-corpus>

## 预备环境

首先，我们应该要导入所需的模块以及设置一些常量。如果你想使用自己的模型，需要保证`MAX_LENGTH`常量设置正确。提醒一下，这个常量定义了在训练过程中允许的最大句子长度以及模型能够产生的最大句子长度输出。

```python
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

MAX_LENGTH = 10 # Maximum sentence length

# Default word tokens
PAD_token = 0 # Used for padding short sentences
SOS_token = 1 # Start-of-sentence token
EOS_token = 2 # End-of-sentence token
```

## 模型概览

正如前文所言，我们使用的[sequence-to-sequence](https://arxiv.org/abs/1409.3215) (seq2seq) 模型。这种类型的模型用于输入是可变长度序列的情况，我们的输出也是一个可变长度序列它不一定是一对一输入映射。`seq2seq` 模型由两个递归神经网络(RNNs)组成：编码器 **encoder**和解码器**decoder**.

[![model](https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/32a87cf8d0353ceb0037776f833b92a7.jpg)](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/32a87cf8d0353ceb0037776f833b92a7.jpg)

图片来源: <https://jeddy92.github.io/JEddy92.github.io/ts_seq2seq_intro/>

### 编码器(Encoder)

编码器RNN在输入语句中每次迭代一个标记(例如单词)，每次步骤输出一个“输出”向量和一个“隐藏状态”向量。”隐藏状态“向量在之后则传递到下一个步骤，同时记录输出向量。编码器将序列中每个坐标代表的文本转换为高维空间中的一组坐标，解码器将使用这些坐标为给定的任务生成有意义的输出。

### 解码器(Decoder)

解码器RNN以逐个令牌的方式生成响应语句。它使用来自于编码器的文本向量和内部隐藏状态来生成序列中的下一个单词。它继续生成单词，直到输出表示句子结束的EOS语句。我们在解码器中使用专注机制[attention mechanism](https://arxiv.org/abs/1409.0473)来帮助它在输入的某些部分生成输出时"保持专注"。对于我们的模型，我们实现了 [Luong et al](https://arxiv.org/abs/1508.04025)等人的“全局关注`Global attention`”模块，并将其作为解码模型中的子模块。

## 数据处理

尽管我们的模型在概念上处理标记序列，但在现实中，它们与所有机器学习模型一样处理数字。在这种情况下，在训练之前建立的模型词汇表中的每个单词都映射到一个整数索引。我们使用`Voc`对象来包含从单词到索引的映射，以及词汇表中的单词总数。我们将在运行模型之前加载对象。

此外，为了能够进行评估，我们必须提供一个处理字符串输入的工具。`normalizeString`函数将字符串中的所有字符转换为小写，并删除所有非字母字符。`indexesFromSentence`函数接受一个单词的句子并返回相应的单词索引序列。

```python
class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count SOS, EOS, PAD

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
```

## 编码器定义

我们通过`torch.nn.GRU`模块实现编码器的RNN。本模块接受一批语句(嵌入单词的向量)的输入，它在内部遍历这些句子，每次一个标记，计算隐藏状态。我们将这个模块初始化为双向的，这意味着我们有两个独立的GRUs:一个按时间顺序遍历序列，另一个按相反顺序遍历序列。我们最终返回这两个GRUs输出的和。由于我们的模型是使用批处理进行训练的，所以我们的`EncoderRNN`模型的`forward`函数需要一个填充的输入批处理。为了批量处理可变长度的句子，我们通过`MAX_LENGTH`令牌允许一个句子中支持的最大长度，并且批处理中所有小于`MAX_LENGTH`令牌的句子都使用我们专用的`PAD_token`令牌填充在最后。要使用带有PyTorch RNN模块的批量填充，我们必须把转发`forward`密令在调用`torch.nn.utils.rnn.pack_padded_sequence`和`torch.nn.utils.rnn.pad_packed_sequence`数据转换时进行打包。注意，`forward`函数还接受一个`input_length`列表，其中包含批处理中每个句子的长度。该输入在填充时通过`torch.nn.utils.rnn.pack_padded_sequence`使用。

### 混合前端笔记:

由于编码器的转发函数`forward`不包含任何依赖于数据的控制流，因此我们将使用**跟踪**`tracing`将其转换为脚本模式`script mode`。在跟踪模块时，我们可以保持模块定义不变。在运行评估之前，我们将在本文末尾初始化所有模型。

```python
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
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
```

## 解码专注模块定义

接下来，我们将定义我们的注意力模块(`Attn`)。请注意，此模块将用作解码器模型中的子模块。Luong等人考虑了各种“分数函数”`score functions`，它们取当前解码器RNN输出和整个编码器输出，并返回关注点“能值”`engergies`。这个关注能值张量`attension energies tensor`与编码器输出的大小相同，两者最终相乘，得到一个加权张量，其最大值表示在特定时间步长解码的查询语句最重要的部分。

```python
# Luong attention layer
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

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
```

## 解码器定义

类似于`EncoderRNN`，我们使用`torch.nn.GRU`模块作为我们的解码器RNN。然而，这一次我们使用单向GRU。需要注意的是，与编码器不同，我们将向解码器RNN每次提供一个单词。我们首先得到当前单词的嵌入并应用抛出功能[dropout](https://pytorch.org/docs/stable/nn.html?highlight=dropout#torch.nn.Dropout)。接下来，我们将嵌入和最后的隐藏状态转发给GRU，得到当前的GRU输出和隐藏状态。然后，我们使用Attn模块作为一个层来获得专注权重，我们将其乘以编码器的输出来获得我们的参与编码器输出。我们使用这个参与编码器输出作为文本`context`张量，它表示一个加权和，表示编码器输出的哪些部分需要注意。在这里，我们使用线性层`linear layer`和`softmax normalization `规范化来选择输出序列中的下一个单词。

### 混合前端笔记:

与`EncoderRNN`类似，此模块不包含任何依赖于数据的控制流。因此，在初始化该模型并加载其参数之后，我们可以再次使用跟踪`tracing`将其转换为Torch脚本。

```python
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
```

## 评估定义

### 贪婪搜索解码器(GreedySearchDecoder)

在聊天机器人教程中，我们使用`GreedySearchDecoder`模块来简化实际的解码过程。该模块将训练好的编码器和解码器模型作为属性，驱动输入语句(词索引向量)的编码过程，并一次一个词(词索引)迭代地解码输出响应序列

对输入序列进行编码很简单:只需将整个序列张量及其对应的长度向量转发给编码器。需要注意的是，这个模块一次只处理一个输入序列，而不是成批的序列。因此，当常数1用于声明张量大小时，它对应于批处理大小为1。要解码给定的解码器输出，我们必须通过解码器模型迭代地向前运行，该解码器模型输出softmax分数，该分数对应于每个单词在解码序列中是正确的下一个单词的概率。我们将`decoder_input`初始化为一个包含SOS_token的张量。在每次通过解码器之后，我们贪婪地将`softmax`概率最高的单词追加到`decoded_words`列表中。我们还使用这个单词作为下一个迭代的decoder_input`。如果``decoded_words`列表的长度达到`MAX_LENGTH`，或者预测的单词是`EOS_token`，那么解码过程将终止。

### 混合前端注释:

该模块的`forward`方法涉及到在每次解码一个单词的输出序列时，遍历/([0,max/_length]/)的范围。因此，我们应该使用脚本将这个模块转换为Torch脚本。与我们可以跟踪的编码器和解码器模型不同，我们必须对`GreedySearchDecoder`模块进行一些必要的更改，以便在不出错的情况下初始化对象。换句话说，我们必须确保我们的模块遵守脚本机制的规则，并且不使用Torch脚本包含的Python子集之外的任何语言特性。

为了了解可能需要的一些操作，我们将回顾聊天机器人教程中的`GreedySearchDecoder`实现与下面单元中使用的实现之间的区别。请注意，用红色突出显示的行是从原始实现中删除的行，而用绿色突出显示的行是新的。

[![diff](https://github.com/apachecn/pytorch-doc-zh/raw/master/docs/1.0/img/452204771a8c708918247913c14bdb7d.jpg)](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/img/452204771a8c708918247913c14bdb7d.jpg)

#### 变更事项:

- `nn.Module` -&gt; `torch.jit.ScriptModule`
  - 为了在模块上使用PyTorch的脚本化机制, 模型需要从 `torch.jit.ScriptModule`继承。
- 将 `decoder_n_layers` 追加到结构参数
  - 这种变化源于这样一个事实，即我们传递给这个模块的编码器和解码器模型将是`TracedModule`(非模块)的子模块。因此，我们无法使用`decoder.n_layers`访问解码器的层数。相反，我们对此进行计划，并在模块构建过程中传入此值。
- 将新属性作为常量保存
  - 在最初的实现中， 我们可以在`GreedySearchDecoder`的`forward`方法中自由地使用来自周围(全局)范围的变量. 然而，现在我们正在使用脚本，我们没有这种自由，因为脚本处理的设想4是我们不一定要保留Python对象，尤其是在导出时。 一个简单的解决方案是将全局作用域中的这些值作为属性存储到构造函数中的模块中， 并将它们添加到一个名为`__constants__`的特殊列表中，以便在`forward`方法中构造图形时将它们用作文本值。这种用法的一个例子在第19行，取代使用 `device` 和 `SOS_token` 全局值，我们使用常量属性 `self._device` 和 `self._SOS_token`。
- 将 `torch.jit.script_method` 装饰器添加到 `forward` 方法
  - 添加这个装饰器可以让JIT编译器知道它所装饰的函数应该是脚本化的。
- 强制 `forward` 方法的参数类型
  - 默认情况下，Torch脚本函数的所有参数都假定为张量。如果需要传递不同类型的参数，可以使用[PEP 3107](https://www.python.org/dev/peps/pep-3107/)中引入的函数类型注释。 此外，还可以使用`MyPy-style`类型的注释声明不同类型的参数(参见(see [doc](https://pytorch.org/docs/master/jit.html#types)))。
- 变更`decoder_input`的初始化
  - 在原有实现中，我们用`torch.LongTensor([[SOS_token]])`初始化了 `decoder_input` 的张量。 当脚本编写时,我们不允许像这样以一种文字方式初始化张量。 取而代之的是，我们可以用一个显式的torch函数，比如`torch.ones`来初始化我们的张量。这种情况下，我们可以很方便的复制标量 `decoder_input` 和通过将1乘以我们存在常量中的` SOS_token`的值 `self._SOS_token`得到的张量。

```python
class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    @torch.jit.script_method
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
```

### 输入评估

接下来，我们定义一些函数来计算输入。求值函数`evaluate`接受一个规范化字符串语句，将其处理为其对应的单词索引张量(批处理大小为1)，并将该张量传递给一个名为`searcher`的`GreedySearchDecoder`实例，以处理编码/解码过程。检索器返回输出的单词索引向量和一个分数张量，该张量对应于每个解码的单词标记的softmax分数。最后一步是使用`voc.index2word`将每个单词索引转换回其字符串表示形式。

我们还定义了两个函数来计算输入语句。`evaluateInput`函数提示用户输入，并计算输入。它持续请求另一次输入，直到用户输入“q”或“quit”。

`evaluateExample`函数只接受一个字符串输入语句作为参数，对其进行规范化、计算并输出响应。

```python
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
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
def evaluateInput(encoder, decoder, searcher, voc):
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
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

# Normalize input sentence and call evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
    print("> " + sentence)
    # Normalize sentence
    input_sentence = normalizeString(sentence)
    # Evaluate sentence
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    print('Bot:', ' '.join(output_words))
```

## 加载预训练参数

好的，是时候加载我们的模型了

### 使用托管模型

托管模型使用步骤:

1. 下载模型 [here](https://download.pytorch.org/models/tutorials/4000_checkpoint.tar).
2. 设置`loadFilename`变量作为下载的检查点文件的路径
3. 将`checkpoint = torch.load(loadFilename)` 行取消注释，表示托管模型在CPU上训练。

### 使用自己的模型

加载自己的预训练模型设计步骤:

1. 将`loadFilename`变量设置为希望加载的检查点文件的路径。注意，如果您遵循从chatbot tutorial中保存模型的协议，这会涉及更改`model_name`、`encoder_n_layers`、`decoder_n_layers`、`hidden_size`和`checkpoint_iter`(因为这些值在模型路径中使用到)。
2. 如果你在CPU上训练，确保你在 `checkpoint = torch.load(loadFilename)` 行打开了检查点。如果你在GPU 上训练，并且在CPU运行这篇教程，解除`checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))` 的注释。

### 混合前端的注释:

请注意，我们像往常一样初始化并将参数加载到编码器和解码器模型中。另外，在跟踪模型之前，我们必须调用`.to(device)`来设置模型的设备选项，调用`.eval()`来设置抛出层`dropout layer`为test mode。`TracedModule`对象不继承`to`或`eval`方法

```python
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
```

Out:

```
Building encoder and decoder ...
Models built and ready to go!

```

## 模型转换为 Torch 脚本

### 编码器

正如前文所述，要将编码器模型转换为Torch脚本，我们需要使用跟踪`Tracing`。跟踪任何需要通过模型的`forward`方法运行一个示例输入，以及跟踪数据相遇时的图形计算。编码器模型接收一个输入序列和一个长度相关的张量。因此，我们创建一个输入序列`test_seq`，配置合适的大小(MAX_LENGTH,1) 包含适当范围内的数值 $$[0,voc.num\_words]$$ 以及搭配的类型(int64)。我们还创建了`test_seq_length`标量，该标量实际包含与`test_seq`中单词数量对应的值。下一步是使用`torch.jit.trace`函数来跟踪模型。注意，我们传递的第一个参数是要跟踪的模块，第二个参数是模块`forward`方法的参数元组。

### 解码器

我们对解码器的跟踪过程与对编码器的跟踪过程相同。请注意，我们对traced_encoder的一组随机输入调用forward，以获得解码器所需的输出。这不是必需的，因为我们也可以简单地生成一个形状、类型和值范围正确的张量。这种方法是可行的，因为在我们的例子中，我们对张量的值没有任何约束，因为我们没有任何操作可能导致超出范围的输入出错。

### 贪婪搜索解码器(GreedySearchDecoder)

回想一下，由于存在依赖于数据的控制流，我们为搜索器模块编写了脚本。在脚本化的情况下，我们通过添加修饰符并确保实现符合脚本规则来预先完成转换工作。我们初始化脚本搜索器的方式与初始化未脚本化变量的方式相同。

```python
### Convert encoder model
# Create artificial inputs
test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words)
test_seq_length = torch.LongTensor([test_seq.size()[0]])
# Trace the model
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))

### Convert decoder model
# Create and generate artificial inputs
test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
# Trace the model
traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))

### Initialize searcher module
scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)

```

## 图形打印

现在我们的模型是Torch脚本形式的，我们可以打印每个模型的图形，以确保适当地捕获计算图形。因为`scripted_searcher`包含`traced_encoder`和`traced_decoder`，所以这些图将以内联方式打印

```python
print('scripted_searcher graph:\n', scripted_searcher.graph)

```

Out:

```python
scripted_searcher graph:
 graph(%input_seq : Tensor
      %input_length : Tensor
      %max_length : int
      %3 : Tensor
      %4 : Tensor
      %5 : Tensor
      %6 : Tensor
      %7 : Tensor
      %8 : Tensor
      %9 : Tensor
      %10 : Tensor
      %11 : Tensor
      %12 : Tensor
      %13 : Tensor
      %14 : Tensor
      %15 : Tensor
      %16 : Tensor
      %17 : Tensor
      %18 : Tensor
      %19 : Tensor
      %118 : Tensor
      %119 : Tensor
      %120 : Tensor
      %121 : Tensor
      %122 : Tensor
      %123 : Tensor
      %124 : Tensor
      %125 : Tensor
      %126 : Tensor
      %127 : Tensor
      %128 : Tensor
      %129 : Tensor
      %130 : Tensor) {
  %58 : int = prim::Constant[value=9223372036854775807](), scope: EncoderRNN
  %53 : float = prim::Constant[value=0](), scope: EncoderRNN
  %43 : float = prim::Constant[value=0.1](), scope: EncoderRNN/GRU[gru]
  %42 : int = prim::Constant[value=2](), scope: EncoderRNN/GRU[gru]
  %41 : bool = prim::Constant[value=1](), scope: EncoderRNN/GRU[gru]
  %36 : int = prim::Constant[value=6](), scope: EncoderRNN/GRU[gru]
  %34 : int = prim::Constant[value=500](), scope: EncoderRNN/GRU[gru]
  %25 : int = prim::Constant[value=4](), scope: EncoderRNN
  %24 : Device = prim::Constant[value="cpu"](), scope: EncoderRNN
  %21 : bool = prim::Constant[value=0](), scope: EncoderRNN/Embedding[embedding]
  %20 : int = prim::Constant[value=-1](), scope: EncoderRNN/Embedding[embedding]
  %90 : int = prim::Constant[value=0]()
  %94 : int = prim::Constant[value=1]()
  %input.7 : Float(10, 1, 500) = aten::embedding(%3, %input_seq, %20, %21, %21), scope: EncoderRNN/Embedding[embedding]
  %lengths : Long(1) = aten::to(%input_length, %24, %25, %21, %21), scope: EncoderRNN
  %input.1 : Float(10, 500), %batch_sizes : Long(10) = aten::_pack_padded_sequence(%input.7, %lengths, %21), scope: EncoderRNN
  %35 : int[] = prim::ListConstruct(%25, %94, %34), scope: EncoderRNN/GRU[gru]
  %hx : Float(4, 1, 500) = aten::zeros(%35, %36, %90, %24), scope: EncoderRNN/GRU[gru]
  %40 : Tensor[] = prim::ListConstruct(%4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19), scope: EncoderRNN/GRU[gru]
  %46 : Float(10, 1000), %encoder_hidden : Float(4, 1, 500) = aten::gru(%input.1, %batch_sizes, %hx, %40, %41, %42, %43, %21, %41), scope: EncoderRNN/GRU[gru]
  %49 : int = aten::size(%batch_sizes, %90), scope: EncoderRNN
  %max_seq_length : Long() = prim::NumToTensor(%49), scope: EncoderRNN
  %51 : int = prim::Int(%max_seq_length), scope: EncoderRNN
  %outputs : Float(10, 1, 1000), %55 : Long(1) = aten::_pad_packed_sequence(%46, %batch_sizes, %21, %53, %51), scope: EncoderRNN
  %60 : Float(10, 1, 1000) = aten::slice(%outputs, %90, %90, %58, %94), scope: EncoderRNN
  %65 : Float(10, 1, 1000) = aten::slice(%60, %94, %90, %58, %94), scope: EncoderRNN
  %70 : Float(10, 1!, 500) = aten::slice(%65, %42, %90, %34, %94), scope: EncoderRNN
  %75 : Float(10, 1, 1000) = aten::slice(%outputs, %90, %90, %58, %94), scope: EncoderRNN
  %80 : Float(10, 1, 1000) = aten::slice(%75, %94, %90, %58, %94), scope: EncoderRNN
  %85 : Float(10, 1!, 500) = aten::slice(%80, %42, %34, %58, %94), scope: EncoderRNN
  %encoder_outputs : Float(10, 1, 500) = aten::add(%70, %85, %94), scope: EncoderRNN
  %decoder_hidden.1 : Tensor = aten::slice(%encoder_hidden, %90, %90, %42, %94)
  %98 : int[] = prim::ListConstruct(%94, %94)
  %100 : Tensor = aten::ones(%98, %25, %90, %24)
  %decoder_input.1 : Tensor = aten::mul(%100, %94)
  %103 : int[] = prim::ListConstruct(%90)
  %all_tokens.1 : Tensor = aten::zeros(%103, %25, %90, %24)
  %108 : int[] = prim::ListConstruct(%90)
  %all_scores.1 : Tensor = aten::zeros(%108, %36, %90, %24)
  %all_scores : Tensor, %all_tokens : Tensor, %decoder_hidden : Tensor, %decoder_input : Tensor = prim::Loop(%max_length, %41, %all_scores.1, %all_tokens.1, %decoder_hidden.1, %decoder_input.1)
    block0(%114 : int, %188 : Tensor, %184 : Tensor, %116 : Tensor, %115 : Tensor) {
      %input.2 : Float(1, 1, 500) = aten::embedding(%118, %115, %20, %21, %21), scope: LuongAttnDecoderRNN/Embedding[embedding]
      %input.3 : Float(1, 1, 500) = aten::dropout(%input.2, %43, %21), scope: LuongAttnDecoderRNN/Dropout[embedding_dropout]
      %138 : Tensor[] = prim::ListConstruct(%119, %120, %121, %122, %123, %124, %125, %126), scope: LuongAttnDecoderRNN/GRU[gru]
      %hidden : Float(1, 1, 500), %decoder_hidden.2 : Float(2, 1, 500) = aten::gru(%input.3, %116, %138, %41, %42, %43, %21, %21, %21), scope: LuongAttnDecoderRNN/GRU[gru]
      %147 : Float(10, 1, 500) = aten::mul(%hidden, %encoder_outputs), scope: LuongAttnDecoderRNN/Attn[attn]
      %149 : int[] = prim::ListConstruct(%42), scope: LuongAttnDecoderRNN/Attn[attn]
      %attn_energies : Float(10, 1) = aten::sum(%147, %149, %21), scope: LuongAttnDecoderRNN/Attn[attn]
      %input.4 : Float(1!, 10) = aten::t(%attn_energies), scope: LuongAttnDecoderRNN/Attn[attn]
      %154 : Float(1, 10) = aten::softmax(%input.4, %94), scope: LuongAttnDecoderRNN/Attn[attn]
      %attn_weights : Float(1, 1, 10) = aten::unsqueeze(%154, %94), scope: LuongAttnDecoderRNN/Attn[attn]
      %159 : Float(1!, 10, 500) = aten::transpose(%encoder_outputs, %90, %94), scope: LuongAttnDecoderRNN
      %context.1 : Float(1, 1, 500) = aten::bmm(%attn_weights, %159), scope: LuongAttnDecoderRNN
      %rnn_output : Float(1, 500) = aten::squeeze(%hidden, %90), scope: LuongAttnDecoderRNN
      %context : Float(1, 500) = aten::squeeze(%context.1, %94), scope: LuongAttnDecoderRNN
      %165 : Tensor[] = prim::ListConstruct(%rnn_output, %context), scope: LuongAttnDecoderRNN
      %input.5 : Float(1, 1000) = aten::cat(%165, %94), scope: LuongAttnDecoderRNN
      %168 : Float(1000!, 500!) = aten::t(%127), scope: LuongAttnDecoderRNN/Linear[concat]
      %171 : Float(1, 500) = aten::addmm(%128, %input.5, %168, %94, %94), scope: LuongAttnDecoderRNN/Linear[concat]
      %input.6 : Float(1, 500) = aten::tanh(%171), scope: LuongAttnDecoderRNN
      %173 : Float(500!, 7826!) = aten::t(%129), scope: LuongAttnDecoderRNN/Linear[out]
      %input : Float(1, 7826) = aten::addmm(%130, %input.6, %173, %94, %94), scope: LuongAttnDecoderRNN/Linear[out]
      %decoder_output : Float(1, 7826) = aten::softmax(%input, %94), scope: LuongAttnDecoderRNN
      %decoder_scores : Tensor, %decoder_input.2 : Tensor = aten::max(%decoder_output, %94, %21)
      %186 : Tensor[] = prim::ListConstruct(%184, %decoder_input.2)
      %all_tokens.2 : Tensor = aten::cat(%186, %90)
      %190 : Tensor[] = prim::ListConstruct(%188, %decoder_scores)
      %all_scores.2 : Tensor = aten::cat(%190, %90)
      %decoder_input.3 : Tensor = aten::unsqueeze(%decoder_input.2, %90)
      -> (%41, %all_scores.2, %all_tokens.2, %decoder_hidden.2, %decoder_input.3)
    }
  %198 : (Tensor, Tensor) = prim::TupleConstruct(%all_tokens, %all_scores)
  return (%198);
}

```

## 运行结果评估

最后，我们将使用Torch脚本模型对聊天机器人模型进行评估。如果转换正确，模型的行为将与它们在即时模式表示中的行为完全相同。

默认情况下，我们计算一些常见的查询语句。如果您想自己与机器人聊天，取消对`evaluateInput`行的注释并让它旋转。

```python
# Evaluate examples
sentences = ["hello", "what's up?", "who are you?", "where am I?", "where are you from?"]
for s in sentences:
    evaluateExample(s, traced_encoder, traced_decoder, scripted_searcher, voc)

# Evaluate your input
#evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)

```

Out:

```python
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

```

## 保存模型

现在我们已经成功地将模型转换为Torch脚本，接下来将对其进行序列化，以便在非python部署环境中使用。为此，我们只需保存`scripted_searcher`模块，因为这是用于对聊天机器人模型运行推理的面向用户的接口。保存脚本模块时，使用`script_module.save(PATH)`代替`torch.save(model, PATH)`。

```python
scripted_searcher.save("scripted_chatbot.pth")

```
