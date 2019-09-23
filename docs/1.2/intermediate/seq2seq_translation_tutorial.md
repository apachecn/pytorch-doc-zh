# NLP从头：用序列到序列网络和翻译注意

**作者** ：[肖恩·罗伯逊](https://github.com/spro/practical-pytorch)

这是在做“NLP的划痕”，在这里我们写我们自己的类和函数对数据进行预处理，以尽我们的NLP建模任务的第三次也是最后的教程。我们希望你完成本教程，你会继续学习如何
torchtext 可以处理很多这样的预处理为你的三个教程立即这个下面了。

在这个项目中，我们将教神经网络翻译从法语译成英语。

    
    
    [KEY: > input, = target, < output]
    
    > il est en train de peindre un tableau .
    = he is painting a picture .
    < he is painting a picture .
    
    > pourquoi ne pas essayer ce vin delicieux ?
    = why not try that delicious wine ?
    < why not try that delicious wine ?
    
    > elle n est pas poete mais romanciere .
    = she is not a poet but a novelist .
    < she not not a poet but a novelist .
    
    > vous etes trop maigre .
    = you re too skinny .
    < you re all alone .
    

......以不同程度的成功。

这是由[序列序网](https://arxiv.org/abs/1409.3215)的简单而有力的想法，其中两个递归神经网络共同努力，一个序列变换到另一个成为可能。编码器网络冷凝的输入序列到载体中，和一个解码器网络展开该载体导入一个新的序列。

![](img/seq2seq.png)

为了改善已在此模型中，我们将使用一个[注意机制](https://arxiv.org/abs/1409.0473)，它可以让解码器学会关注在输入序列的特定范围。

**建议读：**

我假设你已经至少安装PyTorch，知道Python和理解张量：

  * [ https://pytorch.org/ [HTG1对于安装说明](https://pytorch.org/)
  * [ 深，PyTorch学习：60分钟的闪电战 ](../beginner/deep_learning_60min_blitz.html)得到普遍开始PyTorch
  * 与实施例 对于宽和深概述[ 学习PyTorch](../beginner/pytorch_with_examples.html)
  * [ PyTorch为前Torch 用户 ](../beginner/former_torchies_tutorial.html)如果你是前者的LuaTorch 用户

这也将是有益的了解序列具有Sequence网络以及它们是如何工作：

  * [学习使用RNN编码器 - 解码器对统计机器翻译短语表征](https://arxiv.org/abs/1406.1078)
  * [顺序以序列与神经网络的学习](https://arxiv.org/abs/1409.3215)
  * 通过共同学习来调整和翻译[神经机器翻译](https://arxiv.org/abs/1409.0473)
  * [一种神经会话模型](https://arxiv.org/abs/1506.05869)

用字符级RNN 和[ NLP From Scratch的分类名称：生成的名称用字级，你还可以找到[ NLP前面的教程从零开始RNN
](char_rnn_classification_tutorial.html)有益，因为这些概念是非常类似于编码器和解码器模型，分别。](char_rnn_generation_tutorial.html)

而对于更多的，阅读介绍这些主题的论文：

  * [学习使用RNN编码器 - 解码器对统计机器翻译短语表征](https://arxiv.org/abs/1406.1078)
  * [顺序以序列与神经网络的学习](https://arxiv.org/abs/1409.3215)
  * 通过共同学习来调整和翻译[神经机器翻译](https://arxiv.org/abs/1409.0473)
  * [一种神经会话模型](https://arxiv.org/abs/1506.05869)

**需求**

    
    
    from __future__ import unicode_literals, print_function, division
    from io import open
    import unicodedata
    import string
    import re
    import random
    
    import torch
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

## 加载数据文件

该项目的数据是一组成千上万的英语法语翻译对。

[开放式数据堆栈交换](https://opendata.stackexchange.com/questions/3888/dataset-of-
sentences-translated-into-many-languages)这个问题向我指出开放翻译网站[ https://tatoeba.org/
](https://tatoeba.org/)具有可在[
https://tatoeba.org/下载英/下载](https://tatoeba.org/eng/downloads) \-
更好的是，有人却分裂的语言对成单个文本的额外的工作文件的位置：[ https://www.manythings.org/anki/
](https://www.manythings.org/anki/)

英国法国对由于过大的回购协议包括，因此在继续之前下载到`数据/ CHI-fra.txt  [HTG3。该文件是翻译对制表符分隔列表：`

    
    
    I am cold.    J'ai froid.
    

Note

从[此处](https://download.pytorch.org/tutorial/data.zip)下载数据，并将其解压到当前目录。

类似于在字符级RNN教程使用的字符编码，我们将表示一种语言作为一热载体，或除了单一一个（这个词的索引处）的零向量巨每个单词。相比几十个可能在语言中存在的人物，有很多很多的话，那么编码向量大得多。不过，我们会欺骗了一下，修剪数据，每种语言只用几千字。

![](img/word-encoding.png)

我们需要一个唯一索引每字为以后网络的投入和目标使用。要跟踪的这一切，我们将使用名为`郎 `一个辅助类，其中有字→性指数（HTG4]  word2index
）和索引→分词（`index2word`）词典，以及每个字`的计数word2count`使用稍后取代罕见词语。

    
    
    SOS_token = 0
    EOS_token = 1
    
    
    class Lang:
        def __init__(self, name):
            self.name = name
            self.word2index = {}
            self.word2count = {}
            self.index2word = {0: "SOS", 1: "EOS"}
            self.n_words = 2  # Count SOS and EOS
    
        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)
    
        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1
    

这些文件都在Unicode中，为了简化，我们将转向Unicode字符以ASCII，使一切小写和修剪大部分标点符号。

    
    
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    
    # Lowercase, trim, and remove non-letter characters
    
    
    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
    

为了读取数据文件，我们将文件分割成线，然后分割线分成两人一组。这些文件都是英语→其他语言，所以如果我们想从其他语言→英语翻译我加入了`反向
`标志，以扭转对。

    
    
    def readLangs(lang1, lang2, reverse=False):
        print("Reading lines...")
    
        # Read the file and split into lines
        lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
            read().strip().split('\n')
    
        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)
    
        return input_lang, output_lang, pairs
    

由于有 _很多_
举例句，我们希望快速培训的东西，我们会修剪数据设置为仅相对较短和简单的句子。在这里，最大长度为10个字（包括标点符号结束），我们正在筛选到转化为形式的句子“我”或“他”等（占更早替换撇号）。

    
    
    MAX_LENGTH = 10
    
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )
    
    
    def filterPair(p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)
    
    
    def filterPairs(pairs):
        return [pair for pair in pairs if filterPair(pair)]
    

准备好数据的全过程：

  * 阅读文本文件，并分割成线，分割线分成两人一组
  * 由长度和内容正常化文本，过滤器
  * 使Word列出了从成对的句子

    
    
    def prepareData(lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs
    
    
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print(random.choice(pairs))
    

日期：

    
    
    Reading lines...
    Read 135842 sentence pairs
    Trimmed to 10599 sentence pairs
    Counting words...
    Counted words:
    fra 4345
    eng 2803
    ['je ne suis pas embarrassee .', 'i m not embarrassed .']
    

## 所述Seq2Seq模型

甲回归神经网络，或RNN，是操作上的序列，并且使用其自己的输出作为后续步骤输入的网络。

A
[序列到序列网络](https://arxiv.org/abs/1409.3215)，或seq2seq网络，或[编码器解码器网络](https://arxiv.org/pdf/1406.1078v3.pdf)，是由两个RNNs的模型称为编码器和解码器。编码器读取的输入序列和输出的单个载体，和解码器读取矢量以产生一个输出序列。

![](img/seq2seq.png)

不同于序列预测与单个RNN，其中每个输入对应于输出时，seq2seq模型可以让我们从序列长度和顺序，这使得它非常适合在两种语言之间的翻译。

考虑句子“JE NE PAS猪Le Chat
Noir酒店”→“我不是黑猫”。大部分的输入句子的单词在输出句子的直接翻译，但在稍微不同的顺序，例如“聊天比诺”和“黑猫”。因为“NE /
PAS”建设也有在输入句子多了一个字。这将是很难直接从输入字的顺序产生正确的翻译。

用seq2seq模型的编码器创建的单个载体，其在理想情况下，编码的输入序列的“意义”成单个载体 - 在句子在某些N维空间中的单个点。

### 编码器

一个seq2seq网络的编码器是RNN输出用于从所述输入语句的每一个字的一些值。对于每个输入字在编码器输出向量和一个隐藏的状态，并且使用隐藏状态用于下一个输入字。

![](img/encoder-network.png)

    
    
    class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(EncoderRNN, self).__init__()
            self.hidden_size = hidden_size
    
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
    
        def forward(self, input, hidden):
            embedded = self.embedding(input).view(1, 1, -1)
            output = embedded
            output, hidden = self.gru(output, hidden)
            return output, hidden
    
        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
    

### 解码器

解码器是另一RNN，是以编码器输出向量（一个或多个），并输出单词序列来创建翻译。

#### 简单解码器

在最简单的seq2seq解码器，我们只使用了编码器的最后一个输出。有时这最后的输出被称为 _上下文向量，因为它编码从整个序列上下文_
。这个上下文矢量用作解码器的初始隐蔽状态。

在解码的每一个步骤，所述解码器被给定的输入令牌和隐藏状态。初始输入令牌是启动的字符串`& LT ; SOS & GT ;
`标记，并且所述第一隐藏状态是上下文向量（编码器的最后一个隐藏的状态）。

![](img/decoder-network.png)

    
    
    class DecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size):
            super(DecoderRNN, self).__init__()
            self.hidden_size = hidden_size
    
            self.embedding = nn.Embedding(output_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=1)
    
        def forward(self, input, hidden):
            output = self.embedding(input).view(1, 1, -1)
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
            output = self.softmax(self.out(output[0]))
            return output, hidden
    
        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
    

我鼓励你培养和观察这种模式的结果，但为了节省空间，我们将直行的黄金和引入注意机制。

#### 注意解码器

如果只有上下文矢量在编码器和解码器之间传递，即单个载体携带编码整个句子的负担。

注意允许解码器网络以“专注”在编码器的输出的用于解码器自身的输出中的每一个步骤的不同部分。首先，我们计算一组 _关注权重_
的。这些将由编码器输出矢量相乘以产生一个加权组合。的结果（称为`在代码attn_applied
`）应当包含关于输入序列中的特定部分的信息，从而有助于解码器选择合适的输出字。

![](https://i.imgur.com/1152PYf.png)

计算所述关注的权重与另一种前馈层`经办人
`完成后，使用该解码器的输入和隐藏状态作为输入。因为在训练数据各种规模的句子，实际创建和培养这一层，我们必须选择一个最高刑期的长度（输入长度​​，编码器输出），它可以应用到。最大长度的句子将用全部的注意力权重，而较短的句子只会使用前几个。

![](img/attention-decoder-network.png)

    
    
    class AttnDecoderRNN(nn.Module):
        def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
            super(AttnDecoderRNN, self).__init__()
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.dropout_p = dropout_p
            self.max_length = max_length
    
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.dropout = nn.Dropout(self.dropout_p)
            self.gru = nn.GRU(self.hidden_size, self.hidden_size)
            self.out = nn.Linear(self.hidden_size, self.output_size)
    
        def forward(self, input, hidden, encoder_outputs):
            embedded = self.embedding(input).view(1, 1, -1)
            embedded = self.dropout(embedded)
    
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                     encoder_outputs.unsqueeze(0))
    
            output = torch.cat((embedded[0], attn_applied[0]), 1)
            output = self.attn_combine(output).unsqueeze(0)
    
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
    
            output = F.log_softmax(self.out(output[0]), dim=1)
            return output, hidden, attn_weights
    
        def initHidden(self):
            return torch.zeros(1, 1, self.hidden_size, device=device)
    

Note

还有其他的，通过使用相对位置的方法解决长度不限形式的关注。阅读关于在[的有效途径，以诚为本注意神经机器翻译](https://arxiv.org/abs/1508.04025)“当地关注”。

## 培训

### 准备训练数据

为了训练，为每一对，我们需要输入张量（在输入句子中的词索引）和目标张量（在目标句中的指标）。在创建这些载体，我们将追加EOS令牌两个序列。

    
    
    def indexesFromSentence(lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]
    
    
    def tensorFromSentence(lang, sentence):
        indexes = indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    
    
    def tensorsFromPair(pair):
        input_tensor = tensorFromSentence(input_lang, pair[0])
        target_tensor = tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)
    

### 培养模式

为了训练我们通过编码器运行输入句子，并跟踪每一个输出的和最新的隐藏状态。然后，解码器被给出的`& LT ; SOS & GT ;
`令牌作为其第一输入端，和的最后一个隐藏状态编码器作为其第一个隐藏的状态。

“教师迫使”是使用真正的目标输出作为每一个输入，而不是使用该解码器的猜测作为下一个输入的概念。用老师强迫使其收敛快，但[HTG0当训练的网络被利用，则可能出现不稳定[HTG1。

你可以观察到，与相干语法阅读而是从正确的翻译徘徊远老师强制网络输出 -
直觉告诉我已经学会代表输出语法，可以“拿起”的意思，一旦老师告诉它的前几话，但它已经无法正常学习了如何从摆在首位翻译创建了一句。

因为自由的PyTorch的autograd给我们，我们可以随意选用教师强制或不与简单的if语句。转动`teacher_forcing_ratio
`为使用更多。

    
    
    teacher_forcing_ratio = 0.5
    
    
    def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
        encoder_hidden = encoder.initHidden()
    
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
    
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
        loss = 0
    
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
    
        decoder_input = torch.tensor([[SOS_token]], device=device)
    
        decoder_hidden = encoder_hidden
    
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing
    
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
    
                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break
    
        loss.backward()
    
        encoder_optimizer.step()
        decoder_optimizer.step()
    
        return loss.item() / target_length
    

这是一个辅助功能打印时间已过，估计剩余时间给出的当前时间和进度％。

    
    
    import time
    import math
    
    
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    
    def timeSince(since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    

整个训练过程是这样的：

  * 启动计时器
  * 初始化优化和规范
  * 创建集训练对
  * 启动空损失阵列密谋

然后，我们调用`训练 `很多时候，偶尔打印进度（实例％，时间为止，估计时间）和平均损失。

    
    
    def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
    
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        training_pairs = [tensorsFromPair(random.choice(pairs))
                          for i in range(n_iters)]
        criterion = nn.NLLLoss()
    
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
    
            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
    
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
    
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    
        showPlot(plot_losses)
    

### 绘制结果

绘图与matplotlib完成，使用损失值的阵列``保存在训练plot_losses 。

    
    
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import matplotlib.ticker as ticker
    import numpy as np
    
    
    def showPlot(points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
    

## 评价

评估主要是一样的训练，但没有目标，所以我们干脆喂解码器的预测回自己的每一步。每次它预测我们将它添加到输出串词，如果预测EOS原因，我们停在那里。我们还存放了解码器的注意输出，用于显示更高版本。

    
    
    def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()
    
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]
    
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
    
            decoder_hidden = encoder_hidden
    
            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)
    
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])
    
                decoder_input = topi.squeeze().detach()
    
            return decoded_words, decoder_attentions[:di + 1]
    

我们可以从训练集评估随机的句子，并打印出输入，目标和输出做出一些主观质量的判断：

    
    
    def evaluateRandomly(encoder, decoder, n=10):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
    

## 培训和评估

有了这些辅助功能（它看起来像额外的工作，但它可以更容易地运行多个实验），我们实际上可以初始化网络，并开始训练。

请记住，输入句子被大量过滤。对于这个小数据集，我们可以使用256个隐藏节点和一个GRU层相对较小的网络。在MacBook
CPU上约40分钟后，我们会得到一些合理的结果。

Note

如果你运行这个笔记本，你可以训练，中断内核，评估和后继续训练。注释，其中编码器和解码器被初始化线并再次运行`trainIters`。

    
    
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    
    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
    

  * ![img/sphx_glr_seq2seq_translation_tutorial_001.png](img/sphx_glr_seq2seq_translation_tutorial_001.png)
  * ![img/sphx_glr_seq2seq_translation_tutorial_002.png](img/sphx_glr_seq2seq_translation_tutorial_002.png)

Out:

    
    
    1m 44s (- 24m 25s) (5000 6%) 2.8246
    3m 25s (- 22m 14s) (10000 13%) 2.2712
    5m 6s (- 20m 26s) (15000 20%) 1.9838
    6m 45s (- 18m 34s) (20000 26%) 1.6913
    8m 24s (- 16m 48s) (25000 33%) 1.5066
    10m 4s (- 15m 7s) (30000 40%) 1.3337
    11m 45s (- 13m 26s) (35000 46%) 1.1914
    13m 26s (- 11m 45s) (40000 53%) 1.0690
    15m 7s (- 10m 4s) (45000 60%) 0.9474
    16m 49s (- 8m 24s) (50000 66%) 0.8926
    18m 31s (- 6m 44s) (55000 73%) 0.7832
    20m 15s (- 5m 3s) (60000 80%) 0.7254
    21m 58s (- 3m 22s) (65000 86%) 0.6642
    23m 39s (- 1m 41s) (70000 93%) 0.5810
    25m 20s (- 0m 0s) (75000 100%) 0.5430
    
    
    
    evaluateRandomly(encoder1, attn_decoder1)
    

Out:

    
    
    > je n ai pas peur du tout .
    = i m not at all afraid .
    < i m not at all afraid . <EOS>
    
    > je suis ici .
    = i am here .
    < i m here here . <EOS>
    
    > il est ici pour moi .
    = he s here for me .
    < he is here for me . <EOS>
    
    > il est respecte par tout le monde .
    = he is respected by everyone .
    < he is respected by everybody . <EOS>
    
    > j en ai fini .
    = i m done with it .
    < i m done with it . <EOS>
    
    > je ne suis pas l entraineur .
    = i m not the coach .
    < i m not the criminal . <EOS>
    
    > je suis bon .
    = i am good .
    < i m good . <EOS>
    
    > je pars .
    = i m going .
    < i m going . <EOS>
    
    > j ai la baraka .
    = i m very fortunate .
    < i m very fortunate . <EOS>
    
    > tu en fais partie .
    = you re part of this .
    < you re part of this . <EOS>
    

### 注意可视化

注意机制的一个有用特性是它高度可解释的输出。因为它是用于加权输入序列的特定编码器输出，可想而知寻找其中，所述网络被聚焦在每个时间步长最多。

你可以简单地运行`plt.matshow（关注）HTG2] `看到显示的注意输出作为基质，与列在输入步骤和行是输出的步骤：

    
    
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, "je suis trop froid .")
    plt.matshow(attentions.numpy())
    

![img/sphx_glr_seq2seq_translation_tutorial_003.png](img/sphx_glr_seq2seq_translation_tutorial_003.png)

为了更好的观看体验，我们会做的加入轴线和标签的额外工作：

    
    
    def showAttention(input_sentence, output_words, attentions):
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)
    
        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words)
    
        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
        plt.show()
    
    
    def evaluateAndShowAttention(input_sentence):
        output_words, attentions = evaluate(
            encoder1, attn_decoder1, input_sentence)
        print('input =', input_sentence)
        print('output =', ' '.join(output_words))
        showAttention(input_sentence, output_words, attentions)
    
    
    evaluateAndShowAttention("elle a cinq ans de moins que moi .")
    
    evaluateAndShowAttention("elle est trop petit .")
    
    evaluateAndShowAttention("je ne crains pas de mourir .")
    
    evaluateAndShowAttention("c est un jeune directeur plein de talent .")
    

  * ![img/sphx_glr_seq2seq_translation_tutorial_004.png](img/sphx_glr_seq2seq_translation_tutorial_004.png)
  * ![img/sphx_glr_seq2seq_translation_tutorial_005.png](img/sphx_glr_seq2seq_translation_tutorial_005.png)
  * ![img/sphx_glr_seq2seq_translation_tutorial_006.png](img/sphx_glr_seq2seq_translation_tutorial_006.png)
  * ![img/sphx_glr_seq2seq_translation_tutorial_007.png](img/sphx_glr_seq2seq_translation_tutorial_007.png)

Out:

    
    
    input = elle a cinq ans de moins que moi .
    output = she s five years younger than me . <EOS>
    input = elle est trop petit .
    output = she is too short . <EOS>
    input = je ne crains pas de mourir .
    output = i m not scared to die . <EOS>
    input = c est un jeune directeur plein de talent .
    output = he s a talented young director . <EOS>
    

## 练习

  * 用不同的数据集的尝试
    * 另一种语言对
    * 人类→机（例如IOT命令）
    * 聊天→响应
    * 问→答
  * 与预训练字的嵌入，如word2vec或手套更换的嵌入
  * 尝试用更多的层，更隐蔽单位和更多的句子。比较了训练时间和结果。
  * 如果您使用的翻译文件，其中对有两个相同的短语（`我 是 测试 \ T  我 是 测试 `），你可以用这个作为自动编码。尝试这个：
    * 列车为自动编码器
    * 仅保存了网络编码器
    * 从那里培训新的解码器进行翻译

**脚本的总运行时间：** （25分钟27.786秒）

[`Download Python source code:
seq2seq_translation_tutorial.py`](../_downloads/a96a2daac1918ec72f68233dfe3f2c47/seq2seq_translation_tutorial.py)

[`Download Jupyter notebook:
seq2seq_translation_tutorial.ipynb`](../_downloads/a60617788061539b5449701ae76aee56/seq2seq_translation_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../beginner/text_sentiment_ngrams_tutorial.html "Text
Classification with TorchText") [![](../_static/images/chevron-right-
orange.svg) Previous](char_rnn_generation_tutorial.html "NLP From Scratch:
Generating Names with a Character-Level RNN")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * NLP从无到有：用序列到序列网络和翻译注意
    * 加载数据文件
    * [HTG0所述Seq2Seq模型
      * 编码器
      * 解码器
        * 简单解码器
        * [HTG0注意力解码器
    * 培训
      * 准备的训练数据
      * 训练模型
      * 绘图结果
    * 评价
    * 训练和评价
      * 可视注意
    * 练习

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



