
 NLP 从头开始​​：使用序列到序列网络和注意力进行翻译
 [¶](#nlp-from-scratch-translation-with-a-sequence-to-sequence-network-and-attention “此标题的永久链接”）
========================================================================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/seq2seq_translation_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>




**作者** 
 :
 [肖恩·罗伯逊](https://github.com/spro)




 这是从头开始 “NLP 的第三个也是最后一个教程，我们
编写自己的类和函数来预处理数据以执行我们的 NLP
建模任务。我们希望您完成本教程后’
能够继续学习

 torchtext

 如何在紧随本教程之后的
三个教程中为您处理大部分预处理工作。




 在这个项目中，我们将教一个神经网络将
法语翻译成英语。






```
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

```




 … 取得了不同程度的成功。




 这是通过
 [序列到序列网络](https://arxiv.org/abs/1409.3215) 的简单但强大的想法成为可能，其中两个
循环神经网络协同工作来进行变换一个序列到另一个序列。编码器网络将输入序列压缩为向量，
解码器网络将该向量展开为新序列。




![](https://pytorch.org/tutorials/_images/seq2seq.png)


 为了改进这个模型，我们’将使用
 [注意力机制](https://arxiv.org/abs/1409.0473)
，它让解码器
学习关注特定范围的输入序列。




**推荐阅读：**




 我假设您至少已经安装了 PyTorch，了解 Python，并且
了解张量：



* <https://pytorch.org/>
 安装说明
* [使用 PyTorch 进行深度学习：60 分钟闪电战](../beginner/deep_learning_60min_blitz.html)
 一般开始使用 PyTorch\ n* [通过示例学习 PyTorch](../beginner/pytorch_with_examples.html)
 进行广泛而深入的概述
* [针对前 Torch 用户的 PyTorch](../beginner/former_torchies_tutorial.html)
 如果您是前 Lua Torch 用户



 了解序列到序列网络及其工作原理也很有用：



* [使用 RNN 编码器-解码器进行统计机器翻译学习短语表示](https://arxiv.org/abs/1406.1078)
* [使用神经网络进行序列到序列学习](https://arxiv.org /abs/1409.3215)
* [通过联合学习对齐和翻译的神经机器翻译](https://arxiv.org/abs/1409.0473)
* [神经会话模型](https://arxiv.组织/abs/1506.05869）



 您还可以找到之前的教程：
 [NLP From Scratch：使用字符级 RNN 分类名称](char_rnn_classification_tutorial.html)
 和
 [NLP From Scratch：使用字符级 RNN 生成名称] (char_rnn_ Generation_tutorial.html)
 很有帮助，因为这些概念分别与编码器和解码器
模型非常相似。




**要求**






```
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```





 正在加载数据文件
 [¶](#loading-data-files "固定链接到此标题")
--------------------------------------------------------------------------- -



 该项目的数据是一组数千个英语到法语的翻译对。




[关于开放数据堆栈的这个问题
Exchange](https://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages)
 向我指出了开放翻译网站
 <https://tatoeba.org/>
 可以在
下载
 <https://tatoeba.org/eng/downloads>
 - 更好
yet，有人做了额外的分割工作将语言对放入
个文本文件中：
 <https://www.manythings.org/anki/>




 英语到法语对太大，无法包含在存储库中，因此
请在继续之前下载到
 `data/eng-fra.txt`
。该文件是一个制表符
分隔的翻译对列表:






```
I am cold.    J'ai froid.

```





 没有10



 从
 [此处](https://download.pytorch.org/tutorial/data.zip) 下载数据并将其解压到当前目录。





 与字符级 RNN
 教程中使用的字符编码类似，我们将把语言中的每个单词表示为一个单热
向量，或除单个 1 之外的由 0 组成的巨型向量（在索引处） 
这个词）。与语言中可能存在的数十个字符相比，单词的数量要多得多，因此编码向量要大得多。然而，我们会稍微作弊并修剪数据，以便每种语言只使用
几千个单词。




![](https://pytorch.org/tutorials/_images/word-encoding.png)


 我们’将需要每个单词一个唯一的索引，以用作稍后网络的输入和目标。为了跟踪所有这些，我们将使用一个名为 
 `Lang`
 的帮助器类，它具有单词 → 索引 (
 `word2index`
 ) 和索引 → word\ n(
 `index2word`
 ) 个字典，以及每个单词的计数
 `word2count`
，稍后将用于替换稀有单词。






```
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

```




 这些文件都是 Unicode 格式的，为了简化，我们将 Unicode
字符转换为 ASCII，将所有内容变为小写，并修剪大部分
标点符号。






```
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
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

```




 要读取数据文件，我们将文件分成行，然后将行分成对。这些文件都是英语 → 其他语言，因此，如果我们
想要从其他语言 → 英语翻译，我添加了
 `reverse`
 标志来反转这些对。






```
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('')

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

```




 由于有
 *很多* 
 的例句，并且我们希望
快速训练一些东西，\xe2\x80\x99 将把数据集修剪为仅相对较短和
简单的句子。这里最大长度为 10 个单词（包括结尾标点符号），我们\xe2\x80\x99 重新过滤翻译为

形式\xe2\x80\x9cI am\xe2\x80\x9d 或\xe2\x80\ 的句子x9c他是\xe2\x80\x9d等（考虑到撇号已被替换
）。






```
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

```




 准备数据的完整过程是：



* 读取文本文件并分成行，将行分成对
* 规范文本，按长度和内容过滤
* 从成对的句子中创建单词列表





```
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

```






```
Reading lines...
Read 135842 sentence pairs
Trimmed to 11445 sentence pairs
Counting words...
Counted words:
fra 4601
eng 2991
['tu preches une convaincue', 'you re preaching to the choir']

```






 Seq2Seq 模型
 [¶](#the-seq2seq-model "永久链接到此标题")
--------------------------------------------------------------------------- -



 循环神经网络 (RNN) 是一种对序列进行操作并使用其自己的输出作为后续步骤的输入的网络。




 A
 [序列到序列网络](https://arxiv.org/abs/1409.3215) 
 ，或
seq2seq网络，或
 [编码器解码器
网络](https://arxiv.org/pdf/1406.1078v3.pdf) 
 是一个由两个称为编码器和解码器的 RNN 组成的模型。编码器读取
输入序列并输出单个向量，
解码器读取
该向量以生成输出序列。




![](https://pytorch.org/tutorials/_images/seq2seq.png)


 与单个 RNN 的序列预测（其中每个输入
对应一个输出）不同，seq2seq 模型使我们摆脱序列
长度和顺序的影响，这使其成为两种
语言之间翻译的理想选择。




 考虑这句话
 `Je
 

 ne
 

 suis
 

 pas
 

 le
 

 chat
 

黑色`
 →
 `我
 

 是
 

 不是
 

 黑猫
 

` 
 。输入句子中的大多数单词在输出句子中都有直接
翻译，但
顺序略有不同，例如
 `chat
 

 noir`
 和
 `black
 
 
 猫`
.由于
 `ne/pas`
 结构，输入句子中还多了一个单词。直接从输入单词序列
产生正确的翻译是很困难的。




 使用 seq2seq 模型，编码器创建单个向量，在理想情况下，将输入序列的 “meaning” 编码为单个
向量 —某个 N 维句子空间中的单个点。




### 
 编码器
 [¶](#the-encoder "此标题的永久链接")


seq2seq 网络的编码器是一个 RNN，它为输入句子中的每个单词输出一些值。对于每个输入单词，编码器
输出一个向量和一个隐藏状态，并使用
下一个输入单词的隐藏状态。




![](https://pytorch.org/tutorials/_images/encoder-network.png)




```
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

```





### 
 解码器
 [¶](#the-decoder "此标题的永久链接")



 解码器是另一个 RNN，它采用编码器输出向量并
输出单词序列来创建翻译。




#### 
 简单解码器
 [¶](#simple-decoder "此标题的永久链接")



 在最简单的 seq2seq 解码器中，我们仅使用编码器的最后一个输出。
最后一个输出有时称为
 *上下文向量* 
，因为它对整个序列的上下文进行编码。该上下文向量用作
解码器的初始隐藏状态。




 在解码的每一步，解码器都会获得一个输入令牌和
隐藏状态。初始输入标记是字符串开头
 `<SOS>`
 标记，第一个隐藏状态是上下文向量（编码器’s
最后一个隐藏状态）。




![](https://pytorch.org/tutorials/_images/decoder-network.png)




```
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

```




 我鼓励您训练并观察该模型的结果，但为了
节省空间，我们’ 将直接获得金牌并引入
注意力机制。





#### 
 注意力解码器
 [¶](#attention-decoder "永久链接到此标题")



 如果仅在编码器和解码器之间传递上下文向量，
该单个向量就承担了对整个句子进行编码的负担。




 注意力机制允许解码器网络将解码器自己输出的每一步的“聚焦”在编码器’s输出的不同部分上。首先
我们计算一组
 *注意力权重* 
 。这些将乘以编码器输出向量以创建加权组合。结果（在代码中称为“attn_applied”）应包含有关输入序列特定部分的信息，从而帮助解码器选择正确的输出单词。




![](https://i.imgur.com/1152PYf.png)


 计算注意力权重是通过另一个前馈
层
 `attn`
 完成的，使用解码器’s 输入和隐藏状态作为输入。
因为训练数据中有各种大小的句子，为了实际创建和训练该层，我们必须选择它可以应用的最大句子长度（输入长度​​，用于编码器输出）。最大长度的句子将使用所有注意力权重，
而较短的句子将仅使用前几个。




![](https://pytorch.org/tutorials/_images/attention-decoder-network.png)


 Bahdanau 注意力，也称为附加注意力，是序列到序列模型中常用的注意力机制，特别是在神经机器翻译任务中。它是由 Bahdanau 等人提出的。在他们的
标题为
 [联合学习对齐和翻译的神经机器翻译](https://arxiv.org/pdf/1409.0473.pdf) 的论文中
。
这种注意力机制采用学习的对齐模型来计算注意力\编码器和解码器隐藏状态之间的 nscore。它利用前馈
神经网络来计算对齐分数。



然而，还有其他可用的注意力机制，例如 Luong 注意力，它通过解码器隐藏状态和编码器隐藏状态之间的点积来计算注意力分数。它不涉及 Bahdanau 注意力中
使用的非线性变换。




 在本教程中，我们将使用 Bahdanau 注意力。然而，探索修改注意力机制以使用 Luong 注意力将是一项有价值的练习。






```
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

```





 没有10



 还有其他形式的注意力通过使用相对位置方法来解决长度限制。阅读
 [基于注意力的神经机器的有效方法
翻译](https://arxiv.org/abs/1508.04025) 
 中有关 “local
attention” 的内容。
.









 训练
 [¶](#training "固定链接到此标题")
-----------------------------------------------------



### 
 准备训练数据
 [¶](#preparing-training-data "永久链接到此标题")


为了训练，对于每一对，我们需要一个输入张量（输入句子中单词的索引）和目标张量（目标句子中单词的索引）。创建这些向量时，我们会将
EOS 令牌附加到两个序列中。






```
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

```





### 
 训练模型
 [¶](#training-the-model "永久链接到此标题")


为了训练，我们通过编码器运行输入句子，并跟踪每个输出和最新的隐藏状态。然后，解码器被赋予
`<SOS>`
 标记作为其第一个输入，
编码器的最后一个隐藏状态作为其第一个隐藏状态。




 “教师强制” 的概念是使用真实目标输出作为
每个下一个输入，而不是使用解码器’s 猜测作为下一个输入。
使用教师强制会使其收敛得更快，但是
[当训练好的网络被利用时，它可能会表现出稳定性](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.4095&rep=rep1&type =pdf) 
.




 您可以观察教师强制网络的输出，
这些网络使用连贯的语法进行读取，但偏离正确的翻译 - 
直观地它已经学会了表示输出语法，并且可以 \xe2\x80\x9cpick
up\xe2\x80\一旦老师告诉它前几个单词，它就知道了含义，但
它一开始就没有正确学习如何从翻译中创建句子。




 由于 PyTorch’s autograd 给我们带来了自由，我们可以通过简单的 if 语句随机
选择是否使用教师强制。调高
 `teacher_forcing_ratio`
 以使用更多。






```
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

```




 这是一个辅助函数，用于在给定当前时间和进度 % 的情况下打印已用时间和
估计剩余时间。






```
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

```




 整个训练过程是这样的：



* 启动计时器
* 初始化优化器和标准
* 创建训练对集
* 启动空损失数组用于绘图



 然后我们多次调用
 `train`
 并偶尔打印进度（示例的%
、到目前为止的时间、估计时间）和平均损失。






```
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

```





### 
 绘制结果
 [¶](#plotting-results "此标题的固定链接")



 绘图是使用 matplotlib 完成的，使用损失值数组
 `plot_losses`
 在训练时保存。






```
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

```







 评估
 [¶](#evaluation "此标题的永久链接")
--------------------------------------------------------------------------



 评估与训练基本相同，但没有目标，因此
我们只需在每个步骤中将解码器’s 的预测反馈给自身。
每次它预测一个单词时，我们都会将其添加到输出字符串中，如果它预测了 EOS 代币，我们就到此为止。我们还存储解码器’s
attention 输出以供稍后显示。






```
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

```




 我们可以评估训练集中的随机句子并打印
输入、目标和输出，以做出一些主观质量判断：






```
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

```






 训练和评估
 [¶](#training-and-evaluating "永久链接到此标题")
-------------------------------------------------------------------------------------



 有了所有这些辅助函数（看起来像是额外的工作，但
可以更轻松地运行多个实验），我们实际上可以
初始化网络并开始训练。




 请记住，输入的句子经过严格过滤。对于这个小数据集，我们可以使用由 256 个隐藏节点和一个 GRU 层组成的相对较小的网络。在 MacBook CPU 上运行大约 40 分钟后，我们’ 将得到一些
合理的结果。





 注意




 如果您运行此笔记本，您可以训练、中断内核、
评估并稍后继续训练。注释掉编码器和解码器初始化的行，并再次运行
 `trainIters`。







```
hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

```



* ![seq2seq 翻译教程](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_001.png)
* ![seq2seq 翻译教程](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_002.png)





```
Reading lines...
Read 135842 sentence pairs
Trimmed to 11445 sentence pairs
Counting words...
Counted words:
fra 4601
eng 2991
0m 27s (- 6m 51s) (5 6%) 1.5304
0m 54s (- 6m 19s) (10 12%) 0.6776
1m 21s (- 5m 51s) (15 18%) 0.3528
1m 47s (- 5m 23s) (20 25%) 0.1946
2m 14s (- 4m 56s) (25 31%) 0.1205
2m 41s (- 4m 29s) (30 37%) 0.0841
3m 8s (- 4m 2s) (35 43%) 0.0639
3m 35s (- 3m 35s) (40 50%) 0.0521
4m 2s (- 3m 8s) (45 56%) 0.0452
4m 29s (- 2m 41s) (50 62%) 0.0395
4m 55s (- 2m 14s) (55 68%) 0.0377
5m 22s (- 1m 47s) (60 75%) 0.0349
5m 49s (- 1m 20s) (65 81%) 0.0324
6m 16s (- 0m 53s) (70 87%) 0.0316
6m 43s (- 0m 26s) (75 93%) 0.0298
7m 10s (- 0m 0s) (80 100%) 0.0291

```




 将 dropout 层设置为
 `eval`
 模式






```
encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)

```






```
> il est si mignon !
= he s so cute
< he s so cute <EOS>

> je vais me baigner
= i m going to take a bath
< i m going to take a bath <EOS>

> c est un travailleur du batiment
= he s a construction worker
< he s a construction worker <EOS>

> je suis representant de commerce pour notre societe
= i m a salesman for our company
< i m a salesman for our company <EOS>

> vous etes grande
= you re big
< you are big <EOS>

> tu n es pas normale
= you re not normal
< you re not normal <EOS>

> je n en ai pas encore fini avec vous
= i m not done with you yet
< i m not done with you yet <EOS>

> je suis desole pour ce malentendu
= i m sorry about my mistake
< i m sorry about my mistake <EOS>

> nous ne sommes pas impressionnes
= we re not impressed
< we re not impressed <EOS>

> tu as la confiance de tous
= you are trusted by every one of us
< you are trusted by every one of us <EOS>

```




### 
 可视化注意力
 [¶](#visualizing-attention "永久链接到此标题")



 注意力机制的一个有用属性是其高度可解释的
输出。因为它用于对输入序列的
特定编码器输出进行加权，所以我们可以想象在每个时间步
查看网络最关注的位置。




 您可以简单地运行
 `plt.matshow(attentions)`
 来查看显示为矩阵的注意力输出。为了获得更好的观看体验，我们将执行
添加轴和标签的额外工作：






```
def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


defvaluateAndShowAttention(input_sentence):
 输出_words, 注意 = 评估(编码器, 解码器, 输入_sentence, 输入_lang, 输出_lang)
 print('input =', input\ \_sentence)
 print('output =', ' '.join(output_words))
 showAttention(input_sentence, 输出_words, 注意[0, :len(output_words), :])


evaluateAndShowAttention('il n est pas aussi grand que son pere')

evaluateAndShowAttention('je suis trop fatigue pour conduire')

evaluateAndShowAttention('je suis desole si c est une question idiote')

evaluateAndShowAttention('je suis reellement fiere de vous')

```



* ![seq2seq 翻译教程](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_003.png)
* ![seq2seq 翻译教程](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_004.png)
* ![seq2seq 翻译教程](.. /_images/sphx_glr_seq2seq_translation_tutorial_005.png)
* ![seq2seq 翻译教程](https://pytorch.org/tutorials/_images/sphx_glr_seq2seq_translation_tutorial_006.png)





```
input = il n est pas aussi grand que son pere
output = he is not as tall as his father <EOS>
/var/lib/jenkins/workspace/intermediate_source/seq2seq_translation_tutorial.py:823: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

/var/lib/jenkins/workspace/intermediate_source/seq2seq_translation_tutorial.py:825: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

input = je suis trop fatigue pour conduire
output = i m too tired to drive <EOS>
/var/lib/jenkins/workspace/intermediate_source/seq2seq_translation_tutorial.py:823: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

/var/lib/jenkins/workspace/intermediate_source/seq2seq_translation_tutorial.py:825: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

input = je suis desole si c est une question idiote
output = i m sorry if this is a stupid question <EOS>
/var/lib/jenkins/workspace/intermediate_source/seq2seq_translation_tutorial.py:823: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

/var/lib/jenkins/workspace/intermediate_source/seq2seq_translation_tutorial.py:825: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

input = je suis reellement fiere de vous
output = i m really proud of you guys <EOS>
/var/lib/jenkins/workspace/intermediate_source/seq2seq_translation_tutorial.py:823: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

/var/lib/jenkins/workspace/intermediate_source/seq2seq_translation_tutorial.py:825: UserWarning:

set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.

```







 练习
 [¶](#exercises "永久链接到此标题")
------------------------------------------------------------------------------------


* 尝试使用不同的数据集



	+ 另一种语言对
	+ 人类 → 机器（例如 IOT 命令）
	+ 聊天 → 响应
	+ 问题 → 答案
* 替换具有预训练词嵌入的嵌入，例如
 `word2vec`
 或
 `GloVe`
* 尝试使用更多层、更多隐藏单元和更多句子。比较
训练时间和结果。
* 如果您使用的翻译文件中具有两个相同的短语
(
 `I
 

 am
 

 test
 
 
 \t
 

 I
 

 am
 

 test`
 )，您可以将其用作自动编码器。尝试
这个：



	+ 训练为自动编码器
	+ 仅保存编码器网络
	+ 训练新的解码器以从那里进行翻译



**脚本的总运行时间：** 
（7 分 19.245 秒）
