# NLP From Scratch：版本生成的名称用字符级RNN

**作者** ：[肖恩·罗伯逊](https://github.com/spro/practical-pytorch)

这是我们的“NLP的划痕”的三个教程第二。在第一教程& LT ; /中间/ char_rnn_classification_tutorial & GT ;
我们使用了RNN到名称分类到其原籍语言。这一次，我们将回过头来生成语言的名称。

    
    
    > python sample.py Russian RUS
    Rovakov
    Uantov
    Shavakov
    
    > python sample.py German GER
    Gerren
    Ereng
    Rosher
    
    > python sample.py Spanish SPA
    Salla
    Parer
    Allan
    
    > python sample.py Chinese CHI
    Chan
    Hang
    Iun
    

我们还在手工编写一个小RNN与几个线性层。最大的区别是一个名称的所有字母看完之后，而不是预测的一个类别，我们输入一次一个类别，并输出一个字母。反复预测字符，以形成语言（这也可以与词语或其它高阶构建完成的）通常被称为“语言模型”。

**建议读：**

我假设你已经至少安装PyTorch，知道Python和理解张量：

  * [ https://pytorch.org/ [HTG1对于安装说明](https://pytorch.org/)
  * [ 深，PyTorch学习：60分钟的闪电战 ](../beginner/deep_learning_60min_blitz.html)得到普遍开始PyTorch
  * 与实施例 对于宽和深概述[ 学习PyTorch](../beginner/pytorch_with_examples.html)
  * [ PyTorch为前火炬用户 ](../beginner/former_torchies_tutorial.html)如果你是前者的Lua火炬用户

这也将是有益的了解RNNs以及它们如何工作：

  * [回归神经网络](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)不合理有效性示出了一堆真实例子
  * [理解LSTM网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)为约LSTMs具体地说而且翔实约RNNs一般

我也建议前面的教程，[ NLP From Scratch：版本分类名称以字符级RNN
](char_rnn_classification_tutorial.html)

## 准备数据

Note

从[此处](https://download.pytorch.org/tutorial/data.zip)下载数据，并将其解压到当前目录。

看到最后教程，这个过程的更多细节。总之，存在与每一行的名称一堆纯文本文件`数据/名称/ [语言]的.txt
`。我们分割线成一个数组，Unicode转换为ASCII码，并用字典结束`{语言： [名称 ...]}  [ HTG11。`

    
    
    from __future__ import unicode_literals, print_function, division
    from io import open
    import glob
    import os
    import unicodedata
    import string
    
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters) + 1 # Plus EOS marker
    
    def findFiles(path): return glob.glob(path)
    
    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )
    
    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]
    
    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    
    n_categories = len(all_categories)
    
    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')
    
    print('# categories:', n_categories, all_categories)
    print(unicodeToAscii("O'Néàl"))
    

日期：

    
    
    # categories: 18 ['French', 'Czech', 'Dutch', 'Polish', 'Scottish', 'Chinese', 'English', 'Italian', 'Portuguese', 'Japanese', 'German', 'Russian', 'Korean', 'Arabic', 'Greek', 'Vietnamese', 'Spanish', 'Irish']
    O'Neal
    

## 创建网络

该网络已经延伸最后一个教程的RNN 与类别张量，这是与其他人一起串接一个额外的参数。类别张量是一热载体，就像字母输入。

我们将解释输出作为下一个字母的概率。抽样时，最有可能的输出信作为下一个输入字母。

我添加了一个第二线性层`O2O
`（合成后隐藏和输出）给它更多的肌肉一起工作。还有一个漏失层，其[随机归零其输入](https://arxiv.org/abs/1207.0580)的部分具有给定的概率（这里0.1）和通常用于模糊输入，以防止过度拟合。这里，我们使用它向网络末端故意添加一些混乱，并增加抽样品种。

![](https://i.imgur.com/jzVrf7f.png)

    
    
    import torch
    import torch.nn as nn
    
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size
    
            self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
            self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
            self.o2o = nn.Linear(hidden_size + output_size, output_size)
            self.dropout = nn.Dropout(0.1)
            self.softmax = nn.LogSoftmax(dim=1)
    
        def forward(self, category, input, hidden):
            input_combined = torch.cat((category, input, hidden), 1)
            hidden = self.i2h(input_combined)
            output = self.i2o(input_combined)
            output_combined = torch.cat((hidden, output), 1)
            output = self.o2o(output_combined)
            output = self.dropout(output)
            output = self.softmax(output)
            return output, hidden
    
        def initHidden(self):
            return torch.zeros(1, self.hidden_size)
    

## 培训

### 准备训练

首先，辅助函数来获得随机对（类别，行）：

    
    
    import random
    
    # Random item from a list
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]
    
    # Get a random category and random line from that category
    def randomTrainingPair():
        category = randomChoice(all_categories)
        line = randomChoice(category_lines[category])
        return category, line
    

对于每个时间步长（即，用于在训练单词的每个字母）的网络的输入将是`（类别， 电流 信函 隐藏 的状态） `，输出将是`（下一个 信函 下一个 隐藏
的状态） `。因此，对于每一个训练集，我们需要的类别，一组输入字母，和一组输出/目标字母。

由于我们预测从当前字母的下一个字母每个时间步长，信对是从线连续字母组 - 例如为`“ABCD & LT ; EOS & GT ;”
`我们将创建（“A”，“B”），（“ B”，‘C’），（‘C’，‘d’），（‘d’，‘EOS’）。

![](https://i.imgur.com/JH58tXY.png)

类别张量是[独热张量](https://en.wikipedia.org/wiki/One-hot)大小的`& LT ; 1  ×
n_categories [ - - ] GT ;`。当我们训练它在每一个时间步喂到网络 -
这是一个设计选择，它可能已被列入作为初始隐藏状态的部分或其他一些策略。

    
    
    # One-hot vector for category
    def categoryTensor(category):
        li = all_categories.index(category)
        tensor = torch.zeros(1, n_categories)
        tensor[0][li] = 1
        return tensor
    
    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][all_letters.find(letter)] = 1
        return tensor
    
    # LongTensor of second letter to end (EOS) for target
    def targetTensor(line):
        letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(n_letters - 1) # EOS
        return torch.LongTensor(letter_indexes)
    

为了在训练期间的方便，我们会成为一个`randomTrainingExample
`功能，其获取的随机（类别，线）对并将其转换为所需的（类别，输入，目标）张量。

    
    
    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample():
        category, line = randomTrainingPair()
        category_tensor = categoryTensor(category)
        input_line_tensor = inputTensor(line)
        target_line_tensor = targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor
    

### 网络训练

与此相反，以分类，其中仅使用最后的输出中，我们在每个步骤进行预测，所以我们在每一步计算损失。

autograd的魔力让您只需在每一步总结这些损失，并在年底回呼。

    
    
    criterion = nn.NLLLoss()
    
    learning_rate = 0.0005
    
    def train(category_tensor, input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1)
        hidden = rnn.initHidden()
    
        rnn.zero_grad()
    
        loss = 0
    
        for i in range(input_line_tensor.size(0)):
            output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
            l = criterion(output, target_line_tensor[i])
            loss += l
    
        loss.backward()
    
        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)
    
        return output, loss.item() / input_line_tensor.size(0)
    

为了使培训需要多长时间我增加了`timeSince（时间戳）HTG2] `函数返回一个人类可读的字符串轨迹：

    
    
    import time
    import math
    
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    

训练照常营业 - 称火车一堆时间和等待几分钟，打印出当前时间和损耗每`print_every`实例，并保持每一个平均损失[店面HTG4 ]
plot_every  在`实例all_losses`供以后绘制。

    
    
    rnn = RNN(n_letters, 128, n_letters)
    
    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every plot_every iters
    
    start = time.time()
    
    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss
    
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
    
        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
    

Out:

    
    
    0m 17s (5000 5%) 3.5187
    0m 35s (10000 10%) 2.5492
    0m 53s (15000 15%) 2.2320
    1m 11s (20000 20%) 3.2664
    1m 29s (25000 25%) 2.2973
    1m 47s (30000 30%) 1.1620
    2m 5s (35000 35%) 2.8624
    2m 23s (40000 40%) 1.8314
    2m 41s (45000 45%) 2.3952
    2m 58s (50000 50%) 2.7142
    3m 16s (55000 55%) 2.4662
    3m 34s (60000 60%) 2.9410
    3m 53s (65000 65%) 2.5558
    4m 11s (70000 70%) 2.2629
    4m 29s (75000 75%) 2.3106
    4m 47s (80000 80%) 2.2239
    5m 5s (85000 85%) 1.4803
    5m 23s (90000 90%) 2.9525
    5m 42s (95000 95%) 1.9797
    6m 0s (100000 100%) 2.3567
    

### 绘制损失

绘制从all_losses的历史损失显示网络学习：

    
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    plt.figure()
    plt.plot(all_losses)
    

![../_images/sphx_glr_char_rnn_generation_tutorial_001.png](../_images/sphx_glr_char_rnn_generation_tutorial_001.png)

## 采样网络

为了品尝我们给网络中的信，问下一个是什么，喂，在为下一个字母，并重复直到EOS令牌。

  * 创建输入类别，首个字母，而空隐藏状态的张量
  * 创建一个字符串`output_name中 `与首字母
  * 最多输出长度，
    * 饲料当前信网络
    * 获得从最高输出下一个字母，下一个隐藏的状态
    * 如果这封信是EOS，到此为止
    * 如果一个普通的信，添加到`output_name中 `并继续
  * 返回的最终名称

Note

而不是给它的首个字母，另一种策略会一直到包括“字符串的开始”令牌培训，并有网络选择自己的首个字母。

    
    
    max_length = 20
    
    # Sample from a category and starting letter
    def sample(category, start_letter='A'):
        with torch.no_grad():  # no need to track history in sampling
            category_tensor = categoryTensor(category)
            input = inputTensor(start_letter)
            hidden = rnn.initHidden()
    
            output_name = start_letter
    
            for i in range(max_length):
                output, hidden = rnn(category_tensor, input[0], hidden)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == n_letters - 1:
                    break
                else:
                    letter = all_letters[topi]
                    output_name += letter
                input = inputTensor(letter)
    
            return output_name
    
    # Get multiple samples from one category and multiple starting letters
    def samples(category, start_letters='ABC'):
        for start_letter in start_letters:
            print(sample(category, start_letter))
    
    samples('Russian', 'RUS')
    
    samples('German', 'GER')
    
    samples('Spanish', 'SPA')
    
    samples('Chinese', 'CHI')
    

Out:

    
    
    Rovallov
    Uanovakov
    Sanovakov
    Geller
    Eringer
    Raman
    Salos
    Para
    Allan
    Chan
    Hang
    Iun
    

## 练习

  * 尝试使用不同的数据集的类 - & GT ;线，例如：
    * 虚构系列 - & GT ;字符名称
    * 语音的部分 - & GT ;字
    * 国家 - [ - ] GT ;市
  * 使用“句子的开始”标记，这样抽样可以在没有选择的开始字母来完成
  * 取得更好的成绩有更大的和/或更好的网络状
    * 尝试nn.LSTM和nn.GRU层
    * 结合这些RNNs的多为高层网络

**脚本的总运行时间：** （6分钟0.536秒）

[`Download Python source code:
char_rnn_generation_tutorial.py`](../_downloads/8167177b6dd8ddf05bb9fe58744ac406/char_rnn_generation_tutorial.py)

[`Download Jupyter notebook:
char_rnn_generation_tutorial.ipynb`](../_downloads/a35c00bb5afae3962e1e7869c66872fa/char_rnn_generation_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](seq2seq_translation_tutorial.html "NLP From Scratch: Translation
with a Sequence to Sequence Network and Attention")
[![](../_static/images/chevron-right-orange.svg)
Previous](char_rnn_classification_tutorial.html "NLP From Scratch: Classifying
Names with a Character-Level RNN")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。

Built with [Sphinx](http://sphinx-doc.org/) using a
[theme](https://github.com/rtfd/sphinx_rtd_theme) provided by [Read the
Docs](https://readthedocs.org).

  * NLP从头：生成名称与字符级RNN 
    * 准备数据
    * 创建网络
    * 培训
      * 准备训练
      * 训练网络
      * 绘制的损失
    * 采样网络
    * 练习

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)

## 文件

对于PyTorch访问完整的开发文档

[View Docs](https://pytorch.org/docs/stable/index.html)

## 教程

获取详细的教程，对于初学者和高级开发者

[View Tutorials](https://pytorch.org/tutorials)

## 资源

查找开发资源，并得到回答您的问题

[View Resources](https://pytorch.org/resources)

[](https://pytorch.org/)

  * [ PyTorch ](https://pytorch.org/)
  * [入门](https://pytorch.org/get-started)
  * [特点](https://pytorch.org/features)
  * [生态系统](https://pytorch.org/ecosystem)
  * [博客](https://pytorch.org/blog/)
  * [资源](https://pytorch.org/resources)

  * [支持](https://pytorch.org/support)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [讨论](https://discuss.pytorch.org)
  * [ Github的问题](https://github.com/pytorch/pytorch/issues)
  * [松弛](https://pytorch.slack.com)
  * [贡献](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md)

  * 跟着我们
  * 邮箱地址

[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)

  * 入门
  * 特点
  * 生态系统
  * [博客](https://pytorch.org/blog/)
  * [教程](https://pytorch.org/tutorials)
  * [文档](https://pytorch.org/docs/stable/index.html)
  * [资源](https://pytorch.org/resources)
  * [ Github的](https://github.com/pytorch/pytorch)

