# NLP From Scratch：使用char-RNN生成姓氏

> 作者：[Sean Robertson](https://github.com/spro/practical-pytorch)
>
> 译者：[松鼠](https://github.com/HelWireless)
>
> 校验：[松鼠](https://github.com/HelWireless)

这是我们关于“从零开始的NLP”的三个教程中的第二个。在第一个教程`</ intermediate / char_rnn_classification_tutorial>`中，我们使用了RNN将姓氏分类为它们的起源语言。这次，我们将从语言中生成姓氏。
```shell
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
```    

我们之前还在手撸带有一些线性层的小型RNN网络。现在和之前最大的区别在于，我们不再是读取一个姓氏的所有字母来预测是什么类别，而是输入一个类别并同时输出一个字母。这种循环预测出来自语言的字符模型(这也可以用单词或其他高阶结构来完成）通常称为“语言模型”。

**建议：**

假设你已经至少安装PyTorch，知道Python和理解张量：

  * [pytorch](https://pytorch.org/)安装说明
  * 观看[《PyTorch进行深度学习：60分钟速成》](../beginner/deep_learning_60min_blitz.html)来开始学习pytorch
  * [通过实例深入学习PyTorch](../beginner/pytorch_with_examples.html)
  * [pytorch为前torch用户的提供的指南](../beginner/former_torchies_tutorial.html)

下面这些是了解RNNs以及它们如何工作的相关联接：

  * [回归神经网络](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)展示真实生活中的一系列例子
  * [理解LSTM网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)虽然是关于LSTMs的但也对RNNs有很多详细的讲解

我也建议浏览下前面的教程，[NLP From Scratch：使用char-RNN对姓氏进行分类
](../intermediate/char_rnn_classification_tutorial.md)

## 准备数据

>* Note
>从[此处](https://download.pytorch.org/tutorial/data.zip)下载数据，并将其解压到当前目录。

有关此过程的更多详细信息，请参见上一教程。简而言之，有一堆纯文本文件`data/names/[Language].txt`，每行都有一个姓氏。我们将行拆分成一个数组，将`Unicode`转换为`ASCII`，最后得到一个dictionary`{language: [names ...]}`

```python
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
```    

输出： 
```shell    
    # categories: 18 ['French', 'Czech', 'Dutch', 'Polish', 'Scottish', 'Chinese', 'English', 'Italian', 'Portuguese', 'Japanese', 'German', 'Russian', 'Korean', 'Arabic', 'Greek', 'Vietnamese', 'Spanish', 'Irish']
    O'Neal
```    

## 创建网络

该网络 使用类别张量的额外参数扩展了上一教程的RNN，该参数与其他张量串联在一起。类别张量是一个独热向量，就像字母输入一样。

我们将输出解释为下一个字母的概率。采样时，最有可能的输出字母用作下一个输入字母。

我们添加了第二个线性层o2o(将隐藏和输出结合在一起之后），以使它具有更多的性能可以使用。还有一个drop层，它以给定的概率(此处为0.1）将输入的[一部分随机归零](https://arxiv.org/abs/1207.0580)，通常用于模糊输入以防止过拟合。在这里，我们在网络末端使用它来故意添加一些混乱并增加采样种类。

```python
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
```    

## 训练

### 准备训练

先，helper函数获取随机对(类别，行）：

```python
    import random
    
    # Random item from a list
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]
    
    # Get a random category and random line from that category
    def randomTrainingPair():
        category = randomChoice(all_categories)
        line = randomChoice(category_lines[category])
        return category, line
```   

对于每个时间步长(即对于训练单词中的每个字母），网络的输入将为`(category, current letter, hidden state)` ，输出将为`(next letter, next hidden state)`。因此，对于每个训练集，我们都需要类别，一组输入字母和一组输出/目标字母。

由于我们正在预测每个时间步中当前字母的下一个字母，因此字母对是该行中连续的字母组,例如:`"ABCD<EOS>"`我们将创建(“ A”，“ B”），(“ B”，“ C” )，(“ C”，“ D”），(“ D”，“ EOS”）。


类别张量是大小为`<1 x n_categories>`的独热张量。训练时，我们会随时随地将其馈送到网络中。这是一种设计方式，它可能已作为初始隐藏状态或某些其他策略的一部分包含在内。
    
```python    
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
```    
为了方便训练，我们将创建一个`randomTrainingExample`函数以获取随机(类别，行）对并将其转换为所需的(类别，输入，目标）张量。
```python
    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample():
        category, line = randomTrainingPair()
        category_tensor = categoryTensor(category)
        input_line_tensor = inputTensor(line)
        target_line_tensor = targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor
```  

### 网络训练

与仅使用最后一个输出的分类相反，我们在每个步骤进行预测，因此在每个步骤都计算损失。

autograd使您可以简单地将每一步的损失相加，然后在末尾调用。
```python
    
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
```   

为了跟踪训练需要多长时间，我添加了一个`timeSince(timestamp)`返回人类可读字符串的函数：

```python
    
    import time
    import math
    
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
```   

训练通常会需要来回调用很多次，然后等待几分钟，打印每个`print_every`的当前时间和损失值，并保存每个样本的平均损失`plot_every`到`all_losses`供以后绘图用。

```python   
    
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
```   

Out:

```shell   
    
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
```    

### 绘制损失

绘制all_losses中的历史损失值展示网络学习：

    
```python    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    plt.figure()
    plt.plot(all_losses)
```   

![img/sphx_glr_char_rnn_generation_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_generation_tutorial_001.png)

## 采样网络

我们给网络一个字母，问下一个字母是什么，并将其作为下一个字母输入，并重复直到EOS标记。

  * 创建输入类别，首个字母，空的隐藏状态的张量
  * 创建一个以首字母开头的字符串`output_name`
  * 最大输出长度，
    * 喂入当前字母到网络
    * 从输出中获取最可能字母，和下一个隐藏层的状态
    * 如果当前输出是EOS标记，就停止循环
    * 如果是一个正常的字母，就添加到`output_name`中并继续
  * 返回的最终姓氏

>* Note
>相比于给它起一个开始字母，另一种策略是在训练中包括一个“字符串开始”标记，并让网络选择自己的开始字母。
>

```python        
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
```    

输出:
```shell
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
```    

## 练习

  * 尝试使用不同的数据集的category -> line，例如：
    * 虚构系列->角色名称
    * 词性->单词
    * 国家->城市
  * 使用“句子开头”标记，以便无需选择开始字母即可进行采样
  * 通过更大和/或结构更好的网络获得更好的结果
    * 尝试nn.LSTM和nn.GRU层
    * 将多个这些RNN合并为更高级别的网络

**脚本的总运行时间：** (6分钟0.536秒）