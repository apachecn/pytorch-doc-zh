# NLP From Scratch：版本分类名称以字符级RNN

**作者** ：[肖恩·罗伯逊](https://github.com/spro/practical-pytorch)

我们将建设和培训基本字符级RNN分类的话。本教程，伴随着以下两个，说明如何做“从零开始”
NLP建模数据预处理，特别是不使用许多的的便利功能torchtext ，所以你可以看到NLP造型如何预处理在较低水平的作品。

甲字符级RNN读取字作为一系列字符 - 在每个步骤输出预测和“隐藏状态”，喂食其先前的状态隐藏到每个下一步骤。我们采取最终预测是输出，即字属于哪个类。

具体来说，我们从18种语言起源的几千个姓氏训练，并预测该语言的名称是基于拼写：

    
    
    $ python predict.py Hinton
    (-0.47) Scottish
    (-1.52) English
    (-3.57) Irish
    
    $ python predict.py Schmidhuber
    (-0.19) German
    (-2.48) Czech
    (-2.68) Dutch
    

**建议读：**

我假设你已经至少安装PyTorch，知道Python和理解张量：

  * [ https://pytorch.org/ [HTG1对于安装说明](https://pytorch.org/)
  * [ 深，PyTorch学习：60分钟的闪电战 ](../beginner/deep_learning_60min_blitz.html)得到普遍开始PyTorch
  * 与实施例 对于宽和深概述[ 学习PyTorch](../beginner/pytorch_with_examples.html)
  * [ PyTorch为前火炬用户 ](../beginner/former_torchies_tutorial.html)如果你是前者的Lua火炬用户

这也将是有益的了解RNNs以及它们如何工作：

  * [回归神经网络](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)不合理有效性示出了一堆真实例子
  * [理解LSTM网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)为约LSTMs具体地说而且翔实约RNNs一般

## 准备数据

Note

从[此处](https://download.pytorch.org/tutorial/data.zip)下载数据，并将其解压到当前目录。

包括在`数据/名称 `目录被命名为“[语言] .TXT”
18个的文本文件。每个文件都包含了一堆名字，每行一个名字，大多罗马化（但我们仍然需要转换从Unicode到ASCII）。

我们将结束与每种语言的名称列表的字典，`{语言： [名称 ...]}
[HTG7。通用变量“类别”和“行”（在我们的例子中的语言和名称），用于以后的可扩展性。`

    
    
    from __future__ import unicode_literals, print_function, division
    from io import open
    import glob
    import os
    
    def findFiles(path): return glob.glob(path)
    
    print(findFiles('data/names/*.txt'))
    
    import unicodedata
    import string
    
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    
    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )
    
    print(unicodeToAscii('Ślusàrski'))
    
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    
    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]
    
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    
    n_categories = len(all_categories)
    

日期：

    
    
    ['data/names/French.txt', 'data/names/Czech.txt', 'data/names/Dutch.txt', 'data/names/Polish.txt', 'data/names/Scottish.txt', 'data/names/Chinese.txt', 'data/names/English.txt', 'data/names/Italian.txt', 'data/names/Portuguese.txt', 'data/names/Japanese.txt', 'data/names/German.txt', 'data/names/Russian.txt', 'data/names/Korean.txt', 'data/names/Arabic.txt', 'data/names/Greek.txt', 'data/names/Vietnamese.txt', 'data/names/Spanish.txt', 'data/names/Irish.txt']
    Slusarski
    

现在我们有`category_lines`，一个字典映射每个类别（语言）到线（地名）的列表。我们还不断跟踪的`all_categories`
n_categories 以供日后参考（只是一个语言列表）和`[HTG9。`

    
    
    print(category_lines['Italian'][:5])
    

Out:

    
    
    ['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']
    

### 至于名称为张量

现在，我们有所有的名字组织的，我们需要把它们变成张量做任何使用它们。

来表示单个字母，我们使用尺寸`&℃的“一热载体” ; 1  × n_letters [ - - ] GT ;
`。一个一热载体被填充有0以外的一个1在当前字母的索引，例如`“B” =  & LT ; 0  1  0  0  0  ... & GT ;`。

为了使字我们加入了一堆那些成2D矩阵`& LT ; line_length  × 1  X  n_letters & GT ;`。

这额外的一个维是因为PyTorch假设一切都在批 - 我们只是使用1批量大小在这里。

    
    
    import torch
    
    # Find letter index from all_letters, e.g. "a" = 0
    def letterToIndex(letter):
        return all_letters.find(letter)
    
    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letterToTensor(letter):
        tensor = torch.zeros(1, n_letters)
        tensor[0][letterToIndex(letter)] = 1
        return tensor
    
    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][letterToIndex(letter)] = 1
        return tensor
    
    print(letterToTensor('J'))
    
    print(lineToTensor('Jones').size())
    

Out:

    
    
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.]])
    torch.Size([5, 1, 57])
    

## 创建网络

autograd之前，创造了一个火炬回归神经网络参与在几个时间步克隆层的参数。所述层保持隐藏状态和梯度其现在完全由图本身处理。这意味着你可以在一个非常“纯粹”的方式实现RNN，作为常规的前馈层。

此RNN模块（主要来自[的PyTorch火炬用户个别](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#example-2-recurrent-
net)复制）是对输入和隐藏状态下操作，与输出后一个LogSoftmax层仅有2线性层。

![](https://i.imgur.com/Z2xbySO.png)

    
    
    import torch.nn as nn
    
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()
    
            self.hidden_size = hidden_size
    
            self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
            self.i2o = nn.Linear(input_size + hidden_size, output_size)
            self.softmax = nn.LogSoftmax(dim=1)
    
        def forward(self, input, hidden):
            combined = torch.cat((input, hidden), 1)
            hidden = self.i2h(combined)
            output = self.i2o(combined)
            output = self.softmax(output)
            return output, hidden
    
        def initHidden(self):
            return torch.zeros(1, self.hidden_size)
    
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
    

要运行这个网络，我们需要传递一个输入的步骤（在我们的情况下，张量对于当前字母）和先前隐藏状态（这是我们在第一次初始化为零）。我们会回来的输出（每种语言的概率）和下一个隐藏的状态（这是我们保持对下一步）。

    
    
    input = letterToTensor('A')
    hidden =torch.zeros(1, n_hidden)
    
    output, next_hidden = rnn(input, hidden)
    

为了提高效率起见，我们不希望成为创造每一步新的张量，因此我们将使用`lineToTensor`而不是`letterToTensor
`并使用切片。这可以通过张量的预先计算的批次被进一步优化。

    
    
    input = lineToTensor('Albert')
    hidden = torch.zeros(1, n_hidden)
    
    output, next_hidden = rnn(input[0], hidden)
    print(output)
    

Out:

    
    
    tensor([[-2.8636, -2.8199, -2.8899, -2.9073, -2.9117, -2.8644, -2.9027, -2.9334,
             -2.8705, -2.8383, -2.8892, -2.9161, -2.8215, -2.9996, -2.9423, -2.9116,
             -2.8750, -2.8862]], grad_fn=<LogSoftmaxBackward>)
    

正如可以看到的输出为`& LT ; 1  × n_categories & GT ;`张量，其中，每一个项目是该类别的可能性（较高可能性更大）。

## 培训

### 准备训练

之前进入训练中，我们应该做一些辅助功能。首先是要理解网络的输出，这是我们知道的是每个类别的可能性。我们可以使用`Tensor.topk
`来获得最大价值的指标：

    
    
    def categoryFromOutput(output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return all_categories[category_i], category_i
    
    print(categoryFromOutput(output))
    

Out:

    
    
    ('Czech', 1)
    

我们也将需要一个快速的方法来获得一个训练例子（名称和其语言）：

    
    
    import random
    
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]
    
    def randomTrainingExample():
        category = randomChoice(all_categories)
        line = randomChoice(category_lines[category])
        category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
        line_tensor = lineToTensor(line)
        return category, line, category_tensor, line_tensor
    
    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)
    

Out:

    
    
    category = Dutch / line = Sanna
    category = Irish / line = O'Hara
    category = Portuguese / line = Barros
    category = Arabic / line = Mifsud
    category = Polish / line = Wojewodka
    category = Irish / line = O'Kelly
    category = Korean / line = Noh
    category = Korean / line = Byon
    category = Korean / line = Rhee
    category = German / line = Best
    

### 网络训练

现在，一切都需要训练这个网络是表现出来了一堆例子，有它做出猜测，并告诉它，如果它是错的。

的损失函数`nn.NLLLoss`是合适的，因为RNN的最后一层是`nn.LogSoftmax`。

    
    
    criterion = nn.NLLLoss()
    

培训每个循环将：

  * 创建输入和目标张量
  * 创建一个零初始隐藏状态
  * 阅读每个字母和
    * 保持隐藏状态下一封信
  * 比较最后输出到目标
  * 背传播
  * 返回输出和损失

    
    
    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn
    
    def train(category_tensor, line_tensor):
        hidden = rnn.initHidden()
    
        rnn.zero_grad()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
    
        loss = criterion(output, category_tensor)
        loss.backward()
    
        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)
    
        return output, loss.item()
    

现在，我们只需要运行与一堆例子。由于`火车 `函数返回无论是产量和损失，我们可以打印其猜测，并跟踪丢失的密谋。既然有例子，我们1000只打印每`
print_every`实例，并采取损失的平均值。

    
    
    import time
    import math
    
    n_iters = 100000
    print_every = 5000
    plot_every = 1000
    
    
    
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    
    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    
    start = time.time()
    
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss
    
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    

Out:

    
    
    5000 5% (0m 7s) 2.7482 Silje / French ✗ (Dutch)
    10000 10% (0m 15s) 1.5569 Lillis / Greek ✓
    15000 15% (0m 22s) 2.7729 Burt / Korean ✗ (English)
    20000 20% (0m 30s) 1.1036 Zhong / Chinese ✓
    25000 25% (0m 38s) 1.7088 Sarraf / Portuguese ✗ (Arabic)
    30000 30% (0m 45s) 0.7595 Benivieni / Italian ✓
    35000 35% (0m 53s) 1.2900 Arreola / Italian ✗ (Spanish)
    40000 40% (1m 0s) 2.3171 Gass / Arabic ✗ (German)
    45000 45% (1m 8s) 3.1630 Stoppelbein / Dutch ✗ (German)
    50000 50% (1m 15s) 1.7478 Berger / German ✗ (French)
    55000 55% (1m 23s) 1.3516 Almeida / Spanish ✗ (Portuguese)
    60000 60% (1m 31s) 1.8843 Hellewege / Dutch ✗ (German)
    65000 65% (1m 38s) 1.7374 Moreau / French ✓
    70000 70% (1m 46s) 0.5718 Naifeh / Arabic ✓
    75000 75% (1m 53s) 0.6268 Zhui / Chinese ✓
    80000 80% (2m 1s) 2.2226 Dasios / Portuguese ✗ (Greek)
    85000 85% (2m 9s) 1.3690 Walter / Scottish ✗ (German)
    90000 90% (2m 16s) 0.5329 Zhang / Chinese ✓
    95000 95% (2m 24s) 3.4474 Skala / Czech ✗ (Polish)
    100000 100% (2m 31s) 1.4720 Chi / Korean ✗ (Chinese)
    

### 绘制结果

绘制从`all_losses`历史损失示出了网络的学习：

    
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    plt.figure()
    plt.plot(all_losses)
    

![../_images/sphx_glr_char_rnn_classification_tutorial_001.png](../_images/sphx_glr_char_rnn_classification_tutorial_001.png)

## 评价结果

要看到网络表现如何对不同的类别，我们将创建一个混淆矩阵，表示每一个实际的语言（行）的语言，网络的猜测（列）。为了计算混淆矩阵一堆样品的通过网络与`
评价（运行） `，它是相同的`列车（） `减去backprop。

    
    
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    
    # Just return an output given a line
    def evaluate(line_tensor):
        hidden = rnn.initHidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
    
        return output
    
    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    
    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # sphinx_gallery_thumbnail_number = 2
    plt.show()
    

![../_images/sphx_glr_char_rnn_classification_tutorial_002.png](../_images/sphx_glr_char_rnn_classification_tutorial_002.png)

你可以挑选出亮点关闭，显示它猜测的语言错误的主轴，例如中国对韩国，西班牙和意大利。这似乎与希腊做的非常好，也很不好英语（也许是因为与其他语言的重叠）。

### 运行在用户输入

    
    
    def predict(input_line, n_predictions=3):
        print('\n> %s' % input_line)
        with torch.no_grad():
            output = evaluate(lineToTensor(input_line))
    
            # Get top N categories
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []
    
            for i in range(n_predictions):
                value = topv[0][i].item()
                category_index = topi[0][i].item()
                print('(%.2f) %s' % (value, all_categories[category_index]))
                predictions.append([value, all_categories[category_index]])
    
    predict('Dovesky')
    predict('Jackson')
    predict('Satoshi')
    

Out:

    
    
    > Dovesky
    (-0.47) Russian
    (-1.30) Czech
    (-2.90) Polish
    
    > Jackson
    (-1.04) Scottish
    (-1.72) English
    (-1.74) Russian
    
    > Satoshi
    (-0.32) Japanese
    (-2.63) Polish
    (-2.71) Italian
    

在实际PyTorch回购脚本[的最终版本分裂上面的代码到几个文件：](https://github.com/spro/practical-
pytorch/tree/master/char-rnn-classification)

  * `data.py`（加载文件）
  * `model.py`（定义RNN）
  * `train.py`（试验训练）
  * `predict.py`（运行`预测（） `命令行参数）
  * `server.py`（服务预测为具有一个bottle.py JSON API）

运行`train.py`培养和保存网络。

运行`predict.py`使用一个名称，以查看预测：

    
    
    $ python predict.py Hazaki
    (-0.42) Japanese
    (-1.39) Polish
    (-3.51) Czech
    

运行`server.py`，参观[ HTTP：//本地主机：5533 / YOURNAME
](http://localhost:5533/Yourname)获得预测的JSON输出。

## 练习

  * 用不同的数据集线的尝试 - & GT ;类别，例如：
    * 任何字 - & GT ;语言
    * 第一名字 - & GT ;性别
    * 字符的名称 - & GT ;作家
    * 页标题 - & GT ;博客或版（Subreddit）
  * 取得更好的成绩有更大的和/或更好的网络状
    * 添加更多线性层
    * 尝试`nn.LSTM`和`nn.GRU`层
    * 结合这些RNNs的多为高层网络

**脚本的总运行时间：** （2分钟42.458秒）

[`Download Python source code:
char_rnn_classification_tutorial.py`](../_downloads/ccb15f8365bdae22a0a019e57216d7c6/char_rnn_classification_tutorial.py)

[`Download Jupyter notebook:
char_rnn_classification_tutorial.ipynb`](../_downloads/977c14818c75427641ccb85ad21ed6dc/char_rnn_classification_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](char_rnn_generation_tutorial.html "NLP From Scratch: Generating
Names with a Character-Level RNN") [![](../_static/images/chevron-right-
orange.svg) Previous](../beginner/audio_preprocessing_tutorial.html
"torchaudio Tutorial")

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

  * NLP从头：判断名称与字符级RNN 
    * 准备数据
      * [HTG0转到名称成张量
    * 创建网络
    * 培训
      * 准备训练
      * 训练网络
      * 绘制的结果
    * 评估结果
      * 运行于用户输入
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

