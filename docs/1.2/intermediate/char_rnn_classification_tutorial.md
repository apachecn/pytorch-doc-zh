# NLP From Scratch：使用char-RNN对姓氏进行分类

> 作者：[Sean Robertson](https://github.com/spro/practical-pytorch)
>
> 译者：[松鼠](https://github.com/HelWireless)
>
> 校验：[松鼠](https://github.com/HelWireless)、[Aidol](https://github.com/Aidol)

我们将构建和训练基本的char-RNN来对单词进行分类。本教程以及以下两个教程展示了如何“从头开始”为NLP建模进行预处理数据，尤其是不使用Torchtext的许多便利功能，因此您可以了解NLP建模的预处理是如何从低层次进行的。

char-RNN将单词作为一系列字符读取,在每个步骤输出预测和“隐藏状态”，将其先前的隐藏状态输入到每个下一步。我们将最终的预测作为输出，即单词属于哪个类别。

具体来说，我们将训练起源于18种语言的数千种姓氏，并根据拼写来预测姓氏来自哪种语言：

    
    
    $ python predict.py Hinton
    (-0.47) Scottish
    (-1.52) English
    (-3.57) Irish
    
    $ python predict.py Schmidhuber
    (-0.19) German
    (-2.48) Czech
    (-2.68) Dutch
    

**建议：**

假设你已经至少安装PyTorch，知道Python和理解张量：

  * [pytorch](https://pytorch.org/)安装说明
  * 观看[《PyTorch进行深度学习：60分钟速成》](../beginner/deep_learning_60min_blitz.html)来开始学习pytorch
  * [通过实例深入学习PyTorch](../beginner/pytorch_with_examples.html)
  * [pytorch为前torch用户的提供的指南](../beginner/former_torchies_tutorial.html)

下面这些是了解RNNs以及它们如何工作的相关联接：

  * [回归神经网络](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)展示真实生活中的一系列例子
  * [理解LSTM网络](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)虽然是关于LSTMs的但也对RNNs有很多详细的讲解

## 准备数据

>* Note
>从[此处](https://download.pytorch.org/tutorial/data.zip)下载数据，并将其解压到当前目录。

包含了在`data/names `目录被命名为`[Language] .txt`
的18个文本文件。每个文件都包含了一堆姓氏，每行一个名字，大多都已经罗马字母化了(但我们仍然需要从Unicode转换到到ASCII）。

我们将得到一个字典，列出每种语言的名称列表 。通用变量`category`和`line`(在本例中为语言和名称）用于以后的扩展。`{language: [names ...]}`

    
```python
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
    # 作用就是把Unicode转换为ASCII
    def unicodeToAscii(s):
        return ''.join(
        # NFD表示字符应该分解为多个组合字符表示
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
```

输出：
```shell 
    ['data/names/French.txt', 'data/names/Czech.txt', 'data/names/Dutch.txt', 'data/names/Polish.txt', 'data/names/Scottish.txt', 'data/names/Chinese.txt', 'data/names/English.txt', 'data/names/Italian.txt', 'data/names/Portuguese.txt', 'data/names/Japanese.txt', 'data/names/German.txt', 'data/names/Russian.txt', 'data/names/Korean.txt', 'data/names/Arabic.txt', 'data/names/Greek.txt', 'data/names/Vietnamese.txt', 'data/names/Spanish.txt', 'data/names/Irish.txt']
    
    Slusarski
```    

现在，我们有了`category_lines`字典，将每个类别(语言）映射到行(姓氏）列表。我们还保持`all_categories`(只是一种语言列表）和`n_categories`为可追加状态，供后续的调用。
```python
 print(category_lines['Italian'][:5])
```

输出:
```shell
['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']
```    


### 将姓氏转化为张量

我们已经处理好了所有的姓氏，现在我们需要将它们转换为张量以使用它们。

为了表示单个字母，我们使用大小为`<1 x n letters>`的“独热向量” 。一个独热向量就是在字母索引处填充1，其他都填充为0，例，`"b" = <0 1 0 0 0 ...>`

为了表达一个单词，我们将一堆字母合并成2D矩阵，其中矩阵的大小为`<line_length x 1 x n_letters>`

额外的1维是因为PyTorch假设所有东西都是成批的-我们在这里只使用1的批处理大小。

```python    
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
```    

输出:
```shell
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0.]])

torch.Size([5, 1, 57])    
```
## 创建网络

在进行自动求导之前，在Torch中创建一个递归神经网络需要在多个时间状态上克隆图的参数。图保留了隐藏状态和梯度，这些状态和梯度现在完全由图本身处理。这意味着您可以以非常“单纯”的方式将RNN作为常规的前馈网络来实现。

这个RNN模块(大部分是从[PyTorch for Torch用户教程](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#example-2-recurrent-net)中复制的）只有2个线性层，它们在输入和隐藏状态下运行，输出之后是LogSoftmax层。

![RNN.jpg](https://camo.githubusercontent.com/f8a843661e448e1a75f8319a2eea860ebf09794f/68747470733a2f2f692e696d6775722e636f6d2f5a32786279534f2e706e67)

```python
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
```    
运行网络的步骤是，首先我们需要输入(在本例中为当前字母的张量）和先前的隐藏状态(首先将其初始化为零）。我们将返回输出(每种语言的概率）和下一个隐藏状态(我们将其保留用于下一步）。
```python 
    input = letterToTensor('A')
    hidden =torch.zeros(1, n_hidden)
    
    output, next_hidden = rnn(input, hidden)
```

为了提高效率，我们不想为每个步骤都创建一个新的Tensor，因此我们将使用`lineToTensor`加切片的方式来代替`letterToTensor`。这可以通过预先计算一批张量来进一步优化。
```python
    input = lineToTensor('Albert')
    hidden = torch.zeros(1, n_hidden)
    
    output, next_hidden = rnn(input[0], hidden)
    print(output)
```

输出:
```shell 
    tensor([[-2.8636, -2.8199, -2.8899, -2.9073, -2.9117, -2.8644, -2.9027, -2.9334,
             -2.8705, -2.8383, -2.8892, -2.9161, -2.8215, -2.9996, -2.9423, -2.9116,
             -2.8750, -2.8862]], grad_fn=<LogSoftmaxBackward>)
```    

正如你看到的输出为`<1  × n_categories>`的张量，其中每一个值都是该类别的可能性(数值越大可能性越高）。

## 训练

### 准备训练

在训练之前，我们需要做一些辅助函数。首先是解释网络的输出，我们知道这是每个类别的可能性。我们可以用`Tensor.topk`来获取最大值对应的索引：
```python
    def categoryFromOutput(output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return all_categories[category_i], category_i
    
    print(categoryFromOutput(output))
```

输出:
```shell 
    ('Czech', 1)
```

我们也将需要一个快速的方法来获得一个训练例子(姓氏和其所属语言）:
```python
    
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
        print('category =', category, '\t // \t line =', line)
    
```

输出:

```shell    
category =  Dutch 	 // 	 line =  Ryskamp
category =  Spanish 	 // 	 line =  Iniguez
category =  Vietnamese 	 // 	 line =  Thuy
category =  Italian 	 // 	 line =  Nacar
category =  Vietnamese 	 // 	 line =  Le
category =  French 	 // 	 line =  Tremblay
category =  Russian 	 // 	 line =  Bakhchivandzhi
category =  Irish 	 // 	 line =  Kavanagh
category =  Irish 	 // 	 line =  O'Shea
category =  Spanish 	 // 	 line =  Losa
```

### 网络训练

现在，训练该网络所需要做的就是向它喂入大量训练样例，进行预测，并告诉它预测的是否正确。

最后因为RNN的最后一层是`nn.LogSoftmax`,所以我们选择损失函数`nn.NLLLoss`比较合适。
```python 
    criterion = nn.NLLLoss()
```

每个循环的训练将：

  * 创建输入和目标张量
  * 创建一个零初始隐藏状态
  * 读取每个字母
    * 保持隐藏状态到下一个字母
  * 比较最后输出和目标
  * 进行反向传播
  * 返回输出值和损失函数的值

```python
    learning_rate = 0.005 
    # If you set this too high, it might explode. If too low, it might not learn
    
    def train(category_tensor, line_tensor):
        hidden = rnn.initHidden()
    
        rnn.zero_grad()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
    
        loss = criterion(output, category_tensor)
        loss.backward()
    
        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
        # 下面一行代码的作用效果为 p.data = p.data -learning_rate*p.grad.data，更新权重
            p.data.add_(-learning_rate, p.grad.data)
    
        return output, loss.item()
```

现在，我们只需要运行大量样例。由于`train`函数同时返回`output`和`loss`，因此我们可以打印其猜测并跟踪绘制损失。由于有1000个样例，因此我们仅打印每个`print_every`样例，并对损失进行平均。

```python    
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
```

输出:
```shell
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
```

### 绘制结果

从绘制`all_losses`的历史损失图可以看出网络的学习：

```python    
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    plt.figure()
    plt.plot(all_losses) 
```

![img/sphx_glr_char_rnn_classification_tutorial_001.png](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_classification_tutorial_001.png)


## 评价结果

为了了解网络在不同类别上的表现如何，我们将创建一个混淆矩阵，包含姓氏属于的实际语言(行）和网络猜测的是哪种语言(列）。要计算混淆矩阵，将使用`evaluate()`通过网络来评测一些样本。

```python    
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
    
```
![img/sphx_glr_char_rnn_classification_tutorial_002.png](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_classification_tutorial_002.png)

您可以从主轴上挑出一些亮点，以显示错误猜测的语言，例如，中文(朝鲜语）和西班牙语(意大利语）。它似乎与希腊语搭预测得很好，而英语预测的很差(可能是因为与其他语言重叠）。

### 运行用户输入

```python   
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
```    

Out:
```shell
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
```

实际[PyTorch存储库](https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification)中的脚本的最终版本将上述代码分成几个文件：

  * `data.py`(加载文件）
  * `model.py`(定义RNN）
  * `train.py`(训练）
  * `predict.py`(`predict()`与命令行参数一起运行）
  * `server.py`(通过`bottle.py`将预测用作JSON API）

运行`train.py`训练并保存网络。

用`predict.py`脚本并加上姓氏运行以查看预测：

    
```shell    
    $ python predict.py Hazaki
    (-0.42) Japanese
    (-1.39) Polish
    (-3.51) Czech
```    

运行`server.py`，查看[http://localhost:5533/Yourname ](http://localhost:5533/Yourname)获得预测的JSON输出。

## 练习

+ 尝试使用line-> category的其他数据集，例如：
    - 任何单词->语言
    - 名->性别
    - 角色名称->作家
    - 页面标题-> Blog或subreddit
+ 通过更大和/或结构更好的网络获得更好的结果
    - 添加更多线性层
    - 尝试nn.LSTM和nn.GRU图层
    - 将多个这些RNN合并为更高级别的网络

**脚本的总运行时间：** (2分钟42.458秒）
