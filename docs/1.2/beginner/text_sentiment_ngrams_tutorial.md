# 文本分类与TorchText

本教程介绍了如何使用文本分类数据集在`torchtext`，其中包括

    
    
    - AG_NEWS,
    - SogouNews,
    - DBpedia,
    - YelpReviewPolarity,
    - YelpReviewFull,
    - YahooAnswers,
    - AmazonReviewPolarity,
    - AmazonReviewFull
    

这个例子展示了如何训练监督学习算法使用这些`TextClassification`数据集中的一个的分类。

## 与n元语法负载数据

n元语法特征的包被应用到捕获有关地方词序一些部分信息。在实践中，双克或三克被施加比只有一个字，以提供更多的益处为字组。一个例子：

    
    
    "load data with ngrams"
    Bi-grams results: "load data", "data with", "with ngrams"
    Tri-grams results: "load data with", "data with ngrams"
    

`TextClassification`数据集支持n元语法方法。通过n元语法设置为2，数据集中的示例文本将是单个单词加上双克字符串列表。

    
    
    import torch
    import torchtext
    from torchtext.datasets import text_classification
    NGRAMS = 2
    import os
    if not os.path.isdir('./.data'):
        os.mkdir('./.data')
    train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
        root='./.data', ngrams=NGRAMS, vocab=None)
    BATCH_SIZE = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

## 定义模型

该模型是由[ EmbeddingBag
](https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag)层和线性层（参见下图）的。
`nn.EmbeddingBag`计算出的嵌入的“袋”的平均值。这里的文本输入有不同的长度。 `nn.EmbeddingBag
`这里不需要填充因为文本长度被保存在偏移。

另外，由于`nn.EmbeddingBag`积聚在飞跨嵌入物的平均值，`nn.EmbeddingBag
`可以增强的性能和存储器效率处理张量的序列。

![../_images/text_sentiment_ngrams_model.png](../_images/text_sentiment_ngrams_model.png)

    
    
    import torch.nn as nn
    import torch.nn.functional as F
    class TextSentiment(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_class):
            super().__init__()
            self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
            self.fc = nn.Linear(embed_dim, num_class)
            self.init_weights()
    
        def init_weights(self):
            initrange = 0.5
            self.embedding.weight.data.uniform_(-initrange, initrange)
            self.fc.weight.data.uniform_(-initrange, initrange)
            self.fc.bias.data.zero_()
    
        def forward(self, text, offsets):
            embedded = self.embedding(text, offsets)
            return self.fc(embedded)
    

## 发起一个实例

该AG_NEWS数据集有四个标签，因此类的数量是四个。

    
    
    1 : World
    2 : Sports
    3 : Business
    4 : Sci/Tec
    

的翻译大小等于词汇的长度（包括单词和n元语法）。类的数目等于标签的数量，这是四个AG_NEWS情况。

    
    
    VOCAB_SIZE = len(train_dataset.get_vocab())
    EMBED_DIM = 32
    NUN_CLASS = len(train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
    

## 功能用于生成批量

由于文本条目具有不同的长度，自定义函数generate_batch（）被用于生成数据的批次和偏移。的函数传递到`collat​​e_fn`在`
torch.utils.data.DataLoader`。输入至`collat​​e_fn`是具有的batch_size的大小张量的列表，并且`
collat​​e_fn`功能它们打包成一个小批量。这里要注意，确保`collat​​e_fn`被声明为顶级画质。这确保了功能在每个工人可用。

在原始数据批输入的文本项被打包成一个列表，然后连接起来作为一个单一的张量作为`nn.EmbeddingBag
`输入。偏移量是分隔符来表示文字张个人序列的开始索引的张量。标签是一个张保存单个文本输入的标签。

    
    
    def generate_batch(batch):
        label = torch.tensor([entry[0] for entry in batch])
        text = [entry[1] for entry in batch]
        offsets = [0] + [len(entry) for entry in text]
        # torch.Tensor.cumsum returns the cumulative sum
        # of elements in the dimension dim.
        # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
    
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        return text, offsets, label
    

## 定义功能训练模型和评估结果。

[
torch.utils.data.DataLoader建议[HTG1用于PyTorch用户，它使数据加载并行容易（一教程](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)[这里](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)）。我们使用`
的DataLoader`这里载入AG_NEWS数据集，并将其发送到模型的训练/验证。

    
    
    from torch.utils.data import DataLoader
    
    def train_func(sub_train_):
    
        # Train the model
        train_loss = 0
        train_acc = 0
        data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=generate_batch)
        for i, (text, offsets, cls) in enumerate(data):
            optimizer.zero_grad()
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            output = model(text, offsets)
            loss = criterion(output, cls)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == cls).sum().item()
    
        # Adjust the learning rate
        scheduler.step()
    
        return train_loss / len(sub_train_), train_acc / len(sub_train_)
    
    def test(data_):
        loss = 0
        acc = 0
        data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
        for text, offsets, cls in data:
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            with torch.no_grad():
                output = model(text, offsets)
                loss = criterion(output, cls)
                loss += loss.item()
                acc += (output.argmax(1) == cls).sum().item()
    
        return loss / len(data_), acc / len(data_)
    

## 拆分数据集和运行模型

由于原始AG_NEWS没有有效的数据集，我们用的0.95（火车）和0.05（有效）的分流比分割训练数据集到火车/有效集。在这里，我们使用PyTorch核心库[
torch.utils.data.dataset.random_split
](https://pytorch.org/docs/stable/data.html?highlight=random_split#torch.utils.data.random_split)功能。

[ CrossEntropyLoss
](https://pytorch.org/docs/stable/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss)标准在单个类结合nn.LogSoftmax（）和nn.NLLLoss（）。以C类培养了分类问题时是非常有用的。
[ SGD
](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html)实现随机梯度下降法作为优化器。初始学习速率设置为4.0。
[ StepLR
](https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html#StepLR)在此用于调节通过历元的学习速率。

    
    
    import time
    from torch.utils.data.dataset import random_split
    N_EPOCHS = 5
    min_valid_loss = float('inf')
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    
    train_len = int(len(train_dataset) * 0.95)
    sub_train_, sub_valid_ = \
        random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    
    for epoch in range(N_EPOCHS):
    
        start_time = time.time()
        train_loss, train_acc = train_func(sub_train_)
        valid_loss, valid_acc = test(sub_valid_)
    
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
    
        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
    

日期：

    
    
    Epoch: 1  | time in 0 minutes, 8 seconds
            Loss: 0.0261(train)     |       Acc: 84.8%(train)
            Loss: 0.0000(valid)     |       Acc: 90.4%(valid)
    Epoch: 2  | time in 0 minutes, 8 seconds
            Loss: 0.0120(train)     |       Acc: 93.5%(train)
            Loss: 0.0000(valid)     |       Acc: 91.2%(valid)
    Epoch: 3  | time in 0 minutes, 8 seconds
            Loss: 0.0070(train)     |       Acc: 96.4%(train)
            Loss: 0.0000(valid)     |       Acc: 90.8%(valid)
    Epoch: 4  | time in 0 minutes, 8 seconds
            Loss: 0.0039(train)     |       Acc: 98.1%(train)
            Loss: 0.0001(valid)     |       Acc: 91.0%(valid)
    Epoch: 5  | time in 0 minutes, 8 seconds
            Loss: 0.0023(train)     |       Acc: 99.0%(train)
            Loss: 0.0001(valid)     |       Acc: 90.9%(valid)
    

运行在GPU以下信息模型：

大纪元：1 |时间为0分钟，11秒

    
    
    Loss: 0.0263(train)     |       Acc: 84.5%(train)
    Loss: 0.0001(valid)     |       Acc: 89.0%(valid)
    

大纪元：2 |时间0分钟，10秒

    
    
    Loss: 0.0119(train)     |       Acc: 93.6%(train)
    Loss: 0.0000(valid)     |       Acc: 89.6%(valid)
    

大纪元：3 |时间0分钟，9秒

    
    
    Loss: 0.0069(train)     |       Acc: 96.4%(train)
    Loss: 0.0000(valid)     |       Acc: 90.5%(valid)
    

大纪元：4 |时间为0分钟，11秒

    
    
    Loss: 0.0038(train)     |       Acc: 98.2%(train)
    Loss: 0.0000(valid)     |       Acc: 90.4%(valid)
    

大纪元：5 |时间为0分钟，11秒

    
    
    Loss: 0.0022(train)     |       Acc: 99.0%(train)
    Loss: 0.0000(valid)     |       Acc: 91.0%(valid)
    

## 评估与测试数据集的模型

    
    
    print('Checking the results of test dataset...')
    test_loss, test_acc = test(test_dataset)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
    

Out:

    
    
    Checking the results of test dataset...
            Loss: 0.0002(test)      |       Acc: 89.3%(test)
    

检查测试数据集的结果...

    
    
    Loss: 0.0237(test)      |       Acc: 90.5%(test)
    

## 在随机新闻测试

用最好的模式，到目前为止并测试一个高尔夫新闻。标签信息可[此处[HTG1。](https://pytorch.org/text/datasets.html?highlight=ag_news#torchtext.datasets.AG_NEWS)

    
    
    import re
    from torchtext.data.utils import ngrams_iterator
    from torchtext.data.utils import get_tokenizer
    
    ag_news_label = {1 : "World",
                     2 : "Sports",
                     3 : "Business",
                     4 : "Sci/Tec"}
    
    def predict(text, model, vocab, ngrams):
        tokenizer = get_tokenizer("basic_english")
        with torch.no_grad():
            text = torch.tensor([vocab[token]
                                for token in ngrams_iterator(tokenizer(text), ngrams)])
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1
    
    ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
        enduring the season’s worst weather conditions on Sunday at The \
        Open on his way to a closing 75 at Royal Portrush, which \
        considering the wind and the rain was a respectable showing. \
        Thursday’s first round at the WGC-FedEx St. Jude Invitational \
        was another story. With temperatures in the mid-80s and hardly any \
        wind, the Spaniard was 13 strokes better in a flawless round. \
        Thanks to his best putting performance on the PGA Tour, Rahm \
        finished with an 8-under 62 for a three-stroke lead, which \
        was even more impressive considering he’d never played the \
        front nine at TPC Southwind."
    
    vocab = train_dataset.get_vocab()
    model = model.to("cpu")
    
    print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])
    

Out:

    
    
    This is a Sports news
    

这是一个体育新闻

你可以找到本笔记[此处](https://github.com/pytorch/text/tree/master/examples/text_classification)中显示的代码示例。

**脚本的总运行时间：** （1分钟26.424秒）

[`Download Python source code:
text_sentiment_ngrams_tutorial.py`](../_downloads/1824f32965271d21829e1739cc434729/text_sentiment_ngrams_tutorial.py)

[`Download Jupyter notebook:
text_sentiment_ngrams_tutorial.ipynb`](../_downloads/27bd42079e7f46673b53e90153168529/text_sentiment_ngrams_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](torchtext_translation_tutorial.html "Language Translation with
TorchText") [![](../_static/images/chevron-right-orange.svg)
Previous](../intermediate/seq2seq_translation_tutorial.html "NLP From Scratch:
Translation with a Sequence to Sequence Network and Attention")

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

  * 文本分类与TorchText 
    * 与n元语法负载数据
    * 定义模型
    * 启动一个实例
    * 时使用的函数，以产生批次
    * 定义函数来训练模型和评估结果。 
    * 分割数据集和运行模型
    * 评估与测试数据集的模型
    * 上随机新闻测试

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

