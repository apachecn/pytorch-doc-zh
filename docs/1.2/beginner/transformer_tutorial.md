# 序列对序列建模nn.Transformer和TorchText

这是关于如何训练一个使用[ nn.Transformer
](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformer#torch.nn.Transformer)模块的序列到序列模型的教程。

PyTorch 1.2版本包括基于纸张标准变压器模块[注意是所有你需要[HTG1。变压器模型已经证明，同时更可并行是在质量为众多序列到序列问题优越。的`
nn.Transformer`模块完全依赖于注意机制（如最近](https://arxiv.org/pdf/1706.03762.pdf)[
nn.MultiheadAttention
](https://pytorch.org/docs/master/nn.html?highlight=multiheadattention#torch.nn.MultiheadAttention)实现的另一模块）来绘制的输入和输出之间的全局相关性。的`
nn.Transformer`模块现在高度模块化使得单个组分（如[ nn.TransformerEncoder
](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformerencoder#torch.nn.TransformerEncoder)在本教程）可以容易地适应/组成。

![img/transformer_architecture.jpg](img/transformer_architecture.jpg)

## 定义模型

在本教程中，我们训练`nn.TransformerEncoder
`在语言建模任务模式。语言建模任务是分配的概率为给定字（或词的序列）的可能性遵循的字序列。标记序列被传递到埋层第一，接着是位置编码层以考虑字的次序（详见下段）。的`
nn.TransformerEncoder`由[ nn.TransformerEncoderLayer
](https://pytorch.org/docs/master/nn.html?highlight=transformerencoderlayer#torch.nn.TransformerEncoderLayer)多层。随着输入序列，需要多注意口罩，因为`
自注意力层nn.TransformerEncoder`只允许参加序列中的较早位置。对于语言建模任务，对未来位置的任何标记应该屏蔽。有实际的话，的`
输出nn.TransformerEncoder`模型被发送到最终直线层，之后是对数使用SoftMax功能。

    
    
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class TransformerModel(nn.Module):
    
        def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
            super(TransformerModel, self).__init__()
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
            self.model_type = 'Transformer'
            self.src_mask = None
            self.pos_encoder = PositionalEncoding(ninp, dropout)
            encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.ninp = ninp
            self.decoder = nn.Linear(ninp, ntoken)
    
            self.init_weights()
    
        def _generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask
    
        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)
    
        def forward(self, src):
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                device = src.device
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
    
            src = self.encoder(src) * math.sqrt(self.ninp)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, self.src_mask)
            output = self.decoder(output)
            return F.log_softmax(output, dim=-1)
    

`PositionalEncoding
`模块注入大约序列中的令牌的相对或绝对位置的一些信息。的位置编码具有相同的尺寸，使得两个可以概括的嵌入物。在这里，我们使用不同的频率的`正弦 `和`余弦
`功能。

    
    
    class PositionalEncoding(nn.Module):
    
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
    
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
    
        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)
    

## 负载和批数据

训练过程中使用wikitext的-2数据集从`torchtext
`。的翻译对象基于列车数据集构建并用于令牌numericalize成张量。从序列数据开始，`batchify（）
`函数排列数据集到列中，修剪掉剩余的任何令牌中的数据已经被划分成大小为`的batch_size的批次后
`。例如，具有字母的序列（26总长度）和4:1的批量大小，我们将划分成字母长度为6的4个序列：

\\[\begin{split}\begin{bmatrix} \text{A} & \text{B} & \text{C} & \ldots &
\text{X} & \text{Y} & \text{Z} \end{bmatrix} \Rightarrow \begin{bmatrix}
\begin{bmatrix}\text{A} \\\ \text{B} \\\ \text{C} \\\ \text{D} \\\ \text{E}
\\\ \text{F}\end{bmatrix} & \begin{bmatrix}\text{G} \\\ \text{H} \\\ \text{I}
\\\ \text{J} \\\ \text{K} \\\ \text{L}\end{bmatrix} & \begin{bmatrix}\text{M}
\\\ \text{N} \\\ \text{O} \\\ \text{P} \\\ \text{Q} \\\ \text{R}\end{bmatrix}
& \begin{bmatrix}\text{S} \\\ \text{T} \\\ \text{U} \\\ \text{V} \\\ \text{W}
\\\ \text{X}\end{bmatrix} \end{bmatrix}\end{split}\\]

这些列由模型，这意味着`G`和`F`不能被学习，依赖性但允许视为独立更有效的批处理。

    
    
    import torchtext
    from torchtext.data.utils import get_tokenizer
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def batchify(data, bsz):
        data = TEXT.numericalize([data.examples[0].text])
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)
    
    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_txt, batch_size)
    val_data = batchify(val_txt, eval_batch_size)
    test_data = batchify(test_txt, eval_batch_size)
    

日期：

    
    
    downloading wikitext-2-v1.zip
    extracting
    

### 函数来产生输入和目标序列

`get_batch（） `函数生成用于变压器模型的输入和靶序列。它的源数据细分为长度`BPTT`的块。对于语言建模任务，该模型需要以下单词作为`
目标 [HTG11。例如，用`BPTT`的2值，我们会得到以下两个变量为`i的 `= 0：`

![img/transformer_input_target.png](img/transformer_input_target.png)

应当注意的是，块是沿着维度0与`S`在变压器模型尺寸相一致。将批料尺寸`N`是沿着维度1。

    
    
    bptt = 35
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
    

## 发起一个实例

该模型建立与下面的超参数。的词汇尺寸等于词汇对象的长度。

    
    
    ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    

## 运行模型

[ CrossEntropyLoss
](https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss)被施加到跟踪损耗和[
SGD
](https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD)实现随机梯度下降法作为优化器。初始学习速率设置为5.0。
[ StepLR
](https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR)被施加到调节通过历元学习速率。在培训过程中，我们使用[
nn.utils.clip_grad_norm_
](https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_)功能扩展所有梯度在一起，以防止爆炸。

    
    
    criterion = nn.CrossEntropyLoss()
    lr = 5.0 # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    import time
    def train():
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        ntokens = len(TEXT.vocab.stoi)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    
            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
    
    def evaluate(eval_model, data_source):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(TEXT.vocab.stoi)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                output = eval_model(data)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)
    

遍历时期。保存模型如果验证损失是到目前为止我们见过的最好的。每次调整后时代的学习率。

    
    
    best_val_loss = float("inf")
    epochs = 3 # The number of epochs
    best_model = None
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    
        scheduler.step()
    

Out:

    
    
    | epoch   1 |   200/ 2981 batches | lr 5.00 | ms/batch 35.59 | loss  8.12 | ppl  3348.51
    | epoch   1 |   400/ 2981 batches | lr 5.00 | ms/batch 34.57 | loss  6.82 | ppl   912.80
    | epoch   1 |   600/ 2981 batches | lr 5.00 | ms/batch 34.55 | loss  6.39 | ppl   597.41
    | epoch   1 |   800/ 2981 batches | lr 5.00 | ms/batch 34.59 | loss  6.25 | ppl   517.17
    | epoch   1 |  1000/ 2981 batches | lr 5.00 | ms/batch 34.58 | loss  6.12 | ppl   455.67
    | epoch   1 |  1200/ 2981 batches | lr 5.00 | ms/batch 34.59 | loss  6.09 | ppl   442.33
    | epoch   1 |  1400/ 2981 batches | lr 5.00 | ms/batch 34.60 | loss  6.04 | ppl   421.27
    | epoch   1 |  1600/ 2981 batches | lr 5.00 | ms/batch 34.59 | loss  6.05 | ppl   423.61
    | epoch   1 |  1800/ 2981 batches | lr 5.00 | ms/batch 34.60 | loss  5.96 | ppl   386.26
    | epoch   1 |  2000/ 2981 batches | lr 5.00 | ms/batch 34.60 | loss  5.96 | ppl   387.13
    | epoch   1 |  2200/ 2981 batches | lr 5.00 | ms/batch 34.60 | loss  5.85 | ppl   347.56
    | epoch   1 |  2400/ 2981 batches | lr 5.00 | ms/batch 34.60 | loss  5.89 | ppl   362.72
    | epoch   1 |  2600/ 2981 batches | lr 5.00 | ms/batch 34.60 | loss  5.90 | ppl   363.70
    | epoch   1 |  2800/ 2981 batches | lr 5.00 | ms/batch 34.61 | loss  5.80 | ppl   330.43
    -----------------------------------------------------------------------------------------
    | end of epoch   1 | time: 107.65s | valid loss  5.77 | valid ppl   321.01
    -----------------------------------------------------------------------------------------
    | epoch   2 |   200/ 2981 batches | lr 4.75 | ms/batch 34.78 | loss  5.81 | ppl   333.28
    | epoch   2 |   400/ 2981 batches | lr 4.75 | ms/batch 34.63 | loss  5.78 | ppl   324.24
    | epoch   2 |   600/ 2981 batches | lr 4.75 | ms/batch 34.62 | loss  5.61 | ppl   272.10
    | epoch   2 |   800/ 2981 batches | lr 4.75 | ms/batch 34.62 | loss  5.65 | ppl   283.77
    | epoch   2 |  1000/ 2981 batches | lr 4.75 | ms/batch 34.61 | loss  5.60 | ppl   269.12
    | epoch   2 |  1200/ 2981 batches | lr 4.75 | ms/batch 34.63 | loss  5.62 | ppl   275.40
    | epoch   2 |  1400/ 2981 batches | lr 4.75 | ms/batch 34.62 | loss  5.62 | ppl   276.93
    | epoch   2 |  1600/ 2981 batches | lr 4.75 | ms/batch 34.62 | loss  5.66 | ppl   287.64
    | epoch   2 |  1800/ 2981 batches | lr 4.75 | ms/batch 34.63 | loss  5.59 | ppl   268.86
    | epoch   2 |  2000/ 2981 batches | lr 4.75 | ms/batch 34.62 | loss  5.63 | ppl   277.73
    | epoch   2 |  2200/ 2981 batches | lr 4.75 | ms/batch 34.63 | loss  5.52 | ppl   249.01
    | epoch   2 |  2400/ 2981 batches | lr 4.75 | ms/batch 34.61 | loss  5.58 | ppl   265.86
    | epoch   2 |  2600/ 2981 batches | lr 4.75 | ms/batch 34.62 | loss  5.60 | ppl   269.12
    | epoch   2 |  2800/ 2981 batches | lr 4.75 | ms/batch 34.63 | loss  5.51 | ppl   248.37
    -----------------------------------------------------------------------------------------
    | end of epoch   2 | time: 107.58s | valid loss  5.60 | valid ppl   270.75
    -----------------------------------------------------------------------------------------
    | epoch   3 |   200/ 2981 batches | lr 4.51 | ms/batch 34.80 | loss  5.55 | ppl   257.31
    | epoch   3 |   400/ 2981 batches | lr 4.51 | ms/batch 34.63 | loss  5.56 | ppl   259.12
    | epoch   3 |   600/ 2981 batches | lr 4.51 | ms/batch 34.62 | loss  5.36 | ppl   213.08
    | epoch   3 |   800/ 2981 batches | lr 4.51 | ms/batch 34.63 | loss  5.44 | ppl   229.59
    | epoch   3 |  1000/ 2981 batches | lr 4.51 | ms/batch 34.63 | loss  5.37 | ppl   215.90
    | epoch   3 |  1200/ 2981 batches | lr 4.51 | ms/batch 34.64 | loss  5.41 | ppl   223.49
    | epoch   3 |  1400/ 2981 batches | lr 4.51 | ms/batch 34.63 | loss  5.43 | ppl   228.08
    | epoch   3 |  1600/ 2981 batches | lr 4.51 | ms/batch 34.62 | loss  5.47 | ppl   238.36
    | epoch   3 |  1800/ 2981 batches | lr 4.51 | ms/batch 34.58 | loss  5.40 | ppl   222.43
    | epoch   3 |  2000/ 2981 batches | lr 4.51 | ms/batch 34.56 | loss  5.44 | ppl   229.30
    | epoch   3 |  2200/ 2981 batches | lr 4.51 | ms/batch 34.55 | loss  5.32 | ppl   204.63
    | epoch   3 |  2400/ 2981 batches | lr 4.51 | ms/batch 34.54 | loss  5.39 | ppl   220.17
    | epoch   3 |  2600/ 2981 batches | lr 4.51 | ms/batch 34.55 | loss  5.41 | ppl   223.92
    | epoch   3 |  2800/ 2981 batches | lr 4.51 | ms/batch 34.55 | loss  5.34 | ppl   209.22
    -----------------------------------------------------------------------------------------
    | end of epoch   3 | time: 107.47s | valid loss  5.54 | valid ppl   253.71
    -----------------------------------------------------------------------------------------
    

## 评估与所述测试数据集的模型

应用的最佳模式，以检查与测试数据集的结果。

    
    
    test_loss = evaluate(best_model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    

Out:

    
    
    =========================================================================================
    | End of training | test loss  5.43 | test ppl   229.27
    =========================================================================================
    

**脚本的总运行时间：** （5分钟38.763秒）

[`Download Python source code:
transformer_tutorial.py`](../_downloads/f53285338820248a7c04a947c5110f7b/transformer_tutorial.py)

[`Download Jupyter notebook:
transformer_tutorial.ipynb`](../_downloads/dca13261bbb4e9809d1a3aa521d22dd7/transformer_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](../intermediate/reinforcement_q_learning.html "Reinforcement
Learning \(DQN\) Tutorial") [![](../_static/images/chevron-right-orange.svg)
Previous](torchtext_translation_tutorial.html "Language Translation with
TorchText")

* * *

Was this helpful?

Yes

No

Thank you

* * *

©版权所有2017年，PyTorch。



  * 序列到序列与nn.Transformer和TorchText建模
    * 定义模型
    * 负载和批数据
      * 函数来生成输入和目标序列
    * 启动一个实例
    * 运行模型
    * 评估与所述测试数据集的模型

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView

  &noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



