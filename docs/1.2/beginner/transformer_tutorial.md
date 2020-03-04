# 序列对序列建模nn.Transformer和TorchText

> 译者：[dabney777](https://github.com/dabney777)
> 
> 校验：[dabney777](https://github.com/dabney777)

本教程将会使用[ nn.Transformer ](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformer#torch.nn.Transformer)模块训练一个序列到序列模型。

PyTorch 1.2 版本依据论文 [ Attention is All You Need ](https://arxiv.org/pdf/1706.03762.pdf)发布了标准的 transformer 模型。Transformer 模型已被证明在解决序列到序列问题时效果优异。

nn.Transformer 模块通过注意力机制([ nn.MultiheadAttention ](https://pytorch.org/docs/master/nn.html?highlight=multiheadattention#torch.nn.MultiheadAttention))来取得输入与输出之间的全局相关性。nn.Transformer 模块现已高度模块化，可以直接用于构建其他模型(如[ nn.TransformerEncoder](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformerencoder#torch.nn.TransformerEncoder))。

![img/transformer_architecture.jpg](img/transformer_architecture.jpg)

## 定义模型

在本教程中，我们训练 `nn.TransformerEncoder` 用于构建语言模型。语言模型的目标是对给定字/词序列打分，判断该字/词序列出现在文本中的概率。字符序列首先会被传进 embedding 层转化为向量，然后被传入位置编码层 (详见下段）。 `nn.TransformerEncoder` 由多个编码层[nn.TransformerEncoderLayer](https://pytorch.org/docs/master/nn.html?highlight=transformerencoderlayer#torch.nn.TransformerEncoderLayer)组成。对输入序列的每一维需要施加一个自注意力权重影响。`nn.TransformerEncoder` 的自注意力权重只影响序列中靠前的数据，不修改之后位置的数据。在本任务中，`nn.TransformerEncoder` 的输出将会被送至最终的线性层，该层为一个 log-Softmax 层。

    
    
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
    

`PositionalEncoding` 模块将字/词在序列中的绝对位置或相对位置信息编码。 位置编码与嵌入层具有相同的维度，这样位置信息向量和嵌入向量可以直接相加。 这里，我们使用 `sin` 和 `cos` 函数在不同位置的值来作为位置编码的值。具体计算公式见下方代码。

    
    
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
    

## 加载和整合数据

训练过程中使用的数据机是从 `torchtext` 中得到的wikitext的-2数据集。词典对象基于训练数据集进行构建。`batchify(）` 函数把数据集中的数据排到多个列中，在划分成多个大小为 `batch_size` 的集合后，剩下的少于 `batch_size` 个数据会被丢弃。例如，对于字母序列(长度为26, `batch_size` 为4），将按照以下方法划分：

\\[\begin{split}\begin{bmatrix} \text{A} & \text{B} & \text{C} & \ldots &
\text{X} & \text{Y} & \text{Z} \end{bmatrix} \Rightarrow \begin{bmatrix}
\begin{bmatrix}\text{A} \\\ \text{B} \\\ \text{C} \\\ \text{D} \\\ \text{E}
\\\ \text{F}\end{bmatrix} & \begin{bmatrix}\text{G} \\\ \text{H} \\\ \text{I}
\\\ \text{J} \\\ \text{K} \\\ \text{L}\end{bmatrix} & \begin{bmatrix}\text{M}
\\\ \text{N} \\\ \text{O} \\\ \text{P} \\\ \text{Q} \\\ \text{R}\end{bmatrix}
& \begin{bmatrix}\text{S} \\\ \text{T} \\\ \text{U} \\\ \text{V} \\\ \text{W}
\\\ \text{X}\end{bmatrix} \end{bmatrix}\end{split}\\]

对于我们的模型来说，只学习同一列中的数据的关系，不同的列各自独立。即我们的模型无法学习到 `G` 和 `F` 之间的联系，这样可以增加模型的并行度，增加学习效率。

    
    
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
    

输出：

    
    
    downloading wikitext-2-v1.zip
    extracting
    

### 生成训练数据(输入和目标输出)的函数

`get_batch()` 函数生成用于 `transformer` 模型的输入和目标序列。它把源数据细分为长度为 `bptt` 的块。对于语言模型，需要当前词的下一个词作为目标词。例如当 `bptt` 为2， `i` =0 时，该函数会产生以下数据：

![img/transformer_input_target.png](img/transformer_input_target.png)

张量的第0维是不同的块，块的大小与 Transformer 中的编码层大小一致。张量的第1维大小为 `batch` 大小。

    
    
    bptt = 35
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target
    

## 初始化模型

模型的超参数如下，词典大小为 `vocab` 数组的长度。

    
    
    ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
    emsize = 200 # embedding dimension
    nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.2 # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    

## 运行模型

模型使用交叉墒([ CrossEntropyLoss ](https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss))作为损失函数，使用随机梯度下降([ SGD ](https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD))方法更新参数。初始学习率设置为5.0。
[ StepLR ](https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR) 用于调节学习速率。在训练过程中，使用[nn.utils.clip_grad_norm_ ](https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_)函数限制梯度大小以防梯度爆炸。

    
    
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
    
在每个 epoch 结束时，若验证集的损失函数为最低则会更新一次学习率。

    
    
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
    

输出:

    
    
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
    

## 使用测试集评价模型

使用测试集来测试模型。

    
    
    test_loss = evaluate(best_model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    

输出:

    
    
    =========================================================================================
    | End of training | test loss  5.43 | test ppl   229.27
    =========================================================================================
    

**脚本的总运行时间：** (5分钟38.763秒）

[`Download Python source code:transformer_tutorial.py`](../_downloads/f53285338820248a7c04a947c5110f7b/transformer_tutorial.py)

[`Download Jupyter notebook:transformer_tutorial.ipynb`](../_downloads/dca13261bbb4e9809d1a3aa521d22dd7/transformer_tutorial.ipynb)


[Next ![](../_static/images/chevron-right-orange.svg)](../intermediate/reinforcement_q_learning.html "ReinforcementLearning \(DQN\) Tutorial") [![](../_static/images/chevron-right-orange.svg)

[Previous](torchtext_translation_tutorial.html "Language Translation with
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

![](https://www.facebook.com/tr?id=243028289693773&ev=PageView&noscript=1)
![](https://www.googleadservices.com/pagead/conversion/795629140/?label=txkmCPmdtosBENSssfsC&guid=ON&script=0)







 
[](https://www.facebook.com/pytorch) [](https://twitter.com/pytorch)

分析流量和优化经验，我们为这个站点的Cookie。通过点击或导航，您同意我们的cookies的使用。因为这个网站目前维护者，Facebook的Cookie政策的适用。了解更多信息，包括有关可用的控制：[饼干政策[HTG1。](https://www.facebook.com/policies/cookies/)

![](../_static/images/pytorch-x.svg)

[](https://pytorch.org/)



