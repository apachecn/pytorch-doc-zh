# 语言翻译与TorchText

本教程介绍了如何使用`torchtext
`到数据预处理的几种便利类含有英语和德语句子著名的数据集，并用它来训练序列对序列模型的注意，可德国句子翻译成英文。

它是基于关闭的[本教程](https://github.com/bentrevett/pytorch-
seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)从PyTorch社区成员[本Trevett
](https://github.com/bentrevett)，被[赛斯魏德曼](https://github.com/SethHWeidman/)和Ben的许可创建。

在本教程的最后，你将能够：

  * 预处理句子成用于NLP建模通常使用的格式使用以下`torchtext`便利类：
    
    * [ TranslationDataset ](https://torchtext.readthedocs.io/en/latest/datasets.html#torchtext.datasets.TranslationDataset)
    * [领域HTG1]](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field)
    * [ BucketIterator ](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator)

## 领域HTG1]和 TranslationDataset 

`torchtext
`具有创建数据集，可以轻松迭代完成创建语言翻译模型的目的工具。一个键类是[领域HTG5]，指定每个句子应该进行预处理的方法，另一种是在
TranslationDataset  ; `torchtext
`有几个这样的数据集;在本教程中，我们将使用](https://github.com/pytorch/text/blob/master/torchtext/data/field.py#L64)[
Multi30k数据集](https://github.com/multi30k/dataset)，其中包含约30000句子（平均长度约13个字），英语和德语。

注：本教程中的标记化要求[ Spacy ](https://spacy.io)我们使用Spacy，因为它提供了英语以外的语言为符号化的大力支持。 `
torchtext`提供`basic_english
`标记生成器，并支持其他断词的英语（如[摩西](https://bitbucket.org/luismsgomes/mosestokenizer/src/default/)），但语言翻译
- 需要多个语言 - Spacy是你最好的选择。

为了运行该教程，第一安装`spacy`使用`点子 `或`康达 `。接下来，下载的英语和德语Spacy断词的原始数据：

    
    
    python -m spacy download en
    python -m spacy download de
    

与Spacy安装，下面的代码将标记化每个句子中基于`TranslationDataset`上在`领域HTG6] `中定义的标记生成器

    
    
    from torchtext.datasets import Multi30k
    from torchtext.data import Field, BucketIterator
    
    SRC = Field(tokenize = "spacy",
                tokenizer_language="de",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)
    
    TRG = Field(tokenize = "spacy",
                tokenizer_language="en",
                init_token = '<sos>',
                eos_token = '<eos>',
                lower = True)
    
    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                        fields = (SRC, TRG))
    

日期：

    
    
    downloading training.tar.gz
    downloading validation.tar.gz
    downloading mmt_task1_test2016.tar.gz
    

现在，我们已经定义`train_data`，我们可以看到的`torchtext`的`字段一个非常有用的功能 `：将`build_vocab
`方法现在允许我们创建与每个语言相关联的词汇

    
    
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)
    

一旦这些代码行已经在运行，`SRC.vocab.stoi`将与词汇表中的作为键的标记和它们相应的索引作为字典的值; `SRC.vocab.itos
`将是相同的字典，交换了键和值。我们不会广泛使用这一事实在本教程中，但是这可能会在你遇到其他NLP任务有用。

## `BucketIterator`

我们将使用最后一个`torchtext`具体特征是`BucketIterator`，这是很容易使用，因为它需要一个`
TranslationDataset
`作为它的第一个参数。具体而言，作为文档说：定义批处理类似长度的实例一起的迭代器。最小化，同时产生新鲜混洗批次为每个新历元所需要的填充量。参见所用桶装程序池。

    
    
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 128
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = BATCH_SIZE,
        device = device)
    

这些迭代可以被称为就像`DataLoader``s ;  下面， 在 中的 ``培养 `和`评价 `的功能，它们被简单地称为带：

    
    
    for i, batch in enumerate(iterator):
    

每个`批次 `于是具有`SRC`和`TRG`属性：

    
    
    src = batch.src
    trg = batch.trg
    

## 定义我们的`nn.Module`和`优化 `

这主要是从一个`torchtext`perspecive：内置的数据集和定义的迭代器，本教程的其余部分只是我们的模型定义为`nn.Module
`，与`沿优化 `，然后训练它。

我们的模型而言，如下描述HTG0]此处[HTG1（你可以找到一个显著更多评论版[此处](https://github.com/SethHWeidman/pytorch-
seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)）的架构。

注意：这种模式仅仅是可用于语言翻译的示例模型;我们选择它，因为它是该任务的标准模型，而不是因为它是推荐的机型使用进行翻译。正如你可能知道，国家的最先进的机型，目前基于变形金刚;你可以看到PyTorch的能力，实现变压器层[此处](https://pytorch.org/docs/stable/nn.html#transformer-
layers) ;，特别的“关注”在下面的模型中使用的是多头存在于变压器模型自注意不同。

    
    
    import random
    from typing import Tuple
    
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch import Tensor
    
    
    class Encoder(nn.Module):
        def __init__(self,
                     input_dim: int,
                     emb_dim: int,
                     enc_hid_dim: int,
                     dec_hid_dim: int,
                     dropout: float):
            super().__init__()
    
            self.input_dim = input_dim
            self.emb_dim = emb_dim
            self.enc_hid_dim = enc_hid_dim
            self.dec_hid_dim = dec_hid_dim
            self.dropout = dropout
    
            self.embedding = nn.Embedding(input_dim, emb_dim)
    
            self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
    
            self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self,
                    src: Tensor) -> Tuple[Tensor]:
    
            embedded = self.dropout(self.embedding(src))
    
            outputs, hidden = self.rnn(embedded)
    
            hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
    
            return outputs, hidden
    
    
    class Attention(nn.Module):
        def __init__(self,
                     enc_hid_dim: int,
                     dec_hid_dim: int,
                     attn_dim: int):
            super().__init__()
    
            self.enc_hid_dim = enc_hid_dim
            self.dec_hid_dim = dec_hid_dim
    
            self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
    
            self.attn = nn.Linear(self.attn_in, attn_dim)
    
        def forward(self,
                    decoder_hidden: Tensor,
                    encoder_outputs: Tensor) -> Tensor:
    
            src_len = encoder_outputs.shape[0]
    
            repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
    
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
    
            energy = torch.tanh(self.attn(torch.cat((
                repeated_decoder_hidden,
                encoder_outputs),
                dim = 2)))
    
            attention = torch.sum(energy, dim=2)
    
            return F.softmax(attention, dim=1)
    
    
    class Decoder(nn.Module):
        def __init__(self,
                     output_dim: int,
                     emb_dim: int,
                     enc_hid_dim: int,
                     dec_hid_dim: int,
                     dropout: int,
                     attention: nn.Module):
            super().__init__()
    
            self.emb_dim = emb_dim
            self.enc_hid_dim = enc_hid_dim
            self.dec_hid_dim = dec_hid_dim
            self.output_dim = output_dim
            self.dropout = dropout
            self.attention = attention
    
            self.embedding = nn.Embedding(output_dim, emb_dim)
    
            self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
    
            self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
    
            self.dropout = nn.Dropout(dropout)
    
    
        def _weighted_encoder_rep(self,
                                  decoder_hidden: Tensor,
                                  encoder_outputs: Tensor) -> Tensor:
    
            a = self.attention(decoder_hidden, encoder_outputs)
    
            a = a.unsqueeze(1)
    
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
    
            weighted_encoder_rep = torch.bmm(a, encoder_outputs)
    
            weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
    
            return weighted_encoder_rep
    
    
        def forward(self,
                    input: Tensor,
                    decoder_hidden: Tensor,
                    encoder_outputs: Tensor) -> Tuple[Tensor]:
    
            input = input.unsqueeze(0)
    
            embedded = self.dropout(self.embedding(input))
    
            weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                              encoder_outputs)
    
            rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)
    
            output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
    
            embedded = embedded.squeeze(0)
            output = output.squeeze(0)
            weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
    
            output = self.out(torch.cat((output,
                                         weighted_encoder_rep,
                                         embedded), dim = 1))
    
            return output, decoder_hidden.squeeze(0)
    
    
    class Seq2Seq(nn.Module):
        def __init__(self,
                     encoder: nn.Module,
                     decoder: nn.Module,
                     device: torch.device):
            super().__init__()
    
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
    
        def forward(self,
                    src: Tensor,
                    trg: Tensor,
                    teacher_forcing_ratio: float = 0.5) -> Tensor:
    
            batch_size = src.shape[1]
            max_len = trg.shape[0]
            trg_vocab_size = self.decoder.output_dim
    
            outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
    
            encoder_outputs, hidden = self.encoder(src)
    
            # first input to the decoder is the <sos> token
            output = trg[0,:]
    
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_outputs)
                outputs[t] = output
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = output.max(1)[1]
                output = (trg[t] if teacher_force else top1)
    
            return outputs
    
    
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    # ENC_EMB_DIM = 256
    # DEC_EMB_DIM = 256
    # ENC_HID_DIM = 512
    # DEC_HID_DIM = 512
    # ATTN_DIM = 64
    # ENC_DROPOUT = 0.5
    # DEC_DROPOUT = 0.5
    
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    ENC_HID_DIM = 64
    DEC_HID_DIM = 64
    ATTN_DIM = 8
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    
    def init_weights(m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    
    
    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters())
    
    
    def count_parameters(model: nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    

Out:

    
    
    The model has 1,856,685 trainable parameters
    

注：得分尤其是语言翻译模型的性能时，我们必须告诉`nn.CrossEntropyLoss`函数忽略其中目标是简单地填充索引。

    
    
    PAD_IDX = TRG.vocab.stoi['<pad>']
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    

最后，我们可以训练和评价这一模式：

    
    
    import math
    import time
    
    
    def train(model: nn.Module,
              iterator: BucketIterator,
              optimizer: optim.Optimizer,
              criterion: nn.Module,
              clip: float):
    
        model.train()
    
        epoch_loss = 0
    
        for _, batch in enumerate(iterator):
    
            src = batch.src
            trg = batch.trg
    
            optimizer.zero_grad()
    
            output = model(src, trg)
    
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
    
            loss = criterion(output, trg)
    
            loss.backward()
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
            optimizer.step()
    
            epoch_loss += loss.item()
    
        return epoch_loss / len(iterator)
    
    
    def evaluate(model: nn.Module,
                 iterator: BucketIterator,
                 criterion: nn.Module):
    
        model.eval()
    
        epoch_loss = 0
    
        with torch.no_grad():
    
            for _, batch in enumerate(iterator):
    
                src = batch.src
                trg = batch.trg
    
                output = model(src, trg, 0) #turn off teacher forcing
    
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)
    
                loss = criterion(output, trg)
    
                epoch_loss += loss.item()
    
        return epoch_loss / len(iterator)
    
    
    def epoch_time(start_time: int,
                   end_time: int):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    
    N_EPOCHS = 10
    CLIP = 1
    
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
    
        start_time = time.time()
    
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
    
        end_time = time.time()
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    
    test_loss = evaluate(model, test_iterator, criterion)
    
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    

Out:

    
    
    Epoch: 01 | Time: 0m 36s
            Train Loss: 5.686 | Train PPL: 294.579
             Val. Loss: 5.250 |  Val. PPL: 190.638
    Epoch: 02 | Time: 0m 37s
            Train Loss: 5.019 | Train PPL: 151.260
             Val. Loss: 5.155 |  Val. PPL: 173.274
    Epoch: 03 | Time: 0m 37s
            Train Loss: 4.757 | Train PPL: 116.453
             Val. Loss: 4.976 |  Val. PPL: 144.824
    Epoch: 04 | Time: 0m 35s
            Train Loss: 4.574 | Train PPL:  96.914
             Val. Loss: 4.835 |  Val. PPL: 125.834
    Epoch: 05 | Time: 0m 35s
            Train Loss: 4.421 | Train PPL:  83.185
             Val. Loss: 4.783 |  Val. PPL: 119.414
    Epoch: 06 | Time: 0m 38s
            Train Loss: 4.321 | Train PPL:  75.233
             Val. Loss: 4.802 |  Val. PPL: 121.734
    Epoch: 07 | Time: 0m 38s
            Train Loss: 4.233 | Train PPL:  68.957
             Val. Loss: 4.675 |  Val. PPL: 107.180
    Epoch: 08 | Time: 0m 35s
            Train Loss: 4.108 | Train PPL:  60.838
             Val. Loss: 4.622 |  Val. PPL: 101.693
    Epoch: 09 | Time: 0m 34s
            Train Loss: 4.020 | Train PPL:  55.680
             Val. Loss: 4.530 |  Val. PPL:  92.785
    Epoch: 10 | Time: 0m 34s
            Train Loss: 3.919 | Train PPL:  50.367
             Val. Loss: 4.448 |  Val. PPL:  85.441
    | Test Loss: 4.464 | Test PPL:  86.801 |
    

## 接下来的步骤

  * 看看本Trevett的教程的其余部分使用`torchtext`[这里](https://github.com/bentrevett/)
  * 请继续使用其他`torchtext`功能与`一起调整为教程nn.Transformer`通过下一个单词预测语言建模！

**脚本的总运行时间：** （6分钟27.732秒）

[`Download Python source code:
torchtext_translation_tutorial.py`](../_downloads/96d6dc961c7477af88e16ca6c9592240/torchtext_translation_tutorial.py)

[`Download Jupyter notebook:
torchtext_translation_tutorial.ipynb`](../_downloads/05baddac9b2f50d639a62ea5fa6e21e4/torchtext_translation_tutorial.ipynb)

[通过斯芬克斯-廊产生廊](https://sphinx-gallery.readthedocs.io)

[Next ![](../_static/images/chevron-right-
orange.svg)](transformer_tutorial.html "Sequence-to-Sequence Modeling with
nn.Transformer and TorchText") [![](../_static/images/chevron-right-
orange.svg) Previous](text_sentiment_ngrams_tutorial.html "Text Classification
with TorchText")

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

  * 语言翻译与TorchText 
    * 领域HTG2]和 TranslationDataset 
    * `BucketIterator`
    * 定义我们的`nn.Module`和`优化 `
    * [HTG0接下来的步骤

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

