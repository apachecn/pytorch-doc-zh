# 基于TorchText的语言翻译

> 译者：[PengboLiu](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.2/beginner/text_sentiment_ngrams_tutorial.md)
> 
> 校验：[PengboLiu](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.2/beginner/text_sentiment_ngrams_tutorial.md)

本教程介绍如何使用`torchtext
`的几个类来预处理英德数据集，该数据集可以用来训练seq2seq模型，既而能自动把德语句子翻译成英语。

本文基于PyTorch社区成员[Ben Trevett](https://github.com/bentrevett)的[教程](https://github.com/bentrevett/pytorch-
seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)，并得到了他本人的许可。

阅读完本教程，你将能够：

  * 使用以下`torchtext`的类将句子预处理为NLP建模的常用格式：：

    * [TranslationDataset](https://torchtext.readthedocs.io/en/latest/datasets.html#torchtext.datasets.TranslationDataset)
    * [Field](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.Field)
    * [BucketIterator](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator)

## Field和 TranslationDataset 

`torchtext
`具有创建数据集的功能，可以轻松对其迭代以构建机器翻译模型。一个关键的类是Filed，它指定每个句子的预处理方法，另一个类是TranslationDataset  ; `torchtext
`内置了几个翻译数据集；在本教程中，我们将使用 [Multi30k dataset](https://github.com/multi30k/dataset)数据集，其中包含约30000个英德句对(平均长度约13个字）。

注：本教程中的tokenization 需要使用[ Spacy ](https://spacy.io)。Spacy包可以帮助我们对英语以外的语言tokenization。`torchtext`提供了`basic_english
`的tokenizer ，但是对于其他语言，使用Spacy对我们而言是最好的选择。

为了运行该教程，首先要使用pip或conda安装Spacy。接下来，下载英德原始数据：    

    python -m spacy download en
    python -m spacy download de

安装Spacy后，以下代码将根据Field中定义的tokenizer 处理`TranslationDataset`中的每个句子。

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

输出：   
    downloading training.tar.gz  
    downloading validation.tar.gz  
    downloading mmt_task1_test2016.tar.gz  

现在，我们已经定义好了`train_data`，`torchtext`的`Field`有一个非常有用的功能 `：我们可以使用build_vocab
`方法创建每个语言的词汇表。

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

一旦这些代码行被运行，`SRC.vocab.stoi`将成为一个tokens作为key，索引作为value的词典；对应的， `SRC.vocab.itos
`是一个交换了key和value内容相同的字典。在本教程中我们不会广泛使用此功能，但是你可能在遇到其他NLP任务有用。

## `BucketIterator`

我们使用最后一个`torchtext`的特性是`BucketIterator`， 它以TranslationDataset作为第一个参数，所以易于使用。如文档所说：定义一个迭代器，该迭代器将相似长度的数据放在一起。产生每个新bacth时，最大程度地减少所需的填充量。



    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 128
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = BATCH_SIZE,
        device = device)


可以像`DataLoader`一样调用这些迭代器。 在下面的训练和评估函数中，它们可以简单地通过以下方式调用：    

    for i, batch in enumerate(iterator):


每个`batch `于是具有`SRC`和`TRG`属性：

    src = batch.src
    trg = batch.trg


## 定义我们的`nn.Module`和`Optimizer `

解决了数据集的问题并为之定义好迭代器，我们剩下的任务就是定义模型和优化器完成训练过程。

具体来说，我们的模型遵循[此处](<https://arxiv.org/abs/1409.0473>)描述的结构。

注意：我们选择这种模型并不是因为它是目前最优的，而是因为它是机器翻译的标准模型。众所周知，目前机器翻译的最优模型是Transformers。

```
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
```



输出：
    The model has 1,856,685 trainable parameters  


注：当计算模型分数尤其是翻译模型时，我们需要设置`nn.CrossEntropyLoss` 忽略padding。

    PAD_IDX = TRG.vocab.stoi['<pad>']
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


最后，我们可以训练和评价模型：    
    

```
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
```

输出：
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

  * 看看Ben Trevett使用`torchtext`教程的[其余部分](https://github.com/bentrevett/)
  * 请继续关注使用其他`torchtext`功能以及`nn.Transformer`语言建模预测下一个单词的教程！

**脚本的总运行时间：** (6分钟27.732秒）

