
 （测试版）LSTM Word 语言模型上的动态量化
 [¶](#beta-dynamic-quantization-on-an-lstm-word-language-model "永久链接到此标题")
==========================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/dynamic_quantization_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>




**作者** 
 :
 [James Reed](https://github.com/jamesr66a)




**编辑者** 
 :
 [Seth Weidman](https://github.com/SethHWeidman/)





 简介
 [¶](#introduction "此标题的永久链接")
--------------------------------------------------------------------------



 量化涉及将模型的权重和激活从 float
 转换为 int，这可以缩小模型大小并加快推理速度，
对准确性的影响很小。




 在本教程中，我们将应用最简单的量化形式 -
 [动态量化](https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic) 
 -
 到基于 LSTM 的下一个单词预测模型，严格遵循 PyTorch 示例中的
 [单词语言模型](https://github.com/pytorch/examples/tree/master/word_language_model)。






```
# imports
import os
from io import open
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

```






 1. 定义模型
 [¶](#define-the-model "永久链接到此标题")
----------------------------------------------------------------------------



 在这里，我们按照单词语言模型示例中的
 [模型](https://github.com/pytorch/examples/blob/master/word_language_model/model.py) 定义 LSTM 模型架构。\ n





```
class LSTMModel(nn.Module):
 """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

```






 2. 加载文本数据
 [¶](#load-in-the-text-data "固定链接到此标题")
-------------------------------------------------------------------------------------



 接下来，我们将
 [Wikitext-2 数据集](https://www.google.com/search?q=wikitext+2+data)
 加载到
 
 语料库
 
 中， 
再次遵循单词语言模型示例中
的[预处理](https://github.com/pytorch/examples/blob/master/word_language_model/data.py)。






```
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
 """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

model_data_filepath = 'data/'

corpus = Corpus(model_data_filepath + 'wikitext-2')

```






 3. 加载预训练模型
 [¶](#load-the-pretrained-model "永久链接到此标题")
--------------------------------------------------------------------------------------------



 这是关于动态量化的教程，动态量化是一种在模型训练后应用的量化技术。因此，我们’将简单地将一些
预训练的权重加载到该模型架构中；这些权重是通过使用单词语言模型中的默认设置
示例进行五个时期的训练获得的。






```
ntokens = len(corpus.dictionary)

model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)

model.load_state_dict(
    torch.load(
        model_data_filepath + 'word_language_model_quantize.pth',
        map_location=torch.device('cpu')
        )
    )

model.eval()
print(model)

```






```
LSTMModel(
  (drop): Dropout(p=0.5, inplace=False)
  (encoder): Embedding(33278, 512)
  (rnn): LSTM(512, 256, num_layers=5, dropout=0.5)
  (decoder): Linear(in_features=256, out_features=33278, bias=True)
)

```




 现在让’s 生成一些文本以确保预训练模型正常工作
 - 与之前类似，我们按照
 [此处](https://github.com/pytorch/examples/blob/master/word_language_model/generate.py)






```
input_ = torch.randint(ntokens, (1, 1), dtype=torch.long)
hidden = model.init_hidden(1)
temperature = 1.0
num_words = 1000

with open(model_data_filepath + 'out.txt', 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(num_words):
            output, hidden = model(input_, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_.fill_(word_idx)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(str(word.encode('utf-8')) + ('' if i % 20 == 19 else ' '))

            if i % 100 == 0:
                print('| Generated {}/{} words'.format(i, 1000))

with open(model_data_filepath + 'out.txt', 'r') as outf:
    all_output = outf.read()
    print(all_output)

```






```
| Generated 0/1000 words
| Generated 100/1000 words
| Generated 200/1000 words
| Generated 300/1000 words
| Generated 400/1000 words
| Generated 500/1000 words
| Generated 600/1000 words
| Generated 700/1000 words
| Generated 800/1000 words
| Generated 900/1000 words
b'.' b'Ross' b"'" b'final' b'focus' b'respects' b'with' b'rice' b'Rajeev' b'implements' b'.' b'<unk>' b'Darwin' b',' b'a' b'comfortably' b',' b'called' b'that' b'it'
b'is' b'"' b'significant' b'alive' b'"' b'from' b'perform' b'@-@' b'hearted' b',' b'can' b'be' b'among' b'what' b'he' b'is' b'a' b'Sixth' b'minister' b'as'
b'a' b'analysis' b',' b'bathtub' b'for' b'1798' b'and' b'an' b'Nourrit' b'who' b'left' b'the' b'same' b'name' b',' b'which' b'they' b'saw' b'to' b'"'
b'let' b'most' b'or' b'me' b'of' b'its' b'all' b'time' b'that' b'might' b'have' b'done' b'on' b'back' b'on' b'their' b'character' b'position' b'.' b'"'
b'<eos>' b'The' b'2010' b'Peach' b'Bird' b"'" b'Union' b'(' b'1888' b')' b',' b'which' b'could' b'be' b'actively' b'composed' b'in' b'London' b'and' b'in'
b'1609' b'.' b'The' b'work' b'have' b'October' b',' b'but' b',' b'since' b'the' b'parish' b'of' b'times' b'is' b'hard' b'and' b'severely' b'ignored' b'the'
b'plums' b',' b'they' b'<unk>' b'or' b'Giuseppe' b'Leo' b'Rodman' b'for' b'the' b'game' b'<unk>' b',' b'and' b'were' b'released' b'and' b'because' b'it' b'apparently'
b'spent' b'before' b'with' b'those' b'arena' b'to' b'deciding' b'.' b'"' b'strumming' b'on' b'You' b'then' b'heard' b'enough' b'that' b'we' b'have' b'rhythm' b'channels'
b'in' b'a' b'video' b'off' b'his' b'complete' b'novel' b'"' b'.' b'The' b'population' b'of' b'Ceres' b'will' b'be' b'negative' b'for' b'strictly' b'@-@' b'hawk'
b'to' b'come' b'into' b'Year' b'1' b'.' b'There' b'is' b'a' b'pair' b'of' b'using' b'526' b',' b'O2' b',' b'nose' b',' b'<unk>' b'and'
b'coalitions' b'with' b'promyelocytic' b'officials' b'were' b'somewhat' b'developing' b'.' b'The' b'work' b'would' b'be' b'tested' b'as' b'a' b'hunt' b'to' b'Castle' b'network' b'including'
b'possible' b'gear' b'.' b'<eos>' b'<eos>' b'=' b'=' b'Behavior' b'=' b'=' b'<eos>' b'<eos>' b'<unk>' b'Michael' b'David' b'J.' b'M.' b'hilarious' b'(' b'died'
b'port' b'6' b':' b'12' b'<eos>' b'Ffordd' b'admirable' b'reality' b')' b'<eos>' b'trade' b'classifications' b',' b'without' b'a' b'creator' b';' b'of' b'even' b'@-@'
b'narial' b'earth' b',' b'building' b'rare' b'sounds' b',' b'Ridgway' b'contents' b',' b'any' b'GAA' b'in' b'air' b',' b'bleeding' b'.' b'<eos>' b'John' b'Leonard'
b'Rick' b'Smith' b'(' b'Evangeline' b'J.' b'Male' b')' b',' b'who' b'are' b'also' b'known' b'to' b'be' b'generally' b'portrayed' b'as' b'director' b'of' b'the'
b'Roman' b'origin' b'of' b'Sport' b'@-@' b'class' b'consent' b',' b'a' b'new' b'example' b'of' b'high' b'non' b'@-@' b'Crusader' b'forces' b'could' b'be' b'found'
b'by' b'<unk>' b'the' b'death' b'of' b'fish' b'highways' b'.' b'<eos>' b'<eos>' b'=' b'=' b'Background' b'=' b'=' b'<eos>' b'<eos>' b'The' b'majority' b'of'
b'year' b',' b'Superman' b',' b'was' b'also' b'built' b'into' b'alphabet' b'.' b'The' b'NW' b'were' b'written' b'by' b'other' b'astronomers' b'such' b'as' b'<unk>'
b'Jermaine' b'Farr' b',' b'with' b'respond' b'to' b'power' b'(' b'reorganize' b')' b'.' b'These' b'birds' b'have' b'had' b'hosted' b'North' b'AIDS' b'since' b'vocalization'
b'.' b'It' b'depicting' b'an' b'Normal' b'female' b'extended' b'after' b',' b'leaving' b'Petrie' b'resembled' b'Taylor' b'issues' b'has' b'significant' b'governmental' b'features' b',' b'called'
b'it' b',' b'"' b'Parts' b'as' b'well' b'to' b'kill' b'us' b'from' b'Haifa' b'is' b'an' b'gift' b'off' b'them' b'.' b'"' b'In' b'a'
b'review' b'that' b'Downs' b',' b'"' b'Every' b'blames' b'recent' b'human' b'parallels' b'you' b'is' b'Zeller' b'envisioned' b',' b'you' b'The' b'last' b'an' b'middle'
b'adult' b'person' b'in' b'ratio' b'of' b'male' b'throwing' b'lists' b'daily' b'letters' b'even' b',' b'attack' b',' b'and' b'inflict' b'you' b'into' b'Lost' b','
b'but' b'you' b'Rock' b'have' b'access' b'to' b'the' b'Mendip' b'conception' b'who' b"'re" b'overthrow' b'what' b'everything' b'in' b'than' b'store' b'particles' b'.' b'"'
b'The' b'face' b'recognized' b'Innis' b'was' b'of' b'unrepentant' b'Ulaid' b'.' b'glider' b'rent' b'for' b'Sister' b'Weber' b'are' b'exposing' b'to' b'seek' b'during' b'the'
b'hear' b'film' b'dislike' b"'s" b'staged' b'alignment' b'.' b'Another' b'cloth' b'was' b'only' b'impressed' b'by' b'Lab' b',' b'they' b'also' b'occasionally' b'learnt' b'a'
b'listener' b'.' b'<eos>' b'As' b'Plunkett' b"'s" b'death' b',' b'many' b'images' b'entrusted' b'to' b'join' b'items' b'display' b'models' b'than' b'foot' b'in' b'British'
b'countries' b'.' b'<unk>' b'indicated' b'is' b'also' b'safe' b'to' b'decide' b'down' b'McFarland' b',' b'even' b'that' b'searching' b'approaches' b'a' b'winds' b'for' b'two'
b'years' b'of' b'established' b'.' b'It' b'is' b'safe' b'that' b'<unk>' b'responded' b'in' b'(' b'the' b'19th' b'century' b',' b'including' b'A.' b"'\xc3\xa9tat" b';'
b'it' b'will' b'be' b'in' b'their' b'longer' b',' b'propel' b'"' b'<unk>' b'"' b',' b'which' b'aiding' b'God' b'@-@' b'black' b'overly' b',' b'astronomical'
b',' b'business' b',' b'<unk>' b',' b'<unk>' b',' b'or' b'grey' b'timeline' b'by' b'dismissal' b'before' b'mutualistic' b',' b'and' b'substrate' b'attention' b'given' b'as'
b'a' b'certain' b'species' b'of' b'153' b'stages' b'.' b'<unk>' b'in' b'toilet' b'can' b'be' b'found' b'to' b'signs' b'of' b'450' b',' b'compared' b'to'
b'50' b'%' b'closer' b',' b'while' b'manuscripts' b'may' b'be' b'"' b'distinguished' b'it' b'"' b'.' b'Incubation' b'resemble' b'Jordan' b'a' b'extremes' b',' b'Illinois'
b'concluding' b'much' b'of' b'the' b'player' b"'s" b'earlier' b'the' b'<unk>' b'broods' b'policies' b'.' b'<eos>' b'As' b'a' b'year' b',' b'he' b'is' b'found'
b'to' b'scare' b'taking' b'place' b'upon' b'behind' b'other' b'device' b',' b'including' b'its' b'further' b'sequence' b',' b'which' b'saw' b'him' b'a' b'painting' b'of'
b'conspiracy' b'that' b'enters' b'<unk>' b'to' b'cook' b'.' b'By' b'this' b'attacks' b',' b'they' b'are' b'shown' b'that' b'<unk>' b'(' b'an' b'one' b'@-@'
b'year' b')' b',' b'"' b'vision' b'(' b'still' b'most' b'equivalent' b'mourning' b')' b',' b'a' b'high' b'man' b'or' b'sings' b'large' b'Bruins' b'and'
b'rifles' b'all' b'by' b'night' b'<unk>' b',' b'not' b'nursing' b'.' b'"' b'Some' b'authors' b'like' b'H.' b'<unk>' b'<unk>' b'is' b'a' b'pure' b'character'
b'.' b'The' b'Admiralty' b'covers' b'Bob' b'cottonwood' b',' b'a' b'reflection' b'that' b'God' b'heard' b'parallel' b'.' b'reporters' b'went' b'forward' b'with' b'his' b'unusually'
b'controversial' b'Fern\xc3\xa1ndez' b',' b'back' b'"' b'that' b'many' b'authors' b"'re" b'forbidden' b'between' b'Black' b'Island' b'worker' b'!' b"'" b'learns' b'"' b'(' b'2006'
b')' b',' b'whose' b'<unk>' b'will' b'be' b'seen' b'as' b'a' b'child' b'.' b'Scully' b'is' b'trouble' b'apart' b'in' b'the' b'nominally' b',' b'and'
b'only' b'they' b'can' b'not' b'specifically' b'specify' b'after' b'they' b'could' b'be' b'rapidly' b'known' b'.' b'However' b',' b'it' b'may' b'assassinate' b'double' b'in'
b'other' b'ways' b',' b'even' b'because' b'he' b'provide' b'11' b'shock' b',' b'<unk>' b'the' b'Canary' b'Sun' b'breaker' b'.' b'<unk>' b'even' b'<unk>' b'by'
b'a' b'variety' b'of' b'other' b'factors' b',' b'which' b'Canterbury' b'doesn' b"'t" b'be' b'named' b'as' b'they' b'have' b'the' b'127th' b'mention' b'.' b'flocks'
b'fail' b'to' b'be' b'Allah' b',' b'depressed' b'peninsula' b',' b'<unk>' b',' b'and' b'@-@' b'head' b'ice' b'<unk>' b',' b'which' b'may' b'be' b'applied'
b'to' b'both' b'New' b'Zealand' b'.' b'The' b'food' b'and' b'so' b'they' b'can' b'react' b'into' b'Blue' b'or' b'eye' b'itself' b'.' b'They' b'may'
b'improve' b'their' b'position' b'complimented' b'up' b'or' b'place' b'resulted' b'on' b'all' b'Alfa' b'to' b'keep' b'care' b'of' b'Ceres' b',' b'orbiting' b'or' b'wide'
b',' b'then' b'by' b'its' b'space' b'.' b'<unk>' b',' b'they' b'were' b'will' b'try' b'the' b'kakapo' b'of' b'unusual' b',' b'<unk>' b'<unk>' b'or'
b'synthesize' b'Dead' b'(' b'860' b'<unk>' b'<unk>' b')' b'on' b'Activision' b'rather' b'@-@' b'thirds' b'of' b'spotlight' b'its' b'spectrum' b':' b'dying' b',' b'when'
b'British' b'behaviour' b'was' b'a' b'calculate' b'compound' b'to' b'merge' b',' b'with' b'some' b'chicks' b'to' b'use' b'their' b'bestow' b'.' b'It' b'may' b'indicate'

```




’s 没有 GPT-2，但看起来模型已经开始学习
语言的结构！




 我们’ 几乎准备好演示动态量化。我们只需要定义更多
辅助函数:






```
bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# create test data set
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into ``bsz`` parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the ``bsz`` batches.
    return data.view(bsz, -1).t().contiguous()

test_data = batchify(corpus.test, eval_batch_size)

# Evaluation functions
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def repackage_hidden(h):
 """Wraps hidden states in new Tensors, to detach them from their history."""

  if isinstance(h, torch.Tensor):
      return h.detach()
  else:
      return tuple(repackage_hidden(v) for v in h)

def evaluate(model_, data_source):
    # Turn on evaluation mode which disables dropout.
    model_.eval()
    total_loss = 0.
    hidden = model_.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model_(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

```






 4. 测试动态量化
 [¶](#test-dynamic-quantization "固定链接到此标题")
----------------------------------------------------------------------------------------------



 最后，我们可以在模型上调用
 `torch.quantization.quantize_dynamic`
！
具体来说，



* 我们指定我们希望
模型中的
 `nn.LSTM`
 和
 `nn.Linear`
 模块被量化
* 我们指定我们希望将权重转换为
 ` int8`
 值





```
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)

```






```
LSTMModel(
  (drop): Dropout(p=0.5, inplace=False)
  (encoder): Embedding(33278, 512)
  (rnn): DynamicQuantizedLSTM(512, 256, num_layers=5, dropout=0.5)
  (decoder): DynamicQuantizedLinear(in_features=256, out_features=33278, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
)

```




 模型看起来一样；这对我们有什么好处？首先，我们看到
模型大小显着减小：






```
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)

```






```
Size (MB): 113.944064
Size (MB): 79.738484

```




 其次，我们看到推理时间更快，评估损失没有差异：




 注意：我们将线程数设置为 1 以进行单线程比较，因为量化
模型运行单线程。






```
torch.set_num_threads(1)

def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)
    elapsed = time.time() - s
    print('''loss: {0:.3f}elapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

time_model_evaluation(model, test_data)
time_model_evaluation(quantized_model, test_data)

```






```
loss: 5.167
elapsed time (seconds): 200.1
loss: 5.168
elapsed time (seconds): 111.7

```




 在 MacBook Pro 上本地运行此程序，无需量化，推理需要大约 200 秒，
而使用量化则只需大约 100 秒。






 结论
 [¶](#conclusion "此标题的永久链接")
---------------------------------------------------------------------



 动态量化是减小模型大小的简单方法，同时
对准确性的影响有限。




感谢您的阅读！一如既往，我们欢迎任何反馈，因此请在[此处](https://github.com/pytorch/pytorch/issues)
 创建问题
（如果有任何反馈）。




**脚本总运行时间:** 
 ( 5 分 20.298 秒)
