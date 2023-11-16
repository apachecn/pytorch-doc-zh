


# (beta) BERT 上的动态量化 [¶](#beta-dynamic-quantization-on-bert "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/dynamic_quantization_bert_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html>





 提示




 要充分利用本教程，我们建议使用此
 [Colab 版本](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/dynamic_quantization_bert_tutorial.ipynb ) 
 。这将允许您尝试下面提供的信息。





**作者** 
 :
 [Jianyu Huang](https://github.com/jianyuh)




**审阅者** 
 :
 [Raghuraman Krishnamoorthi](https://github.com/raghuramank100)




**编辑者** 
 :
 [Jessica Lin](https://github.com/jlin27)





## 简介 [¶](#introduction "此标题的永久链接")




 在本教程中，我们将在 BERT 模型上应用动态量化，紧密遵循 [HuggingFace
Transformers 示例](https://github.com/huggingface/transformers) 中的 BERT 模型。
 。\通过这个循序渐进的旅程，我们希望演示如何
将 BERT 等众所周知的最先进模型转换为动态
量化模型。



* BERT，即来自 Transformers 的双向嵌入表示，
是一种预训练语言表示的新方法，
可在许多流行的
自然语言处理 (NLP) 任务（例如问答）上实现最先进的准确度结果、
文本分类等。原始论文可以在
 [此处](https://arxiv.org/pdf/1810.04805.pdf)
.
* PyTorch 中的动态量化支持将浮点模型转换为具有静态 int8 或
的量化模型用于激活权重和动态量化的 float16 数据类型。当权重量化为 int8 时，激活会动态（每批次）量化为 int8。在 PyTorch 中，我们有
 [torch.quantization.quantize_dynamic API](https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic) 
 ，
它将指定的模块替换为动态仅权重量化版本并输出量化模型。
* 我们在 [Microsoft Research Paraphrase Corpus (MRPC) 任务](https://www.microsoft.com/en) 上演示了准确性和推理性能结果-us/download/details.aspx?id=52398) 
 通用语言理解评估基准
 [(GLUE)](https://gluebenchmark.com/) 
 。 MRPC（Dolan 和 Brockett，2005）是从在线新闻源中自动提取的句子对语料库，并由人工注释该对中的句子是否在语义上等效。由于类别不平衡（68%
正，32% 负），我们遵循常见做法并报告
 [F1 分数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics。 f1_score.html) 
.
MRPC 是语言对分类的常见 NLP 任务，如下所示
。


![https://pytorch.org/tutorials/_images/bert.png](https://pytorch.org/tutorials/_images/bert.png)


## 1. 设置 [¶](#setup "固定链接到此标题")




### 1.1 安装 PyTorch 和 HuggingFace Transformers [¶](#install-pytorch-and-huggingface-transformers "永久链接到此标题")



 要开始本教程，请让’s 首先按照安装说明进行操作
 PyTorch
 [此处](https://github.com/pytorch/pytorch/#installation)
 和 HuggingFace Github Repo\ n [此处](https://github.com/huggingface/transformers#installation) 
.
此外，我们还安装
 [scikit-learn](https://github.com/scikit-learn/scikit -learn)
 包，因为我们将重用其
内置的 F1 分数计算辅助函数。






```
pip install sklearn
pip install transformers==4.29.2

```




 因为我们将使用 PyTorch 的测试版，所以
建议安装最新版本的 torch 和 torchvision。您
可以在[此处](https://pytorch.org/get-started/locally/)
 找到有关本地安装的最新说明。例如，要在
Mac 上安装：






```
yes y | pip uninstall torch tochvision
yes y | pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html

```





### 1.2 导入必要的模块 [¶](#import-the-necessary-modules "永久链接到此标题")



 在此步骤中，我们导入本教程所需的 Python 模块。






```
import logging
import numpy as np
import os
import random
import sys
import time
import torch

from argparse import Namespace
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

logging.getLogger("transformers.modeling_utils").setLevel(
   logging.WARN)  # Reduce logging

print(torch.__version__)

```




 我们设置线程数来比较 FP32 和 INT8 性能之间的单线程性能。
在教程的最后，用户可以通过构建具有右并行后端的 PyTorch 来设置其他线程数。






```
torch.set_num_threads(1)
print(torch.__config__.parallel_info())

```





### 1.3 了解辅助函数 [¶](#learn-about-helper-functions "永久链接到此标题")



 辅助函数内置于转换器库中。我们主要使用
以下辅助函数：一个用于将文本示例转换
为特征向量；另一个用于测量
预测结果的 F1 分数。




 [glue_convert_examples_to_features](https://github.com/huggingface/transformers/blob/master/transformers/data/processors/glue.py) 
 函数将文本转换为输入特征:



* 对输入序列进行标记；
* 在开头插入 [CLS]；
* 在第一个句子和第二个句子之间插入 [SEP]，以及
* 生成标记类型 ID 以指示是否令牌属于第一个序列或第二个序列。



 [glue_compute_metrics](https://github.com/huggingface/transformers/blob/master/transformers/data/processors/glue.py) 
 函数的计算指标为\ 
 [F1 分数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) 
 ，可以解释为精度和召回率的加权平均值， 
其中 F1 分数达到其最佳值为 1，最差分数为 0。
精确率和召回率对 F1 分数的相对贡献是相等的。



* F1 分数的公式为：



 \[F1 = 2 * (\text{精度} * \text{召回}) /(\text{精度} + \text{召回})

\ ]



### 1.4 下载数据集 [¶](#download-the-dataset "永久链接到此标题")



 在运行 MRPC 任务之前，我们通过运行
 [此脚本](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)下载
 [GLUE 数据](https://gluebenchmark.com/tasks)
 ) 
 并将其解压到目录
 `glue_data`
.






```
python download_glue_data.py --data_dir='glue_data' --tasks='MRPC'

```





## 2. 微调 BERT 模型 [¶](#fine-tune-the-bert-model "Permalink to this header")



BERT 的精神是预训练语言表示，然后以最小的任务相关参数对各种任务上的深度双向表示进行微调，并实现状态艺术成果。在本教程中，我们将重点关注
使用预训练的 BERT 模型进行微调，以对 MRPC 任务上的语义等效
句子对进行分类。




 要针对 MRPC 任务微调预训练的 BERT 模型（
 HuggingFace 转换器中的 
 `bert-base-uncased`
 模型），您可以按照命令

 [示例](https://github.com/huggingface/transformers/tree/master/examples#mrpc) 
 :






```
export GLUE_DIR=./glue_data
export TASK_NAME=MRPC
export OUT_DIR=./$TASK_NAME/
python ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 100000 \
    --output_dir $OUT_DIR

```




 我们为 MRPC 任务提供了经过微调的 BERT 模型
 [此处](https://download.pytorch.org/tutorial/MRPC.zip) 
.
为了节省时间，您可以下载该模型文件 (~400 MB) 直接复制到本地文件夹
 `$OUT_DIR`
.




### 2.1 设置全局配置 [¶](#set-global-configurations "永久链接到此标题")



 这里我们设置全局配置，用于评估动态量化之前和之后的微调 BERT
模型。






```
configs = Namespace()

# The output directory for the fine-tuned model, $OUT_DIR.
configs.output_dir = "./MRPC/"

# The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
configs.data_dir = "./glue_data/MRPC"

# The model name or path for the pre-trained model.
configs.model_name_or_path = "bert-base-uncased"
# The maximum length of an input sequence
configs.max_seq_length = 128

# Prepare GLUE task.
configs.task_name = "MRPC".lower()
configs.processor = processors[configs.task_name]()
configs.output_mode = output_modes[configs.task_name]
configs.label_list = configs.processor.get_labels()
configs.model_type = "bert".lower()
configs.do_lower_case = True

# Set the device, batch size, topology, and caching flags.
configs.device = "cpu"
configs.per_gpu_eval_batch_size = 8
configs.n_gpu = 0
configs.local_rank = -1
configs.overwrite_cache = False


# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

```





### 2.2 加载微调后的 BERT 模型 [¶](#load-the-fine-tuned-bert-model "永久链接到此标题")



 我们从
 `configs.output_dir`
 加载分词器和微调的 BERT 序列分类器模型
(FP32)。






```
tokenizer = BertTokenizer.from_pretrained(
    configs.output_dir, do_lower_case=configs.do_lower_case)

model = BertForSequenceClassification.from_pretrained(configs.output_dir)
model.to(configs.device)

```





### 2.3 定义标记化和评估函数 [¶](#define-the-tokenize-and-evaluation-function "Permalink to this header")



 我们重用来自 [Huggingface](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py) 的标记化和评估函数 
 。






```
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info(" Num examples = %d", len(eval_dataset))
        logger.info(" Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'labels':         batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info(" %s = %s", key, str(result[key]))
                writer.write("%s = %s" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

```





## 3. 应用动态量化 [¶](#apply-the-dynamic-quantization "永久链接到此标题")




 我们在模型上调用
 `torch.quantization.quantize_dynamic`
 以在 HuggingFace BERT 模型上应用
动态量化。具体来说，



* 我们指定我们希望模型中的 torch.nn.Linear 模块
被量化；
* 我们指定我们希望权重转换为量化的 int8
值。





```
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print(quantized_model)

```




### 3.1 检查模型大小 [¶](#check-the-model-size "永久链接到此标题")



 让’s 首先检查模型大小。我们可以观察到模型大小显着减小
（FP32 总大小：438 MB；INT8 总大小：181 MB）：






```
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)

```




 本教程中使用的 BERT 模型 (
 `bert-base-uncased`
 ) 的词汇大小 V 为 30522。嵌入大小为 768，词嵌入表的总大小为~ 4（字节/FP32）* 30522 * 768 =
90 MB。因此，借助量化，
非嵌入表部分的模型大小从 350 MB（FP32 模型）减少到 90 MB
（INT8 模型）。





### 3.2 评估推理精度和时间 [¶](#evaluate-the-inference-accuracy-and-time "Permalink to this header")



 接下来，让’s 比较原始 FP32 模型与动态量化后
INT8 模型之间的推理时间和评估准确度。






```
def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    result = evaluate(configs, model, tokenizer, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))

# Evaluate the original FP32 BERT model
time_model_evaluation(model, configs, tokenizer)

# Evaluate the INT8 BERT model after the dynamic quantization
time_model_evaluation(quantized_model, configs, tokenizer)

```




 在 MacBook Pro 上本地运行此程序，不进行量化、推理
（对于 MRPC 数据集中的所有 408 个示例）大约需要 160 秒，而
进行量化则只需要大约 90 秒。我们将在 Macbook Pro 上运行量化 BERT 模型推理的结果总结如下：






```
| Prec | F1 score | Model Size | 1 thread | 4 threads |
| FP32 |  0.9019  |   438 MB   | 160 sec  | 85 sec    |
| INT8 |  0.902   |   181 MB   |  90 sec  | 46 sec    |

```


在 MRPC 任务的微调 BERT 模型上应用训练后动态量化后，我们的 F1 分数准确度降低了 0.6%。作为比较，在[最近的论文](https://arxiv.org/pdf/1910.06188.pdf)（表 1）中，通过应用训练后动态量化实现了 0.8788，通过应用训练后动态量化实现了 0.8956通过应用量化感知训练。主要区别在于我们在 PyTorch 中支持对称量化，而该论文仅支持对称量化。




 请注意，在本教程中，我们将线程数设置为 1 以进行单线程
比较。我们还支持这些量化 INT8 运算符的运算内并行化。用户现在可以通过
 `torch.set_num_threads(N)`
 设置多线程（
 `N`
 是
tra-op 并行化线程的数量）。启用操作内并行化支持的一个初步要求是使用正确的[后端](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#build-options)构建 PyTorch，例如作为 OpenMP、Native 或 TBB。
您可以使用
 `torch.__config__.parallel_info()`
 检查并行化设置。在使用 PyTorch 和本机后端进行并行化的同一台 MacBook Pro 上，
我们可以花费大约 46 秒来
处理 MRPC 数据集的评估。





### 3.3 序列化量化模型 [¶](#serialize-the-quantized-model "永久链接到此标题")



 我们可以在跟踪模型后使用 
 
 torch.jit.save
 
 序列化并保存量化模型以供将来使用。






```
def ids_tensor(shape, vocab_size):
    # Creates a random int32 tensor of the shape within the vocab size
    return torch.randint(0, vocab_size, shape=shape, dtype=torch.int, device='cpu')

input_ids = ids_tensor([8, 128], 2)
token_type_ids = ids_tensor([8, 128], 2)
attention_mask = ids_tensor([8, 128], vocab_size=2)
dummy_input = (input_ids, attention_mask, token_type_ids)
traced_model = torch.jit.trace(quantized_model, dummy_input)
torch.jit.save(traced_model, "bert_traced_eager_quant.pt")

```




 要加载量化模型，我们可以使用
 
 torch.jit.load







```
loaded_quantized_model = torch.jit.load("bert_traced_eager_quant.pt")

```





## 结论 [¶](#conclusion "永久链接到此标题")




 在本教程中，我们演示了如何将 BERT 等众所周知的最先进的 NLP 模型转换为动态量化
模型。动态量化可以减小模型的大小，但
对精度的影响有限。




感谢您的阅读！一如既往，我们欢迎任何反馈，因此请在[此处](https://github.com/pytorch/pytorch/issues)创建
问题
（如果您有
任何问题）。





## 参考文献 [¶](#references "此标题的永久链接")




 [1] J.Devlin、M. Chang、K. Lee 和 K. Toutanova，
 [BERT：用于语言理解的深度双向变换器的预训练 (2018)](https://arxiv.org/pdf/1810.04805.pdf) 
.




 [2]
 [HuggingFace 变压器](https://github.com/huggingface/transformers) 
.




 [3] O. Zafrir、G. Boudoukh、P. Izsak 和 M. Wasserblat (2019)。
 [Q8BERT:
量化 8 位 BERT](https://arxiv.org/pdf/1910.06188.pdf) 
.









