# Torchmultimodal 教程：微调 Flava

> 译者：[方小生]()
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/beginner/flava_finetuning_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/beginner/flava_finetuning_tutorial.html>

多模态人工智能最近变得非常流行，因为它无处不在，从图像字幕和视觉搜索等用例到从文本生成图像等最近的应用。 **TorchMultimodal 是一个由 Pytorch 提供支持的库，由构建块和端到端示例组成，旨在促进和加速多模态研究。**

 在本教程中，我们将演示如何使用 TorchMultimodal 库中名为 FLAVA 的**预训练 SoTA 模型来微调多模态任务，即视觉问答 (VQA)**。该模型由两个基于单模态变压器的文本和图像编码器以及一个结合两种嵌入的多模态编码器组成。它使用对比、图像文本匹配以及文本、图像和多模态掩蔽损失进行预训练。

##  安装

本教程将使用 TextVQA 数据集和 Hugging Face 数据集。因此，除了 TorchMultimodal.bert tokenizer 之外，您还需要安装数据集和transformers。

**NOTE:**

在 Google Colab 中运行本教程时，通过创建新单元并运行以下命令来安装所需的包：

```
!pip install torchmultimodal-nightly
!pip install datasets
!pip install transformers
```

## 步骤

1. 通过运行以下命令将 Hugging Face 数据集下载到计算机上的目录：

   ```bash
   wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz
   tar xf vocab.tar.gz
   ```

   **NOTE:**

   如果您在 Google Colab 中运行本教程，请在新单元中运行这些命令，并在这些命令前面加上感叹号 (!)

2. 在本教程中，我们将 VQA 视为分类任务，其中输入是图像和问题(文本)，输出是答案类。因此，我们需要下载带有答案类的词汇文件并创建标签映射的答案。

   我们还加载了来自 Hugging Face 的包含 34602 个训练样本(图像、问题和答案)的 textvqa 数据集。我们看到有 3997 个答案类，其中包括代表未知答案的类。

   

   ```python
   with open("data/vocabs/answers_textvqa_more_than_1.txt") as f:
     vocab = f.readlines()
   
   answer_to_idx = {}
   for idx, entry in enumerate(vocab):
     answer_to_idx[entry.strip("\n")] = idx
   print(len(vocab))
   print(vocab[:5])
   
   from datasets import load_dataset
   dataset = load_dataset("textvqa")
   ```

   **Out:**

   ```
   3997
   ['<unk>\n', 'nokia\n', 'ec\n', 'virgin\n', '2011\n']
   
   Downloading builder script:   0%|          | 0.00/5.02k [00:00<?, ?B/s]
   Downloading builder script: 100%|##########| 5.02k/5.02k [00:00<00:00, 27.0MB/s]
   
   Downloading metadata:   0%|          | 0.00/13.3k [00:00<?, ?B/s]
   Downloading metadata: 100%|##########| 13.3k/13.3k [00:00<00:00, 39.3MB/s]
   
   Downloading readme:   0%|          | 0.00/13.2k [00:00<?, ?B/s]
   Downloading readme: 100%|##########| 13.2k/13.2k [00:00<00:00, 42.7MB/s]
   
   Downloading data files:   0%|          | 0/5 [00:00<?, ?it/s]
   因为太多省略。。。。。
   ```

   让我们显示数据集中的示例条目：

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   idx = 5
   print("Question: ", dataset["train"][idx]["question"])
   print("Answers: " ,dataset["train"][idx]["answers"])
   im = np.asarray(dataset["train"][idx]["image"].resize((500,500)))
   plt.imshow(im)
   plt.show()
   ```

   ![sphx_glr_flava_finetuning_tutorial_001](../../img/sphx_glr_flava_finetuning_tutorial_001.png)

   **Out:**

   ```
   Question:  what year is shown in the photo?
   Answers:  ['2011', '2011', '2011', '2011', '2011', '2011', '2011', '2011', '2011', '2011']
   ```

3. 接下来，我们编写transform 函数，将图像和文本转换为我们的模型可使用的tensor - 对于图像，我们使用 torchvision 中的变换来转换为tensor并调整大小为统一大小 - 对于文本，我们使用它们来标记(和填充)它们来自 Hugging Face - 对于答案(即标签)，我们采用最常出现的答案作为训练标签：BertTokenizer

   ```python
   import torch
   from torchvision import transforms
   from collections import defaultdict
   from transformers import BertTokenizer
   from functools import partial
   
   def transform(tokenizer, input):
     batch = {}
     image_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224,224])])
     image = image_transform(input["image"][0].convert("RGB"))
     batch["image"] = [image]
   
     tokenized=tokenizer(input["question"],return_tensors='pt',padding="max_length",max_length=512)
     batch.update(tokenized)
   
   
     ans_to_count = defaultdict(int)
     for ans in input["answers"][0]:
       ans_to_count[ans] += 1
     max_value = max(ans_to_count, key=ans_to_count.get)
     ans_idx = answer_to_idx.get(max_value,0)
     batch["answers"] = torch.as_tensor([ans_idx])
     return batch
   
   tokenizer=BertTokenizer.from_pretrained("bert-base-uncased",padding="max_length",max_length=512)
   transform=partial(transform,tokenizer)
   dataset.set_transform(transform)
   ```

   **Out:**

   ```
   Downloading (…)solve/main/vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]
   Downloading (…)solve/main/vocab.txt: 100%|##########| 232k/232k [00:00<00:00, 191MB/s]
   
   Downloading (…)okenizer_config.json:   0%|          | 0.00/28.0 [00:00<?, ?B/s]
   Downloading (…)okenizer_config.json: 100%|##########| 28.0/28.0 [00:00<00:00, 197kB/s]
   
   Downloading (…)lve/main/config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]
   Downloading (…)lve/main/config.json: 100%|##########| 570/570 [00:00<00:00, 4.05MB/s]
   ```

4. 最后，我们导入 from 。它默认加载预训练的 FLAVA 检查点，并包含分类头。flava_model_for_classificationtorchmultimodal 模型前向函数通过视觉编码器传递图像，通过文本编码器传递问题。然后图像和问题嵌入通过多模态编码器。与 CLS 令牌相对应的最终嵌入通过 MLP 头传递，最终给出每个可能答案的概率分布。

   ```python
   from torchmultimodal.models.flava.model import flava_model_for_classification
   model = flava_model_for_classification(num_classes=len(vocab))
   ```

   **Out:**

   ```
   flava_for_pretraining_unified_text_encoder.pt: 0.00B [00:00, ?B/s]
   flava_for_pretraining_unified_text_encoder.pt:   1%|          | 8.99M/1.43G [00:00<00:15, 89.9MB/s]
   flava_for_pretraining_unified_text_encoder.pt:   2%|2         | 30.7M/1.43G [00:00<00:08, 165MB/s]
   flava_for_pretraining_unified_text_encoder.pt:   4%|3         | 52.5M/1.43G [00:00<00:07, 189MB/s]
   因为太多省略。。。。。
   ```

5. 我们将数据集和模型放在一个玩具训练循环中，以演示如何训练模型进行 3 次迭代：

   ```python
   from torch import nn
   BATCH_SIZE = 2
   MAX_STEPS = 3
   from torch.utils.data import DataLoader
   
   train_dataloader = DataLoader(dataset["train"], batch_size= BATCH_SIZE)
   optimizer = torch.optim.AdamW(model.parameters())
   
   
   epochs = 1
   for _ in range(epochs):
     for idx, batch in enumerate(train_dataloader):
       optimizer.zero_grad()
       out = model(text = batch["input_ids"], image = batch["image"], labels = batch["answers"])
       loss = out.loss
       loss.backward()
       optimizer.step()
       print(f"Loss at step {idx} = {loss}")
       if idx >= MAX_STEPS-1:
         break
   ```

   **Out:**

   ```
   /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/transforms/functional.py:1603: UserWarning:
   
   The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
   
   Loss at step 0 = 8.290154457092285
   Loss at step 1 = 8.367232322692871
   Loss at step 2 = 8.210197448730469
   ```

   ## 结论

   本教程介绍了如何使用 TorchMultimodal 中的 FLAVA 微调多模式任务的基础知识。另请查看该库中的其他示例，例如 MDETR(用于对象检测的多模态模型)和 Omnivore(涵盖图像、视频和 3D 分类的多任务模型)。

    脚本总运行时间：(2分49.785秒)



