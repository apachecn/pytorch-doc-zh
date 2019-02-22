# 为 pytorch 自定义 C 扩展

> 译者：[@飞龙](https://github.com/wizardforcel)

**作者**: [Soumith Chintala](http://soumith.ch)

## 第一步. 准备你的 C 代码

首先, 你需要编写你的 C 函数.

下面你可以找到模块的正向和反向函数的示例实现, 它将两个输入相加.

在你的 `.c` 文件中, 你可以使用 `#include &lt;TH/TH.h&gt;` 直接包含 TH, 以及使用 `#include &lt;THC/THC.h&gt;` 包含 THC.

ffi (外来函数接口) 工具会确保编译器可以在构建过程中找到它们.

```py
/* src/my_lib.c */
#include <TH/TH.h>

int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2,
THFloatTensor *output)
{
    if (!THFloatTensor_isSameSizeAs(input1, input2))
        return 0;
    THFloatTensor_resizeAs(output, input1);
    THFloatTensor_cadd(output, input1, 1.0, input2);
    return 1;
}

int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input)
{
    THFloatTensor_resizeAs(grad_input, grad_output);
    THFloatTensor_fill(grad_input, 1);
    return 1;
}

```

代码没有任何限制, 除了你必须准备单个头文件, 它会列出所有你想要从 Python 调用的函数.

它会由 ffi 用于生成合适的包装.

```py
/* src/my_lib.h */
int my_lib_add_forward(THFloatTensor *input1, THFloatTensor *input2, THFloatTensor *output);
int my_lib_add_backward(THFloatTensor *grad_output, THFloatTensor *grad_input);

```

现在, 你需要一个超短的文件, 它会构建你的自定义扩展:

```py
# build.py
from torch.utils.ffi import create_extension
ffi = create_extension(
name='_ext.my_lib',
headers='src/my_lib.h',
sources=['src/my_lib.c'],
with_cuda=False
)
ffi.build()

```

## 第二步: 在你的 Python 代码中包含它

你运行它之后, pytorch 会创建一个 `_ext` 目录, 并把 `my_lib` 放到里面.

包名称可以在最终模块名称之前, 包含任意数量的包 (包括没有). 如果构建成功, 你可以导入你的扩展, 就像普通的 Python 文件.

```py
# functions/add.py
import torch
from torch.autograd import Function
from _ext import my_lib

class MyAddFunction(Function):
    def forward(self, input1, input2):
        output = torch.FloatTensor()
        my_lib.my_lib_add_forward(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input = torch.FloatTensor()
        my_lib.my_lib_add_backward(grad_output, grad_input)
        return grad_input

```

```py
# modules/add.py
from torch.nn import Module
from functions.add import MyAddFunction

class MyAddModule(Module):
    def forward(self, input1, input2):
        return MyAddFunction()(input1, input2)

```

```py
# main.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from modules.add import MyAddModule

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)

model = MyNetwork()
input1, input2 = Variable(torch.randn(5, 5)), Variable(torch.randn(5, 5))
print(model(input1, input2))
print(input1 + input2)

```