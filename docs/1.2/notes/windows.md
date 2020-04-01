# Windows FAQ  

> 译者：[冯宝宝](https://github.com/PEGASUS1993)

## 从源码中构建  

### 包含可选组件  

Windows PyTorch有两个受支持的组件：MKL和MAGMA。 以下是使用它们构建的步骤。  

```py
REM Make sure you have 7z and curl installed.

REM Download MKL files
curl https://s3.amazonaws.com/ossci-windows/mkl_2018.2.185.7z -k -O
7z x -aoa mkl_2018.2.185.7z -omkl

REM Download MAGMA files
REM cuda90/cuda92/cuda100 is also available in the following line.
set CUDA_PREFIX=cuda80
curl -k https://s3.amazonaws.com/ossci-windows/magma_2.4.0_%CUDA_PREFIX%_release.7z -o magma.7z
7z x -aoa magma.7z -omagma

REM Setting essential environment variables
set "CMAKE_INCLUDE_PATH=%cd%\\mkl\\include"
set "LIB=%cd%\\mkl\\lib;%LIB%"
set "MAGMA_HOME=%cd%\\magma"

```

### 为Windows构建加速CUDA  

Visual Studio当前不支持并行自定义任务。 作为替代方案，我们可以使用Ninja来并行化CUDA构建任务。 只需键入几行代码即可使用它。 

```
REM Let's install ninja first.
pip install ninja

REM Set it as the cmake generator
set CMAKE_GENERATOR=Ninja  
``` 

### 脚本一键安装  

你可以参考[这些脚本](https://github.com/peterjc123/pytorch-scripts)。它会给你指导方向。  

## 扩展 

### CFEI扩展  

对[CFFI](https://cffi.readthedocs.io/en/latest/)扩展的支持是非常试验性的。在Windows下启用它通常有两个步骤。

首先，在Extension对象中指定其他库以使其在Windows上构建。   

```py
ffi = create_extension(
    '_ext.my_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_compile_args=["-std=c99"],
    libraries=['ATen', '_C'] # Append cuda libaries when necessary, like cudart
)

```  
其次，这是“由`extern THCState *state`状态引起的未解决的外部符号状态”的工作场所;

将源代码从C更改为C ++。 下面列出了一个例子。 

```py
#include <THC/THC.h>
#include <ATen/ATen.h>

THCState *state = at::globalContext().thc_state;

extern "C" int my_lib_add_forward_cuda(THCudaTensor *input1, THCudaTensor *input2,
                                        THCudaTensor *output)
{
    if (!THCudaTensor_isSameSizeAs(state, input1, input2))
    return 0;
    THCudaTensor_resizeAs(state, output, input1);
    THCudaTensor_cadd(state, output, input1, 1.0, input2);
    return 1;
}

extern "C" int my_lib_add_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input)
{
    THCudaTensor_resizeAs(state, grad_input, grad_output);
    THCudaTensor_fill(state, grad_input, 1);
    return 1;
}

```  

### C++扩展  

与前一种类型相比，这种类型的扩展具有更好的支持。不过它仍然需要一些手动配置。首先，打开VS 2017的x86_x64交叉工具命令提示符。然后，在其中打开Git-Bash。它通常位于C：\Program Files\Git\git-bash.exe中。最后，您可以开始编译过程。  

## 安装  

### 在Win32 找不到安装包  

```py
Solving environment: failed

PackagesNotFoundError: The following packages are not available from current channels:

- pytorch

Current channels:
- https://conda.anaconda.org/pytorch/win-32
- https://conda.anaconda.org/pytorch/noarch
- https://repo.continuum.io/pkgs/main/win-32
- https://repo.continuum.io/pkgs/main/noarch
- https://repo.continuum.io/pkgs/free/win-32
- https://repo.continuum.io/pkgs/free/noarch
- https://repo.continuum.io/pkgs/r/win-32
- https://repo.continuum.io/pkgs/r/noarch
- https://repo.continuum.io/pkgs/pro/win-32
- https://repo.continuum.io/pkgs/pro/noarch
- https://repo.continuum.io/pkgs/msys2/win-32
- https://repo.continuum.io/pkgs/msys2/noarch

```
Pytorch不能在32位系统中工作运行。请安装使用64位的Windows和Python。  

### 导入错误  

```
from torch._C import *

ImportError: DLL load failed: The specified module could not be found.
```

问题是由基本文件丢失导致的。实际上，除了VC2017可再发行组件和一些mkl库之外，我们几乎包含了PyTorch对conda包所需的所有基本文件。您可以通过键入以下命令来解决此问题。

```
conda install -c peterjc123 vc vs2017_runtime
conda install mkl_fft intel_openmp numpy mkl
```

至于wheel包(轮子)，由于我们没有包含一些库和VS2017可再发行文件，请手动安装它们。可以下载[VS 2017可再发行安装程序]((https://aka.ms/vs/15/release/VC_redist.x64.exe))。你还应该注意你的Numpy的安装。 确保它使用MKL而不是OpenBLAS版本的。您可以输入以下命令。  

```
pip install numpy mkl intel-openmp mkl_fft
```  

另外一种可能是你安装了GPU版本的Pytorch但是电脑中并没有NVIDIA的显卡。碰到这种情况，就把GPU版本的Pytorch换成CPU版本的就好了。  


```
from torch._C import *

ImportError: DLL load failed: The operating system cannot run %1.
```

这实际上是Anaconda的上游问题。使用conda-forge通道初始化环境时,将出现此问题。您可以通过此命令修复intel-openmp库。  

## 使用(并行处理）  

### 无if语句保护的多进程处理错误  

```py
RuntimeError:
    An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

   This probably means that you are not using fork to start your
   child processes and you have forgotten to use the proper idiom
   in the main module:

       if __name__ == '__main__':
           freeze_support()
           ...

   The "freeze_support()" line can be omitted if the program
   is not going to be frozen to produce an executable.

```  

在Windows上实现`多进程处理`是不同的，它使用的是spawn而不是fork。 因此，我们必须使用if子句包装代码，以防止代码执行多次。将您的代码重构为以下结构。 

```
import torch

def main()
    for i, data in enumerate(dataloader):
        # do something here

if __name__ == '__main__':
    main()
```

### 多进程处理错误“坏道”  

```
ForkingPickler(file, protocol).dump(obj)

BrokenPipeError: [Errno 32] Broken pipe
```

当在父进程完成发送数据之前子进程结束时，会发生此问题。您的代码可能有问题。您可以通过将DataLoader的num_worker减少为零来调试代码，并查看问题是否仍然存在。  

### 多进程处理错误“驱动程序关闭”  

```
Couldn’t open shared file mapping: <torch_14808_1591070686>, error code: <1455> at torch\lib\TH\THAllocator.c:154

[windows] driver shut down
```

请更新您的显卡驱动程序。如果这种情况持续存在，则可能是您的显卡太旧或所需要的计算能力对您的显卡负担太重。请根据[这篇文章]((https://www.pugetsystems.com/labs/hpc/Working-around-TDR-in-Windows-for-a-better-GPU-computing-experience-777/).)更新TDR设置。

### CUDA IPC操作  

```
THCudaCheck FAIL file=torch\csrc\generic\StorageSharing.cpp line=252 error=63 : OS call failed or operation not supported on this OS
```

Windows不支持它们。在CUDA张量上进行并行处理这样的事情无法成功，有两种选择:  

1\.不要使用并行处理。将Data Loader的num_worker设置为零。  

2\.采用共享CPU张量方法。确保您的自定义`DataSet`返回CPU张量。

