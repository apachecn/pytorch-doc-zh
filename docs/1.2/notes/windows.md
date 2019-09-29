# Windows 常见问题

## 从源大厦

### 包括任选的组分

目前的Windows PyTorch支持的两个组成部分：MKL与岩浆。这里有与他们建立的步骤。

    
    
    REM Make sure you have 7z and curl installed.
    
    REM Download MKL files
    curl https://s3.amazonaws.com/ossci-windows/mkl_2018.2.185.7z -k -O
    7z x -aoa mkl_2018.2.185.7z -omkl
    
    REM Download MAGMA files
    REM cuda100/cuda101 is also available for `CUDA_PREFIX`. There are also 2.4.0 binaries for cuda80/cuda92.
    REM The configuration could be `debug`or `release`for 2.5.0. Only `release`is available for 2.4.0.
    set CUDA_PREFIX=cuda90
    set CONFIG=release
    curl -k https://s3.amazonaws.com/ossci-windows/magma_2.5.0_%CUDA_PREFIX%_%CONFIG%.7z -o magma.7z
    7z x -aoa magma.7z -omagma
    
    REM Setting essential environment variables
    set "CMAKE_INCLUDE_PATH=%cd%\\mkl\\include"
    set "LIB=%cd%\\mkl\\lib;%LIB%"
    set "MAGMA_HOME=%cd%\\magma"
    

### CUDA加速构建于Windows

Visual Studio中不支持当前并行自定义任务。作为替代方案，我们可以使用`忍者 `并行CUDA建设任务。它可以通过输入代码只有几行使用。

    
    
    REM Let's install ninja first.
    pip install ninja
    
    REM Set it as the cmake generator
    set CMAKE_GENERATOR=Ninja
    

### 一个键安装脚本

你可以看看[这套脚本[HTG1。它会带路为您服务。](https://github.com/peterjc123/pytorch-scripts)

## 分机

### CFFI扩展

对于CFFI扩展的支持是非常实验性的。一般是有两个步骤，使其能够在Windows下。

首先，请在`分机 `对象的附加`库 `，使其在Windows上构建。

    
    
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
    

其次，在这里是“引发的`的extern  THCState  *状态解析外部符号状态;`”用于workground

改变源代码从C到C ++。一个例子如下所列。

    
    
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
    

### CPP扩展

与以往的相比这种类型的扩展有更好的支持。但是，它仍然需要一些手动配置。首先，你应该打开 **x86_x64跨工具命令提示符为VS
2017年[HTG1。然后，你就可以开始你的编译过程。**

## 安装

### 包装在Win-32频道未找到。

    
    
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
    

PyTorch不能在32位系统中工作。请使用Windows和Python的64位版本。

### 为什么没有Python的2包的Windows？

因为它不够稳定。还有一些是需要之前，我们正式发布它要解决的问题。您可以通过建立它自己。

### 导入错误

    
    
    from torch._C import *
    
    ImportError: DLL load failed: The specified module could not be found.
    

该问题是由重要文件丢失引起的。其实，我们几乎都可以看到PyTorch需要为康达包装除了VC2017可再发行组件和一些MKL库的基本文件。您可以通过键入以下命令来解决此问题。

    
    
    conda install -c peterjc123 vc vs2017_runtime
    conda install mkl_fft intel_openmp numpy mkl
    

至于这些轮子包，因为我们没有收拾一些libaries和VS2017再发行的文件，请确保您手动安装它们。在[ VS
2017年再分发安装](https://aka.ms/vs/15/release/VC_redist.x64.exe)可以下载。而且你还要注意你的NumPy的安装。确保它使用的不是OpenBLAS
MKL。您可以在下面的命令类型。

    
    
    pip install numpy mkl intel-openmp mkl_fft
    

另一个可能的原因可能是您使用的GPU版本，而NVIDIA显卡。请与CPU更换你的GPU封装。

    
    
    from torch._C import *
    
    ImportError: DLL load failed: The operating system cannot run %1.
    

这实际上是蟒蛇的上游问题。当你初始化畅达锻通道的环境中，这个问题将会出现。您可以通过此命令修复英特尔的OpenMP库。

    
    
    conda install -c defaults intel-openmp -f
    

## 使用量（多处理）

### 多处理错误，而不if子句保护

    
    
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
    

多处理 的`实施是不同的Windows，其使用`菌种 `而非`叉 `。因此，我们必须使用if子句来包装代码保护代码执行多次。重构代码为如下的结构。`

    
    
    import torch
    
    def main()
        for i, data in enumerate(dataloader):
            # do something here
    
    if __name__ == '__main__':
        main()
    

### 多处理错误“断管”

    
    
    ForkingPickler(file, protocol).dump(obj)
    
    BrokenPipeError: [Errno 32] Broken pipe
    

当子进程结束父进程完成发送数据之前，这个问题会发生。可能有一些错误代码。您可以通过减少`num_worker`为零[ `的DataLoader`
](../data.html#torch.utils.data.DataLoader
"torch.utils.data.DataLoader")调试代码，看看问题是否依然存在。

### 多处理错误“司机关闭”

    
    
    Couldn’t open shared file mapping: <torch_14808_1591070686>, error code: <1455> at torch\lib\TH\THAllocator.c:154
    
    [windows] driver shut down
    

请更新您的显卡驱动程序。如果这仍然存在，这可能是你的显卡太旧或计算是你的卡太重。请你照这个[信息](https://www.pugetsystems.com/labs/hpc/Working-
around-TDR-in-Windows-for-a-better-GPU-computing-experience-777/)更新TDR设置。

### CUDA IPC操作

    
    
    THCudaCheck FAIL file=torch\csrc\generic\StorageSharing.cpp line=252 error=63 : OS call failed or operation not supported on this OS
    

他们不支持Windows。喜欢的东西做的CUDA张量多能不能成功，有两个备选方案这一点。

1.不要用`多重处理 [HTG3。将`num_worker`为零的[ `的DataLoader`
](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader")。`

2.共享CPU张量来代替。请确保您的自定义`数据集 `返回CPU张量。

[Next ![](../_static/images/chevron-right-
orange.svg)](../community/contribution_guide.html "PyTorch Contribution
Guide") [![](../_static/images/chevron-right-orange.svg)
Previous](serialization.html "Serialization semantics")

* * *

©版权所有2019年，Torch 贡献者。
