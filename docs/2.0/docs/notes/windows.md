# Windows 常见问题解答 [¶](#windows-faq "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/windows>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/windows.html>


## 从源代码构建 [¶](#building-from-source "此标题的永久链接")


### 包含可选组件 [¶](#include-optional-components "永久链接到此标题")


 Windows PyTorch 有两个受支持的组件：MKL 和 MAGMA。以下是使用它们进行构建的步骤。


```
REM Make sure you have 7z and curl installed.

REM Download MKL files
curl https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z -k -O
7z x -aoa mkl_2020.2.254.7z -omkl

REM Download MAGMA files
REM version available:
REM 2.5.4 (CUDA 10.1 10.2 11.0 11.1) x (Debug Release)
REM 2.5.3 (CUDA 10.1 10.2 11.0) x (Debug Release)
REM 2.5.2 (CUDA 9.2 10.0 10.1 10.2) x (Debug Release)
REM 2.5.1 (CUDA 9.2 10.0 10.1 10.2) x (Debug Release)
set CUDA_PREFIX=cuda102
set CONFIG=release
curl -k https://s3.amazonaws.com/ossci-windows/magma_2.5.4_%CUDA_PREFIX%_%CONFIG%.7z -o magma.7z
7z x -aoa magma.7z -omagma

REM Setting essential environment variables
set "CMAKE_INCLUDE_PATH=%cd%\mkl\include"
set "LIB=%cd%\mkl\lib;%LIB%"
set "MAGMA_HOME=%cd%\magma"

```


### 加速 Windows 的 CUDA 构建 [¶](#speeding-cuda-build-for-windows "此标题的永久链接")


 Visual Studio 目前不支持并行自定义任务。作为替代方案，我们可以使用“Ninja”来并行化 CUDAbuild 任务。只需键入几行代码即可使用它。


```
REM Let's install ninja first.
pip install ninja

REM Set it as the cmake generator
set CMAKE_GENERATOR=Ninja

```


### 一键安装脚本 [¶](#one-key-install-script "永久链接到此标题")


 你可以看一下 [这套脚本](https://github.com/peterjc123/pytorch-scripts)，它会为你引路。


## 扩展 [¶](#extension "此标题的永久链接")


### CFFI 扩展 [¶](#cffi-extension "此标题的永久链接")


 对 CFFI 扩展的支持是非常实验性的。您必须在“Extension”对象中指定额外的“libraries”才能使其在 Windows 上构建。


```
ffi = create_extension(
    '_ext.my_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_compile_args=["-std=c99"],
    libraries=['ATen', '_C'] # Append cuda libraries when necessary, like cudart
)

```


### Cpp 扩展 [¶](#cpp-extension "此标题的永久链接")


 与前一种扩展相比，这种类型的扩展具有更好的支持。然而，它仍然需要一些手动配置。首先，您应该打开 **x86_x64 Cross Tools Command Prompt for VS 2017** 。然后，您可以开始编译过程。


## 安装[¶](#installation“永久链接到此标题”)


### 在 win-32 通道中找不到软件包。 [¶](#package-not-found-in-win-32-channel“此标题的永久链接”)


```
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


 PyTorch 不适用于 32 位系统。请使用Windows和Python 64位版本。


### 导入错误 [¶](#import-error "此标题的永久链接")


```
from torch._C import *

ImportError: DLL load failed: The specified module could not be found.

```


 该问题是由于缺少必要文件引起的。实际上，除了 VC2017 可再发行组件和一些 mkl 库之外，我们几乎包含了 PyTorch condapackage 所需的所有基本文件。您可以通过键入以下命令来解决此问题。


```
conda install -c peterjc123 vc vs2017_runtime
conda install mkl_fft intel_openmp numpy mkl

```


 至于wheels包，由于我们没有打包一些库和VS2017redistributable文件，请确保手动安装它们。 [VS 2017 redistributable installer](https://aka.ms/vs/15/release/VC_redist.x64.exe)可以下载。并且你还应该注意你的Numpy安装。确保它使用 MKL 而不是 OpenBLAS。您可以输入以下命令。


```
pip install numpy mkl intel-openmp mkl_fft

```


 另一个可能的原因可能是您使用的是没有 NVIDIA 显卡的 GPU 版本。请将您的 GPU 套件更换为 CPU 套件。


```
from torch._C import *

ImportError: DLL load failed: The operating system cannot run %1.

```


 这实际上是Anaconda的上游问题。当您使用 conda-forge 通道初始化环境时，就会出现此问题。您可以通过此命令修复 intel-openmp 库。


```
conda install -c defaults intel-openmp -f

```


## 用法(多处理)[¶](#usage-multiprocessing "此标题的永久链接")



### 没有 if-clause 保护的多处理错误  [¶](#multiprocessing-error-without-if-clause-protection "Permalink to this header")


```
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


 Windows 上“multiprocessing”的实现有所不同，它使用“spawn”而不是“fork”。所以我们必须用 anif 子句包裹代码，以防止代码被多次执行。将您的代码重构为以下结构。


```
import torch

def main()
    for i, data in enumerate(dataloader):
        # do something here

if __name__ == '__main__':
    main()

```


### 多处理错误“管道损坏”[¶](#multiprocessing-error-broken-pipe“永久链接到此标题”)


```
ForkingPickler(file, protocol).dump(obj)

BrokenPipeError: [Errno 32] Broken pipe

```


 当子进程在父进程完成发送数据之前结束时，就会出现此问题。您的代码可能有问题。您可以通过将 [`DataLoader`](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") 的 `num_worker` 减少到零来调试代码，看看问题是否存在持续存在。


### 多处理错误“驱动程序关闭”[¶](#multiprocessing-error-driver-shut-down“永久链接到此标题”)


```
Couldn’t open shared file mapping: <torch_14808_1591070686>, error code: <1455> at torch\lib\TH\THAllocator.c:154

[windows] driver shut down

```


 请更新您的显卡驱动程序。如果这种情况持续存在，则可能是您的显卡太旧或计算量对于您的卡来说太重了。请根据此[帖子](https://www.pugetsystems.com/labs/hpc/Working-around-TDR-in-Windows-for-a-better-GPU-computing-experience-777/)更新TDR设置。


### CUDA IPC 操作 [¶](#cuda-ipc-operations "永久链接到此标题")


```
THCudaCheck FAIL file=torch\csrc\generic\StorageSharing.cpp line=252 error=63 : OS call failed or operation not supported on this OS

```


 Windows 不支持它们。在 CUDAtensors 上进行多重处理之类的事情无法成功，有两种替代方案。


 1. 不要使用`multiprocessing`。将 [`DataLoader`](../data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") 的 `num_worker` 设置为零。


 2.改为共享CPUtensor。确保您的自定义“DataSet”返回 CPU tensor。