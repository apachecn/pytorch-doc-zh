


 使用 Cpp 扩展自定义进程组后端
 [¶](#customize-process-group-backends-using-cpp-extensions "永久链接到此标题")
=================================================================================================================================================

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/intermediate/process_group_cpp_extension_tutorial>
>
> 原始地址：<https://pytorch.org/tutorials/intermediate/process_group_cpp_extension_tutorial.html>




**作者** 
 :
 
 Howard Huang <https://github.com/H-Huang>
 
 ,
 [冯天](https://github.com/ftian1 ) 
 ,
 [沉力](https://mrshenli.github.io/) 
 ,
 [敏斯](https://minsii.github.io/)





 没有10



[![edit](https://pytorch.org/tutorials/_images/pencil-16.png)](https://pytorch.org/tutorials/_images/pencil-16.png)
 在 [github](https://github.com/pytorch/tutorials/blob/main/intermediate_source/process_group_cpp_extension_tutorial.rst) 
.





 先决条件:



* [PyTorch 分布式概述](../beginner/dist_overview.html)
* [PyTorch 集体通信包](https://pytorch.org/docs/stable/distributed.html)
* [PyTorch Cpp 扩展] (https://pytorch.org/docs/stable/cpp_extension.html)
* [使用 PyTorch 编写分布式应用程序](https://pytorch.org/tutorials/intermediate/dist_tuto.html)



 本教程演示如何实现自定义
 `后端`
 并将其插入
 [PyTorch 分布式包](https://pytorch.org/docs/stable/distributed.html)
 使用
 [cpp 扩展](https://pytorch.org/docs/stable/cpp_extension.html) 
 。当您需要针对硬件
的专用软件堆栈，或者当您想要尝试新的
集体通信算法时，这会很有帮助。





 基础知识
 [¶](#basics "此标题的永久链接")
-----------------------------------------------------------------



 PyTorch 集体通信支持多种广泛采用的分布式
训练功能，包括
 [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 
 ,
 [ZeroRedundancyOptimizer](https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer) 
 ,
 [FullyShardedDataParallel](https://github.com/pytorch/pytorch /blob/master/torch/distributed/_fsdp/fully_sharded_data_parallel.py) 
 。
为了使同一个集体通信 API 能够与
不同的通信后端一起工作，分布式包将集体
通信操作抽象为
 [后端] （https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp）
 类。然后可以使用首选第三方库将不同的后端实现为“后端”的子类。 PyTorch 分布式附带三个默认后端：
 `ProcessGroupNCCL`
 、
 `ProcessGroupGloo`
 和 
 `ProcessGroupMPI`
 。然而，除了这三个后端之外，还有其他通信库（例如，[UCC](https://github.com/openucx/ucc)、[OneCCL](https://github.com/oneapi-src/oneCCL) 
 )，不同类型的硬件
（例如，
 [TPU](https://cloud.google.com/tpu) 
 ,
 [Trainum]( https://aws.amazon.com/machine-learning/trainium/) 
 )，以及新兴的
通信算法（例如，
 [Herring](https://www.amazon.science/publications/herring-rethinking -the-parameter-server-at-scale-for-the-cloud) 
 ,
 [缩减服务器](https://cloud.google.com/blog/topics/developers-practitioners/optimize-training-performance -reduction-server-vertex-ai) 
 )。
因此，分布式包公开扩展 API 以允许自定义
集体通信后端。




 下面的 4 个步骤展示了如何实现虚拟
 `后端`
 后端
并在 Python 应用程序代码中使用它。请注意，本教程的重点
不是演示扩展 API，而不是开发
正常运行的通信后端。因此，
 `dummy`
 后端仅覆盖 
API 的一个子集（
 `all_reduce`
 和
 `all_gather`
 ），并简单地设置张量的值
到 0。






 第 1 步：实现
 `Backend`的子类
[¶](#step-1-implement-a-subclass-of-backend "永久链接到此标题")
-------------------------------------------------------------------------------------------------------------------------



 第一步是实现一个
 `Backend`
 子类，该子类覆盖
目标集体通信 API 并运行自定义通信算法。
该扩展还需要实现
 
 `Work`
 子类，该子类
作为通信结果的未来，并允许在应用程序代码中异步执行。如果扩展使用第三方库，则它可以包含标头并从“BackendDummy”子类调用库 API。下面的两个代码片段展示了
 `dummy.h`
 和
 `dummy.cpp`
 的实现。请参阅
 [虚拟集体](https://github.com/H-Huang/torch_collective_extension)
 存储库以了解完整实现。






```
// file name: dummy.hpp
#include <torch/python.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <pybind11/chrono.h>

namespace c10d {

class BackendDummy : public Backend {
 public:
 BackendDummy(int rank, int size);

 c10::intrusive_ptr<Work> allgather(
 std::vector<std::vector<at::Tensor>>& outputTensors,
 std::vector<at::Tensor>& inputTensors,
 const AllgatherOptions& opts = AllgatherOptions()) override;

 c10::intrusive_ptr<Work> allreduce(
 std::vector<at::Tensor>& tensors,
 const AllreduceOptions& opts = AllreduceOptions()) override;

 // The collective communication APIs without a custom implementation
 // will error out if invoked by application code.
};

class WorkDummy : public Work {
 public:
 WorkDummy(
 OpType opType,
 c10::intrusive_ptr<c10::ivalue::Future> future) // future of the output
 : Work(
 -1, // rank, only used by recvAnySource, irrelevant in this demo
 opType),
 future_(std::move(future)) {}
 bool isCompleted() override;
 bool isSuccess() const override;
 bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
 virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

 private:
 c10::intrusive_ptr<c10::ivalue::Future> future_;
};
} // namespace c10d

```






```
// file name: dummy.cpp
#include "dummy.hpp"

namespace c10d {

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allgather(
 std::vector<std::vector<at::Tensor>>& outputTensors,
 std::vector<at::Tensor>& inputTensors,
 const AllgatherOptions& /* unused */) {
 for (auto& outputTensorVec : outputTensors) {
 for (auto& outputTensor : outputTensorVec) {
 outputTensor.zero_();
 }
 }

 auto future = c10::make_intrusive<c10::ivalue::Future>(
 c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
 future->markCompleted(c10::IValue(outputTensors));
 return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allreduce(
 std::vector<at::Tensor>& tensors,
 const AllreduceOptions& opts) {
 for (auto& tensor : tensors) {
 tensor.zero_();
 }

 auto future = c10::make_intrusive<c10::ivalue::Future>(
 c10::ListType::create(c10::TensorType::get()));
 future->markCompleted(c10::IValue(tensors));
 return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}
} // namespace c10d

```






 步骤 2：公开扩展 Python API
 [¶](#step-2-expose-the-extension-python-apis "永久链接到此标题")
------------------------------------------------------------------------------------------------------------------------



 后端构造函数被调用
 [从 Python 端](https://github.com/pytorch/pytorch/blob/v1.9.0/torch/distributed/distributed_c10d.py#L643-L650) 
 ,
so该扩展还需要向 Python 公开构造函数 API。这可以通过添加以下方法来完成。在此示例中，
 `store`
 和
 `timeout`
 被
 `BackendDummy`
 实例化方法忽略，因为
在此虚拟实现中未使用它们。但是，实际扩展
应考虑使用
 `store`
 来执行交会并支持
 `timeout`
 参数。






```
// file name: dummy.hpp
class BackendDummy : public Backend {
 ...
 <Step 1 code>
 ...

 static c10::intrusive_ptr<Backend> createBackendDummy(
 const c10::intrusive_ptr<::c10d::Store>& store,
 int rank,
 int size,
 const std::chrono::duration<float>& timeout);

 static void BackendDummyConstructor() __attribute__((constructor)) {
 py::object module = py::module::import("torch.distributed");
 py::object register_backend =
 module.attr("Backend").attr("register_backend");
 // torch.distributed.Backend.register_backend will add `dummy` as a
 // new valid backend.
 register_backend("dummy", py::cpp_function(createBackendDummy));
 }
}

```






```
// file name: dummy.cpp
c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
 const c10::intrusive_ptr<::c10d::Store>& /* unused */,
 int rank,
 int size,
 const std::chrono::duration<float>& /* unused */) {
 return c10::make_intrusive<BackendDummy>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("createBackendDummy", &BackendDummy::createBackendDummy);
}

```






 步骤 3：构建自定义扩展
 [¶](#step-3-build-the-custom-extension "永久链接到此标题")
------------------------------------------------------------------------------------------------------------------------



 现在，扩展源代码文件已准备就绪。然后我们可以使用 [cpp 扩展](https://pytorch.org/docs/stable/cpp_extension.html) 来构建它。为此，请创建一个
 `setup.py`
 文件来准备路径和
命令。然后调用
 `python
 

 setup.py
 

develop`
 来安装扩展。




 如果扩展依赖于第三方库，您还可以为 cpp 扩展 API 指定
 `libraries_dirs`
 和
 `libraries`
。请参阅
 [torch ucc](https://github.com/openucx/torch-ucc)
 项目作为实际示例。






```
# file name: setup.py
import os
import sys
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["src/dummy.cpp"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/"]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name = "dummy_collectives",
        sources = sources,
        include_dirs = include_dirs,
    )
else:
    module = cpp_extension.CppExtension(
        name = "dummy_collectives",
        sources = sources,
        include_dirs = include_dirs,
    )

setup(
    name = "Dummy-Collectives",
    version = "0.0.1",
    ext_modules = [module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

```






 步骤 4：在应用程序中使用扩展
 [¶](#step-4-use-the-extension-in-application "永久链接到此标题")
------------------------------------------------------------------------------------------------------------------------



 安装后，您可以在调用时方便地使用
 `dummy`
 后端
 [init_process_group](https://pytorch.org/docs/stable/distributed.html#torch. distribution.init_process_group) 
 就好像它是内置后端一样。




 我们可以通过更改
 `init_process_group`
 的
 `backend`
 参数来指定基于后端的调度。我们可以通过
指定
 `cpu:gloo,cuda:dummy`
 作为后端参数。




 要将所有张量发送到
 `dummy`
 后端，我们只需指定
 `dummy`
 作为后端参数。






```
import os

import torch
# importing dummy_collectives makes torch.distributed recognize `dummy`
# as a valid backend.
import dummy_collectives

import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Alternatively:
# dist.init_process_group("dummy", rank=0, world_size=1)
dist.init_process_group("cpu:gloo,cuda:dummy", rank=0, world_size=1)

# this goes through gloo
x = torch.ones(6)
dist.all_reduce(x)
print(f"cpu allreduce: {x}")

# this goes through dummy
if torch.cuda.is_available():
    y = x.cuda()
    dist.all_reduce(y)
    print(f"cuda allreduce: {y}")

    try:
        dist.broadcast(y, 0)
    except RuntimeError:
        print("got RuntimeError when calling broadcast")

```









