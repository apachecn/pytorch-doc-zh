


# 使用自定义 C++ 类扩展 TorchScript [¶](#extending-torchscript-with-custom-c-classes "永久链接到此标题")


> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/tutorials/advanced/torch_script_custom_classes>
>
> 原始地址：<https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html>




 本教程是
 [自定义运算符](torch_script_custom_ops.html)
 教程的后续内容，介绍了我们为将 C++ 类同时绑定到 TorchScript
 和 Python 而构建的 API’。该 API 与 [pybind11](https://github.com/pybind/pybind11) 非常相似，如果您熟悉该系统，大多数概念都会转移。\ n


## 在 C++ 中实现和绑定类 [¶](#implementing-and-binding-the-class-in-c "永久链接到此标题")




 对于本教程，我们将定义一个简单的 C++ 类，用于在成员变量中维护
持久状态。






```
// This header is all you need to do the C++ portions of this
// tutorial
#include <torch/script.h>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>

#include <string>
#include <vector>

template <class T>
struct MyStackClass : torch::CustomClassHolder {
 std::vector<T> stack_;
 MyStackClass(std::vector<T> init) : stack_(init.begin(), init.end()) {}

 void push(T x) {
 stack_.push_back(x);
 }
 T pop() {
 auto val = stack_.back();
 stack_.pop_back();
 return val;
 }

 c10::intrusive_ptr<MyStackClass> clone() const {
 return c10::make_intrusive<MyStackClass>(stack_);
 }

 void merge(const c10::intrusive_ptr<MyStackClass>& c) {
 for (auto& elem : c->stack_) {
 push(elem);
 }
 }
};

```




 有几点需要注意：



* `torch/custom_class.h`
 是您需要包含的标头，以便使用您的自定义类扩展 TorchScript
。
* 请注意，每当我们使用自定义
类的实例时，我们都会通过
 `c10::intrusive_ptr<>`
 的实例。将
 `intrusive_ptr`
 视为像
 `std::shared_ptr`
 一样的智能指针，但引用计数
直接存储在对象中，而不是单独的元数据块（如
 `std::shared_ptr`
 中所做的那样。
 `torch::Tensor`
 内部使用相同的指针类型；
并且自定义类也必须使用此指针类型，以便我们可以
一致地管理不同的对象类型。
* 第二个需要注意的是，用户定义的类必须继承自
 `torch::CustomClassHolder`
 。这可以确保自定义类有
空间来存储引用数数。



 现在让’s 看看如何使此类对 TorchScript 可见，这个过程称为
 *绑定* 
 类：






```
// Notice a few things:
// - We pass the class to be registered as a template parameter to
// `torch::class_`. In this instance, we've passed the
// specialization of the MyStackClass class ``MyStackClass<std::string>``.
// In general, you cannot register a non-specialized template
// class. For non-templated classes, you can just pass the
// class name directly as the template parameter.
// - The arguments passed to the constructor make up the "qualified name"
// of the class. In this case, the registered class will appear in
// Python and C++ as `torch.classes.my_classes.MyStackClass`. We call
// the first argument the "namespace" and the second argument the
// actual class name.
TORCH_LIBRARY(my_classes, m) {
 m.class_<MyStackClass<std::string>>("MyStackClass")
 // The following line registers the contructor of our MyStackClass
 // class that takes a single `std::vector<std::string>` argument,
 // i.e. it exposes the C++ method `MyStackClass(std::vector<T> init)`.
 // Currently, we do not support registering overloaded
 // constructors, so for now you can only `def()` one instance of
 // `torch::init`.
 .def(torch::init<std::vector<std::string>>())
 // The next line registers a stateless (i.e. no captures) C++ lambda
 // function as a method. Note that a lambda function must take a
 // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
 // as the first argument. Other arguments can be whatever you want.
 .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
 return self->stack_.back();
 })
 // The following four lines expose methods of the MyStackClass<std::string>
 // class as-is. `torch::class_` will automatically examine the
 // argument and return types of the passed-in method pointers and
 // expose these to Python and TorchScript accordingly. Finally, notice
 // that we must take the *address* of the fully-qualified method name,
 // i.e. use the unary `&` operator, due to C++ typing rules.
 .def("push", &MyStackClass<std::string>::push)
 .def("pop", &MyStackClass<std::string>::pop)
 .def("clone", &MyStackClass<std::string>::clone)
 .def("merge", &MyStackClass<std::string>::merge)
 ;
}

```





## 使用 CMake 将示例构建为 C++ 项目 [¶](#building-the-example-as-a-c-project-with-cmake "永久链接到此标题")




 现在，我们’ 将使用
 [CMake](https://cmake.org) 
 构建系统构建上述 C++ 代码。首先，将目前为止涵盖的所有 C++ 代码
we’ 放入一个名为
 `class.cpp`
 的文件中。
然后，编写一个简单的
 `CMakeLists.txt`
文件并将其放在
同一目录中。 
 `CMakeLists.txt`
 应如下所示：






```
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_class SHARED class.cpp)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(custom_class "${TORCH_LIBRARIES}")

```




 另外，创建一个
 `build`
 目录。您的文件树应如下所示：






```
custom_class_project/
  class.cpp
  CMakeLists.txt
  build/

```




 我们假设您’ 已经按照
[上一教程](torch_script_custom_ops.html)
 中描述的相同方式设置了环境。
继续调用 cmake，然后进行构建项目：






```
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
 -- The C compiler identification is GNU 7.3.1
 -- The CXX compiler identification is GNU 7.3.1
 -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc
 -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc -- works
 -- Detecting C compiler ABI info
 -- Detecting C compiler ABI info - done
 -- Detecting C compile features
 -- Detecting C compile features - done
 -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++
 -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++ -- works
 -- Detecting CXX compiler ABI info
 -- Detecting CXX compiler ABI info - done
 -- Detecting CXX compile features
 -- Detecting CXX compile features - done
 -- Looking for pthread.h
 -- Looking for pthread.h - found
 -- Looking for pthread_create
 -- Looking for pthread_create - not found
 -- Looking for pthread_create in pthreads
 -- Looking for pthread_create in pthreads - not found
 -- Looking for pthread_create in pthread
 -- Looking for pthread_create in pthread - found
 -- Found Threads: TRUE
 -- Found torch: /torchbind_tutorial/libtorch/lib/libtorch.so
 -- Configuring done
 -- Generating done
 -- Build files have been written to: /torchbind_tutorial/build
$ make -j
 Scanning dependencies of target custom_class
 [ 50%] Building CXX object CMakeFiles/custom_class.dir/class.cpp.o
 [100%] Linking CXX shared library libcustom_class.so
 [100%] Built target custom_class

```




 您将发现’ 现在（除其他外）在构建目录中存在一个动态库
文件。在 Linux 上，这可能被命名为
 `libcustom_class.so`
 。因此文件树应如下所示：






```
custom_class_project/
  class.cpp
  CMakeLists.txt
  build/
    libcustom_class.so

```





## 使用 Python 和 TorchScript 中的 C++ 类 [¶](#using-the-c-class-from-python-and-torchscript "永久链接到此标题")




 现在我们已经将类及其注册编译成
 `.so`
 文件，
我们可以将
 
.so
 
 加载到Python 中并尝试一下。这里’ 是一个脚本，
演示：






```
import torch

# `torch.classes.load_library()` allows you to pass the path to your .so file
# to load it in and make the custom C++ classes available to both Python and
# TorchScript
torch.classes.load_library("build/libcustom_class.so")
# You can query the loaded libraries like this:
print(torch.classes.loaded_libraries)
# prints {'/custom_class_project/build/libcustom_class.so'}

# We can find and instantiate our custom C++ class in python by using the
# `torch.classes` namespace:
#
# This instantiation will invoke the MyStackClass(std::vector<T> init)
# constructor we registered earlier
s = torch.classes.my_classes.MyStackClass(["foo", "bar"])

# We can call methods in Python
s.push("pushed")
assert s.pop() == "pushed"

# Test custom operator
s.push("pushed")
torch.ops.my_classes.manipulate_instance(s)  # acting as s.pop()
assert s.top() == "bar" 

# Returning and passing instances of custom classes works as you'd expect
s2 = s.clone()
s.merge(s2)
for expected in ["bar", "foo", "bar", "foo"]:
    assert s.pop() == expected

# We can also use the class in TorchScript
# For now, we need to assign the class's type to a local in order to
# annotate the type on the TorchScript function. This may change
# in the future.
MyStackClass = torch.classes.my_classes.MyStackClass


@torch.jit.script
def do_stacks(s: MyStackClass): # 我们可以传递自定义类实例
 # 我们可以实例化该类
 s2 = torch.classes.my_classes.MyStackClass([ "hi", "mom"])
 s2.merge(s) # 我们可以调用类上的方法
 # 我们还可以返回类的实例
 # 从 TorchScript 函数/方法
 return s2。克隆（），s2.top（）


stack, top = do_stacks(torch.classes.my_classes.MyStackClass(["wow"]))
assert top == "wow"
对于 ["wow", "mom", "hi 中的预期"]:
 断言 stack.pop() == 预期


```





## 使用自定义类保存、加载和运行 TorchScript 代码 [¶](# saving-loading-and-running-torchscript-code-using-custom-classes "永久链接到此标题")
- 


我们还可以使用 libtorch 在 C++ 进程中使用自定义注册的 C++ 类。例如，让’s 定义一个简单
 `nn.Module`
，
 实例化并调用我们的 MyStackClass 类上的方法：






```
import torch

torch.classes.load_library('build/libcustom_class.so')


Foo 类(torch.nn.Module):
 def __init__(self):
 super().__init__()\ n
 defforward(self, s: str) -> str:
 stack = torch.classes.my_classes.MyStackClass(["hi", "mom"])
 return stack.pop() + s


scripted_foo = torch.jit.script(Foo())
print(scripted_foo.graph)

scripted_foo.save('foo.pt')

```


我们的文件系统中的 

`foo.pt`
 现在包含我们刚刚定义的序列化 TorchScript
 程序’ve。




 现在，我们’ 将定义一个新的 CMake 项目来展示如何加载
此模型及其所需的.so 文件。要全面了解如何执行此操作，
请查看
 [在 C++ 教程中加载 TorchScript 模型](https://pytorch.org/tutorials/advanced/cpp_export.html) 
 。




 与之前类似，让’s 创建一个包含以下内容的文件结构:






```
cpp_inference_example/
  infer.cpp
  CMakeLists.txt
  foo.pt
  build/
  custom_class_project/
    class.cpp
    CMakeLists.txt
    build/

```




 请注意，我们’ 已经复制了序列化
 `foo.pt`
 文件，以及上面
 `custom_class_project`
 中的源
树。我们将添加 
 `custom_class_project`
 作为此 C++ 项目的依赖项，以便我们可以
将自定义类构建到二进制文件中。




 让’s 使用以下内容填充
 `infer.cpp`
:






```
#include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
 torch::jit::Module module;
 try {
 // Deserialize the ScriptModule from a file using torch::jit::load().
 module = torch::jit::load("foo.pt");
 }
 catch (const c10::Error& e) {
 std::cerr << "error loading the model";
 return -1;
 }

 std::vector<c10::IValue> inputs = {"foobarbaz"};
 auto output = module.forward(inputs).toString();
 std::cout << output->string() << std::endl;
}

```




 同样让’s 定义我们的 CMakeLists.txt 文件：






```
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(infer)

find_package(Torch REQUIRED)

add_subdirectory(custom_class_project)

# Define our library target
add_executable(infer infer.cpp)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(infer "${TORCH_LIBRARIES}")
# This is where we link in our libcustom_class code, making our
# custom class available in our binary.
target_link_libraries(infer -Wl,--no-as-needed custom_class)

```




 你知道该怎么做：
 `cd
 

 build`
 、
 `cmake`
 和
 `make`
 :






```
$ cd build
$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
 -- The C compiler identification is GNU 7.3.1
 -- The CXX compiler identification is GNU 7.3.1
 -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc
 -- Check for working C compiler: /opt/rh/devtoolset-7/root/usr/bin/cc -- works
 -- Detecting C compiler ABI info
 -- Detecting C compiler ABI info - done
 -- Detecting C compile features
 -- Detecting C compile features - done
 -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++
 -- Check for working CXX compiler: /opt/rh/devtoolset-7/root/usr/bin/c++ -- works
 -- Detecting CXX compiler ABI info
 -- Detecting CXX compiler ABI info - done
 -- Detecting CXX compile features
 -- Detecting CXX compile features - done
 -- Looking for pthread.h
 -- Looking for pthread.h - found
 -- Looking for pthread_create
 -- Looking for pthread_create - not found
 -- Looking for pthread_create in pthreads
 -- Looking for pthread_create in pthreads - not found
 -- Looking for pthread_create in pthread
 -- Looking for pthread_create in pthread - found
 -- Found Threads: TRUE
 -- Found torch: /local/miniconda3/lib/python3.7/site-packages/torch/lib/libtorch.so
 -- Configuring done
 -- Generating done
 -- Build files have been written to: /cpp_inference_example/build
$ make -j
 Scanning dependencies of target custom_class
 [ 25%] Building CXX object custom_class_project/CMakeFiles/custom_class.dir/class.cpp.o
 [ 50%] Linking CXX shared library libcustom_class.so
 [ 50%] Built target custom_class
 Scanning dependencies of target infer
 [ 75%] Building CXX object CMakeFiles/infer.dir/infer.cpp.o
 [100%] Linking CXX executable infer
 [100%] Built target infer

```




 现在我们可以运行令人兴奋的 C++ 二进制文件了：






```
$ ./infer
 momfoobarbaz

```




 难以置信！





## 将自定义类移入/移出 IValues [¶](#moving-custom-classes-to-from-ivalues "永久链接到此标题")




’s 也可能需要将自定义类移入或移出
 `IValue``，
 

 例如
 

 如
 
\ n 当
 

 你
 

 采取
 

 或
 

 返回
 

 ``IValue``s
 

 from\ n 

 TorchScript
 

 方法
 

 或
 

 您
 

 想要
 

 实例化
 

 

 a
 

 自定义
 

 类
 

 属性
 

 在
 

 C++ 中。
 

 对于
 \ n
 从自定义 C++ 类实例创建
 

 一个
 

 ``IValue`
:



* `torch::make_custom_class<T>()`
 提供了一个类似于 c10::intrusive_ptr<T>
 的 API，它将接受您提供给它的任何参数集，调用与该参数集匹配的 T 的
构造函数，并将该实例包装起来并返回它。
但是，它不是仅返回指向自定义类对象的指针，而是返回
an
 `IValue`
 包装目的。然后，您可以将此
 `IValue`
 直接传递给
TorchScript。
* 如果您已有一个
 `intrusive_ptr`
 指向您的类，
您可以直接构造一个 IValue使用构造函数
 `IValue(intrusive_ptr<T>)`
 从中获取。



 用于将
 `IValue`
 转换回自定义类:



* `IValue::toCustomClass<T>()`
 将返回一个
 `intrusive_ptr<T>`
 指向
 `IValue`
 包含的
自定义类。在内部，此函数正在检查
“T”
 是否已注册为自定义类，以及
“IValue”
 实际上是否包含
 自定义类。您可以通过调用
 `isCustomClass()`
 手动检查
 `IValue`
 是否包含自定义类。





## 为自定义 C++ 类定义序列化/反序列化方法 [¶](#defining-serialization-deserialization-methods-for-custom-c-classes "永久链接到此标题")




 如果您尝试将具有自定义绑定 C++ 类的
 `ScriptModule`
 保存为
an 属性，您’ 将收到以下错误:






```
# export_attr.py
import torch

torch.classes.load_library('build/libcustom_class.so')


Foo 类(torch.nn.Module):
 def __init__(self):
 super().__init__()\ n self.stack = torch.classes.my_classes.MyStackClass(["just", "testing"])

 defforward(self, s: str) -> str:
 返回 self.stack。弹出() + s


scripted_foo = torch.jit.script(Foo())

scripted_foo.save('foo.pt')
loaded = torch.jit.load('foo.pt')

print(loaded.stack.pop())

```






```
$ python export_attr.py
RuntimeError: Cannot serialize custom bound C++ class __torch__.torch.classes.my_classes.MyStackClass. Please define serialization methods via def_pickle for this class. (pushIValueImpl at ../torch/csrc/jit/pickler.cpp:128)

```




 这是因为 TorchScript 无法自动找出 C++ 类中保存的信息。您必须手动指定。执行此操作的方法
 是在类上定义
 `__getstate__`
 和
 `__setstate__`
 方法在
 `class_`
 上使用
特殊
 `def_pickle`
 方法。





 注意




 TorchScript 中
 `__getstate__`
 和
 `__setstate__`
 的语义与 
 等效Python pickle 模块。您可以
 [阅读更多](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md#getstate-and-setstate)
 了解我们如何使用这些方法。





 下面是
 `def_pickle`
 调用的示例，我们可以将其添加到
 `MyStackClass`
 的注册中以包含序列化方法:






```
 // class_<>::def_pickle allows you to define the serialization
 // and deserialization methods for your C++ class.
 // Currently, we only support passing stateless lambda functions
 // as arguments to def_pickle
 .def_pickle(
 // __getstate__
 // This function defines what data structure should be produced
 // when we serialize an instance of this class. The function
 // must take a single `self` argument, which is an intrusive_ptr
 // to the instance of the object. The function can return
 // any type that is supported as a return value of the TorchScript
 // custom operator API. In this instance, we've chosen to return
 // a std::vector<std::string> as the salient data to preserve
 // from the class.
 [](const c10::intrusive_ptr<MyStackClass<std::string>>& self)
 -> std::vector<std::string> {
 return self->stack_;
 },
 // __setstate__
 // This function defines how to create a new instance of the C++
 // class when we are deserializing. The function must take a
 // single argument of the same type as the return value of
 // `__getstate__`. The function must return an intrusive_ptr
 // to a new instance of the C++ class, initialized however
 // you would like given the serialized state.
 [](std::vector<std::string> state)
 -> c10::intrusive_ptr<MyStackClass<std::string>> {
 // A convenient way to instantiate an object and get an
 // intrusive_ptr to it is via `make_intrusive`. We use
 // that here to allocate an instance of MyStackClass<std::string>
 // and call the single-argument std::vector<std::string>
 // constructor with the serialized state.
 return c10::make_intrusive<MyStackClass<std::string>>(std::move(state));
 });

```





 没有10



 我们在 pickle API 中采用了与 pybind11 不同的方法。而 pybind11
 是一个特殊函数
 `pybind11::pickle()`
 ，您可以将其传递给
 `class_::def()`
 ，
我们有一个单独的方法
 `def\ \_pickle`
 为此目的。这是因为
名称
 `torch::jit::pickle`
 已被占用，并且我们不想’ 造成混乱。





 一旦我们以这种方式定义了（反）序列化行为，我们的脚本
现在就可以成功运行：






```
$ python ../export_attr.py
testing

```





## 定义采用或返回绑定 C++ 类的自定义运算符 [¶](#defining-custom-operators-that-take-or-return-bound-c-classes "此标题的永久链接")
 



 一旦您’ 定义了自定义 C++ 类，您还可以将该类用作参数或从自定义运算符（即自由函数）返回。假设
您有以下自由函数:






```
c10::intrusive_ptr<MyStackClass<std::string>> manipulate_instance(const c10::intrusive_ptr<MyStackClass<std::string>>& instance) {
 instance->pop();
 return instance;
}

```




 您可以在
 `TORCH_LIBRARY` 块中运行以下代码来注册它：






```
 m.def(
 "manipulate_instance(__torch__.torch.classes.my_classes.MyStackClass x) -> __torch__.torch.classes.my_classes.MyStackClass Y",
 manipulate_instance
 );

```




 请参阅
 [自定义操作教程](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html) 
 有关注册 API 的更多详细信息。




 完成此操作后，您可以使用如下示例所示的操作：






```
class TryCustomOp(torch.nn.Module):
    def __init__(self):
        super(TryCustomOp, self).__init__()
        self.f = torch.classes.my_classes.MyStackClass(["foo", "bar"])

    def forward(self):
        return torch.ops.my_classes.manipulate_instance(self.f)

```





 没有10



 注册以 C++ 类作为参数的运算符需要
已注册自定义类。您可以通过
确保自定义类注册和您的自由函数定义
位于同一个
 `TORCH_LIBRARY`
 块中，并且自定义类
注册位于第一位来强制执行此操作。将来，我们可能会放宽这一要求，
以便这些可以按任意顺序注册。





## 结论 [¶](#conclusion "此标题的永久链接")




 本教程向您介绍了如何向 TorchScript
（以及扩展 Python）公开 C++ 类、如何注册其方法、如何从 Python 和 TorchScript 使用该类
以及如何使用
保存和加载代码类并在独立的 C++ 进程中运行该代码。您现在已准备好
使用与第三方 C++ 库交互的 C++ 类来扩展您的 TorchScript 模型，
或者实现需要 Python、TorchScript 和 C++ 之间的
流畅融合的任何其他用例。




 一如既往，如果您遇到任何问题或有疑问，可以使用我们的
 [论坛](https://discuss.pytorch.org/) 
 或
 [GitHub issues](https://github.com/pytorch/pytorch/issues) 
 取得联系。此外，我们的
 [常见问题 (FAQ) 页面](https://pytorch.org/cppdocs/notes/faq.html)
 可能包含有用的信息。









