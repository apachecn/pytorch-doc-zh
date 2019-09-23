# 多包 - torch.multiprocessing

torch.multiprocessing是天然[ `多处理 `
](https://docs.python.org/3/library/multiprocessing.html#module-
multiprocessing "\(in Python
v3.7\)")模块周围的包装。它注册的自定义减速器，其使用共享存储器，以提供在不同的处理相同的数据共享视图。一旦张量/存储移动到shared_memory（见[
`share_memory_（） `](tensors.html#torch.Tensor.share_memory_
"torch.Tensor.share_memory_")），将有可能将其发送到其他过程，而不进行任何拷贝。

API是与原模块100％兼容 - 这是足以改变`进口 多重处理 `至`进口 Torch .multiprocessing
`将所有经由其他机制通过队列发送或共享的张量，移动到共享存储器。

由于原料药的相似性，我们不记录大部分这个包的内容，我们建议参考原来的内存模块的很好的文档。

警告

如果主过程退出突然（例如，由于输入信号的），Python的`多处理
`有时不能清理其子女。这是一个已知的警告，所以如果你打断解释后，看不到任何资源泄漏，这可能意味着这只是发生在你身上。

## 战略管理

`torch.multiprocessing.``get_all_sharing_strategies`()[[source]](_modules/torch/multiprocessing.html#get_all_sharing_strategies)

    

返回一组支持的当前系统上共享战略。

`torch.multiprocessing.``get_sharing_strategy`()[[source]](_modules/torch/multiprocessing.html#get_sharing_strategy)

    

返回共享CPU张量目前的策略。

`torch.multiprocessing.``set_sharing_strategy`( _new_strategy_
)[[source]](_modules/torch/multiprocessing.html#set_sharing_strategy)

    

设置共享CPU张量的策略。

Parameters

    

**new_strategy** （[ _STR_
](https://docs.python.org/3/library/stdtypes.html#str "\(in Python v3.7\)")） -
所选择的策略的名称。应该是由 `返回的值中的一个get_all_sharing_strategies（） `。

## 分享CUDA张量

进程之间共享CUDA张量仅在Python 3被支撑，利用`产卵 `或`forkserver`开始的方法。 [ `多处理 `
](https://docs.python.org/3/library/multiprocessing.html#module-
multiprocessing "\(in Python v3.7\)")在Python 2可使用`叉 `仅创建子过程，并且它不被CUDA运行时的支持。

不同于CPU张量，需要在发送过程中保持原有的张量，只要该接收处理保留了张量的副本。该引用计数是引擎盖下实现的，但要求用户按照下面的最佳实践。

Warning

如果消费者进程异常死亡的致命信号，共享的张量可能会永远只要发送进程正在运行保存在内存中。

  1. 在消费者尽快释放内存。

    
    
    ## Good
    x = queue.get()
    # do somethings with x
    del x
    
    
    
    ## Bad
    x = queue.get()
    # do somethings with x
    # do everything else (producer have to keep x in memory)
    

2.保持生产过程中运行，直到所有的消费者退出。这将防止这种情况，当生产者进程释放内存仍处于由消费者使用。

    
    
    ## producer
    # send tensors, do something
    event.wait()
    
    
    
    ## consumer
    # receive tensors and use them
    event.set()
    

  3. 千万不要错过收到张量。

    
    
    # not going to work
    x = queue.get()
    queue_2.put(x)
    
    
    
    # you need to create a process-local copy
    x = queue.get()
    x_clone = x.clone()
    queue_2.put(x_clone)
    
    
    
    # putting and getting from the same queue in the same process will likely end up with segfault
    queue.put(tensor)
    x = queue.get()
    

## 共享策略

本节提供了一个简要介绍如何将不同的共享战略方面的工作。请注意，它仅适用于CPU张量 - CUDA张量将始终使用CUDA
API，因为这是他们可以共享的唯一途径。

### 文件描述符 - `类file_descriptor`

注意

这是（如果它不支持除MacOS和OS X）默认的策略。

这一战略将使用文件描述符共享内存句柄。每当一个存储被移动到共享存储器，从`获得的文件描述符的shm_open
`被高速缓存与所述对象，并​​且当它要被发送到其他过程，文件描述符将被转移（通过UNIX插座EG）给它。接收机还将缓存文件描述符和`MMAP
`它，以获得共享视图到存储数据。

请注意，如果将有很多共享的张量，这一战略将保留大量文件描述符打开的大部分时间。如果你的系统有打开的文件描述符的数量下限，并且你不能养宠物，你应该使用`
将file_system`策略。

### 文件系统 - `将file_system`

这一战略将使用给予`的shm_open
`文件名称来标识的共享内存区域。这具有不需要缓存从中获得的文件描述符执行的一个好处，但同时又是易共享内存泄漏。该文件不能在创建后删除，因为其他进程需要访问它打开了自己的看法。如果流程致命崩溃，或者被打死，不叫存储析构函数，该文件将保留在系统中。这是非常严重的，因为他们继续使用了内存，直至系统重新启动，或者他们正在手动释放。

为了对抗共享存储器文件泄漏的问题， `torch.multiprocessing`将产生一个守护进程名为`torch_shm_manager
`为将本身从当前进程组隔离，并且将跟踪所有共享存储器分配。一旦连接到它退出所有进程，它会等待片刻，以确保不会有新的连接，并会遍历由组分配的所有共享内存文件。如果发现其中的任何依然存在，他们将被释放。我们已经测试过这种方法，它被证明是稳健的各种故障。不过，如果你的系统有足够高的限制，`
类file_descriptor`被支持的战略，我们不建议切换到这一个。

## 产卵子过程

Note

可用于Python & GT ; = 3.4。

这取决于在Python的`多处理 `包`菌种 `启动方法。

产卵一些子来执行一些功能可以通过创建`过程 `实例，并调用`加入
`等待其完成来完成。与单个子打交道时，这种方法工作正常，但有多个进程打交道时具有潜在的问题。

即，在加入过程依次意味着它们将按顺序终止。如果他们不这样做，和第一进程不会终止，进程终止将被忽视。另外，还有一些错误传播没有原生的设施。

下面的`产卵 `功能解决了这些问题，并采取错误传播的护理，乱序终止，并将积极终止进程当在其中的一个检测错误。

`torch.multiprocessing.``spawn`( _fn_ , _args=()_ , _nprocs=1_ , _join=True_ ,
_daemon=False_ )[[source]](_modules/torch/multiprocessing/spawn.html#spawn)

    

产卵`nprocs`流程运行`FN`与`ARGS`。

如果其中一个进程与非零状态退出，剩下的进程被终止，并抛出一个异常与终止的原因。在一个异常被夹在子进程的情况下，被转发和回溯包含在父进程引发的异常。

Parameters

    

  * **FN** （ _函数_ ） - 

功能被称为衍生进程的入口点。该功能必须在模块的顶部电平被限定以便它可以进行酸洗和衍生。这是一个由多处理施加的规定。

该函数被称为`FN（I， *参数） `，其中`i的 `是进程索引和`ARGS`在贯通的参数元组通过。

  * **ARGS** （[ _元组_ ](https://docs.python.org/3/library/stdtypes.html#tuple "\(in Python v3.7\)")） - 参数传递给`FN`。

  * **nprocs** （[ _INT_ ](https://docs.python.org/3/library/functions.html#int "\(in Python v3.7\)")） - 过程编号产卵。

  * **加入** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 执行上的所有进程的阻挡加入。

  * **守护程序** （[ _布尔_ ](https://docs.python.org/3/library/functions.html#bool "\(in Python v3.7\)")） - 衍生的进程的守护进程标记。如果设置为True一样，后台进程将被创建。

Returns

    

无如果`加入 `是`真 `， `SpawnContext`如果`加入 `是`假 `

_class_`torch.multiprocessing.``SpawnContext`[[source]](_modules/torch/multiprocessing/spawn.html#SpawnContext)

    

通过 `产卵返回（） `当使用`加入=假 `调用。

`join`( _timeout=None_
)[[source]](_modules/torch/multiprocessing/spawn.html#SpawnContext.join)

    

试图在此背景下产卵加入一个或多个进程。如果其中一人非零退出状态退出，此功能杀死剩余的所有进程，并提出与离开第一工艺的原因的异常。

返回`真 `如果所有的过程已经成功加入，`假 `如果有需要连接多个进程。

Parameters

    

**超时** （[ _浮动_ ](https://docs.python.org/3/library/functions.html#float "\(in
Python v3.7\)")） - 上等待放弃之前长时间等待此。

[Next ![](_static/images/chevron-right-orange.svg)](random.html
"torch.random") [![](_static/images/chevron-right-orange.svg)
Previous](jit.html "TorchScript")

* * *

©版权所有2019年，Torch 贡献者。