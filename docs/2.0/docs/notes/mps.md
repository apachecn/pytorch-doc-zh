# MPS 后端 [¶](#mps-backend "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/notes/mps>
>
> 原始地址：<https://pytorch.org/docs/stable/notes/mps.html>


“mps”设备支持使用 Metal 编程框架在 MacOS 设备上进行 GPU 高性能训练。它引入了一种新设备，可将机器学习计算图和基元分别映射到高效的 Metal Performance Shaders Graph 框架和 Metal Performance Shaders 框架提供的调整内核上。


 新的 MPS 后端扩展了 PyTorch 生态系统，并提供现有脚本功能来在 GPU 上设置和运行操作。


 首先，只需将tensor和模块移动到“mps”设备：


```
# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happens on the GPU
    y = x * 2

    # Move your model to mps just like any other device
    model = YourFavoriteNet()
    model.to(mps_device)

    # Now every call runs on the GPU
    pred = model(x)

```