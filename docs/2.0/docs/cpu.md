# torch.cpu [¶](#module-torch.cpu "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/cpu>
>
> 原始地址：<https://pytorch.org/docs/stable/cpu.html>


 该包实现了“torch.cuda”中的抽象，以方便编写与设备无关的代码。


|  |  |
| --- | --- |
| [`current_stream`](generated/torch.cpu.current_stream.html#torch.cpu.current_stream "torch.cpu.current_stream") |返回给定设备当前选择的 [`Stream`]( generated/torch.cpu.Stream.html#torch.cpu.Stream "torch.cpu.Stream")。 |
| [`is_available`](generated/torch.cpu.is_available.html#torch.cpu.is_available "torch.cpu.is_available") |返回一个布尔值，指示 CPU 当前是否可用。 |
| [`synchronize`](generated/torch.cpu.synchronize.html#torch.cpu.synchronize "torch.cpu.synchronize") |等待 CPU 设备上所有流中的所有内核完成。 |
| [`stream`](generated/torch.cpu.stream.html#torch.cpu.stream "torch.cpu.stream") |围绕选择给定流的上下文管理器 StreamContext 的包装。 |
| [`device_count`](generated/torch.cpu.device_count.html#torch.cpu.device_count "torch.cpu.device_count") |返回 CPU 设备(不是核心)的数量。 |
| [`StreamContext`](generated/torch.cpu.StreamContext.html#torch.cpu.StreamContext "torch.cpu.StreamContext") |选择给定流的上下文管理器。 |


## 流和事件 [¶](#streams-and-events "此标题的永久链接")


|  |  |
| --- | --- |
| [`Stream`](generated/torch.cpu.Stream.html#torch.cpu.Stream "torch.cpu.Stream") | 注意： |
