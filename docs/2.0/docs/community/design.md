# PyTorch 设计哲学 [¶](#pytorch-design-philosophy "永久链接到此标题")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/community/design>
>
> 原始地址：<https://pytorch.org/docs/stable/community/design.html>


 本文档旨在帮助贡献者和模块维护者了解 PyTorch 中长期开发的高级设计原则。这些并不是硬性规则，而是作为指南，帮助权衡不同的问题并解决开发 PyTorch 时可能出现的协议。有关贡献、模块维护以及如何将分歧升级给核心维护者的更多信息，请参阅 [PyTorchGovernance](https://pytorch.org/docs/main/community/governance.html)。


## 设计原则 [¶](#design-principles "此标题的固定链接")


### 原则 1：可用性高于性能 [¶](#principle-1-usability-over-performance "Permalink to this header")


 这个原理可能会令人惊讶！正如一位黑客新闻海报所写：*PyTorch 太棒了！ […]虽然我很困惑。 ML 框架如何才能不痴迷于速度/性能？* 请参阅 [PyTorch 上的黑客新闻讨论](https://news.ycombinator.com/item?id=28066093) 。


 Soumith 关于 [发展 PyTorchCommunity](https://soumith.ch/posts/2021/02/forming-opensource/?fbclid=IwAR1bvN_xZ8avGvu14ODJzS8Zp7jX1BOyfuGUf-zoRawpyL-s95Vjxf88W7s) 的博客文章对此进行了一定深度的探讨，但处于较高的层次：



* PyTorch 的主要目标是可用性
* 次要目标是拥有“合理的”性能


 我们相信，保持灵活性以支持在我们的抽象基础上进行构建的研究人员的能力仍然至关重要。我们无法预见工作负载的未来，但我们知道我们希望它们首先构建在 PyTorch 上，而这需要灵活性。


 更具体地说，我们以“可用性优先”的方式进行操作，并尽量避免在没有清晰地权衡观点的情况下跳转到“限制优先”的制度(例如，静态形状，仅限图形模式)。通常会倾向于预先施加严格的用户限制，因为它可以简化实施，但这会带来风险：



* 性能可能不值得用户的摩擦，要么是因为性能优势不够引人注目，要么它只适用于相对狭窄的一组子问题。 
* 即使性能优势很引人注目，这些限制也会将生态系统分割成不同的限制集，很快就会变得对用户来说难以理解。


 我们希望用户能够将他们的 PyTorch 代码无缝移动到不同的硬件和软件平台，与不同的库和框架进行互操作，并体验 PyTorch 用户体验的全部丰富性，而不是最小公分母子集。


### 原则 2：简单胜过简单 [¶](#principle-2-simple-over-easy "永久链接到此标题")


 这里，我们借用[The Zen ofPython](https://peps.python.org/pep-0020/)：



* *显式优于隐式*
* *简单优于复杂*


 描述这两个目标的更简洁的方法是 [Simple OverEasy](https://www.infoq.com/presentations/Simple-Made-Easy/) 。让我们从一个例子开始，因为“简单”和“简单”在日常英语中经常互换使用。考虑一下如何在 PyTorch 中对设备进行建模：



* **简单/显式(理解、调试)：** 每个tensor都与一个设备关联。用户明确指定tensor设备移动。需要跨设备移动的操作会导致错误。
* **简单/隐式(使用)：** 用户不必担心设备；系统计算出全局最佳的设备布局。


 在这种特定情况下，作为一般设计理念，PyTorch 倾向于公开简单且明确的构建块，而不是易于从业者使用的 API。新的 PyTorch 用户可以立即理解和调试简单版本：如果您在程序中实际调用操作符的位置调用需要跨设备移动的操作符，您会得到一个明显的错误。简单的解决方案可能会让新用户一开始行动得更快，但调试这样的系统可能很复杂：系统如何做出决定？用于插入此类系统的 API 是什么？对象在其 IR 中如何表示？


 支持这种设计的一些经典论点来自[分布式计算的注释](https://dl.acm.org/doi/book/10.5555/974938)(TLDR：不要对具有不同性能特征的资源进行统一建模，详细信息会泄漏)和[End-to-EndPrinciple](http://web.mit.edu/Saltzer/www/publications/endtoend/endtoend.pdf)(TLDR：将智能构建到堆栈的较低层可以防止构建堆栈中较高层的高性能功能，并且通常无论如何都不起作用)。例如，我们可以构建运营商级别或全局设备移动规则，但精确的选择并不明显，并且构建可扩展的机制具有不可避免的复杂性和延迟成本。


 这里需要注意的是，这并不意味着更高级别的“简单”API 没有价值；而是意味着更高级别的“简单”API 没有价值。当然，例如，堆栈中的更高级别有一个价值，可以支持大型集群中跨异构计算的高效tensor计算。相反，我们的意思是，专注于简单的较低级别构建块有助于为简单的 API 提供信息，同时在用户需要离开人迹罕至的地方时仍然保持良好的体验。它还为创新提供了空间，并以我们无法在 PyTorch 核心库中支持的速度增长更多固执己见的工具，但最终从中受益，正如我们的[丰富的生态系统](https://pytorch.org/ecosystem/) 所证明的那样。换句话说，一开始就不自动化可以让我们更快地达到良好的自动化水平。


### 原则 3：Python 优先，具有一流的语言互操作性 [¶](#principle-3-python-first-with-best-in-class-language-interoperability "永久链接到此标题")


 这一原则始于 ** Python First** ：



> 
> 
> 
> PyTorch 不是将 Python 绑定到整体 C++ 框架中。
> 它的构建目的是深度集成到 Python 中。你可以像使用它一样自然地使用它
> [NumPy](https://www.numpy.org/) 
> ,
> [SciPy](https://www.scipy.org/) 
> ,
> [scikit-学习](https://scikit-learn.org/) 
> ,
> 或其他 Python 库。您可以使用您最喜欢的库并使用 
> [Cython] 等包在 Python 中编写新的神经网络
> 层(https://cython.org/) 
> 和
> [Numba](http://numba.pydata.org/) 
> 。我们的目标是在适当的情况下不重新发明轮子。
> 
> 
> 


 PyTorch 多年来需要处理的一件事是 Python 开销：我们首先用 C++ 重写了 autograd 引擎，然后是大部分运算符定义，然后开发了 TorchScript 和 C++ 前端。


 尽管如此，使用 Python 工作仍然可以轻松地为我们的用户提供最佳体验：它灵活、熟悉，也许最重要的是，它拥有一个可供使用的庞大的科学计算库和扩展生态系统。这一事实激发了我们最近的一些贡献，这些贡献试图达到接近 Python 可用性曲线末端的帕累托最优点：



* [TorchDynamo](https://dev-discuss.pytorch.org/t/torchdynamo-an-experiment-in-dynamic-python-bytecode-transformation/361) ，一个能够加速现有 eager-mode 的 Python 框架评估工具PyTorch 程序只需最少的用户干预。
* [torch_function](https://pytorch.org/docs/main/notes/extending.html#extending-torch) 和 [torch_dispatch](https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557)扩展点，这些扩展点使得 Python 优先的功能能够构建在 C++ 内部之上，例如 [torch.fxtracer](https分别为://pytorch.org/docs/stable/fx.html)和[functorch](https://github.com/pytorch/functorch)。


 这些设计原则并不是硬性规定，而是艰难的选择，并锚定了我们如何将 PyTorch 构建为当今可调试、可破解且灵活的框架。随着我们拥有更多的贡献者和维护者，我们期待与您一起在我们的图书馆和生态系统中应用这些核心原则。我们也愿意随着我们学习新事物和人工智能领域的发展而不断发展它们，正如我们所知道的那样。