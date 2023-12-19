# PyTorch 贡献指南 [¶](#pytorch-contribution-guide "此标题的永久链接")

> 译者：[片刻小哥哥](https://github.com/jiangzhonglian)
>
> 项目地址：<https://pytorch.apachecn.org/2.0/docs/community/contribution_guide>
>
> 原始地址：<https://pytorch.org/docs/stable/community/contribution_guide.html>


 PyTorch 是一个 GPU 加速的 Python 张量计算包，用于使用基于磁带的 autograd 系统构建深度神经网络。


## 贡献过程 [¶](#contribution-process "此标题的固定链接")


 PyTorch 组织由 [PyTorchGovernance](governance.html) 管理，贡献的技术指南可以在 [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) 中找到。


 PyTorch 开发过程涉及核心开发团队和社区之间进行大量的公开讨论。


 PyTorch 的操作方式与 GitHub 上的大多数开源项目类似。但是，如果您以前从未为开源项目做出过贡献，那么以下是基本流程。



* **弄清楚你要做什么。** 大多数开源贡献都来自于那些自己痒痒的人。但是，如果你不知道你想做什么，或者只是想获得更多熟悉该项目后，这里有一些关于如何找到合适任务的提示：
    + 查看[问题跟踪器](https://github.com/pytorch/pytorch/issues/)，看看是否有任何您知道如何解决的问题。其他贡献者确认的问题往往更容易调查。我们还为可​​能对新人有利的问题维护了一些标签，例如 **bootcamp** 和 **1hr** ，尽管这些标签维护得不太好。 
    + 加入我们的 [开发讨论](https://dev-discuss.pytorch.org/)，让我们知道您有兴趣了解 PyTorch。我们非常乐意帮助研究人员和合作伙伴快速了解代码库。

* **找出更改的范围，如果问题较大，请寻求有关 GitHub 问题的设计评论。** 大多数拉取请求都很小;在这种情况下，无需让我们知道您想做什么，只需开始吧。但如果更改很大，通常最好先通过[提交 RFC](https://github.com/pytorch/rfcs/blob/master/README.md) 获得一些设计评论。
    + 如果您不知道会有多大的变化，我们可以帮助您弄清楚！只需在 [问题](https://github.com/pytorch/pytorch/issues/) 或 [开发讨论](https://dev-discuss.pytorch.org/) 上发布相关内容即可。 
    + 一些功能的添加非常标准化；例如，很多人向 PyTorch 添加新的运算符或优化器。这些情况下的设计讨论主要归结为“我们需要这个运算符/优化器吗？”提供其实用性的证据，例如在同行评审论文中的使用，或在其他框架中的存在，在说明这一点时会有所帮助。 
        - **添加最近发布的研究中的运算符/算法**通常不被接受，除非有压倒性的证据表明这项新发表的工作具有突破性的成果并最终成为该领域的标准。如果您不确定自己的方法适用于哪里，请在实施 PR 之前先提出问题。 
    + 核心变更和重构可能非常难以协调，因为 PyTorch 主分支的开发速度非常快。一定要接触根本性或跨领域的变革；我们通常可以提供有关如何将此类更改分成更容易审查的部分的指导。

* **将其编码出来！**
    + 请参阅 [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md) 文件以获取有关以技术形式使用 PyTorch 的建议。

* **打开拉取请求。**
    + 如果您还没有准备好审查拉取请求，请先创建草稿拉取请求 
        - 您可以稍后按“准备审查”按钮将其转换为完整的 PR。您还可以在 PR 仍处于草稿状态时在其标题前加上“[WIP]”(“正在进行的工作")。在进行审核时，我们将忽略草稿 PR。如果您正在进行一项复杂的变更，最好从草稿开始，因为您需要花时间查看 CI 结果，看看事情是否成功。
    + 为您的变更寻找合适的审阅者。我们有一些人定期检查 PR 队列并尝试审查所有内容，但如果您碰巧知道受补丁影响的给定子系统的维护者是谁，请随时将他们直接包含在拉取请求中。您可以了解有关 [Persons of Interest](https://pytorch.org/docs/main/community/persons_of_interest.html) 的更多信息，他们可以审查您的代码。

* **迭代拉取请求，直到它被接受！**
    + 我们会尽力减少评审往返次数，只有在出现重大问题时才屏蔽 PR。对于拉取请求中最常见的问题，请查看[常见错误](#common-mistakes-to-avoid)。 
    + 一旦拉取请求被接受并且 CI 通过，您无需执行任何其他操作；我们将为您合并 PR。


## 入门 [¶](#getting-started "此标题的永久链接")


### 提出新功能 [¶](#propusing-new-features "永久链接到此标题")


 新功能的想法最好针对特定问题进行讨论。请提供尽可能多的信息、任何随附数据以及您建议的解决方案。 PyTorch 团队和社区经常审查他们认为可以提供帮助的新问题和评论。如果您对自己的解决方案充满信心，请继续实施它。


### 报告问题 [¶](#reporting-issues "此标题的永久链接")


 如果您发现了问题，请首先搜索那里的[现有问题列表](https://github.com/pytorch/pytorch/issues)。如果您无法找到类似的问题，请创建一个新问题。提供尽可能多的信息以重现有问题的行为。另外，请包括任何其他见解，例如您期望的行为。


### 实现功能或修复错误 [¶](#implementing-features-or-fixing-bugs "永久链接到此标题")


 如果您想解决特定问题，最好根据您的意图对单个问题进行评论。但是，我们不会锁定或分配问题，除非我们之前与开发人员合作过。最好就该问题进行对话并讨论您建议的解决方案。 PyTorch 团队可以提供节省您时间的指导。


 标记为第一新问题、低优先级或中等优先级的问题提供了最佳切入点，也是很好的起点。


### 添加教程 [¶](#adding-tutorials "永久链接到此标题")


 [pytorch.org](https://pytorch.org/) 上的大量教程来自社区本身，我们欢迎额外的贡献。要了解有关如何贡献新教程的更多信息，您可以在此处了解更多信息：[PyTorch GitHub 上的.org 教程贡献指南](https://github.com/pytorch/tutorials/#contributing)


### 改进文档和教程 [¶](#improving-documentation-tutorials "永久链接到此标题")


 我们的目标是制作高质量的文档和教程。在极少数情况下，内容会包含拼写错误或错误。如果您发现可以修复的问题，请向我们发送拉取请求以供考虑。


 查看[文档](#on-documentation) 部分以了解我们的系统如何工作。


### 参与在线讨论 [¶](#participating-in-online-discussions "永久链接到此标题")


 您可以在 [PyTorch 讨论论坛](https://discuss.pytorch.org/) 以及 [PyTorch 开发讨论论坛](https://dev-discuss.pytorch.org/) 上找到活跃的讨论。对于开发人员和维护人员。


### 提交拉取请求以修复未解决的问题 [¶](#submitting-pull-requests-to-fix-open-issues "永久链接到此标题")


 您可以在[此处](https://github.com/pytorch/pytorch/issues)查看所有未解决问题的列表。对问题发表评论是引起团队注意的好方法。您可以在这里分享您的想法以及您计划如何解决问题。


 对于更具挑战性的问题，团队将提供反馈和指导，以最好地解决问题。


 如果您无法自行解决问题，发表评论并分享您是否可以重现该问题可以帮助团队识别问题区域。


### 审查开放拉取请求 [¶](#reviewing-open-pull-requests "此标题的永久链接")


 我们感谢您帮助审查和评论拉取请求。我们的团队努力将开放拉取请求的数量保持在可管理的范围内，如果需要，我们会快速响应以获取更多信息，并合并我们认为有用的 PR。然而，由于人们的兴趣很高，对拉取请求的更多关注总是值得赞赏的。


### 提高代码可读性 [¶](#improving-code-readability "永久链接到此标题")


 提高代码可读性对每个人都有帮助。提交涉及几个文件的少量拉取请求通常比提交涉及许多文件的大量拉取请求更好。在 PyTorch 论坛 [此处](https://discuss.pytorch.org/) 或就与您的改进相关的问题开始讨论是最好的开始方式。


### 添加测试用例以使代码库更加健壮 [¶](#adding-test-cases-to-make-the-codebase-more-robust "永久链接到此标题")


 额外的测试覆盖率值得赞赏。


### 推广 PyTorch [¶](#promoting-pytorch "永久链接到此标题")


 您在项目、研究论文、文章、博客或互联网上的一般讨论中使用 PyTorch 有助于提高对 PyTorch 和我们不断发展的社区的认识。请联系[marketing@pytorch.org](mailto:marketing@pytorch.org) 获取营销支持。


### 分类问题 [¶](#triaging-issues "此标题的永久链接")


 如果您认为某个问题可以从特定标签或复杂程度中受益，请对该问题发表评论并分享您的意见。如果您觉得问题没有正确分类，请发表评论并告知团队。


## 关于开源开发 [¶](#about-open-source-development "此标题的永久链接")


 如果这是您第一次为开源项目做出贡献，那么开发过程的某些方面对您来说可能会显得不寻常。



* **没有办法“声明”问题。** 人们在决定处理某个问题时通常希望“声明”该问题，以确保当其他人最终处理该问题时不会浪费工作。这在开源中并不太有效，因为有人可能决定做某事，但最终没有时间去做。请随意以咨询方式提供信息，但最终，我们将通过运行代码和粗略共识来快速推进。
* **新功能的门槛很高。** 与企业环境不同，在企业环境中，编写代码的人隐式地“拥有”它，并且可以期望在代码的生命周期中照顾它，一旦拉取请求被合并到开源项目中，它立即成为该项目所有维护者的集体责任。当我们合并代码时，我们是说我们，维护者，可以审查后续更改并对代码进行错误修复。这自然会带来更高的贡献标准。


## 要避免的常见错误[¶](#common-mistakes-to-avoid "永久链接到此标题")



* **您添加了测试吗？**(或者如果更改很难测试，您是否描述了如何测试更改？)
    + 我们要求进行测试有几个动机： 
        1. 帮助我们判断稍后是否会破坏它 
        2. 首先帮助我们判断补丁是否正确(是的，我们确实对其进行了审查，但正如 Knuth 所言，“当心下面的代码，因为我没有运行它，只是证明它是正确的“)
    + 什么时候可以不添加测试？有时无法方便地测试某个更改，或者该更改显然是正确的(并且不太可能被破坏)，因此不测试它也没关系。相反，如果某个更改看起来可能(或已知可能)被意外破坏，那么花时间制定测试策略就很重要。

* **您的 PR 是否太长？**
    + 我们更容易审查和合并小型 PR。审查 PR 的难度与其大小呈非线性关系。 
    + 什么时候可以提交大型 PR？如果某个问题中有相应的设计讨论，并由将要审查您的差异的人员签字同意，这会很有帮助。我们还可以帮助提供有关如何将大型变更拆分为可单独交付的部分的建议。同样，如果对 PR 的内容有完整的描述，也会有所帮助：如果我们知道里面的内容，那么审查代码就更容易了！
* **对微妙事物的评论？** 如果您的代码的行为有细微差别，请包括额外的注释和文档可以让我们更好地理解代码的意图。
* **您添加了 hack 吗？** 有时，正确的答案是 hack。但通常情况下，我们必须讨论它。
* **你想接触一个非常核心的组件吗？** 为了防止重大回归，接触核心组件的拉取请求会受到额外的审查。在进行重大更改之前，请确保您已与团队讨论了您的更改。
* **想要添加新功能？** 如果您想添加新功能，请在相关问题上评论您的意图。我们的团队尝试向社区发表评论并向社区提供反馈。在构建新功能之前，最好与团队和社区其他成员进行公开讨论。这有助于我们了解您正在处理的内容，并增加合并的机会。
* **您是否接触过与 PR 无关的代码？** 为了帮助进行代码审查，请仅在您的拉取请求中包含文件与您的更改直接相关。


## 常见问题 [¶](#frequently-asked-questions "此标题的永久链接")



* **我如何作为审阅者做出贡献？** 如果社区开发人员重现问题、尝试新功能或以其他方式帮助我们识别或解决问题，那么将会有很多价值。使用您的环境详细信息对任务或拉取请求进行评论很有帮助，值得赞赏。
* **CI 测试失败，这意味着什么？** 也许您的 PR 是基于损坏的主分支？您可以尝试在最新的主分支之上重新调整您的更改。您还可以在 <https://hud.pytorch.org/> 上查看主分支 CI 的当前状态。
* **最高风险的更改是什么？** 任何涉及构建配置的内容都是风险区域。请避免更改这些内容，除非您事先与团队进行了讨论。
* **嘿，我的分支上出现了一个提交，这是怎么回事？** 有时另一个社区成员会为您的拉取请求或分支提供补丁或修复。这通常是通过 CI 测试所必需的。


## 关于文档 [¶](#on-documentation "此标题的永久链接")


### Python 文档 [¶](#python-docs "此标题的永久链接")


 PyTorch 文档是使用 [Sphinx](https://www.sphinx-doc.org/en/master/) 从 python 源生成的。生成的 HTML 复制到 [pytorch.github.io](https://github.com/pytorch/pytorch.github.io/tree/master/docs) 主分支中的 docs 文件夹，并通过 GitHub 页面提供服务。



* 站点：<https://pytorch.org/docs>
* GitHub：<https://github.com/pytorch/pytorch/tree/main/docs>
* 服务来源：<https://github.com/pytorch /pytorch.github.io/tree/master/docs>


### C++ 文档 [¶](#c-docs "此标题的永久链接")


 对于 C++ 代码，我们使用 Doxygen 生成内容文件。 C++ 文档构建在特殊服务器上，生成的文件被复制到 <https://github.com/pytorch/cppdocs
> 存储库，并从 GitHubpages 提供服务。



* 站点：<https://pytorch.org/cppdocs>
* GitHub：<https://github.com/pytorch/pytorch/tree/main/docs/cpp>
* 服务来源：<https://github.com /pytorch/cppdocs>


## 教程 [¶](#tutorials "此标题的永久链接")


 PyTorch 教程是用于帮助理解使用 PyTorch 完成特定任务或理解更全面的概念的文档。教程是使用 [Sphinx-Gallery](https://sphinx-gallery.readthedocs.io/en/latest/index.html) 构建的来自可执行的 python 源文件，或来自重组文本 (rst) 文件。



* 站点：<https://pytorch.org/tutorials>
* GitHub：<https://github.com/pytorch/tutorials>


### 教程构建概述 [¶](#tutorials-build-overview "此标题的固定链接")


 对于教程，[pullrequests](https://github.com/pytorch/tutorials/pulls) 触发器是使用 CircleCI 构建整个站点来测试更改的效果。此构建分为 9 个工作构建，总共需要大约 40 分钟。同时，我们使用 *makehtml-noplot* 进行 Netlify 构建，它构建网站时无需将笔记本输出渲染到页面中以供快速查看。


 PR 被接受后，将使用 GitHubActions 重建并部署站点。


### 贡献新教程 [¶](#contributing-a-new-tutorial "永久链接到此标题")


 请参阅 [PyTorch.org 贡献者教程](https://github.com/pytorch/tutorials/#contributing) 。