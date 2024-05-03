<div align="center">
<h2>Personal Paper Notes</h2>
</div>

---

 
> **知乎:** https://www.zhihu.com/people/friedrichor (continuously update)  
> **CSDN:** https://blog.csdn.net/Friedrichor (out of use)  

- [Multimodal](#multimodal)
  - [Large Multimodal Models](#large-multimodal-models)
    - [AnyGPT](#anygpt)
    - [VL-GPT](#vl-gpt)
    - [Kosmos-G](#kosmos-g)
    - [DreamLLM](#dreamllm)
    - [LaVIT](#lavit)
    - [SpeechGPT](#speechgpt)
  - [Multimodal Dialogue](#multimodal-dialogue)
    - [MMDialog](#mmdialog)
    - [MDRG / Divter](#mdrg--divter)
  - [Multimodal Learning](#multimodal-learning)
    - [CLIP](#clip)
    - [ViLT](#vilt)


# Multimodal

## Large Multimodal Models

### AnyGPT
  - **Paper:** [AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](https://arxiv.org/abs/2402.12226)  
  [![Star](https://img.shields.io/github/stars/OpenMOSS/AnyGPT.svg?style=social&label=Star)](https://github.com/OpenMOSS/AnyGPT)
  [![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://junzhan2000.github.io/AnyGPT.github.io/)
  [![zhihu](https://img.shields.io/badge/-知乎-000000?logo=zhihu&logoColor=0084FF)](https://zhuanlan.zhihu.com/p/683058051)
  - **Accepted:** None (temporal)
  - **Submission Date:** 2024-02-19
  - **Authors:** 复旦大学; Multimodal Art Projection Research Community; 上海AI Lab
  - **Tags:** 每个模态都通过相应 tokenizer 处理
  - **Abstract:** 我们介绍了 AnyGPT，这是一种 any-to-any 的多模态语言模型，它使用离散表示来统一处理各种模态，包括语音、文本、图像和音乐。AnyGPT 可以稳定地训练，而不需要对当前的大型语言模型 (LLM) 体系结构或训练范例进行任何更改。相反，它完全依赖于数据级的预处理，便于将新的motai无缝地整合到 LLMs 中，类似于合并新的语言。我们构建了一个以文本为中心的多模态数据集，用于多模态对齐的预训练。利用生成式模型，我们合成了第一个大规模 any-to-any 的多模态指令数据集。它由 108K 个多轮对话样本组成，这些样本错综复杂地交织着各种模态，从而使该模型能够处理多模态输入和输出的任意组合。实验结果表明，AnyGPT 能够促进 any-to-any 多模态对话，同时在所有模态上获得与专用模型相当的性能，证明了离散表示可以有效且方便地将多个模态统一在一个语言模型中。
  - **Personal Note**: [[HTML]](https://friedrichor.github.io/Awesome-Multimodal-Papers/notes/AnyGPT/index.html)

### VL-GPT
  - **Paper:** [VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation](https://arxiv.org/abs/2312.09251)  
  [![Star](https://img.shields.io/github/stars/AILab-CVC/VL-GPT.svg?style=social&label=Star)](https://github.com/AILab-CVC/VL-GPT)
  - **Accepted:** None (temporal)
  - **Submission Date:** 2023-12-14
  - **Authors:** 西安交通大学; 腾讯; 香港大学
  - **Tags:** vision tokenizer
  - **Abstract:** 在这项工作中，我们介绍了 Vision-Language Generative Pre-trained Transformer (VL-GPT)，这是一种能够同时感知和生成视觉和语言数据的 transformer 模型。. VL-GPT 通过采用直接的自回归目标实现了图像和文本模态的统一预训练方法，从而使模型能够像语言模型处理文本一样无缝地处理图像和文本。为了实现这一目标，我们最初提出了一种新的图像 tokenizer-detokenizer 框架，用于视觉数据，专门设计用于将原始图像转换为连续 embeddings 序列并相应地重建它们。结合现有的文本 tokenizer 和 detokenizer，该框架允许将交错的图像-文本数据编码为多模态序列，该序列随后可以输入到 transformer 模型中。因此，VL-GPT 可以利用统一的自回归目标 (即 next-token prediction) 对多模态语料库进行大规模预训练。在完成预训练后，VL-GPT 在各种视觉和语言理解和生成任务中表现出显著的 zero-shot 和 few-shot 性能，包括 image captioning、视觉问答、text-to-image 生成等。此外，预先训练的模型在提供多模态提示时重新训练上下文学习能力。我们进一步对我们的 VL-GPT 进行指令微调，突出其在多模态方面的特殊潜力。
  - **Personal Note**: [[HTML]](https://friedrichor.github.io/Awesome-Multimodal-Papers/notes/VL-GPT/index.html)

### Kosmos-G
  - **Paper:** [Kosmos-G: Generating Images in Context with Multimodal Large Language Models](https://arxiv.org/abs/2310.02992)  
  [![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)](https://github.com/microsoft/unilm/tree/master/kosmos-g)
  [![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://xichenpan.com/kosmosg/)
  - **Accepted:** ICLR 2024
  - **Submission Date:** 2023-10-04
  - **Authors:** 微软研究院; 纽约大学; 滑铁卢大学
  - **Tags:** vision-language-to-image, “image as a foreign language in image generation.”
  - **Abstract:** 最近在 text-to-image (T2I) 和 vision-language-to-image (VL2I) 生成方面取得了重大进展。然而，广义的视觉-语言输入的生成，特别是涉及多个图像的生成，仍然没有得到充分的探索。本文提出了一种利用多模态大语言模型 (Multimodal Large Language Models, MLLMs) 的高级感知能力来解决上述挑战的模型 KOSMOS-G。我们的方法使用文本模态作为锚，将 MLLM 的输出空间与 CLIP 对齐，并对精选数据执行组合指令调优。KOSMOS-G 展示了 zero-shot 多实体主体驱动生成的独特能力。值得注意的是，分数蒸馏指令调优不需要修改 image decoder。这允许 CLIP 的无缝替代和与无数 U-Net 技术的轻松集成，从细粒度控制到个性化 image decoder 变体。我们将 KOSMOS-G 作为 “图像作为图像生成中的外语” 这一目标的初步尝试。
  - **Personal Note**: [[HTML]](https://friedrichor.github.io/Awesome-Multimodal-Papers/notes/Kosmos-G/index.html)

### DreamLLM
  - **Paper:** [DreamLLM: Synergistic Multimodal Comprehension and Creation](https://arxiv.org/abs/2309.11499)  
  [![Star](https://img.shields.io/github/stars/RunpeiDong/DreamLLM.svg?style=social&label=Star)](https://github.com/RunpeiDong/DreamLLM)
  [![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://dreamllm.github.io/)
  - **Accepted:** ICLR 2024
  - **Submission Date:** 2023-09-20
  - **Authors:** 西安交通大学; 旷视科技; 清华大学; 华中科技大学
  - **Tags:** vision tokenizer
  - **Abstract:** 本文介绍了 DreamLLM，这是一个学习框架，它首先实现了多功能多模态大型语言模型 (MLLMs)，该模型具有多模态理解和创造之间经常被忽视的协同作用。DreamLLM 基于两个基本原则运行。第一个重点是通过在原始多模态空间中直接采样对语言和图像后验进行生成建模。该方法规避了外部特征提取器 (如 CLIP) 固有的局限性和信息丢失，获得了更彻底的多模态理解。其次，DreamLLM 促进生成原始的交错文档，对文本和图像内容以及非结构化布局进行建模。这使得DreamLLM 可以有效地学习所有条件分布、边际分布和联合多模态分布。因此，DreamLLM 是第一个能够生成自由格式交错内容的 MLLM。综合实验表明，DreamLLM 作为 zero-shot 多模态通才的优异表现，得益于增强的学习协同效应。
  - **Personal Note**: [[HTML]](https://friedrichor.github.io/Awesome-Multimodal-Papers/notes/DreamLLM/index.html) [[知乎]](https://zhuanlan.zhihu.com/p/695700682)


### LaVIT
  - **Paper:** [Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization](https://arxiv.org/abs/2309.04669)  
  [![Star](https://img.shields.io/github/stars/jy0205/LaVIT.svg?style=social&label=Star)](https://github.com/jy0205/LaVIT)
  - **Accepted:** ICLR 2024
  - **Submission Date:** 2023-09-09
  - **Authors:** 北京大学，快手科技
  - **Tags:** vision tokenizer
  - **Abstract:** 最近，大型语言模型 (LLM) 的显著进步激发了研究人员将其非凡的推理能力转移到视觉和语言数据上。然而，主流的方法主要是将视觉输入作为提示，并专注于通过冻结的 LLM 优化以视觉内容为条件的文本生成过程。这种对视觉和语言的不公平对待严重限制了模型的潜力。在本文中，我们突破了这一局限，将视觉和语言以统一的形式表现出来。具体来说，我们引入了一个设计良好的视觉 tokenizer，将非语言的图像翻译成一系列离散的 tokens，就像 LLM 可以阅读的外语一样。生成的视觉 tokens 包含与单词相当的高级语义，并且还支持随图像变化的动态序列长度。利用这个 tokenizer，所提出的基础模型 **LaVIT** 可以在相同的生成学习范式下对图像和文本进行无差别处理。这种统一使 LaVIT 成为一个令人印象深刻的通用界面，可以同时理解和生成多模态内容。大量的实验进一步表明，在大量的视觉语言任务中，它比现有的模型表现得更好。
  - **Personal Note**: [[HTML]](https://friedrichor.github.io/Awesome-Multimodal-Papers/notes/LaVIT/index.html)

### SpeechGPT
  - **Paper:** [SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities](https://aclanthology.org/2023.findings-emnlp.1055/)  
  [![Star](https://img.shields.io/github/stars/0nutation/SpeechGPT.svg?style=social&label=Star)](https://github.com/0nutation/SpeechGPT)
  [![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://0nutation.github.io/SpeechGPT.github.io/)  
  - **Accepted:** EMNLP 2023 (Findings)
  - **Submission Date:** 2023-05-18
  - **Authors:** 复旦大学
  - **Tags:** 音频理解+生成, 离散token
  - **Abstract:** 多模态大型语言模型被认为是实现人工通用智能（AGI）的关键一步，并随着 ChatGPT 的出现而备受关注。然而，目前的语音语言模型通常采用级联范式，阻碍了跨模态知识的传递。在本文中，我们提出了 SpeechGPT，这是一种具有内在跨模态会话能力的大型语言模型，能够感知和生成多模型内容。通过离散语音表征，我们首先构建了大规模跨模态语音教学数据集 SpeechInstruct。此外，我们还采用了三阶段训练策略，包括模态适应预训练、跨模态指令微调和模态链指令微调。实验结果表明，SpeechGPT 在遵循人类多模态指令方面具有令人印象深刻的能力，并凸显了用一个模型处理多种模态指令的潜力。
  - **Personal Note**: [[HTML]](https://friedrichor.github.io/Awesome-Multimodal-Papers/notes/SpeechGPT/index.html)


## Multimodal Dialogue

### MMDialog
  - **Paper:** [MMDialog: A Large-scale Multi-turn Dialogue Dataset Towards Multi-modal Open-domain Conversation](https://aclanthology.org/2023.acl-long.405/)  
  [![Star](https://img.shields.io/github/stars/victorsungo/MMDialog.svg?style=social&label=Star)](https://github.com/victorsungo/MMDialog)
  - **Accepted:** ACL 2023
  - **Submission Date:** 2022-11-10
  - **Authors:** 北京大学; 微软
  - **Tags:** 大规模多模态对话数据集, 多模态对话评价指标 MM-Relevance
  - **Abstract:** 对多模态内容进行响应已被认为是智能会话代理的基本功能。在本文中，我们引入了 MMDialog 数据集来更好地促进多模态会话。MMDialog 由 108 万个真实世界的对话组成，其中包含 4184 个主题的 153 万张独特图像。MMDialog 有两个主要且独特的优点。首先，它是最大的多模式对话数据集，对话数量是88x。其次，它包含大量的主题来概括开放领域。为了使用该数据集构建一个引人入胜的对话系统，我们提出并规范了两个基于检索和生成场景的响应预测任务。此外，我们用最先进的技术为上述任务建立了两条基线，并报告了它们的实验性能。我们还提出了一个新的评价指标 MM-Relevance 来衡量多模态回复。
  - **Personal Note**: [[HTML]](https://friedrichor.github.io/Awesome-Multimodal-Papers/notes/MMDialog/index.html)

### MDRG / Divter
  - **Paper:** [Multimodal Dialogue Response Generation](https://aclanthology.org/2022.acl-long.204/)  
  - **Accepted:** ACL 2022  
  - **Submission Date:** 2021-10-16  
  - **Authors:** 北京大学; 微软
  - **Tags:** 多模态对话生成
  - **Abstract:** 图像响应已被认为是智能会话代理的一项重要功能。然而，现有的研究只关注于基于检索方法的多模态对话模型的探索，而忽略了生成方法。为了填补空白，我们首先提出了一个新的任务:给定对话上下文的多模态对话响应生成(MDRG)，一个模型需要生成文本或图像作为响应。学习这样的MDRG模型通常需要包含文本和图像的多模态对话，而这些对话很难获得。由于实践中的挑战，我们在一个自然的假设下考虑MDRG，即只有有限的训练样本可用。在这种低资源设置下，我们设计了一种新的会话代理，Divter，以便从整个生成模型中分离依赖于多模态对话的参数。通过这种方法，模型的主要部分可以分别从大量的纯文本对话和文本对中学习，然后只用几个训练样例就可以很好地拟合整个参数。大量的实验表明，我们的方法在自动和人工评估中都取得了最先进的结果，并且可以生成信息丰富的文本和高分辨率的图像响应。
  - **Personal Note**: [[HTML]](https://friedrichor.github.io/Awesome-Multimodal-Papers/notes/MDRG/index.html) [[知乎]](https://zhuanlan.zhihu.com/p/692609405) [[CSDN]](https://blog.csdn.net/Friedrichor/article/details/127833890)



## Multimodal Learning

### CLIP
  - **Paper:** [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
  [![Star](https://img.shields.io/github/stars/OpenAI/CLIP.svg?style=social&label=Star)](https://github.com/OpenAI/CLIP)
  [![Blog](https://img.shields.io/badge/OpenAI-Blog-orange.svg)](https://openai.com/index/clip)  
  - **Accepted:** ICML 2021
  - **Submission Date:** 2021-02-26  
  - **Authors:** OpenAI  
  - **Tags:** 对比学习
  - **Abstract:** 最先进的计算机视觉系统被训练来预测一组固定的预定 object 类别。这种受限制的监督形式限制了它们的通用性和可用性，因为需要额外的标记数据来指定任何其他视觉概念。直接从原始文本中学习图像是一种很有前途的选择，它利用了更广泛的监督来源。我们证明了预测哪个标题与哪个图像相匹配的简单预训练任务是一种有效且可扩展的方法，可以在从互联网收集的 4 亿对 (图像，文本) 数据集上从头开始学习 SOTA 图像表示。在预训练之后，使用自然语言来参考学习到的视觉概念(或描述新的概念)，从而使模型 zero-shot 转移到下游任务。我们通过对 30 多个不同的现有计算机视觉数据集进行基准测试来研究这种方法的性能，这些数据集涵盖了 OCR、视频中的动作识别、地理定位和许多类型的细粒度对象分类等任务。该模型不平凡地转移到大多数任务，并且通常与完全监督的基线竞争，而不需要任何数据集特定的训练。例如，我们在 ImageNet zero-shot 上达到原始 ResNet-50 的精度，而不需要使用它所训练的 128 万个训练样本中的任何一个。
  - **Personal Note**: [[知乎]](https://zhuanlan.zhihu.com/p/691509430) [[CSDN]](https://blog.csdn.net/Friedrichor/article/details/127272167)

### ViLT
  - **Paper:** [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)  
  [![Star](https://img.shields.io/github/stars/dandelin/vilt.svg?style=social&label=Star)](https://github.com/dandelin/vilt)  
  - **Accepted:** ICML 2021
  - **Submission Date:** 2021-02-05
  - **Tags:** Transformer 统一处理视觉+文本
  - **Abstract:** 视觉和语言预训练 (Vision-and-Language Pre-training, VLP) 在各种联合视觉和语言下游任务中提高了性能。目前的 VLP 方法严重依赖于图像特征提取过程，其中大多数涉及区域监督 (例如，目标检测) 和卷积架构 (例如，ResNet)。尽管在文献中被忽视，但我们发现它在两个方面都存在问题:(1)效率/速度，简单地提取输入特征比多模态交互步骤需要更多的计算;(2)表达能力，因为它是视觉嵌入器及其预定义视觉词汇的表达能力的上限。在本文中，我们提出了一个最小的VLP模型，视觉和语言 Transformer (ViLT)，它的整体意义在于视觉输入的处理被大大简化为与我们处理文本输入相同的无卷积方式。我们表明，ViLT 比以前的 VLP 模型快几十倍，但具有竞争力或更好的下游任务性能。
  - **Personal Note**: [[知乎]](https://zhuanlan.zhihu.com/p/692450450) [[CSDN]](https://blog.csdn.net/Friedrichor/article/details/127167784)
