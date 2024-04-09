
<head>
<style>
    body {text-align: justify;}
</style>
</head>

[TOC]

<p style="text-align: center; font-weight: bold; font-size: 24px;">
MMDialog: A Large-scale Multi-turn Dialogue Dataset Towards Multi-modal Open-domain Conversation
</p>
**Title:** MMDialog: A Large-scale Multi-turn Dialogue Dataset Towards Multi-modal Open-domain Conversation
**Paper:** https://arxiv.org/abs/2211.05719
**Submission Date:** 2022.11.10
**Accepted:** ACL 2023
**Github:** https://github.com/victorsungo/MMDialog
**Authors:**  北京大学; 微软

<hr>

# 5 任务定义

&emsp;&emsp;假设有一个多模态对话数据集 $\mathcal D=\{(U_i, R_i)\}_{i=1}^n$，其中 $\forall i\in\{1,\ldots,n\}$，$U_i$ 是多轮对话上下文，$R_i$ 是对于 $U_i$ 的回复。$U_i$ 和 $R_i$ 可以包含多模态组件：文本元素 (如 utterances) 和 视觉元素 (如 图像)。对于任意的 $U$ 和 $R$，我们定义 $U_i=\{u_k^m\}_{k=1}^K$ 和 $R_i=\{r_l^m\}_{l=1}^L$ 作为包含文本 utterances 和视觉图像的多模态元素序列。$K$ 和 $L$ 分别是上下文和回复的元素数。$m\in\{t,v\}$ 表示元素的模态类型，$t$ 表示文本 utterances 而 $v$ 表示视觉图像。 我们的目标是从 $\mathcal D$ 中学习一个多模态对话模型 $g$，对于任何一个新的上下文 $U$，都能够使用 $g$ 来预测一个多模态回复 $R$。

&emsp;&emsp;目前关于纯文本开放域对话系统的研究主要包括基于检索的方法和生成的方法。我们将其适应于多模态场景，并定义了以下两个构建多模态开放域对话系统所必需的任务：

**任务1：多模态回复生成 (Task-1: Multi-modal Response Generation)**

&emsp;&emsp;为了生成多模态回复 $R$，需要学习一个多模态生成模型 $P(R|U;\theta)$，其中 $\theta$ 为模型参数。因此，给定一个新的对话上下文 $U$，$P(R|U;\theta)$ 可以直接合成由文本 utterances  和/或 视觉图像 组成的多模态回复 $\tilde R$。

**任务2：多模态回复检索 (Task-2:Multi-modal Response Retrieval)**

&emsp;&emsp;对于基于检索的模型，每一个对话样本 $(U,R)$ 都额外提供了一系列 negative 多模态元素作为干扰。然后我们将 $R$ 中的 ground-truth 的文本 utterances $\{r_l^t\}$ 和 negative 样本组合成一个用于文本检索的候选集 $C^t=\{r_z^t\}_{z=1}^Z$，其中 $Z$ 是 $C$ 的 size。同样，我们可以搭建图像候选集 $C^v=\{r_z^v\}_{z=1}^Z$。因此，回复检索模型的目标是在预测每个元素 $r_l^m$ 时，从给定的元素候选集 $C^t$ 或 $C^v$ 中逐步提取元素。通过这种自回归风格的检索过程，我们最终可以获得一个完整的检索到的多模态回复 $\tilde R$。

**回复模态意图预测 (Response Modal Intent Prediction)**

&emsp;&emsp;在 MMDialog 中，文本 utterances 和视觉图像能够自由地位于多模态回复中的任何位置。因此，回复元素模态的生成或检索顺序对多模态会话也具有重要意义。意图预测任务旨在给定对话上下文 $U$ 时预测回复 $\tilde R$ 中不同模态的顺序。因此，意图预测可以定义为一个分类任务：
$$
\forall j\in[1,J],\mathcal I(U,\tilde R_{<j})\in {0,1,2}
$$
其中，$\mathcal I(\cdot,\cdot)$ 是意图预测模型，它将对话上下文 $U$ 和在第 $j$ 步之前生成/检索出来的回复 $\tilde R_{<j}$ 作为输入，从而得到下一个回复的模态类型。具体来说，当 $r_j$ 是一个文本 utterance 时应该预测为 0，当 $r_j$ 是视觉图像时，应该预测为 1。我们也定义为 2 表示为回复 $\tilde R$ 结束，模型应该停止生成/检索。

<hr>



# 6 多模态对话任务的评价

&emsp;&emsp;因为大多数用于文本生成 (如 BLEU，ROUGE) 或图像生成任务 (如 DALLE 中使用的 FID，IS) 或检索 (如 Recall) 的评价指标只能在单一模态中进行评价。同时，多模态对话回复中元素的模态顺序可能与 ground-truth 回复不一样。因此，对交叉模态回复元素进行评价是非常重要的。

<img src="https://img-blog.csdnimg.cn/d091d6291d5b460e9f3038d3d9ba09a3.png">

&emsp;&emsp;在 **Task-1** 中，我们可以通过将生成的文本部分与来自左侧的 ground-truth 回复进行对齐来获得 BLEU 和 ROUGE 分数。在预测回复的下一个文本元素时，如果模型生成没有生成与当前步 ground-truth 元素相对应的文本元素，那么我们可以将该步骤的评价结果赋值为 0。然而，我们不能对文本回复生成和图像生成任务的评价指标 (如 PPL 和  FID) 直接采用相同的策略，因为默认零值的设置是不平凡的。同样，我们只对生成的图像计算 IS。

&emsp;&emsp;在 **Task-2** 中，我们也可以用类似的方法计算 Recall 分数。具体来说，我们首先对齐在检索到的回复中的文本 (或视觉) 元素 和左侧 ground-truth 回复中的文本 (或图像) 元素。当预测回复中下一个文本 (或视觉) 元素时，如果模型没有检索到与 ground-truth 回复相一致的任何文本 (或视觉) 元素，我们也可以将这一步的 Recall 值赋为 0。如果先前检索到的文本 (或视觉) 元素在第 $j$ 步之前就达到 ground-truth 回复的文本 (或视觉) 元素数量，第 $j$ 步的 Recall 分数将不被考虑，并且模型只能从候选集 $C^t$ (或 $C^v$) 中给定的 negative 元素中检索文本 (或视觉) 元素。示例的 Recall 分数是相同模态中所有蒜素的平局分数。然而，Recall 分数只能反应单个模态的模型表现，不能全面衡量多模态回复的整体质量。

&emsp;&emsp;为了解决上述两个任务的评价问题，我们提出了一个新的评价指标 **MM-Relevance**，该指标基于大规模预训练多模态模型 CLIP 对多模态对话回复生成和检索任务进行 visual-language 匹配。CLIP 被训练于一个巨大的 image-caption 对的语料库。它学会了通过对比目标将两种模态 (视觉和文本)的 embeddings 结合起来。因此，我们利用这个对比模型来评价生成/检索出来的回复和 ground-truth 的回复之间的相关性，以缓解模态不对齐的问题。具体来说，假设我们获得一个生成/检索出来的多模态回复 $\tilde R=\{\tilde r_j^m\}_{j=1}^J$，其对应的 ground-truth 的回复 $R=\{r_l^m\}_{l=1}^L$。我们首先从左边对其这两个序列，然后分别通过 CLIP 预训练的 text encoder 或 image encoder 对文本回复或视觉图像进行编码，得到每个元素的表征向量。定义两个回复的编码向量为：$\tilde E=\{\tilde e_j^m\}_{j=1}^J$ 和 $E=\{e_l^m\}_{l=1}^L$。然后逐位计算两个元素间的 CLIP 分数，直到它们不能对齐：
$$
{\rm MM_{Rel}}(R,\tilde R)=\sum_{i=1}^{min\{L,J\}}(e_i^m)^T\cdot\tilde e_i^m
$$
为了惩罚生成/检索出的序列太长或太短，进一步改进了这个评价指标：
$$
{\rm P_{MM}}=\frac{{\rm MM_{Rel}}(R,\tilde R)}{J}  \\
{\rm R_{MM}}=\frac{{\rm MM_{Rel}}(R,\tilde R)}{L}  \\
{\rm F1_{MM}}=\frac{2{\rm P_{MM}}{\rm R_{MM}}}{{\rm P_{MM}}+{\rm R_{MM}}}
$$
${\rm P_{MM}}$，${\rm R_{MM}}$，${\rm F1_{MM}}$ 分别表示 soft-precision，soft-recall 和 soft-F1。我们将 ${\rm F1_{MM}}$ 作为 **MM-Relevance**。因此，现在可以计算两个未对齐模态回复 $R$  和 $\tilde R$ 之间的相关度了。

<hr>

&emsp;&emsp;对于意图预测，遵循 [PhotoChat](https://aclanthology.org/2021.acl-long.479/) 的方法，采用 F1 分数作为评价指标，衡量模型对对话中模态顺序预测的准确性。具体来说，我们首先分别得到生成/检索出来的和 ground-truth 的模态序列：$\tilde M=\{\tilde m_j\}_{j=1}^J$ 和 $M=\{m_l\}_{l=1}^L$。 F1 分数计算如下：
$$
{\rm Match(M,\tilde M)}=\sum_{i=1}^{min\{L,J\}}\mathbb 1(m_i,\tilde m_i)   \\
{\rm P_{intent}}=\frac{{\rm Match(M,\tilde M)}}{J}  \\
{\rm R_{intent}}=\frac{{\rm Match(M,\tilde M)}}{L}  \\
{\rm F1_{Intent}}=\frac{2{\rm P_{intent}}{\rm R_{intent}}}{{\rm P_{intent}}+{\rm R_{intent}}}
$$
其中 $\mathbb 1$ 是一个指示函数，当 $m_i=\tilde m_i$ 时它的值是 1，否则为 0。在这两个任务中，$J$ 是根据生成/检索出的回复 $\tilde R$ 的模态序列所决定的。

<hr>

 # 7 Baselines

&emsp;&emsp;如 Figure 2 所示，我们利用 baseline 模型来评估 MMDialog 利用上述两个新型多模式任务。

## 7.1 多模态回复生成模型

&emsp;&emsp;我们考虑实现 [Multimodal Dialogue Response Generation](https://aclanthology.org/2022.acl-long.204/) 提出的 SOTA 的多模态对话回复生成模型 Divter (Figure 2a)，该模型由两个组件组成：文本对话回复生成器 $\mathcal G$ 和 description-to-image 转换器 $\mathcal F$。

&emsp;&emsp;具体来说，$\mathcal G$ 以对话上下文 $U$ 为输入，然后生成一个文本序列，该文本序列可以包含一个文本回复 $r^t$ 或一个文本图像描述 $r^c$ 或两个都有。注意，在我们的 MMDialog 设置中，在多轮对话上下文中也可能会有一些图像 $u^v$，因此我们在 image-to-description 转换模型的帮助下，用它们的描述 $u^c$ 替换这些图像。 这样，我们就可以将文本 utterances $u^t$ 和图像描述连接成一个序列，作为 $\mathcal G$ 的输入。此外，我们在文本 utterances 和图像描述的开头分别加上 $\rm [UTT]$ 和 $\rm [DST]$，以区分它们。然后，对于生成的开头为 $\rm [DST]$ 的图像描述 $r^c$，$\mathcal F$ 将其作为条件输入，生成一个逼真连续的高分辨率图像 $r^v$ 作为真实回复。

## 7.2 多模态回复检索模型

略

# 8 实验

&emsp;&emsp;在 MMDialog 数据集上进行了实验，以评估 baselines 在提出的多模态对话任务的性能。我们对每个对话的第一轮之外的所有轮进行回复/意图预测，并将之前的所有轮视为上下文。

## 8.1 实验设置

&emsp;&emsp;首先对 10K 和 10K 对话分别进行验证和测试，详细统计数据见 Table 2。对于检索任务，我们为每个对话随机抽取 999 个 negative 文本 utterances 和 999 个 negative 视觉图像，保持候选元素总数为 1K。

<img src="https://img-blog.csdnimg.cn/417ed7f659204c9c9a803b6012a53c3a.png" style="zoom: 50%;" />

&emsp;&emsp;在训练阶段，像 [Learning Transferable Visual Models From Natural Language Supervision](http://proceedings.mlr.press/v139/radford21a.html) 一样，对 negative 样本进行批量采样。对于文本对话回复生成器，使用 huggingface 提供的 transformers 库来 fine-tune “DialoGPT-medium”，这与 [Multimodal Dialogue Response Generation](https://aclanthology.org/2022.acl-long.204/) 一致。

&emsp;&emsp;对于 description-to-image 转换器，使用 [dalle-mini](https://github.com/borisdayma/dalle-mini) 中的 “mega” 版本实现的 DALL-E，这也与 [Multimodal Dialogue Response Generation](https://aclanthology.org/2022.acl-long.204/) 的设置一致。我们 fine-tune DALL-E mega，初始学习率为 1e-8，mini-batch size 为 64，所有图像都处理成 $256\times256$ RGB 格式。

&emsp;&emsp;为了在 MMDialog 中获取图像描述，采用了 OFA-huge ([OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework](https://proceedings.mlr.press/v162/wang22al.html)) 使用代码 [https://github.com/OFA-Sys/OFA/tree/feature/add_transformers](https://github.com/OFA-Sys/OFA/tree/feature/add_transformers)。

&emsp;&emsp;在本文中所有使用的所有 CLIP 模型版本都是 “[openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)“。

&emsp;&emsp;在实现 Divter 时，我们遵循相同的实验配置。

&emsp;&emsp;对于检索 baseline，通过 CLIP 模型获得两种模态的表征向量，并在训练过程中固定。检索任务中使用的 transformers 由 4 个 Transformer layers 组成，其中 hidden size 为 512 且 heads 数为 8。训练检索模型时初始学习率为 5e-7，mini-batch size 为 512。

&emsp;&emsp;对于所有 baselines，在验证集中采用 early stopping 策略，并根据验证集的性能采选择最好的模型。这两个任务的训练都是在 8 块 Nvidia Tesla A100 80G GPU 上进行的。BLEU 和 ROUGE 分数时通过 [https://github.com/Maluuba/nlg-eval](https://github.com/Maluuba/nlg-eval) 的代码进行计算。IS 通过 [https://github.com/toshas/torch-fidelity](https://github.com/toshas/torch-fidelity) 计算。

## 8.2 多模态 baselines 结果

&emsp;&emsp;Table 3 展示了多模态回复生成 baseline 的评价结果。遵循 [Multimodal Dialogue Response Generation](https://aclanthology.org/2022.acl-long.204/)，评价了文本回复生成、图像生成和意图预测任务。首先，可以发现，在提出的 MMDialog 中，SOTA 模型 Divter 实现了相对较低的文本回复生成性能 (BLEU-1 为 9.44，ROUGE-L 为 11.19)，这验证了多模态回复生成任务的困难，也证明了为构建数据驱动的大规模多模态对话数据集的必要性。其次，与文本生成的结果对比，该模型在图像生成任务中上有着更好的性能，IS 为 20.53。第三，我们观察到 baseline 在意图预测任务上取得了 71.77 的 F1 分数，这表明模型在对话过程中判断生成文本还是图像的能力相当强。最后，我们还利用提出的 MM-Relevance 来评价生成的多模态对话回复与 ground-truth 回复之间的整体相关度，baseline 达到了 61.85 分。

<img src="https://img-blog.csdnimg.cn/7cab278cb32547b29772f4b930a69e30.png">

检索模型结果：

<img src="https://img-blog.csdnimg.cn/c3ece75ef36a40f8b7e5f3d04d985041.png">

## 8.3 案例研究

![](https://img-blog.csdnimg.cn/27be014d095a4effb2d8125d7b1f4ede.png)

&emsp;&emsp;为了进一步研究我们提出的 baseline 预测的多模态回复的质量，在 Figure 3 中展示了 MMDialog 测试集数据的一个示例。左侧为 ”A“ 和 ”B“ 之间的多轮对话上下文，右侧为 我们设计的 baselines 生成/检索出来的多模态回复。可以看到，Divter 生成的文本回复与对话上下文是一致的，并且它还可以在最后一轮语境中生成一个逼真的关于 ”Power Station“ 的高分辨率图像，这体现了我们设计的生成式 baseline 的多模态生成能力。在检索模型上，baseline 还提取了与对话上下文语义相关的 ”PinkFloyd“ 的文本回复，以及 ”Power Station“ 的图像，验证了 baseline 的有效性。

# 9 结论

我们提出了 MMDialog，一个面向多模态开放域会话的大规模多轮对话数据集。通过从超过 4K 个主题中提取与图像及其周围上下文相关的对话，MMDialog 提供了一个多样化的开放域数据集。为了促进构建更吸引人的多模态对话系统的研究，我们定义了多模态回复生成任务和检索任务，以及基于 MMDialog 的 MM-Relevance 评价指标。我们还搭建了 baseline 模型，并对其性能进行了一些分析。



​    
