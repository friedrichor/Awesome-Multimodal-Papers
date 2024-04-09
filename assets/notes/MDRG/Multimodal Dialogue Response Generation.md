[TOC]

<center><B><font size=5>Multimodal Dialogue Response Generation</font></B></center>

<p align="right">——ACL 2022</p>

# 前言

论文标题：[Multimodal Dialogue Response Generation](https://aclanthology.org/2022.acl-long.204.pdf)
论文网址：[https://aclanthology.org/2022.acl-long.204/](https://aclanthology.org/2022.acl-long.204/)
收  录  于：ACL 2022

<hr>

**<font size=5 face="楷体">省流版：</font>**

**<font size=4>动机：</font>**

- 多模态开放域对话的回复生成，目前研究者们基本都是围绕检索任务进行研究，很少涉猎 **生成任务**
  - 检索模型会受训练数据集的制约，无法在新场景下获得良好表现
  - 多模态对话生成任务除文本生成外，还涉及难度较大的图片生成
- 多模态对话 **数据集** 由于人工构造难度大，真实数据涉及隐私等原因，可用数量很少
- 图像与文本难以 **联合表示** 的问题依然存在，图片会含有大量难以用文本表示的抽象信息

<center><img src="https://img-blog.csdnimg.cn/0f167bde74734a81958d9f33eb55fe2b.png" width="20%"></center>
<I>Figure 3：</I>提出的方法的抽象逻辑。实线表示存在大规模训练集对生成模型进行预训练，虚线表示只有很少的训练样例可用，“×” 表示生成质量差。其中 $U$ 为对话文本， $r^e$ 为文本回复，$c$ 为图片描述，$r^v$ 为图片回复，${\mathcal D}_C,{\mathcal D}_P,\widetilde{\mathcal D}_S$ 分别为相应的数据集 (后文有详细说明)。

<center><img src="https://img-blog.csdnimg.cn/e9d1ab5ff8f74a78adf8eecf36c11de5.png"></center>
*Figure 2*：本文多模态对话回复生成模型的整体结构。**Textual Dialogue Response Generator** 将文本对话上下文 $U$ 作为输入，生成包含文本回复和图像描述的序列（例如，“a parrot with red belly and green back is standing on the railing.”(一只红腹绿背的鹦鹉正站在栏杆上)）。以描述为条件，**Text-to-Image Translator** 生成图像表示 $\hat{z}$。 **Image Decoder** $V_D$ 将 $\hat{z}$ 重建为逼真连续的高分辨率图像。 

**<font size=4>方法：</font>**

- 虽然文本对话+图片的相关数据集较少，但图片描述+图片的数据集很多
  - 对于文本回复生成，采用常规的开放域 **对话回复** 生成方法
  - 对于图片的生成，采用间接生成的策略，先依据对话文本生成 **图片描述**，再根据图片描述文本生成图片
- 采用基于 Transformer 的端到端模型 (Divter)，分别单独预训练两个子模型
  - Text-to-Text 模型，依据对话文本生成文本回复和图片描述
  - Text-to-Image 模型，依据图片描述生成图片

**<font size=4>实验结果及结论：</font>**

- 机器评价
  - 文本生成任务采用 PPL、BLEU 和 Rouge 作为评价指标，图片生成任务采用 FID 和 IS 评价图片质量
  - Divter 模型在图片描述生成、文本回复生成和图片生成三个任务上均取得了高于 baseline 的表现

<center><img src="https://img-blog.csdnimg.cn/29117c7819f54d3c93ce09ee9211f365.png"></center>

- 人工评价
  - Divter 模型在文本和图片方面都获得了更高的人工评价分数

<center><img src="https://img-blog.csdnimg.cn/1d1b8db01fab430dbd433ab880c37d96.png" width="60%"></center>

<hr>

# <font face="Times New Roman">0 摘要</font>

&emsp;&emsp;图像响应已经被认为是智能会话代理的一项重要能力。然而，现有的研究主要集中在基于检索的多模态对话模型，而忽略了生成方法。

&emsp;&emsp;为了填补这一空白，作者首先提出了一个新的任务：多模态对话响应生成 (multi- modal dialogue response generation，**MDRG**) —— 在给定的对话上下文中，一个模型需要生成文本或图像作为回复。

&emsp;&emsp;学习这样的 MDRG 模型通常需要包含文本和图像的多模态对话，而这些对话很难获得。出于实践中的挑战，我们在一个自然的假设下考虑 MDRG，即只有有限的训练样例可用。在这样的一个低资源环境下，我们设计了一个新的会话代理 **Divter**，以便从整个生成模型中分离出依赖于多模态对话的参数。通过这种方法，模型的主要部分可以分别从大量 纯文本对话 和 文本-图像对 中学习，然后只需要几个训练样例就可以很好地拟合整个参数。

&emsp;&emsp;大量实验表明，该方法在自动评估和人工评估方面都达到了 SOTA，并能生成信息丰富的文本和高分辨率的图像回复。

<hr>

# <font face="Times New Roman">1 Introduction</font>

&emsp;&emsp;近几十年来，随着实时通信技术的发展，网络对话的媒介也从单纯的文本转变为多种视觉模态 (如图像、GIF动画、短视频)。与现实中通过通信工具 (如 Facebook，WhatsApp，WeChat) 进行交流类似，一个优秀的智能会话代理应该不仅能够用纯文本自由对话，还要具备感知和分享真实视觉物理世界的能力。

&emsp;&emsp;尽管最近一些大规模预训练纯文本对话生成模型，如 DialoGPT, Blender, Meena，表现出了优异的性能，但他们仍然不能完全依赖纯文本来完全模拟视觉感知的丰富体验。最近，各种 vision-language 任务被引入并引起广泛关注，如 视觉问答 (visual question answering)，图像描述 (image captioning)，基于图像的对话 (image-grounded dialogue)。
- DialoGPT：
论文标题：[DIALOGPT : Large-Scale Generative Pre-training for Conversational Response Generation](https://aclanthology.org/2020.acl-demos.30.pdf)
论文网址：[https://aclanthology.org/2020.acl-demos.30/](https://aclanthology.org/2020.acl-demos.30/)
- Blender
论文标题：[Recipes for Building an Open-Domain Chatbot](https://aclanthology.org/2021.eacl-main.24.pdf)
论文网址：[https://aclanthology.org/2021.eacl-main.24/](https://aclanthology.org/2021.eacl-main.24/)
- Meena：
论文标题：Towards a Human-like Open-Domain Chatbot
论文网址：[https://arxiv.org/abs/2001.09977](https://arxiv.org/abs/2001.09977)


<center><img src="https://img-blog.csdnimg.cn/71307b909c1e47fd9c5e8481276a68d8.png" width="40%">

&emsp;&emsp;在人类对话中，图像很容易表现出丰富的视觉感知，这是纯文本难以表达的。 如 *Figure 1* 所示，至少在三种情况下需要图像： 

1. 另一个说话者对只有你见过的物体知之甚少(例如，在第一幅图像中的 colorful Burano)；
2. 分享物品的更多细节 (例如，在第二幅图像中的红酒和意大利面)；
3. 表达你对某一特定时间的情绪 (例如，在第三幅图的开心)

&emsp;&emsp;现有的一个相关任务是图片分享，其目的是基于文本上下文来选择和分享图像，这是一个具有挑战性的任务，需要模型理解由人类想象补充的背景故事，而不是像以前工作那样定位相关的视觉目标或明确提到图像中的主要可见内容。[PhotoChat: A Human-Human Dialogue Dataset With Photo Sharing Behavior For Joint Image-Text Modeling](https://aclanthology.org/2021.acl-long.479/) 提出一种基于检索的方法来解决上述挑战。然而，基于检索的方法在特定领域受到预先构建的会话历史存储库的大小的限制，特别是在历史对话中没有涉及的长尾 (long-tail) 上下文，其中检索系统的图像回复集也是固定的。另一方面，一个更好的方法是相应地生成一个新的回复。

&emsp;&emsp;本文提出了一个新的问题：多模态对话响应生成(**M**ultimodal **D**ialogue **R**esponse **G**eneration, **MDRG**)，即在给定对话上下文的情况下，模型不仅要生成纯文本回复，还要具有生成多模态回复的能力 (例如，同时包含图像和文本)。 

作者认为在实际应用中仍然存在一些障碍，因为：
1. 复杂的神经端到端结构将过拟合极少的标注好的训练数据 (例如，少数现有的 10k 多模态对话)。当讨论训练数据领域外的话题时，其性能急剧下降。
2. 由于人力资源昂贵，为一个新的领域收集足够的训练数据并不容易。

基于以上事实，我们进一步将 MDRG 的假设扩展到只有少量多模态对话可用的低资源环境中。

&emsp;&emsp;为了解决上述问题，我们的主要思想是通过分离文本回复生成和图像回复生成，使依赖于多模态对话的参数变得小且独立，从而从纯文本对话和更容易获得的 <image description, image> pairs 中学习生成模型的主要部分。 具体来说，作者提出了 **Divter**，一个由大规模视觉世界体验驱动的新型会话代理。

<center><img src="https://img-blog.csdnimg.cn/e9d1ab5ff8f74a78adf8eecf36c11de5.png"></center>
*Figure 2*：本文多模态对话回复生成模型的整体结构。**Textual Dialogue Response Generator** 将文本对话上下文 $U$ 作为输入，生成包含文本回复和图像描述的序列（例如，“a parrot with red belly and green back is standing on the railing.”(一只红腹绿背的鹦鹉正站在栏杆上)）。以描述为条件，**Text-to-Image Translator** 生成图像表示 $\hat{z}$。 **Image Decoder** $V_D$ 将 $\hat{z}$ 重建为逼真连续的高分辨率图像。 



&emsp;&emsp;如 *Figure 2* 所示，Divter 由两个基于 Transformer 的组件组成：一个多模态对话回复生成器，和一个 text-to-image 转换器。Divter 将对话上下文作为输入，生成文本序列，该序列可以包含一个文本回复或一个文本形式的图像描述，也可以包含两者。text-to-image 转换器将以上的图像描述作为条件，生成逼真连续的高分辨率图像。这两个组件都是独立的，具有相反的知识，因此可以分别使用大量的纯文本对话和 <image description, image> pairs。 端到端的 Divter 依赖于以元组形式构造的多模态对话： *(dialogue context, text response / <image description, image>)*，但是这两个组件的联合学习和评估仅需要一些训练样例，具体取决于特定领域。

本文的贡献有三个方面：

- 这是第一项在多模态对话回复生成的工作。作者在低资源环境下探索这个任务，其中只有一些多模态对话被假定为可用。
- 本文提出 Divter，一个新颖的会话代理，它可以有效地理解对话上下文并生成信息丰富的文本和高分辨率图像回复。
- 在 PhotoChat Corpus 上进行大量实验证明了 Divter 的有效性，它通过纯文本对话生成模型和基于检索的图像分享方法获得了显著的改进。

# <font face="Times New Roman">2 相关工作</font>

## <font face="Times New Roman">2.1 文本对话回复生成</font>

&emsp;文本开放领域对话的 end-to-end 回复生成受到神经 sequence-to-sequence 在机器翻译上的成功启发。在这个基础结构之上，vanilla encoder-decoder 方法被广泛研究，以应对开放域对话系统中的关键挑战，包括改善回复的多样性、建模会话上下文、控制回复的属性、对某些特定人物角色的偏置、将额外知识纳入生成器、构建通用的预训练 agent。不同于以往对开放域对话回复生成的研究，本文的工作主要是对多模态回复生成的研究。

## <font face="Times New Roman">2.2 Text-to-Image 生成</font>

在 text-to-image 生成的研究中，许多工作都得到了广泛的研究。

- [Generating Images from Captions with Attention](https://arxiv.org/abs/1511.02793) 展示了 Draw 生成模型。

- [DRAW: A Recurrent Neural Network For Image Generation](https://proceedings.mlr.press/v37/gregor15.html) 可以从自然语言描述生成图像。

- [Generative Adversarial Text to Image Synthesis](https://proceedings.mlr.press/v48/reed16.html) 提出了生成对抗网络来提高图像的保真度。然后一些改进方法继续优化生成架构：
	- 堆叠生成器：[StackGAN: Text to Photo-Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://ieeexplore.ieee.org/document/8237891)
	- 注意力网络：[AttnGAN: Fine-Grained Text to Image Generation With Attentional Generative Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.html)
	- 额外知识：[Object-Driven Text-To-Image Synthesis via Adversarial Training](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Object-Driven_Text-To-Image_Synthesis_via_Adversarial_Training_CVPR_2019_paper.html)
- [Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space](https://ieeexplore.ieee.org/document/8099857) 提供了相关激活最大化方法的统一概率解释，以产生更高分辨率的高质量图像。
- 使用具有大范围的掩码比率的均匀 mask，并将合适的预训练数据集与合适的物体对齐。
- [X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers](https://aclanthology.org/2020.emnlp-main.707/) 和 [CogView: Mastering Text-to-Image Generation via Transformers](https://arxiv.org/abs/2105.13290) 采用基于 Transformer 的方法，将文本和图像 token 自回归地建模为单个数据流。

对于这种多模态回复生成场景，作者使用文本形式的图像描述来连接文本对话生成和 text-to-image 生成模型，其中图像描述是前者的输出，是后者在低资源环境下的输入。

# <font face="Times New Roman">3 Problem Formailzation</font>

&emsp;&emsp;假设有数据集 $\mathcal D_S=\{(U_i,R_i)\}_{i=1}^n$，其中 $\forall i \in \{1,\ldots, n\}$，$U_i=\{u_{i,1}, \ldots, u_{i,n_i}\}$ 是对话上下文， $u_{i,j}$ 是第 $j$ 个 utterance，$R_i$ 是关于 $U_i$ 的回复。$u_{i,j}$ 和 $R_i$ 可以包含两种模态：文本 和 图像。目标是使用 $\mathcal D_S$ 来学习一个**生成**模型 $P(R\mid U;\theta)$，其中 $\theta$ 为模型参数。因此，给定一个新的对话上下文 $U$，可以通过 $P(R\mid U;\theta)$ 生成一个多模态回复 $R$。

# <font face="Times New Roman">4 Approach</font>

&emsp;&emsp;本章节首先阐述了用于多模态对话的统一的 tokenization 方法。 然后介绍了低资源场景下多模态对话回复生成模型 (**Divter**) 中的两个重要组成部分：(1) 文本对话回复生成器； (2) text-to-image 转换器。 *Figure 2* 展示了 **Divter** 的整体框架。 

## <font face="Times New Roman">4.1 多模态 Tokenization</font>

为了学习一个多模态生成模型，我们首先要对文本和图像的统一表示进行建模。 受DALLE 和 VQGAN 的成功启发，为了利用高度表达的 Transformer 结构来进行 text-to-image 的生成，我们需要以序列的形式表达图像，类似于我们通常对纯文本 tokenization 所做的事情。 

- DALLE：
论文：[Zero-Shot Text-to-Image Generation](https://proceedings.mlr.press/v139/ramesh21a.html)
- VQGAN：
论文：[Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
讲解：[详解VQGAN（一）| 结合离散化编码与Transformer的百万像素图像生成](https://zhuanlan.zhihu.com/p/515214329)
有关后文中 codebook 的概念。

### <font face="Times New Roman">4.1.1 文本 Tokenization</font>

&emsp;&emsp;文本的 tokenization 已经得到了很好的研究，例如 BPE。本工作使用 50257 BPE-endoded tokens 和 Transformer 的分布式 embedding，对对话中的文本进行建模。 

### <font face="Times New Roman">4.1.2 图像 Tokenization</font>

&emsp;&emsp;图像的 tokenization 是一个离散 Auto-Encoder (VQGAN, [https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)) $\mathcal V$，如图2所示。 $\mathcal V$ 利用编码器 ${\mathcal V}_E$ 将形状为 $H\times W \times 3$ 的每个图像 $r^v$ 压缩成形状为 $h \times w \times d_z$ 的 $\hat z$，然后在 element-wise 量化 **q(·)** 的作用下，将每个维数为 $d_z$ 的向量在学习的离散 codebook $\mathcal Z=\{z_k\}_{k=1}^K \in {\mathbb R}^{d_z}$ 中量化到其最接近 embedding $z_k$。
$$
z_{\bf q}={\bf q}(\hat z):= \left( \underset{z_k \in \mathcal Z}{\rm arg\ min}\lVert \hat z_{ij}-z_k \rVert \right) \in \mathbb R^{h \times w \times d_z}
$$
&emsp;&emsp;因此 $r^v$ 可以用 codebook $z_{\bf q}\in \mathbb R^{h\times w\times d_z}$ 的空间集合来表示。解码器 ${\mathcal V}_D$ 将 $z_{\bf q}$ 映射回图像 $\hat{r^v}$ 用来重建输入。在本工作中，$H=W=256$，$h=w=16$，$K=16384$，$d_z=256$。$\mathcal V$ 和 $\mathcal Z$ 的实现细节可以在 [Zero-Shot Text-to-Image Generation](https://proceedings.mlr.press/v139/ramesh21a.html) 中找到。

## <font face="Times New Roman">4.2 低资源学习模型</font>

&emsp;&emsp;用单一的 sequense-to-sequence 模型学习一个有效的多模态生成模型往往需要大量的训练样例。然而，由于社交媒体上的隐私原先和昂贵的人工费，导致只有很少的多模态对话可用。

<center><img src="https://img-blog.csdnimg.cn/0f167bde74734a81958d9f33eb55fe2b.png" width="20%"></center>
<I>Figure 3：</I>提出的方法的抽象逻辑。实线表示存在大规模训练集对生成模型进行预训练，虚线表示只有很少的训练样例可用，“×” 表示生成质量差。其中 $U$ 为对话文本， $r^e$ 为文本回复，$c$ 为图片描述，$r^v$ 为图片回复，${\mathcal D}_C,{\mathcal D}_P,\widetilde{\mathcal D}_S$ 均为数据集，下文有详细说明。

&emsp;&emsp;另一方面，如 *Figure 3* 所示，存在大量的开放的纯文本对话和大量的 <image description, image> pairs 的数据集：
- 纯文本对话：例如，[Reddit comments](https://files.pushshift.io/reddit/)，表示为 ${\mathcal D}_C=\{(U_i,r_i^e)\}_{i=1}^N$，其中 $(U_i,r_i^e)$ 是一个 <text dialogue context, text response> pair)
- <image description, image> pairs：例如，[YFCC100M](https://dl.acm.org/doi/10.1145/2812802)，表示为 ${\mathcal D}_P=\{(c_j,r_j^v)\}_{j=1}^M$，其中 $(c_j,r_j^v)$ 是一个 <textual image-description, image> pair。

基于以上事实和在 MDRG任务中的低资源挑战，作者将生成式 text-to-image 转换引入到纯文本开放域对话回复生成中。更具体地说：
- (i) 如果多模态对话上下文包含一个图像，我们就用它的描述来代替原本的图像，形成一个纯文本的语境，并将这个上下文作为纯文本对话生成模型 $\mathcal G$ 的输入，其中 $\mathcal G$ 使用 $\mathcal D_C$ 预训练。
- (ii) 如果需要生成图像作为回复的一部分，我们可以先用 $\mathcal G$ 来生成文本形式的描述，然后用 text-to-image 转换模块 $\mathcal F$ (用 $\mathcal D_P$ 预训练) 将描述转换成同义图像。为了桥接 $\mathcal G$ 和 $\mathcal F$，我们进一步将 $\mathcal D_S$ 的形式扩展为一个新的 $\widetilde{\mathcal D}_S$，其中每个图像 $r^v$ 与其文本描述 $c$ 配对。

(i) 和 (ii) 都可以独立学习，这成为用大的 $\mathcal D_C$ 和 $\mathcal D_P$ 辅助小的 $\widetilde{\mathcal D}_S$ 的关键。

&emsp;&emsp;通过这种方法，当前的目标是学习一个具有 $\mathcal D=\{\widetilde{\mathcal D}_S, \mathcal D_C, \mathcal D_P\}$ 的生成模型 $P(R\mid U;\theta)$。利用预训练好的 $\mathcal G$ 和 $\mathcal F$，最终使用 $\widetilde{\mathcal D}_S$ 对  $\mathcal G$ 和 $\mathcal F$ 联合地 finetune，以获得生成多模态回复的能力。

&emsp;&emsp;*Figure 2* 阐述了 Divter 的结构。该模型又两个部分组成：一个文本对话回复生成器 $\mathcal G$ 和 一个 text-to-image 转换器 $\mathcal F$。

### <font face="Times New Roman">4.2.1 文本对话回复生成器 (Textual Dialogue Response Generator)</font>

&emsp;&emsp;文本对话回复生成器 $\mathcal G$ 是一个基于 Transformer 的sequence-to-sequence 模型，它由一个 24 层的 Transformer (其中隐藏大小为 1024 且有 16 个头) 组成。具体地说，给定一个来源于 $\widetilde D_S$ 的文本对话上下文 $U=\{u_1,\ldots,u_l\}$，目标是一个文本 $\widetilde R=\{w_1,\cdots,{\rm [SEP],[DST],\cdots,[SEP],\cdots},w_T\}$，其中 $w_t$ 是第 $t$ 个单词，$\rm [DST]$ token 表示接下来的子序列是一个文本形式的图像描述 $c$。生成器的 loss 被定义为：
$$
\mathcal{L_G}=\mathbb E_{(U,\widetilde R)\sim\widetilde D_S} [-\log p(\widetilde R)]
$$

$$
p(\widetilde R)=\prod_t p(w_t|U,w_{1:t-1})
$$

**推理：** 给定一个新的文本对话上下文 $U$，当一个生成的图像描述 $c$ 出现时，它将被送入后面的 text-to-image 转换器，然后构造其同义图像的 codebook embeddings。

### <font face="Times New Roman">4.2.2 Text-to-Image 转换器 (Text-to-Image Translator)</font>

&emsp;&emsp;text-to-image 转换器 $\mathcal F$ 也是一个基于 Transformer 的 sequence-to-sequence 生成模型，它由一个 24 层的 Transformer (其中隐藏大小为 1024 且有 16 个注意力头) 组成。给定一个图像 $r^v\in \mathbb R^{H\times W\times 3}$ 及其来源于 $\widetilde D_S$ 的文本描述 $c=\{w_1,\cdots,w_T\}$，在 $\mathcal V_E$ 和 $\mathcal Z$ 可用的情况下，我们可以用编码的 codebook 索引来表示 $r^v$。更准确地说，图像 $r^v$ 的量化编码由 $z_{\bf q}={\bf q}(\mathcal V_E(r^v)) \in \mathbb R^{h\times w\times d_z}$ 给出，并且可以转换为 codebook $\mathcal Z$ 中的索引序列 $s\in \{0,\cdots,|\mathcal Z|-1\}^{h\times w}$，该序列通过 codebook $\mathcal Z$ 中的索引替换每一个 code 而获得。
$$
s_{i,j}=k\quad {\rm such\ that}\quad (z_{\bf q})_{i,j}=z_k
$$
然后将 tokenize 后的 $c$ 和 $s$ concat 到一个单个的 token 流中：
$$
x=\{w_1,\cdots,w_T,{\rm [SEP]},s_1,\cdots,s_{h\times w}\}
$$
训练一个自回归 Transformer 来模拟文本和图像 token 的联合分布，生成器的 loss 被定义为：
$$
\mathcal{L_F}=\mathbb E_{(c,r^v)\sim\widetilde D_S} [-\log p(x)]
$$

$$
p(x)=\prod_t p(w_t|w_{1:t-1})\prod_i p(s_i|c,s_{1:i-1})
$$

**推理：**给定一个描述 $c$，利用 text-to-image 转换器生成其同义图像的表征 $\hat z=\mathcal F(c)\in \mathbb R^{h\times w\times d_z}$。

### <font face="Times New Roman">4.2.3 学习细节</font>

&emsp;&emsp;定义 $\{\theta_g,\theta_{\pi},\theta_{\phi}\}$ 分别为 文本对话回复生成器 $\mathcal G$，图像 tokenizer $\mathcal V$ 和 text-to-image 转换器 $\mathcal F$ 的参数。在预训练阶段，使用文本对话 $\mathcal D_C$ 来评估 $\theta_g$，用 ImageNet 来评估 $\theta_{\pi}$，用 <image description, image> pairs $\mathcal D_P$ 来评估 $\theta_{\phi}$。然后拟合 $\theta_{\pi}$，并用 $\widetilde D_S$ 将 $\theta_g$ 和 $\theta_{\phi}$ 联合 finetune，因此最终目标是最小化整体 loss：
$$
\mathcal L=\mathcal L_{\mathcal G}+\lambda \mathcal L_{\mathcal F}
\label{eq:8}
$$
其中 $\lambda$ 是一个超参数。

**讨论：**在这项工作中，我们主要集中在整合文本和图像回复生成，但我们提出的方法实际上提供了一个低资源 MDRG 的通用解决方案，其中目标模态可以是 gif、视频或语音等。要做到这一点，我们只需要修改 text-to-image 转换器，使其与特定的模态类型兼容，然后预先训练一个新的 text-to-\<target modality> 转换器。 

# <font face="Times New Roman">5 实验</font>

## <font face="Times New Roman">5.1 数据集</font>

在 PhotoChat 数据集上进行了广泛的实验来评价 Divter 的性能，这是一个由 10917 个图像和 12286 个对话组成的多模态会话数据集，每段对话都与会话过程中分享的用户图像配对，每幅图像都与其文本描述配对。数据集已被拆分为 10286 个训练集、1000 个验证集和 1000 个测试集样例。更多细节参考附录A.1。

<center><img src="https://img-blog.csdnimg.cn/c6e29e7a0e3648cdb168637436a777e1.png" width="50%"></center>

## <font face="Times New Roman">5.2 评价指标</font>

使用自动评价和人工评价进行评估。
- 对于自动评价，主要关注四个方面：
  1. 图像意图预测，该任务的目标是预测在给定的背景下是否应该在下一轮生成一幅图像；
  2. 文本描述生成；
  3. 图像生成质量； 
  4. 文本回复生成。 

  对于 (1)，遵循 [PhotoChat: A Human-Human Dialogue Dataset With Photo Sharing Behavior For Joint Image-Text Modeling](https://aclanthology.org/2021.acl-long.479/) 的观点，将问题制定为二分类任务，使用 **F1** 作为评价指标；对于 (2) 和 (4)，使用 **PPL**，**BLEU**、**Rouge **和 **F1**； 对于 (3) 遵循 DALLE 的观点，使用 Frechet Inception Distance (**FID**) 和 Inception Score (**IS**)。 
  
- 对于人工评价，随机抽取 200 个对话上下文，并从 PhotoChat 中生成 Divter 和 baselines 的回复。要求 3 位人类注释者从以下 4 个方面对回复质量进行评分，评分范围为 $\{0, 1, 2\}$：

  1. **语境连贯：**文本回复是否与语境连贯；
  2. **文本流畅性：**文本回复是否自然、流畅；
  3. **图像质量：**图像回复的质量 (包括清晰度和完整性)；
  4. **图像背景一致性：**对于每一个对话，我们选择 top-8 生成/检索出的图像组，并要求注释者判断该组是否与对话背景一致。
  
  定性评估如 *Figure 5* 所示。本文展示了 3 个注释者的平均分数，分数越高越好。

<center><img src="https://img-blog.csdnimg.cn/bd394c9263d2409baa15bf6e524cf6ec.png" width="70%"><img src="https://img-blog.csdnimg.cn/d1e7f1226fc04b959c72ec476378c289.png" width="30%"></center><center><I>Figure 5</I>： Divter 生成的图像和 SCAN 检索出的图像示例。对话背景见附录 A.2。</center>


&emsp;&emsp;作者还将纯文本 Divter 和多模态 Divter 分别与 DialoGPT 进行比较。纯文本 Divter 意味着我们在解码阶段屏蔽词汇表中的 $\rm[DST]$ token，使回复中只包含文本。我们还随机抽取了 200 个对话。对于每一个注释者，来自不同模型的两个回复被提出，这两个回复被随机打乱以隐藏他们的来源。然后，注释者判断哪种回复更能有效地提高对话的体验和吸引力。注释者之间的一致性是通过 Fleiss’s Kappa 来衡量的。

## <font face="Times New Roman">5.3 实现细节</font>

&emsp;&emsp;对于文本对话回复生成器 $\mathcal G$，使用 DialoGPT 作为预训练模型初始化，在 2005 年到 2017 年的 Reddit 评论链中提取的 147M 对话式交流中训练。在 fine-tune 阶段，用 $\rm[SEP]$  concat 上下文的单个序列，采用 Adam 优化器且初始学习率为 1e-5，batch size 设为 256。使用 beam search (size=5) 来解码文本序列。

&emsp;&emsp;对于图像 tokenizer $\mathcal V$，直接使用 VQGAN。

&emsp;&emsp;对于 text-to-image 转换器 $\mathcal F$，作者随机选取了 5M 个来源于 ImageNet 中的 <categorical image description, image> pairs 和来源于 YFCC100M 中的 <image description, image> pairs 作为训练数据。将最大图像描述长度设置为 32，batch size 设为 256，预训练 $\mathcal F$ 3.5million 个 step。在 fine-tune 阶段，训练 PhotoChat 50000 个 step。 在推理阶段，使用 CLIP 对生成的 256 个样本进行重新排序。

&emsp;&emsp;在联合学习中，先训练 $\mathcal F$ 48000 个 step，然后联合训练 $\mathcal G$ 和 $\mathcal F$ 2000 个 step。公式 $\eqref{eq:8}$ 中的 $\lambda$ 为 0.2。验证时 early stopping 是一种规范的策略，所有的超参数都是通过 grid search 来确定。更多细节参考附件 A.3。

<hr>

<font face="Times New Roman">**A.3 More Implementation Details**</font>

&emsp;&emsp;CLIP 模型根据图像与描述的匹配程度给出评分，并利用 CLIP 对生成的256个样本进行重新排序，选择最佳的图像作为最终的回复。为了获得高质量的训练集，丢弃了描述中以 “The photo has your * #” 为前缀的样例，其中 “*” 包括 “mom”，“dad”，“daughter”，“sister”，“uncle” 等，“#” 是一个人的名字。 为了从ImageNet 中构建 text-to-image 转换器 $\mathcal F$ 的训练集，我们结合文本 “Objects in the photo:” 和每个图像的文本分类名称来构建 <categorical image description, image> pair。为了训练 baseline S2S-TF 模型，我们还使用图像 tokenizer $\mathcal V$ 对每个图像进行 tokenize，并将图像 tokens 与文本 tokens 结合形成单一流作为生成的来源或目标。 

<hr>

实现代码：

- 图像 Auto-Encoder：[https://github.com/CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
- 文本对话回复生成器：[https://github.com/microsoft/DialoGPT](https://github.com/microsoft/DialoGPT)
- Text-to-Image 转换器：[https://github.com/lucidrains/DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch)

## <font face="Times New Roman">5.4 Baselines</font>

&emsp;&emsp;选择两个预训练模型 **BERT-base** 和 **T5-3B** 作为 baseline 来衡量 5.2节 中的 “图像意图预测” 任务。它们将文本对话上下文作为输入，预测 “一张图像是否会在下一轮被分享” (在给定的背景下是否应该在下一轮生成一幅图像)。 

&emsp;&emsp;**SCAN**：该模型获取图像区域和文本 tokens 之间的相互作用来推断 image-text 的相似性，SCAN 在 PhotoChat 上实现了 “图像检索” 任务的 SOTA。

&emsp;&emsp;**S2S-TF** 是一个具有 24 层 Transformer 的单一的 sequence-to-sequence 模型，本文只使用 PhotoChat 来训练这个多模态生成模型。 

## <font face="Times New Roman">5.5 评价结果</font>

<center><img src="https://img-blog.csdnimg.cn/e035758b3d3c4987ac0a3e20f44c6ab3.png"></center>

*Table 1*：测试集上 Divter 和 baseline 的自动评价结果。($w/o$ joint learning) 表示分别 fine-tune $\mathcal G$ 和 $\mathcal F$，而不是使用公式 $\eqref{eq:8}$。

&emsp;&emsp;如 *Table 1* 所示， Divter 不仅获得了与基于检索的图像回复意图预测的 SOTA 模型相当的性能，而且在所有生成部分的评价指标中都获得了优越的性能。这标志着 Divter 能够准确地判断在给定对话上下文时生成图像回复的时机，并生成语上下文连贯的文本回复，也能够生成高质量的图像回复。Divter 与未进行预训练的 baseline (如 S2S-TF，Divter 的变体) 之间显著的性能差距表明了作者提出的学习策略的优越性。

<center><img src="https://img-blog.csdnimg.cn/d33ca61982a94013844489e59b55425d.png" width="60%"></center>

*Table 2* 展示了人工评价的结果，Divter 在大多数方面都由于 baseline。

<center><img src="https://img-blog.csdnimg.cn/52405db1124e44578e03d9cc37ec64b3.png" width="65%"></center>

*Table 3* 的对比结果表明：
1. Divter 在纯文本回复生成方面可以达到与 DialoGPT 相当的性能；
2. 与纯文本对话模型 DialoGPT 相比， Divter 生成的多模态回复在对话体验和吸引力上有了显著的提高。

## <font face="Times New Roman">5.6 消融实验</font>

<center><img src="https://img-blog.csdnimg.cn/e035758b3d3c4987ac0a3e20f44c6ab3.png"></center>

*Table 1*：测试集上 Divter 和 baseline 的自动评价结果。($w/o$ joint learning) 表示分别 fine-tune $\mathcal G$ 和 $\mathcal F$，而不是使用公式 $\eqref{eq:8}$。

&emsp;&emsp;如 *Table 1* 所示，所有变体都在大多数评价指标中的性能更差。

<center><img src="https://img-blog.csdnimg.cn/1b66e6f96c674f68969ba803cbd04ca3.png" width="50%"></center>

*Figure 4*：在 PhotoChat 测试集中，输入相同的上下文对图像生成的各种变体进行定性评估。第1列：Divter。 第2列：Divter $w/o\ \mathcal G$ pre-train。 第3列：Divter $w/o\ \mathcal F$ pre-train。 

&emsp;&emsp;为了更直观的比较，定性评估结果也如图4所示。 特别是，消融研究的定量和定性结果都验证了：

1. 预训练对于低资源多模态对话回复生成至关重要，因为当训练数据较小时，从预训练中删除任何分量都会导致性能下降；
2. 在对图像生成性能的影响方面 $\mathcal F>\mathcal G$，在对文本生成性能方面 $\mathcal G>\mathcal F$ (由 *Table 1* 中倒数 3、4 行可得)；
3. 联合学习对于 Divter 也有贡献，表明利用文本上下文和视觉图像的集成学习比任何单一的学习都要好。

## <font face="Times New Roman">5.7 案例分析</font>

<center><img src="https://img-blog.csdnimg.cn/875905cf366b48d7958c486746c1962f.png"></center>

*Tabel4*：PhotoChat 测试集样例。在每个例子中，前缀为 “A” 或 “B” 的是给定的上下文，<font color="blue">蓝色</font>文本是 Divter 生成的文本描述，左边的图像和<font color="red">红色</font>的回复是由 Divter 生成，右边的图像是 ground-truth 图像。

&emsp;&emsp;为了进一步研究 Divter 生成的多模态回复的质量，在 *Tabel4* 中展示了 PhotoChat测试集上的两个例子。第一个给定的上下文是关于 “ice-cream” 的，第二个是关于 “honey bee” 的。Divter 不仅可以生成与背景一致的逼真的高分辨率图像，而且可以生成基于该图像的信息丰富的文本回复。另外，生成的高质量图像与真实世界的 ground truths 相媲美，证明了 Divter 的实用性。 

## <font face="Times New Roman">5.8 讨论</font>

<center><img src="https://img-blog.csdnimg.cn/bd394c9263d2409baa15bf6e524cf6ec.png" width="70%"><img src="https://img-blog.csdnimg.cn/d1e7f1226fc04b959c72ec476378c289.png" width="30%"></center><center><I>Figure 5</I>： Divter 生成的图像和 SCAN 检索出的图像示例。对话背景见附录 A.2。</center>

**优于基于检索的方法**

&emsp;&emsp;为了进一步研究和比较 Divter 和基于检索的方法的泛用性，作者还获取了在给定相同上下文的条件下，从 Divter 生成的 top-10 的图像和从 SCAN 模型中等效检索出来的图像。如 *Figure 5* 所示，一方面，生成的图像的多样性和丰富性是令人满意的，另一方面，这些检索的结果往往和对话背景不一致。例如：
- 在第二个例子中，对话是在讨论 “coffee”，但检索到的图像包含一些不相关的物体，如 “milk”，“cake”，“dog”，和 “snack”。
- 在第三个例子中，由于训练和检索空间中几乎没有 ”curtain“，所以所有的检索结果都是错的。

&emsp;&emsp;这表明基于检索的方法在特定领域的性能收到预先构建的会话历史存储库规模的限制，特别是在低资源的情况下。此外，本文提出的基于生成的方法展示出了更好的泛化能力，以解决低资源的挑战。

# <font face="Times New Roman">6 结论</font>

&emsp;&emsp;本文研究了低资源环境下的多模态对话回复生成问题。为了克服新任务和训练数据不足带来的挑战，提出了一种神经会话代理 Divter，它将 text-to-image 生成与纯文本对话回复生成结合起来，其中大部分参数不再依赖于训练数据，而是可以从大规模文本开放域对话和<image description, image> pairs 中估计。大量的实验表明，Divter 在自动和人工评价方面达到了 SOTA。在未来，作者将探索更高效的方法，为回复生成注入更多的模态。 
