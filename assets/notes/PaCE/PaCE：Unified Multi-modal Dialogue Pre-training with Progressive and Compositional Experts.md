[TOC]

<center><B><font size=5>PaCE: Unified Multi-modal Dialogue Pre-training with Progressive and Compositional Experts</font></B></center>

<p align="right">——ACL 2023</p>

# 前言

论文标题：[PaCE: Unified Multi-modal Dialogue Pre-training with Progressive and Compositional Experts](https://aclanthology.org/2023.acl-long.749.pdf)
论文网址：[https://aclanthology.org/2023.acl-long.749/](https://aclanthology.org/2023.acl-long.749/)
收  录  于：ACL 2023

# <font face="Times New Roman">4 Pre-training Method</font>

&emsp;&emsp;给定一个包含 $n$ 个多模态对话样本的集合 $\mathcal{D}=\{(U_i,R_i)\}_{i=1}^n$，其中 $U_i$ 和 $R_i$ 分别表示对话上下文和回复。与传统的文本对话相比，$U_i=\{u_k^m\}_{k=1}^K$ 和 $R_i=\{r_q^m\}_{q=1}^Q$ 都可以包含包括文本和视觉图像在内的各种类型的信息，其中 $K$ 和 $Q$ 是元素的数量，$m\in\{t,v\}$ 表示 $U_i$  (或 $R_i$) 的模态。 符号 $t$ 表示文本语句，$v$ 表示视觉图像。

&emsp;&emsp;<font face="Times New Roman">我们设计了一个分而治之的针对多模态对话的预训练策略。具体来说，我们将复杂的多模态对话分解为五个基本的子功能，并设计了五个相应的专家 (即，caption、context、image、grounding(语境) 和 generation)。在此基础上，我们提出了一种渐进的训练策略，通过在不同的预训练阶段控制专家的组合来演化模型。接下来，我们详细描述了输入表征学习模块 (input representation learning module)、分治式预训练策略 (divide-and-conquer pre-training strategy)、预训练目标 (pre-training objectives) 和微调过程 (fine-tuning process)。</font>

<img src="https://github.com/friedrichor/Deep-Learning-Paper-List/raw/main/photo/PaCE_figure3.png" width="80%">

## <font face="Times New Roman">4.1 Input Representation Learning</font>

&emsp;&emsp;提出的模型被设计用来处理两种模态的输入数据：视觉表征和文本表征。

<font face="Times New Roman">**Visual Representations** 对话上下文和回复可以是视觉数据或文本数据、我们使用 Vision Transformer 来学习图像的视觉表征。形式上，我们对视觉图像 $v\in \mathbb R^{H\times W\times \mathcal C}$ 进行处理，将其分为 $N=HW/P^2$ 个 patches $v^p\in \mathbb R^{N\times (P^2 \mathcal C)}$，其中 $\mathcal C$ 为通道数， $(H,W)$ 是输入图像的分辨率，$P$ 为 patch 的分辨率。这使得模型可以将图像视为一小组区域， 而不是单个大的像素数组，从而从图像中提取有意义的特征。然后将图像 patches 展平成向量，使用一个权重矩阵 $W_V\in \mathbb R^{(P^2 \mathcal C)\times E}$ 和 position embedding $W_V^{pos}\in \mathbb R^{(N+1)\times E}$ 进行线性投影，得到 patch embedding $\bar{v}\in \mathbb R^{(N+1)\times E}$，其中 $E$ 为 embedding 的维度。position embedding 用于添加关于图像中 patch 位置的附加信息。最后，我们对 patch embedding 和 position embedding 进行累加，得到视觉表征 $H_0^v$。</font> 

<font face="Times New Roman"> **Textual Representations** 通过使用一个 word embedding 矩阵 $W_T\in \mathbb R^{|O| \times E}$ 和一个 position embedding 矩阵 $W_T^{pos}\in \mathbb R^{(\mathcal L + 1) \times E}$ 将输入文本 $t\in \mathbb R^{\mathcal L\times|O|}$ embed 到稠密表征 $\bar{t}\in \mathbb R^{\mathcal{L} \times E}$，其中 $|O|$ 是单词表的大小，$\mathcal L$ 是文本的长度，$E$ 是 embedding 的维度。指的注意的是，我们通常将上下文和当前的语句连接起来形成最终的文本输入。文本表征可以表示为  $H_0^t$。</font> 

## <font face="Times New Roman">4.2 Divide-and-Conquer Pre-training Strategy</font>
&emsp;&emsp;我们以一种分治的方式设计了一种新的预训练策略。具体来说，我们首先将复杂的多模型对话分解为几个子问题，这样学习起来更容易。然后结合子问题的解决方案，给出不同下游多模态对话任务的解决方案。

<font face="Times New Roman">**Multi-expert Architecture** PaCE 采用了标准的 Transformer 的扩展，它学习多个语义专家，而不是像原始 Transformer 那样学习单个前馈网络 (FFN)。具体来说，专家通过多头注意力机制 (MSA) 共享文本和视觉模态的信息，而每个专家 $\rm FFN^{expert}$ 都有自己独特的参数来学习不同的语义表征。在形式上，每个分块的专家交换得到的唯一信息可以表示为：</font> 
$$
H_l'={\rm{MSA}}({\rm{LN}}(H_{l-1})) + H_{l-1}\\
H_l^{{\rm expert}_k}={{\rm FFN}^{{\rm expert}_k}}({\rm LM}(H_l'))+H_l'
$$
<font face="Times New Roman">其中 $H_{l-1}$ ($l\in [1,L]$) 表示 $l-1$ 层的输出表征，$L$ 是 Transformer 块的数量。$H_l^{{\rm expert}_k}$ 是第 $k$ 个专家的表征。输入表征可以形式化为 $H_0=[H_0^v,H_0^t]$。在这里，MSA 和 LN 分别是标准的多头自注意力和层归一化 (layer normalization)。</font> 

<font face="Times New Roman">**Modality and Capability Experts** 如图 3 所示，我们将复杂的多模态对话任务分解成五个简单的子问题，包括 CAPTION 建模、CONTEXT 建模、IMAGE 建模、GROUNDING 和 GENERATION。我们设计了一个语义专家来解决每个子问题。这五个专家可以分为两类：模态专家 (CAPTION 和 IMAGE 专家) 和能力专家 (GROUNDING、CONTEXT 建模、GENERATION 专家)。最终，我们以一种分级的方式激活模态专家和能力专家，底下的 $(L-F)$ 层只激活模态专家，上面的 $F$ 层激活能力专家，其中 $F$ 是预先定义的超参数。</font> 

<font face="Times New Roman">**Experts Combination for Different Tasks** 我们提出了一种递进的级联预训练策略，通过自适应地组合子问题的解决方案来解决不同的多模态对话任务。我们将在 4.3 节介绍递进级联预训练的细节。</font> 

## <font face="Times New Roman">4.3 Pre-training Objectives</font>

我们的递进级联训练前过程包括三个阶段，每个阶段都有一个量身定制的预训练目标。

<font face="Times New Roman">**Stage I: Image-Text Matching** 在阶段 I，与 ViLT 类似，我们使用非对话多模态数据 $\mathcal D_n$ 来学习基本的多模态对齐，这一阶段只涉及三个专家，包括 CAPTION 专家、IMAGE 专家和 GROUNDING 专家。如 Figure 3(a) 所示，遵循在 word 和 patch embeddings，文本和图像被专门的 CAPTION 和 IMAGE 专家分别处理成文本和图像表征。然后，这些表征被融合并输入到 GROUNDING 专家，生成文本和图像的统一表征。然后，我们使用专家输出的 [CLS] token 的表征作为二分类网络的输入，以预测当前文本和图像之间的对齐。image-text matching 的损失函数定义为：</font> 
$$
{\mathcal L}_{\rm itm}={\mathbb E}_{(V,T)\sim D_n}{\rm CE}({\boldsymbol y}_{\rm itm},{\boldsymbol p}_{\rm itm}(V,T))
$$
<font face="Times New Roman">除了 ${\mathcal L}_{\rm itm}$ 之外，我们还在这个阶段使用 MLM 损失 ${\mathcal L}_{\rm mlm}$ 来理解独特的文本形式。具体来说，按照 BERT 的方法，我们在文本序列中随机选择 tokens，并将其替换为 [MASK] token。模型被训练来预测这些被 mask 掉的 tokens，使用剩余的未 mask 的 tokens 的上下文和视觉线索。我们采用 15% 的 mask 概率。然后将被 mask 掉的tokens 的最终输出向量输入到整个文本词汇表 (vocabulary) 的分类器中，训练损失为交叉熵损失。</font> 
$$
{\mathcal L}_{\rm mlm}={\mathbb E}_{(V,\hat T)\sim \{D_n\cup D_d\}}{\rm CE}({\boldsymbol y}_{\rm mask},{\boldsymbol p}_{\rm mask}(V,\hat T))
$$
<font face="Times New Roman">其中 $\hat T$ 是一个被 mask 的文本，$V$ 是一个原始图像，${\boldsymbol p}_{\rm mask}(V,\hat T)$ 表示模型对被 mask 的 token $\hat T$ 的预测概率。$D_n$ 和 $D_d$ 分别表示多模态非对话数据和对话数据。</font> 

<font face="Times New Roman">Stage I 的联合损失可以形式化为：</font> 
$$
{\mathcal L}_{\rm stage}^{\rm I}={\mathcal L}_{\rm itm}+{\mathcal L}_{\rm mlm}
$$
<font face="Times New Roman">**Stage II: Image-Context Matching** 在阶段 II，我们使用多模态对话数据 $D_d$ 对 PaCE 进行预训练，旨在为多模态对话任务建模对话上下文。在这一阶段，除了第一阶段的三个专家外，还启用了 CAPTION 专家。具体来说，在第二阶段，对话上下文 $C$ 输入给 CONTEXT 专家，图像 $V$ 输入给 IMAGE 专家，其对应的图像描述 $T$ 输入给 CAPTION 专家。image-context matching 的损失函数被定义为：</font> 
$$
{\mathcal L}_{\rm icm}={\mathbb E}_{(V,T,C)\sim D_d}{\rm CE}({\boldsymbol y}_{\rm icm},{\boldsymbol p}_{\rm icm}(V,T,C))
$$
<font face="Times New Roman">另外，我们使用第一阶段学习到的 CAPTION 专家作为 teacher 来促进 CONTEXT 专家的学习。</font> 
$$
{\mathcal L}_{\rm tca}=\lVert H_{L-F}^t-H_{L-F}^c\rVert_2^2
$$
其中 $H_{L-F}^t$ 和 $H_{L-F}^c$ 分别是 CAPTION 专家和 CONTEXT 专家第$\{L-F\}$ 层的输出。

<font face="Times New Roman">此外，我们还在 stage II 采用 stage I 定义过的 MLM 损失，stage II 的联合损失 ${\mathcal L}_{\rm stage}^{\rm II}$ 可以形式化为：</font> 
$$
{\mathcal L}_{\rm stage}^{\rm II}={\mathcal L}_{\rm icm}+{\mathcal L}_{\rm tca}+{\mathcal L}_{\rm mlm}
$$
<font face="Times New Roman">**Stage III: Generation Modeling** 第三阶段的目标是使模型能够生成回复。GENERATION 专家被激活，输入由 COTEXT 专家和 IMAGE 专家组成。第三阶段的损失函数定义如下：</font> 
$$
{\mathcal L}_{\rm stage}^{\rm III}=-\sum_{n=1}^N {\rm log}\ {\boldsymbol p}_{\rm rgm}(C_n\mid V,C_{<n})
$$
<font face="Times New Roman">在这里，我们通过自回归建模生成能力，即使用过去的对话历史 $C_{<n}$ 和相关的图像 $V$ 来预测对话当前的轮次(turn) $C_n$。</font> 

## <font face="Times New Roman">4.4 Fine-Tuning on Downstream Tasks</font>

<font face="Times New Roman">一旦 PaCE 的预训练完成，我们就会对特定的下游任务进行微调。由于我们的分治预训练方法，我们可以灵活地选择不同能力的专家来解决特定的下游任务，包括意图预测，对话检索和对话状态跟踪，我们激活 CONTEXT 专家、IMAGE 专家和 GROUNDING 专家。对于生成任务，如对话状态跟踪和回复生成，我们激活 CONTEXT 专家、IMAGE 专家和 GENERATION 专家。</font> 