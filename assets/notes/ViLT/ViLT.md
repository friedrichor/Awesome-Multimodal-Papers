[TOC]

# 0 前言
ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision
论文网址：[https://arxiv.org/abs/2102.03334](https://arxiv.org/abs/2102.03334)
源码网址：[https://github.com/dandelin/vilt](https://github.com/dandelin/vilt)

# 1 摘要
&emsp;&emsp;视觉和语言预训练（Vision-and-Language Pre-training，VLP）提高了各种视觉和语言联合下游任务的性能。目前的VLP方法在很大程度上依赖于图像特征提取过程（以往大家发现视觉的网络越好，最终的效果也越好），其中大部分涉及区域超视距（如目标检测，Figure 1中的Region Feature）和卷积结构（如ResNet，Figure 1中的Grid Feature）。但我们发现它在以下两个方面存在问题：
（1）效率/速度。提取图像特征需要比多模态融合花的时间还要多。按理来说，如果只是将多个单模态做的很好，并不能保证最终多模态的效果依旧很好，而多模态的效果大部分要取决于融合时的效果。
（2）表达能力。如果只用一个预训练好的模型去提取特征，那么这个多模态模型的效果是很受限的。比如用一个预训练好的目标检测器，目前目标检测数据集的类别并不大，检测的类别很有限，并不能涵盖所有范围，所以模型如果不是端到端，仅仅是使用预训练模型去提取特征，那么大概率将不是最优解。

&emsp;&emsp;在本文中，作者提出了一个最小的VLP模型--Vision-and-Language Transformer（ViLT），在这个意义上，视觉输入的处理被大幅简化为与我们处理文本输入一样的无卷积方式。作者表明，ViLT比以前的VLP模型快几十倍，但其下游任务的性能具有竞争力或更好。

<center><img src="https://img-blog.csdnimg.cn/fd4dc88931bc46578b01994a145e4b87.png" width="60%">


图 *Figure1*：传统VLP架构与ViLT的比较。

在图像方面，这里分了三类 ：
1. Region Feature：如ViLBERT、UNITER等等，这些是要提取区域特征的。给定一个图像，然后通过一个卷积神经网络的Backbone（如 ResNet50，ResNet101 等），再通过Region Operations（如 RoI，NMS 等，下侧图中深紫色~810ms下方的文字中有列举）获取到一些区域性的特征。这就相当于做了个目标检测方面的任务，得到的是一些bounding box，都是离散形式的，也就可以想象成一堆序列。但这样的缺点很明显，如图，使用目标检测方法的，整个模型的运行时间是900ms，但其中视觉这部分就占了885ms（75+810），文本仅仅占15ms。所以说为了使模型达到很好的效果，模型在视觉这方面浪费了太多资源。
2. Grid Feature：如Pixel-BERT，它是用了一个在 ImageNet 上预训练好的 ResNet，然后把 ResNet 得到的特征图当做离散的序列传给Transformer去学（就和 ViT hybrid 一样），就只有 CNN Backbone，而没有后面的目标检测相关的 RoI 和 NMS（Non-Maximum Suppression，非极大值抑制）等等。这个方法虽然运行时间上大大缩短，但性能下降的过多，因此并不是很好。
3. Patch Projection：也就是ViLT的方法，其实和 Vision Transformer（简称ViT）的预处理一样，通过一个Linear Embedding层实现，将 patch 变成 token。ViLT在视觉方面的运行时间仅仅需要0.4ms，相比传统模型的运行时间大大减少，且模型效果并不会下降很多。

在文本方面，这些模型都是基本一样的，通过一个Embedding矩阵，变成一个个的word token。
得到了视觉的序列和文本的序列后输入到Modality Interaction（基本都是Transformer）进行模态之间的融合。

&emsp;&emsp;虽然这里提出了ViLT的运行时间相比传统模型的运行时间大大缩短，但ViLT的 训练时间 并不短，甚至比以往的很多方法还要久。而且ViLT的效果也不强于以往 Region Feature 方法。但ViLT最主要的成就就是它的运行时间特别短。

<hr>

# 2 Introduction
&emsp;&emsp;在先前的研究中，基本都是依靠 图像文本对儿 来进行预训练，它们的目标函数也基本上是 image text matching（图像-文本的匹配的loss）和 masked language modeling（NLP中如BERT等使用的掩码学习）【尽管某些工作采用了其他目标函数，但这两个目标函数在所有的VLP模型都用到了】，然后在下游任务中进行微调，其中往往都是涉及两个模态。
&emsp;&emsp;如果要做VLP，文本毋庸置疑要用Transformer，那么图像的像素就需要变成某一种形式，如一种离散的但带有很高语义的特征表示形式，这样图像才能和 language tokens 匹配起来，然后再一起传给 Transformer。Vision Transformer中是将图像分成 16 ×16 patch，然后将这些 patch 传给 Transformer 去学。在 Vision Transformer 提出之前，VLP 大部分还是依赖于一个目标检测器，目前 VLP 模型基本都使用一个预训练好的目标检测器，是在 Visual Genome 数据集（有1600个物体类别和400个属性类别）上预训练的。

为什么用目标检测器？
1. 上面提到了VLP想要的是 **离散的且语义性强的特征表示形式**，而目标检测正好是一个离散化的国策很，返回的是 bounding box，是检测到的物体，bounding box有明确的语义信息，且是离散的，直接用RoI提取特征就好了。这里面目标检测的 region，其实就可以想象成文本那里句子中的某一个单词。
2. 跟当时的 VLP 下游任务有关，当时主要是 VQA（Visual Question Answering）、Image Captioning、Image Retrieval 等等（这些任务的简介可以参考[VL (Vision and Language) 任务简介及数据集](https://blog.csdn.net/Friedrichor/article/details/127126679)），这些任务往往都跟物体有非常直接的联系，有非常强的对物体的依赖性。

&emsp;&emsp;但使用目标检测器来提取图像特征实在是太浪费资源，于是也开始尝试把视觉这里的计算量降下来。其中一个尝试就是Pixel-BERT，它是用了一个在 ImageNet 上预训练好的 ResNet，然后把 ResNet 得到的特征图当做离散的序列传给Transformer去学（就和 ViT hybrid 一样），这样计算量就只有 CNN Backbone了，而没有后面的目标检测相关的 RoI 和 NMS（Non-Maximum Suppression，非极大值抑制）等等，运行时间快了不少。

&emsp;&emsp;但作者认为还是不够，目前 VLP 的研究还是聚焦在如何通过提升 visual embedders 来提高性能。在实验中大家经常忽略提取特征的时间，在训练时，可以先把数据集的特征提取好存在硬盘上，真正训练时直接用就好了，但在实际应用时，使用的数据都是实时的新数据，都要去提取新特征，这部分是无法提前存在硬盘上的，这部分时间的花费是无法忽略的。

&emsp;&emsp;于是作者将重心放在如何设计一个更轻量更快速的提取图像特征的方法。作者参考了 ViT（[An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/abs/2010.11929)，相关讲解可参考[【论文&模型讲解】Vision Transformer](https://blog.csdn.net/Friedrichor/article/details/124194428)），将图像分成若干个patch，然后通过 linear projection 层把 patch 变成 embedding，从而替代了以往繁琐的提取图像特征的过程。

ViLT的贡献：
- ViLT是迄今为止最简单的 vision- and-language 模型，除了模态融合用的 Transformer 外就没用其它模型了（以往还需要用到 ResNet，甚至还有目标检测的网络）。ViLT带来了非常短的运行时间和参数量的减少。
- 在减少计算复杂度的同时，ViLT能够保证在不使用 Region Features 和 CNN 的情况下性能下降很少甚至不降，这是以往工作中达不到的。
- 在训练时用了更多的数据增强方式，如在文本部分将整个词 mask 掉，在图像部分使用图像增强，使得模型性能得到提高。

<hr>

# 3 背景（小综述）
## 3.1 Vision-and-Language 模型分类
作者根据两点对当前 VLP 模型做了一个分类：
1. 图像和文本的表达力度是否平衡，即两种模态所用的参数量和计算量。按理来说图像和文本的重要性应该差不多，而不应像以往大多数方法那样图像比文本重要特别多。
2. 两个模态怎么去融合

根据以上两点，作者将 VLP 模型分类四类，如图 *Figure2* 所示，其中VE，TE，MI分别表示visual embedder，textual embedder，和 modality interaction。
![在这里插入图片描述](https://img-blog.csdnimg.cn/99d4884cd445494699f182a931d04512.png)

- (a) VE > TE > MI：如 VSE（visual semantic embedding） 系列，文本和融合比较轻量，但图像比较繁琐。
- (b) VE = TE > MI：如CLIP，图像和文本的表达力度基本一样，计算量也差不多，在模态融合中比较轻量，直接将两个模态的特征做点乘。比较适合做提取特征、 Retrieval 任务，但像 VQA 等任务就不适合做了，因为CLIP只是简单获取了两个模态的特征，而 VQA 需要把两个模态的特征好好融合一下才能知道其中的对应关系。
- (c) VE > MI > TE：如 ViLBERT、UNITER，图像部分用目标检测，融合部分用 Transformer。如上述提到的，这种方法的性能确实很好，在各个下游任务中都取得不错的成果
- (d) MI > VE = TE：也就是本文的模型，ViLT，借助 ViT 的想法把图像部分也变得非常轻量。
## 3.2 模态融合的方式
主要分为两类：
- single-stream approaches：只用一个模型。那么怎么解决两个模态的输入呢？最简单的方法就是将两个输入 concat 起来，把两个序列合并成一个序列就可以输入到一个模型去了。
- dual-stream approaches：用两个模型。这两个模型先各自对各自的输入做一些处理，充分挖掘单模态中的信息，然后在后面再做一些融合。

&emsp;&emsp;作者使用的是 single-stream，也就是将两个模态的特征 concat 后传给 Transformer 学习。而 dual-stream 是两个模型，所以需要的参数量更大一些。
## 3.3 Visual Embedding 方法
&emsp;&emsp;在文本端，大多都是用的是预训练好的 BERT 里的embedder-tokenizer，所以这部分都一样，而且还很轻量。所以文本就不过多赘述，主要讲视觉的特征提取。

**Region Feature**
&emsp;&emsp;首先是通过一个 Backbone（如ResNet-101，ResNext-152）抽取一些特征，然后通过 FPN 抽取一些 RoI，再通过 NMS 将 RoI 降到一定的数量，那么这个数量其实就是序列的长度。得到的就是一堆 bounding box，然后通过 RoI head 就能得到一些一维的向量，这些就是 Region Feature 了。
&emsp;&emsp;这里虽然听起来很合理，把一些连续的图片，变成一些离散的 bounding box，每个box都有一定的特征，这就跟文本那里匹配起来了。但这整个过程都非常费资源。即是现在目标检测有一些快速、轻量的模型，但也依旧不如一个简单的 backbone 或 patch embedding 快。

**Grid Feature**
&emsp;&emsp;作者列举了几个 Grid Feature 的方法，但依旧不够轻量，且性能下降了很多。

**Patch Projection**
&emsp;&emsp;参考 ViT 的 patch projection embedding，不仅视觉部分轻量的很多，且模型性能相较于原先的 Region Feature 方法基本不变。

<hr>

# 4 ViLT（Vision-and-Language Transformer）
## 4.1 模型
![在这里插入图片描述](https://img-blog.csdnimg.cn/f2fc58f584f74e77a4597896e5df3562.png)
如图 *Figure3*：

**输入部分：**

输入分别是文本和图像：
- 文本序列通过 BERT tokenizer 得到 word embedding，假如文本序列的长度为 L，token embedding 的维度为 H，所以文本输入到 Transformer 的是 L×H 的矩阵。
<img src="https://img-blog.csdnimg.cn/5bd7212c1e454896b2a5ea9c63996aac.png" width="60%">
- 图像先是分成若干个 patch，每个 patch 通过 patch embedding 就又变成一系列的 token。假设图像 token 的序列长度是 N，token embedding 的维度依旧为 H，所以图像输入到 Transformer 的是 N×H 的矩阵。
<img src="https://img-blog.csdnimg.cn/705b5c0e5ddc4537b685005767c0b92f.png" width="80%">


以下部分就是文本和图像通过 embedding 层后生成的 token。

![在这里插入图片描述](https://img-blog.csdnimg.cn/db44bc426edb41c6b818dd52a76c389d.png)

- 星号部分就是[CLS] token，左边的就是文本的[CLS] token，右边的就是图像的[CLS] token。
- 灰色部分表示模态的类别，如图文本部分都是 0，图像部分就是 1。对于 single-stream 方法，是将文本和图像 concat 起来成一系列 token 作为 Transformer 的输入，如果不告诉模型哪块是文本，哪块是图像，可能不利于模型的学习；如果告诉模型哪块是文本，哪块是图像，模型可能会去训练文本与图像之间的关系，更有利于学习。
- 深绿色和深紫色（对应图中的0 1 2 3 4 5 6）分别是文本和图像中的position embedding。
- 浅绿色和浅紫色分别是文本和图像通过 embedding 层后生成的 token。

&emsp;&emsp;但这三个（灰色 - Modal-type embedding、深绿/紫色 - Token/Patch position embedding、浅绿/紫色 - 文本和图像生成的 token）并不是像图中显现的那样 concat 起来的，而是相加起来的（如下代码，来自[源码](https://github.com/dandelin/vilt)中 vilt/modules/vilt_module.py）。

<img src="https://img-blog.csdnimg.cn/f986fdf023d64699963558e9fbe9767e.png" width="80%">


concat 的部分是将上述相加后得到的整体 进行 concat，变成一个序列。

<img src="https://img-blog.csdnimg.cn/afb065e5dc744bd488ed5c11d94961aa.png" width="60%">

这样就是 Transformer 模型的输入了，那么这个输入的序列的长度就是 1 + L + 1 + N = 2 + L + N，那么整个输入就是 (2 + L + N) × H。

**输出部分**

&emsp;&emsp;Image Text Matching 和 Word Patch Alignment 都是用来计算文本和图片之间的相似性，是否匹配。Masked Language Modeling 用于文本部分。

&emsp;&emsp;Image Text Matching 就相当于一个二分类任务，看文本和图片是否匹配，它用到的输出就是整个序列输出的第一个位置上的，就像[CLS] token 一样，并不是用了所有的输出，图中的 pooler 是 H×H 的 matrix，然后变成 1×H，再经过一个FC，就可以做 Image Text Matching了。大部分 VLP 模型都用到了 Image Text Matching 这个 目标函数。

&emsp;&emsp;Word Patch Alignment 也是用来计算文本特征和图像特征之间的相似度，利用 optimal transport（最有运输理论），可以理解成将文本的输出和图像的输出当成一个概率分布，然后计算这两个分布之间的距离，这里当然也是越小越好。

&emsp;&emsp;Masked Language Modeling 相当于完形填空，把其中某个单词 mask 掉，再通过模型重建，这是基本所有 NLP 任务中都会用到的。

&emsp;&emsp;这里其中一个可改进的地方就是在图像部分也加入“完形填空”的目标函数，但由于当时 BEiT 和 MAE 等工作还没做出，图像领域还无法很有效的去做重建任务，所以作者这里就没有加。现在就有 VL-BEiT 在视觉部分加入了重建 loss

## 4.2 Whole Word Masking
&emsp;&emsp;将整个词都 mask 掉。作者在论文中举例，如果有单词 "giraffe"，如果用 tokenizer 如 BPE 等，那么 "giraffe" 就会被分成 ["gi", "##raf", "##fe"]，此时这些小部分才是一个个的 token，假设此时把其中的一个 token "##raf" mask 掉，即变成 ["gi", "[MASK]", "##fe"]，英文中以 "gi" 开头，"##fe" 结尾的单词很少，那么模型很容易就知道中间填 "##raf"，这就导致再做多模态时，根本不需要借助图像这边的信息，仅根据文本就能判断中间填 "##raf"，这样这个 loss 就失去了意义。既然如此，作者就将整个单词都 mask 掉，也就是在整个句子中去掉了 "giraffe"，这时模型再想把 "giraffe" 重建出来，就必须要借助图像的信息，进一步加强图像与文本间的联系。

## 4.3 Image Augmentation
&emsp;&emsp;在以往的研究中，VLP 模型中基本没有用到数据增强。作者使用颜色反转（因为文本通常也包含颜色信息）和 cutout（因为它可以清除分散在整个图像中的小但重要的对象），并进行了一些调参，最终取得了不错的效果。

<hr>

# 5 实验

## 5.1 数据集
<center><img src="https://img-blog.csdnimg.cn/f56e6b9d3c0f4ab6a2bd75a64e773da4.png" width="50%">

## 5.2 对比实验
**分类任务**
<center><img src="https://img-blog.csdnimg.cn/6d45aa5b8e864a00a578764bb00361b0.png" width="60%">

&emsp;&emsp;传统的 Region Feature 方法在性能上还是比较好的。ViLT的运行时间相比于传统方法都大大缩短。相比于VisualBERT，ViLT的性能更好，然相比于最好的 OSCAR 和 VinVL，ViLT 在 VQAv2 数据集上还有竞争力，但在NLVR2就明显不行了。ViLT 最主要的成就还是在运行速度和精度上的结合比以往的模型都要好——运行速度极短且精度也具有很强的竞争力。

**Retrieval 任务**

zero-shot：

![在这里插入图片描述](https://img-blog.csdnimg.cn/7ccc33dc9f0f4aa0bdc53dd0c6e7c697.png)

ﬁne-tuning：

![在这里插入图片描述](https://img-blog.csdnimg.cn/f2f113955a8b4af0a3310da3d8cea65d.png)

&emsp;&emsp;总之，ViLT在性能上并不如以往最好的模型，但其在时间和性能的取舍上做的比较好，性能略微降低一些，简易性和快速性得到了大幅提升。



## 5.3 消融实验
![在这里插入图片描述](https://img-blog.csdnimg.cn/cc5969157ec54a858cd5a49c76bbd8d4.png)
Training Steps 列证明了训练时间越长模型性能越好。
**w** 表示是否使用 whole word masking，使用它对于模型有提升，但相对而言是比较小的。
**m** 表示是否使用 MPP objective（图像上的“完形填空”，重建图像），实验表明使用MPP后性能并没有上升，作者也就没有在后面用这个（当时BEiT，MAE还未发表，图像重建领域还不是很好）。
**a** 表示是否使用 RandAugment，也就是图像上的数据增强，可以看到对于模型性能的提升是很有效的。

## 5.4 VLP 模型对比
<center><img src="https://img-blog.csdnimg.cn/f7525bc118464654a3334b06b17fef66.png" width="50%"><img src="https://img-blog.csdnimg.cn/bd355764dba340d9913950992defa453.png" width="50%">

<hr>

# 6 结论
&emsp;&emsp;本文提出了一个极小化的 VLP 模型 ViLT，它没有使用以往需要大量配置的 embedding 方式（如Faster R-CNN 和 ResNet来提取特征），仅仅使用一个简单的 Patch Embedding 就解决了图像特征提取。整个模型更简单，速度又快，结果又还可以。ViLT 虽然性能比不上 SOTA，但它提供了一种不需要 convolution 和 region 的 supervision 方法。


作者提供了未来的几个可能的研究方向：
- **Scalability**：如果模型更大，用的数据集更多，那么结果也会更好。
- **Masked Modeling for Visual Inputs**：在图像上也使用重建方法。这部分当前已经有 BEiT 和 MAE 了，也已经有论文对 ViLT 在这部分进行了图像重建方面的改进。
- **Augmentation Strategies**：根据消融实验来看，数据增强确实是很有效果的。