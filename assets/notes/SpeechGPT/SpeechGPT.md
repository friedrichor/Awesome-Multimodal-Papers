<head>
<style>
    body {text-align: justify;}
</style>
</head>

[TOC]

<p style="text-align: center; font-weight: bold; font-size: 24px;">
SpeechGPT
</p>
**Paper:** [SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities](https://arxiv.org/abs/2305.11000)
**Submission Date:** 2023-05-18
**Accepted:** EMNLP 2023 (Findings)
**Project Page:** https://0nutation.github.io/SpeechGPT.github.io/
**Github:** https://github.com/0nutation/SpeechGPT
**Authors:** 复旦大学


## 模型结构

<div class="columns is-centered" id="figure_2">
<center><img src="figure_2.png" width="50%"></center>
</div>

由三部分组成：

- 离散单元提取器<font face="Times New Roman" size=4>(discrete unit extractor)</font>：可以理解为音频模态的编码器。使用 <font face="Times New Roman">Hidden-unit BERT(HuBERT)</font>，将语音信号转为离散的单元序列，即对音频进行 embedding。
- 大语言模型<font face="Times New Roman" size=4>(large language model)</font>：使用 <font face="Times New Roman">LLaMA</font>
- 单元声码器<font face="Times New Roman" size=4>(unit vocoder)</font>：可以理解为音频模态的解码器。使用 <font face="Times New Roman">HiFi-GAN</font>，将离散的单元序列解码为语音

## 训练

首先扩展单词表，在文本单词表的基础上添加了用于表示音频的 token 和 embedding

三阶段：

1. 对未配对语音数据的模态适应预训练<font face="Times New Roman">(Modality-Adaptation Pre-training on unpaired speech data)</font>
2. 跨模态指令微调<font face="Times New Roman">(Cross-modal Instruction Fine-Tuning)</font>
3. 模态链指令微调<font face="Times New Roman">(Chain-of-Modality Instruction Fine-Tuning)</font>

### <font face="Times New Roman">Stage 1: Modality-Adaptation Pre-training</font>

<font face="Times New Roman">利用一个无标记的语音语料库来训练 LLM 进行 next-token 预测任务 (类似于MLM)</font>

### <font face="Times New Roman">Stage 2: Cross-modal Instruction Fine-Tuning</font>

利用成对的语音-文本数据对齐语音和文本模态

<div class="columns is-centered" id="table_1">
<center><img src="table_1.png" width="75%"></center>
</div>

### <font face="Times New Roman">Stage 3: Chain-of-Modality Instruction Fine-Tuning</font>

<font face="Times New Roman">对第二步得到的模型，利用 LoRA 对语音指导中的模态链指令进行微调</font>

<font face="Times New Roman">这步类似于思维链(Chain-of-Thought)的方法，让模型逐步地得到最终的结果</font>

<div class="columns is-centered" id="appendix_C">
<center><img src="appendix_C.png" width="75%"></center>
</div>

## <font face="Times New Roman">利用 GPT-4 生成指令</font>

<div class="columns is-centered" id="appendix_A">
<center><img src="appendix_A.png" width="75%"></center>
</div>

<div class="columns is-centered" id="appendix_B">
<center><img src="appendix_B.png" width="75%"></center>
</div>
