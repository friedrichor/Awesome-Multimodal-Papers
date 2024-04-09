
<center><B><font size=5> Transformer</font></B></center>

# 讲解

[Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE)(李沐)

[史上最小白之Transformer详解](https://blog.csdn.net/Tink1995/article/details/105080033)

[transformer的简要解读（应该都能看懂）](https://blog.csdn.net/qq_40663469/article/details/123609730)

[详解Transformer中Self-Attention以及Multi-Head Attention](https://blog.csdn.net/qq_37541097/article/details/117691873)

# 7 结论

&emsp;&emsp;在这项工作中，我们提出了 Transformer，第一个完全基于注意的序列转换模型，用 multi-headed self-attention 取代了 encoder-decoder 架构中最常用的循环层。

&emsp;&emsp;对于翻译任务，Transformer 的训练速度比基于循环层或卷积层的架构快得多。在 WMT 2014 English-to-German 和 WMT 2014 English-to-French 的翻译任务中，我们都达到了一个新的水平。在前一个任务中，我们的最佳模型甚至优于所有之前报告的集合。

&emsp;&emsp;我们对基于注意力的模型的未来感到兴奋，并计划将其应用到其他任务中。我们计划将 Transformer 扩展到涉及文本以外的输入和输出模式的问题，并研究局部的、受限制的注意机制，以有效地处理大的输入和输出，如图像、音频和视频。减少时序生成是我们的另一个研究目标。

# 1 引言

&emsp;&emsp;用 RNN 处理时序信息时，因为是时序关系，是一步一步计算来的，难以并行，计算上性能比较差；此外，由于历史信息是一步一步地往后传递，如果时序比较长的话，那么很早期的那些时序信息在后面可能会丢掉。

# 2 相关工作

&emsp;&emsp;用卷积神经网络对于比较长的序列难以建模，因为卷积做计算时每次都只能看一个比较小的窗口，比如 $3\times3$ 的像素块。如果两个像素比较远的话需要用很多层卷积，最后才能把这两个隔得远的像素融合起来。如果使用 Transformer 中的注意力机制的话，每一次能够看到所有的像素，一层就能把整个序列都看到，就没有卷积神经网络的那个问题。但卷积神经网络比较好的就是可以做多个输出通道，一个输出通道可以认为是去识别不一样的模式，于是提出了 Multi-Head Attention。

&emsp;&emsp;Transformer 是第一个只依赖于**自注意力**来做这种 encoder-decoder 架构的模型。

# 3 模型

## 3.1 Encoder and Decoder Stacks

LayerNorm：

<img src="https://img-blog.csdnimg.cn/e8eda9a441ec4895a753650c55796cfc.png" width="40%">

&emsp;&emsp;decoder 是自回归的，也就是之前的输出也会作为后面的输入。

&emsp;&emsp;Masked Multi-Head Attention 保证输入进来时，在 $t$ 时间不会看到 $t$ 时间以后的那些输入，从而保证训练和预测时的行为时一致的。

## 3.2 Attention

&emsp;&emsp;注意力函数是将一个 query 和一些 key-value 对映射成一个 output 的函数，这里的 query、key、value、output 都是向量，具体来说 output 就是 value 的一个加权和，所以 output 和 value 的维度是一样的。对于每一个 value 的权重，它是每个 value 对应的 key 和 query 的相似度计算得来的

<img src="https://img-blog.csdnimg.cn/1d9cba85840d4e8eb0bf20141b151a30.png" alt="在这里插入图片描述" style="zoom:50%;" />

&emsp;&emsp;$QK^T$ 相当于做内积（余弦相似度），值越大说明 query 和 key 越相似。$\rm softmax (\it \frac{QK^T}{\sqrt{d_k}})$ 即为权重，

<img src="https://img-blog.csdnimg.cn/6a502314cb3e4b909ac939d495e3afe1.png" style="zoom:60%;" />

Scaled Dot-Product Attention 里面几乎是没有可学的参数的。

Multi-Head Attention 里面有可学的参数，为了用于不同的任务。

## 3.2.3 Applications of Attention in our Model

encoder：

<img src="https://img-blog.csdnimg.cn/cb678b3b27e04068a88701a4eb6d928c.png" alt="在这里插入图片描述" style="zoom:45%;" />

&emsp;&emsp;红圈处的三个分支分别表示 key，value 和 query。因为图中是一根线过来，然后复制成了三份（三个分支）说明这三个输入都是一样的东西，既作为 key，也作为 value，又作为 query，所以叫自注意力机制。

decoder：

<img src="https://img-blog.csdnimg.cn/262a9bfa469d475ba5bf35b3e7bba645.png" alt="在这里插入图片描述" style="zoom:50%;" />

&emsp;&emsp;这里大体上与 encoder 部分的 Attention 一样，唯一不同的是多了个 Masked，这就是前面说过的保证在算 $t$ 时刻 query 的输出时不会看到 $t$ 时间以后的那些输入，从而保证训练和预测时的行为时一致的。

<img src="https://img-blog.csdnimg.cn/814429c2ecba4bc19ba9f91ac51109a7.png" alt="在这里插入图片描述" style="zoom:33%;" />

&emsp;&emsp;key，value 来自于 encoder 的输出，query 来自于之前 Masked Multi-Head Attention 的输出。这个 Attention 的作用就是有效地把编码器中的一些输出根据我想要的东西给他拎出来。比如说中英文翻译 “hello world” 和 “你好世界”，key 为 “hello world”，当 query 为 “你” 和 “好” 时，会把对应 “hello” 所对应的权重调高，而 “world” 的权重调低；当 query 为 “世“ 和 ”界” 时，会把对应 “world” 所对应的权重调高，而 “hello” 的权重调低。意味着根据在解码器时的输入的不一样，那么会根据当前的那一个向量去在编码器的输出中挑感兴趣的东西。

## 3.3 Position-wise Feed-Forward Networks

其实就是 MLP，不过是 position-wise 的。

$$
{\rm FFN}(x) = {\rm max}(0, xW1 + b1)W2 + b2
$$
$x$ 原本是 512 的，经过 $W1$ 变成 2048，经过 $W2$ 又变回 512 了。

## 3.5 Positional Encoding

&emsp;&emsp;Attention 是不会有时序信息的，输出是 value 的加权和，其中的权重是 query 和 key 之间的距离，跟序列信息是无关的，不会看 key-value 对在序列中的哪些地方，这就意味着如果把一句话打乱之后 attention 出来的结果都是一样的，顺序会变但值不会变，假如在实际应用中把一句话中词的顺序打乱，语义发生变化了，但 attention 并不会处理这个情况。所以需要把时序信息加进来，RNN 是之前的输出作为新的输入了，而 Transformer 就是利用了 Positional Encoding，记录位置信息。

# 4 Why Self-Attention







