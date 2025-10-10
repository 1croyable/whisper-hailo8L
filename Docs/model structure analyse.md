我们看`model.py`中定义的类和函数，他们反映了whisper的整体架构

**ModelDimensions**

```python
@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
```

定义数据类型

整体的 **音频编码器**-**文本解码器** 结构的重要参数

- n_mels 音频特征的频谱维度（mel 频带数）
- n_audio_ctx 音频上下文长度（mel 时间帧数）。encoder 的时间轴长度，输入的行数
- n_audio_state 音频编码器的隐藏维度，输入的列数
- n_audio_head 音频编码器中多头注意力的头数
- n_audio_layer 音频编码器里残差注意力块的层数
- n_vocab 文本端的词表大小（tokenizer 的词汇数量）。
- n_text_ctx 文本解码器的上下文长度，输入的行数
- n_text_state 文本解码器的隐藏维度，输入的列数
- n_text_head 文本解码器中多头注意力的头数。
- n_text_layer 文本解码器里的残差注意力块层数

**LayerNorm**

```python
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype) #先把输入 x 转为 float32，调用父类的 LayerNorm 执行计算，然后把输出用 .type(x.dtype) 转回原始 dtype
```

- 把每个样本的特征向量居中并标准化为零均值、单位方差。避免梯度爆炸或过小梯度，归一化后有2个可学习参数
- 让每个样本的特征在每一层都保持零均值、单位方差，使得不同层之间的数值分布保持稳定。

对输入向量 $x \in \mathbb{R}^{d}$：
$$
\text{LayerNorm}(x)
= \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta
$$
其中
$$
\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad
\sigma = \sqrt{ \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2 }
$$

- $\gamma, \beta$ 是两个可学习参数（缩放和偏移）。

**Linear**

```python
class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )
```

线性层，同样是统一了运算精度

> 对于归一化/统计操作（mean/var、sqrt、除法），需要更高精度保证数值稳定，所以选择上采样到 float32。
>
> 对于大规模的矩阵乘加，使用输入的低精度（如 float16）能显著提高吞吐量且通常精度足够，因此把权重降到输入 dtype 更高效；同时避免在前向时把输入从低精度升到高精度（那会增加内存/带宽开销）。

**Conv1d**

```python
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )
```

对一维卷积层的封装。在混合精度场景下避免 dtype 不匹配或不必要的隐式转换，保证计算使用与输入一致的精度。

**函数sinusoids**

```python
def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

```

这个函数给梅尔频谱矩阵添加上位置信息`(T, dmodel)`大小的矩阵。这个位置信息的计算只依赖于数据的元信息

输入参数：length 时间步，上下文长度；channels 通道数（第一行用于验证channels必须是偶数）、max_timescale 默认值和原始的transformer一致
$$
PE(pos, 2i) = sin(pos / 10000^{(2i / d_{model})})\\
PE(pos, 2i+1) = cos(pos / 10000^{(2i / d_{model})})
$$
代码逻辑是：因为它按照指数方式增长，不好算，所以先取对数，然后除以一半的通道数，就变成了增长的补偿，然后再计算，最终就得到一个维度是`(T, dmodel)`的位置信息矩阵，加到梅尔频谱矩阵上

**函数disable_sdpa**

这里是控制pytorch是否可用sdpa，这是内置的用于加速多头注意力计算QKV的方法

**MultiHeadAttention**

```python
class MultiHeadAttention(nn.Module):
    use_sdpa = True # 是否使用pytorch加速

    def __init__(self, n_state: int, n_head: int): # n_state就是d_model 每条数据的特征数
        super().__init__()
        self.n_head = n_head # 头数
        # 这里做多头的输入是(T, d_model)，数学上每个小头都有自己的维度dim_head，每个小矩阵是(d_model, dim_head)然后拼接，现在这里是只做一次大乘(T, d_model) * (d_model, d_model)^T = (T, d_model)得到结果
        self.query = Linear(n_state, n_state) # 线性层 Q
        self.key = Linear(n_state, n_state, bias=False) # 线性层 K
        self.value = Linear(n_state, n_state) # 线性层 V
        self.out = Linear(n_state, n_state) # 线性层 把多头注意力每个头的输出拼接后的向量做一次可学习的线性组合，得到最终的 d_model 维度表示，供后续残差相加和 MLP 使用。这里提供了额外的一次融合前后数据的机会

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x) # 第一步：对输入x应用线性层得到Q，此时形状是(batch, T, d_model)

        # 判断是否有缓存，在解码的时候会有缓存，K和V是固定的
        # 或者有xa说明是交叉注意力，这样的话k和v应该由xa生成而不是x（此时x是已生成的文本经过带掩码的注意力机制后得到的）
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask) # 混合q k v矩阵
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # q k v 是三个矩阵，可能带掩码，可能没有，在解码器里，单词过来要执行带掩码的计算。
        n_batch, n_ctx, n_state = q.shape # 处理Q，分成batch数 T上下文数 d_model特征数
        scale = (n_state // self.n_head) ** -0.25 # 假如说特征数80，分8个头，那就是10开四次根号作为这个scale
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) #这里相当于n_batch, n_ctx, n_head, n_state 再做转调换变成(n_batch, n_head, n_ctx, n_state)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            # 加速
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            # 不加速
            # 这里执行QK^T的乘积，作为评分矩阵，但是评分矩阵的数字可能很大，所以这里它提出了乘上一个scale，这里是每个头的维度数开四次根号（transform架构原文好像是二次根号），四次根号可能他认为更稳定吧。总之，这里是（n_batch, n_head, n_ctx, n_state）与（n_batch, n_head, n_state, n_ctx）相乘，得到(n_batch, n_head, n_ctx, n_ctx)的评分矩阵，每一行表示q里的一行数据对k的一行数据的分数
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx] # 用 -inf 遮住未来信息，再softmax的时候被加上 -inf 的未知就变了0
            qk = qk.float() # 变成float32

            w = F.softmax(qk, dim=-1).to(q.dtype) # softmax，让q的每一行的评分归一化
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2) # w: (B, H, T_q, T_k), v: (B, H, T_k, head_dim)，w@v 对每个 query 在每个头上做加权求和，permute+flatten 将多头结果合并为 (B, T_q, d_model)
            qk = qk.detach() # 回收计算图，前向传播过程用不到梯度计算

        return out, qk

```

**ResidualAttentionBlock**

Transformer 残差块

```python
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head) # 初始化一个多头注意力机制实例
        self.attn_ln = LayerNorm(n_state) # 初始化多头注意力后的Norm层

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None # 交叉注意力机制，默认不创建
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None # 交叉注意力后的Norm层

        # 前馈神经网络 引入非线性
        n_mlp = n_state * 4 # 先把维度扩大以捕获更丰富的组合特征，再投回原维度，经验上效果好。
        # 神经网络：Linear(n_state, 4n_state) -> GELU -> Linear(4n_state, n_state)；
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state) # Norm层

        # 在 PyTorch 中调用模块实例 self.attn(...) 实际上会执行 nn.Module.call，其内部会：
		# 调用模块的 forward 方法（即最终会执行 forward），
		# 在调用前后处理 hooks（pre_hook / forward_hook / backward_hook 等）、处理 training/eval 状态、处理注册的 hooks/参数/buffers 等。
		# 所以写 self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache) 等价于 self.attn.forward(... )，但更安全、会触发注册的 forward hooks。
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0] #  注意力层输出 (out, qk) ，用[0]获取注意力输出；self.attn_ln(x)返回的是Norm归一化后的x，作为输入
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0] # 如果有交叉注意力，在多头注意力后进行一次交叉注意力，解码器里是先做mask的self-attention再进入这里
        x = x + self.mlp(self.mlp_ln(x)) # 前馈神经网络
        return x
```

**AudioEncoder**

到此已经有了所有的模块，可以开始写编码器/解码器结构

编码器输出音频的注意力处理后的数据，形状是`(batch, 原始行大小/2, d_model)`

```python
class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1) # 卷积层，3个卷积核
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1) # 步长是2，输出后行数减半
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state)) # 存储positional_embedding到buffer

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        ) # n_layer层注意力机制，内部有Norm，最后一次出来，内部没做，外边做一次Norm
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x)) # 卷积 -> 激活
        x = F.gelu(self.conv2(x)) # 卷积 -> 激活
        x = x.permute(0, 2, 1) # 交换行列
		# 确保位置向量的形状和这个一样，才可以加和
        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        
        # 进入注意力模块
        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x) # Norm层
        return x
```

**TextDecoder**

```python
class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
	
    	# 把“文本的离散符号 ID（数字）”转成可以被 Transformer 处理的“连续向量”。在这一步有一个embedding矩阵可以被训练。这个过程理解成one-hot矩阵与embedding矩阵的内积，查表。这里每行对应一个token的embedding
        self.token_embedding = nn.Embedding(n_vocab, n_state) 
        
        # 占位，给位置向量开辟一个空间
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        # 残差注意力模块，交叉注意力，这在内部会多创建一个交叉注意力层
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state) # Norm层

        # 一个上三角掩码，会被加到注意力矩阵qk上，注册到buffer里
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        首先要对x做变换才能和xa做交叉注意力
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        # 自注意力机制的每一次的k 和 v都是由文本生成的
        # 每次都生成前面已经生成过的k和v很浪费资源，时间复杂度t^2，所以引入缓存
        # 检查是否已有缓存（kv_cache）；
		# 如果有，取出缓存里任意一个张量；
		# 读取它的时间步长度（.shape[1]）；
		# 把这个长度作为 offset，告诉模型“我已经生成到第几个 token 了”，
		# 从而在下一步解码时加上正确的 位置编码。
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)
        
        # 到这里，x经过了embedding和位置编码，已经可以用于注意力训练了，并且它的形状应该和编码器输出一样(batch_size, T, d_model)
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache) #在这里是残差注意力块，包含了掩码子注意力 -> 交叉注意力 -> 前馈神经网络 这一套会重复n_layers次

        x = self.ln(x) # 我们内部用的是前norm层，现在出来就应该有一次norm
        logits = (
            # 输出矩阵与词表embedding矩阵相乘结果 logits 形状为 (batch, seq_len, n_vocab)
            # 这里的 n_vocab 就是此表大小，这样乘起来后经过归一化后的列就表示这个词的概率
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits
```

**Whisper**

```python
class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims # 存储了模型基本信息，超参数
        self.encoder = AudioEncoder( # 构造编码器和解码器
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)
        """
        把 “用于时间对齐” 的注意力头选在 decoder 最后一半层（即靠近输出的那些层），因为靠后层的表示更靠近最终预测、对时序对齐通常更可靠。
        OpenAI 团队就预先分析好了哪些 heads 是“alignment heads”并打包掩码到dump文件中，(n_text_layer,n_text_head)本身是一个布尔表，保存到alignment_heads中
        
        """
    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel) # 编码器处理音频
    
    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features) # 解码器处理文字，decoder的forward接收第一个参数是token，第二个是用于交叉注意力的编码器输出

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel)) # 先编码器后解码器，这是主要入口

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865 # 属性表示此表大小判断是否多语言

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual) # 减去1是一个特殊token，比如51865大小的基础词表就会表示99个语言条目

    # 一个存储k v缓存的机制|，在模型推理的时候需要用到，避免重复计算
    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks
	
    # 三个辅助函数，由另外其他的文件提供
    # detect_language 在正式 decode 之前用来做语言识别（取最可能的 language token），它会走 encoder + decoder 一次浅前向，但被 @torch.no_grad() 包裹，不构建梯度图，且不干扰 kv-cache。常用于多语模型先确定语言。
    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function

```

时间对齐的方法：

首先，我们有帧长度比如10ms

**步骤一：取出这些 heads 的注意力矩阵**

每个解码层的 cross-attention 输出 attention 权重矩阵：
$$
A^{(l,h)} \in \mathbb{R}^{T_{text} \times T_{audio}}
$$
选出 alignment_heads 指定的那些与事件相关的权重矩阵，取它们的平均：
$$
A_{align} = \frac{1}{N} \sum_{(l,h)\in \text{alignment heads}} A^{(l,h)}
$$
**步骤二：对于每个 token，找到最大注意力帧**

对每个 token i：
$$
t_i = \arg\max_j A_{align}[i, j]
$$
→ 这就是 “token i 最强关注的音频帧 j”。

把帧号 j 转成时间：
$$
\text{time}(i) = j \times 10\text{ms (或对应帧长度)}
$$
