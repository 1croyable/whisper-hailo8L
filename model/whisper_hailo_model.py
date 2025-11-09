import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

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

class Linear(nn.Module):
    """
    4D Linear 层 (B, C, T, 1)
    实质是 1x1 Conv2d：
        输入通道 = in_features = C
        输出通道 = out_features
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 注意：权重是可训练参数（或可冻结后导出）
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, 1, 1)
        )

        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        输入:  x (B, C, T, 1)
        输出:  (B, out_features, T, 1)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,T,1), got {x.shape}")

        return F.conv2d(x, self.weight, self.bias)

class LayerNorm(nn.Module):
    """
    (B, C, T, 1) 输入的 LayerNorm 等价形式 (GroupNorm 实现)。
    使用 GroupNorm(1, C) 在样本内部跨所有通道做归一化。
    """
    def __init__(self, n_state: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(1, n_state, eps=eps, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T, 1)
        return: (B, C, T, 1)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,T,1), got {x.shape}")
        return self.norm(x)
    
class RMSNorm4D(nn.Module):
    """
    4D 版本 RMSNorm，按通道维度 C 做归一化。
    输入/输出形状: (B, C, T, 1)
    """
    def __init__(self, n_state: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习缩放参数，按通道
        # 修正权重形状为标准 4D 广播形式
        self.weight = nn.Parameter(torch.ones(1, n_state, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,T,1), got {x.shape}")
        # 按通道维做均方根归一化
        rms = (x.pow(2).mean(dim=1, keepdim=True) + self.eps).rsqrt()
        y = x * rms * self.weight
        return y

class SSM4D(nn.Module):
    """
    每个通道独立指数衰减 + 相位旋转记忆:
    y_t = Σ β * α^τ * cos(τ * θ) * x_{t-τ}
    """
    def __init__(self, n_state: int, kernel_size: int = 24):
        super().__init__()
        self.n_state = n_state
        self.kernel_size = kernel_size
        self.alpha_logit = nn.Parameter(torch.zeros(n_state))
        self.beta_param = nn.Parameter(torch.ones(n_state) * 0.5)
        self.theta_param = nn.Parameter(torch.linspace(0, float(np.pi / 4), n_state))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, 1)
        B, C, T, _ = x.shape
        assert C == self.n_state, f"Channel mismatch ({C} vs {self.n_state})"

        # 指数衰减参数
        alpha = torch.sigmoid(self.alpha_logit).clamp(1e-4, 1 - 1e-4).to(x.dtype)
        beta  = F.softplus(self.beta_param).to(x.dtype)
        theta = self.theta_param.to(x.dtype)

        # 生成指数卷积核 (C,K)
        h = torch.arange(self.kernel_size, device=x.device, dtype=x.dtype)
        decay = torch.pow(alpha.unsqueeze(1), h.unsqueeze(0))
        phase = torch.cos(h.unsqueeze(0) * theta.unsqueeze(1))
        kernel = beta.unsqueeze(1) * decay * phase
        weight = kernel.view(C, 1, self.kernel_size, 1)

        # 深度可分离因果卷积
        y = F.conv2d(x, weight, bias=None, stride=1,
                     padding=(self.kernel_size - 1, 0), groups=C)
        # 替换切片为 torch.narrow，避免 ONNX Slice 参数问题
        y = torch.narrow(y, dim=2, start=0, length=T)
        return y

class MultiHeadAttention(nn.Module):
    """
    4D版本多头注意力，保持 (B, C, T, 1)
    - 自注意力: 采用核注意力近似 + 因果 cumsum
    - 交叉注意力: 使用标准缩放点积注意力（支持不同长度的键值序列）
    """
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.d_head = n_state // n_head

        self.query = Linear(n_state, n_state)
        self.key   = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out   = Linear(n_state, n_state)

    def phi(self, x: Tensor) -> Tensor:
        # 正值核，防止全 0
        return F.silu(x) + 1.0

    def forward(self, x: Tensor, xa: Optional[Tensor] = None) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,T,1), got {x.shape}")

        if xa is None:
            q = self.query(x)   # (B, C, T, 1)
            k = self.key(x)     # (B, C, T, 1)
            v = self.value(x)   # (B, C, T, 1)
            out = self._self_attention_linear(q, k, v)
            return self.out(out)
        else:
            q = self.query(x)
            k = self.key(xa)
            v = self.value(xa)
            out = self._cross_attention_linear(q, k, v)
            return self.out(out)

    def _self_attention_linear(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, C, T, _ = q.shape
        H = self.n_head
        Dh = self.d_head

        phi_q = self.phi(q).reshape(B, H, Dh, T, 1)
        phi_k = self.phi(k).reshape(B, H, Dh, T, 1)
        v = v.reshape(B, H, Dh, T, 1)

        # 因果累计 (Hailo 支持 cumsum)
        k_sum = torch.cumsum(phi_k, dim=3)
        kv_sum = torch.cumsum(phi_k * v, dim=3)

        den = torch.sum(phi_q * k_sum, dim=2, keepdim=True) + 1e-6
        out = (phi_q * kv_sum) / den

        out = out.reshape(B, C, T, 1)
        out = out / torch.sqrt(torch.tensor(Dh, dtype=out.dtype, device=out.device))
        return out

    def _cross_attention_linear(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, C, Tq, _ = q.shape
        _, Ck, Tk, _ = k.shape

        H = self.n_head
        Dh = self.d_head

        phi_q = self.phi(q).reshape(B, H, Dh, Tq, 1)
        phi_k = self.phi(k).reshape(B, H, Dh, Tk, 1)
        v = v.reshape(B, H, Dh, Tk, 1)

        # 对 encoder 的时间维 Tk 做全局聚合
        k_sum  = torch.sum(phi_k, dim=3, keepdim=True)          # (B,H,Dh,1,1)
        kv_sum = torch.sum(phi_k * v, dim=3, keepdim=True)      # (B,H,Dh,1,1)

        den = torch.sum(phi_q * k_sum, dim=2, keepdim=True)     # (B,H,1,Tq,1)
        den = den + 1e-6
        out = (phi_q * kv_sum) / den                            # (B,H,Dh,Tq,1)

        # 合并 heads 回到 4D
        out = out.reshape(B, C, Tq, 1)
        out = out / torch.sqrt(torch.tensor(Dh, dtype=out.dtype, device=out.device))
        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.SiLU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
    ):
        x = x + self.attn(self.attn_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x

class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_mels,
            out_channels=n_state,
            kernel_size=(3, 1),
            padding=(1, 0)
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_state,
            out_channels=n_state,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0)
        )
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, pos_emb: Optional[Tensor] = None) -> Tensor:
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))  # (B, n_state, T/2, 1)

        if pos_emb is not None:
            # 截到同样的时间长度
            x = (x + pos_emb[..., :x.shape[2], :]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)
    
class MambaLikeBlock4D(nn.Module):
    """
    一个 NPU 友好的、mamba 风格的 4D block
    输入输出: (B, C, T, 1) 和注意力块一样
    核心步骤: RMS Norm, projection, convolution, SiLU, 选择性SSM, output projection, 残差
    """
    def __init__(self, n_state: int, d_conv: int = 3, ssm_kernel: int = 24):
        super().__init__()
        self.n_state = n_state

        # 进来先做归一化
        self.ln = RMSNorm4D(n_state)

        # 把 C 打成 2C，前 C 做内容，后 C 做 gate
        self.in_proj = Linear(n_state, 2 * n_state)

        # depthwise conv：对每个通道单独卷，沿时间维
        # 输入是 (B, 2C, T, 1)
        self.d_conv = nn.Conv2d(
            in_channels=2 * n_state,
            out_channels=2 * n_state,
            kernel_size=(d_conv, 1),
            # 直接在 conv 上做时间维左侧 padding，实现因果卷积
            padding=(d_conv - 1, 0),
            groups=2 * n_state
        )

        self.ssm = SSM4D(n_state, kernel_size=ssm_kernel)

        # 输出再压回 C
        self.out_proj = Linear(n_state, n_state)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, 1)
        B, C, T, _ = x.shape

        h = self.ln(x)                  # (B,C,T,1)
        h = self.in_proj(h)             # (B,2C,T,1)
        # 直接使用带 padding 的 depthwise conv，再做因果裁剪，保持与残差同长
        h = self.d_conv(h)
        h = torch.narrow(h, dim=2, start=0, length=T)

        # 4) 拆成内容 / gate 两半
        h_content, h_gate = torch.split(h, C, dim=1)   # (B,C,T,1), (B,C,T,1)

        # 内容走 SiLU -> SSM，gate 经 sigmoid，再做选择性门控
        h_content = F.silu(h_content)
        h_ssm = self.ssm(h_content)     # (B,C,T,1)
        gate = torch.sigmoid(h_gate)
        h_sel = h_ssm * gate            # (B,C,T,1)

        # 5) 输出投影
        h_out = self.out_proj(h_sel)    # (B,C,T,1)

        # 6) 残差
        return x + h_out

class ResidualMambaCrossBlock(nn.Module):
    """
    解码器的基本块：
    - 自身路径使用 MambaLikeBlock4D（阶段性 SSM 近似，自回归因果）
    - 然后做一次交叉注意力（对编码器特征），再接 MLP
    每步都有残差；交叉注意力与 MLP 前置 LayerNorm。
    """
    def __init__(self, n_state: int, n_head: int, d_conv: int = 3, ssm_kernel: int = 24):
        super().__init__()
        self.mamba = MambaLikeBlock4D(n_state, d_conv=d_conv, ssm_kernel=ssm_kernel)
        self.cross_attn = MultiHeadAttention(n_state, n_head)
        self.cross_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.SiLU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        # Mamba 自身路径（内部自带残差）
        x = self.mamba(x)
        # 交叉注意力 + 残差
        x = x + self.cross_attn(self.cross_ln(x), xa)
        # MLP + 残差
        x = x + self.mlp(self.mlp_ln(x))
        return x

class Decoder(nn.Module):
    """
    基于 Mamba 自注意 + 交叉注意力 的解码器，实现与 Whisper 类似的结构：
    - token embedding + learnable pos embedding
    - 多层 (Mamba + CrossAttn + MLP)
    - 最后 LayerNorm 与词表投影（权重共享）
    形状约定: 文本张量保持 4D: (B, C, T, 1)
    """
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,
                 d_conv: int = 3, ssm_kernel: int = 24):
        super().__init__()
        self.n_state = n_state
        self.n_ctx = n_ctx

        # 替换 nn.Embedding 为 nn.Conv2d
        self.token_proj = nn.Conv2d(n_vocab, n_state, kernel_size=1, bias=False)

        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        self.out_proj = nn.Conv2d(
            in_channels=n_state,
            out_channels=n_vocab,
            kernel_size=1,
            bias=False
        )

        self.blocks = nn.ModuleList([
            ResidualMambaCrossBlock(n_state, n_head, d_conv=d_conv, ssm_kernel=ssm_kernel)
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm(n_state)

        self.n_vocab = n_vocab

    def forward(self, tokens: Tensor, xa: Tensor) -> Tensor:
        """
        tokens: LongTensor, (B, T_text)
        xa: 编码器输出 (B, C, T_audio, 1)
        返回: logits (B, n_vocab, T_text, 1)
        """
        if tokens.dim() != 2:
            raise ValueError("tokens should be (B, T_text)")
        B, Tt = tokens.shape
        if xa.dim() != 4:
            raise ValueError("xa should be 4D (B,C,T,1)")

        # 1) One-hot + Conv2d 替代 embedding
        one_hot = F.one_hot(tokens, num_classes=self.n_vocab).float()  # (B, T, V)
        x = one_hot.transpose(1, 2).unsqueeze(-1)  # (B, V, T, 1)
        x = self.token_proj(x)                     # (B, C, T, 1)

        # 2) 加上位置编码 (已是 (B,C,T,1) 形状，不需要 permute)
        pos = self.positional_embedding[:Tt, :].T.unsqueeze(0).unsqueeze(-1)  # (1,C,T,1)
        x = (x + pos).to(xa.dtype)  # (B,C,T,1)

        # 3) 多层解码块
        for block in self.blocks:
            x = block(x, xa)

        # 4) 输出层归一化 + 投影
        x = self.ln(x)            # (B,C,T,1)
        logits = self.out_proj(x) # (B,V,T,1)
        return logits
    
class WhisperHailoModel(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

        # 语音编码器
        self.encoder = AudioEncoder(
            n_mels=dims.n_mels,
            n_ctx=dims.n_audio_ctx,
            n_state=dims.n_audio_state,
            n_head=dims.n_audio_head,
            n_layer=dims.n_audio_layer,
        )

        # 文本解码器（Mamba + CrossAttention）
        self.decoder = Decoder(
            n_vocab=dims.n_vocab,
            n_ctx=dims.n_text_ctx,
            n_state=dims.n_text_state,
            n_head=dims.n_text_head,
            n_layer=dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """编码音频特征 -> (B, C, T_audio, 1)"""
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """解码 logits"""
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        audio_features = self.encoder(mel)
        logits = self.decoder(tokens, audio_features)
        return logits