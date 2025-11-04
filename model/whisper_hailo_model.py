import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

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
    

class MultiHeadAttention(nn.Module):
    """
    4D版本多头注意力，保持 (B, C, T, 1)
    - 使用 φ(x)=silu(x)+1 代替 softmax
    - 不含 permute、transpose、view
    - 全部运算为 conv + mul + sum，可被 Hailo 编译
    """
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.d_head = n_state // n_head

        # 用 1×1 Conv 表示线性映射
        self.query = Linear(n_state, n_state)
        self.key   = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out   = Linear(n_state, n_state)

    # φ(x)：softmax 近似
    def phi(self, x: Tensor) -> Tensor:
        return F.silu(x) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, C, T, 1)
        return: (B, C, T, 1)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,T,1), got {x.shape}")

        q = self.query(x)   # (B, C, T, 1)
        k = self.key(x)     # (B, C, T, 1)
        v = self.value(x)   # (B, C, T, 1)

        # 注意力主体
        out = self.qkv_attention(q, k, v)
        return self.out(out)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        线性注意力 (kernel attention)
        """
        B, C, T, _ = q.shape
        H = self.n_head
        Dh = self.d_head

        # 对 q,k 应用核函数
        phi_q = self.phi(q)
        phi_k = self.phi(k)

        # 分头 reshape
        phi_q = phi_q.reshape(B, H, Dh, T, 1)
        phi_k = phi_k.reshape(B, H, Dh, T, 1)
        v = v.reshape(B, H, Dh, T, 1)

        # 分子：Σ_t φ(k_t) * v_t
        num = torch.sum(phi_k * v, dim=3, keepdim=True) # (B,H,Dh,1,1)

        # 分母：Σ_t φ(k_t)
        den = torch.sum(phi_k, dim=3, keepdim=True) # (B,H,Dh,1,1)
        den = den + 1e-6

        norm = phi_q * den
        out = (phi_q * num) / norm # (B,H,Dh,T,1)

        # 合并 heads → (B,C,T,1)
        return out.reshape(B, C, T, 1)
    
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
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        # 第一层卷积：提升通道数 n_mels → n_state
        self.conv1 = nn.Conv2d(
            in_channels=n_mels,
            out_channels=n_state,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

        # 第二层卷积：时间降采样
        self.conv2 = nn.Conv2d(
            in_channels=n_state,
            out_channels=n_state,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0)
        )

        # 残差注意力块
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )

        # 归一化层仿照LayerNorm
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, pos_emb: Optional[Tensor] = None) -> Tensor:
        """
        x : (B, n_mels, T, 1)
        pos_emb : (B, n_state, T, 1)
        return : (B, n_state, T_out, 1)
        """
        # 两层卷积特征提取
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))  # (B, n_state, T/2, 1)

        # 叠加位置编码
        x = (x + pos_emb[..., :x.shape[2], :]).to(x.dtype)

        # 注意力块
        for block in self.blocks:
            x = block(x)  # (B, n_state, T', 1)

        # 最后归一化
        return self.ln_post(x)


class EncoderCTC(nn.Module):
    def __init__(self, encoder: AudioEncoder, vocab_size: int, n_state: int):
        super().__init__()
        self.encoder = encoder
        self.ctc_linear = Linear(n_state, vocab_size)

    def forward(self, mel: Tensor, pos_emb: Optional[Tensor]) -> Tensor:
        """
        mel: (B, n_mels, T, 1)
        pos_emb: (B, n_state, T, 1)
        return: (B, vocab_size, T_out, 1)
        """
        x = self.encoder(mel, pos_emb)         # (B, n_state, T', 1)
        logits = self.ctc_linear(x)            # (B, vocab_size, T', 1)
        return logits