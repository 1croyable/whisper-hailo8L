import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )
    
class InstanceNorm(nn.Module):
    def __init__(self, n_state: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_state, affine=True, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, T, d_model)
        """
        # 转置并调整形状以适配 InstanceNorm2d
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (B, d_model, T, 1)
        x = self.norm(x)                      # 应用 InstanceNorm
        x = x.squeeze(-1).permute(0, 2, 1)    # 转回原始形状 (B, T, d_model)
        return x
    
class MultiHeadAttention(nn.Module):
    use_sdpa = True

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv = self.qkv_attention(q, k, v)
        return self.out(wv)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        q = q * scale
        k = k * scale

        def phi(x: Tensor) -> Tensor:
            return F.silu(x) + 1

        phi_q = phi(q)
        phi_k = phi(k)

        B, H, T, Dh = phi_q.shape
        phi_q = phi_q.reshape(B * H, T, Dh)
        phi_k = phi_k.reshape(B * H, T, Dh)
        v = v.reshape(B * H, T, Dh)

        kv = phi_k.transpose(-2, -1) @ v
        out = phi_q @ kv

        out = out.reshape(B, H, T, Dh).permute(0, 2, 1, 3).flatten(start_dim=2)

        return out
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = InstanceNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.SiLU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = InstanceNorm(n_state)

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
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = InstanceNorm(n_state)

    def forward(self, x: Tensor, pos_emb: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.silu(self.conv1(x))
        x = F.silu(self.conv2(x))
        
        x = x.permute(0, 2, 1)

        x = (x + pos_emb).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
    
class EncoderCTC(nn.Module):
    def __init__(self, encoder: AudioEncoder, vocab_size: int, n_state: int):
        super().__init__()
        self.encoder = encoder
        self.ctc_linear = nn.Linear(n_state, vocab_size)

    def forward(self, mel, pos_emb):
        x = self.encoder(mel, pos_emb)
        logits = self.ctc_linear(x)  # (B, T, vocab)
        return logits