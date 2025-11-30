import base64
import gzip
import math
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

class LayerNorm4D(nn.Module):
    """
    Hailo-friendly LN for (B, C, T, 1)
    - 不用 GroupNorm
    - 不用 **2 / Pow
    - 只在通道维 C 上做归一化
    """
    def __init__(self, n_state: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, n_state, 1, 1))
        self.bias   = nn.Parameter(torch.zeros(1, n_state, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, 1)
        # 1) 按通道求均值
        mean = x.mean(dim=1, keepdim=True)                # (B,1,T,1)

        # 2) 方差用乘法，不用 **2
        diff = x - mean                                   # (B,C,T,1)
        diff_sq = diff * diff                             # (B,C,T,1)
        var = diff_sq.mean(dim=1, keepdim=True)           # (B,1,T,1)

        # 3) 标准化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)   # (B,C,T,1)

        # 4) 仿射
        return x_hat * self.weight + self.bias
    
class RMSNorm4D(nn.Module):
    """
    4D 版本 RMSNorm，按通道维度 C 做归一化。
    输入/输出形状: (B, C, T, 1)
    """
    def __init__(self, n_state: int, eps: float = 1e-6):
        super().__init__()
        # GroupNorm(1, C) normalizes over all channels
        self.norm = nn.GroupNorm(1, n_state, eps=eps, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)
    
class SSM4D(nn.Module):
    """
    Hailo-friendly 版本 SSM：
    - 动态 kernel（alpha, beta, theta）照算；
    - 不用 F.conv2d；
    - 全静态 shape：kernel_size=24, pad_top=23；
    - 输入输出恒定为 4D (B, C, T, 1)。
    """
    def __init__(self, n_state: int, kernel_size: int = 24):
        super().__init__()
        self.n_state = n_state
        self.kernel_size = int(kernel_size)
        time_indices = torch.arange(self.kernel_size).view(1, 1, self.kernel_size)
        self.register_buffer("time_indices", time_indices, persistent=False)

    def _cos_approx(self, x: Tensor) -> Tensor:
        # 使用泰勒展开近似 cos(x) ≈ 1 - x^2/2 + x^4/24
        x2 = x * x
        x4 = x2 * x2
        return 1.0 - 0.5 * x2 + (1.0 / 24.0) * x4

    def forward(self, x: Tensor, alpha: Tensor, beta: Tensor, theta: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,T,1), got {x.shape}")

        B, C, T, _ = x.shape
        t_idx = self.time_indices.to(dtype=x.dtype, device=x.device)

        alpha = alpha.view(1, C, 1).to(dtype=x.dtype, device=x.device)
        beta = beta.view(1, C, 1).to(dtype=x.dtype, device=x.device)
        theta = theta.view(1, C, 1).to(dtype=x.dtype, device=x.device)

        log_alpha = torch.log(alpha.clamp(min=1e-6))
        decay = torch.exp(log_alpha * t_idx)                 # (1,C,K)
        phase = self._cos_approx(theta * t_idx)              # (1,C,K)
        kernel = (beta * decay * phase).view(C, 1, self.kernel_size)

        x_1d = x.squeeze(-1)
        x_padded = F.pad(x_1d, (self.kernel_size - 1, 0))
        y = F.conv1d(x_padded, kernel, groups=C)
        return y.unsqueeze(-1)

class Conv1x1NoUnsqueeze(nn.Conv2d):
    def forward(self, x):
        # 强制 weight 为 buffer，不走参数路径 → 不触发 Constant→Unsqueeze
        weight = self.weight
        bias = self.bias
        return F.conv2d(x, weight, bias)

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
            out = self._self_attention_kernel(q, k, v)
            return self.out(out)
        else:
            q = self.query(x)
            k = self.key(xa)
            v = self.value(xa)
            out = self._cross_attention_linear(q, k, v)
            return self.out(out)

    def _self_attention_kernel(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, C, T, _ = q.shape
        H = self.n_head
        Dh = C // H

        # 1) 核映射 + 分头
        phi_q = self.phi(q).reshape(B, H, Dh, T, 1)
        phi_k = self.phi(k).reshape(B, H, Dh, T, 1)
        v = v.reshape(B, H, Dh, T, 1)
        # 2) 全局聚合（注意不是 cumsum，是 sum）
        #    S_k  : Σ_t φ(k_t)
        #    S_kv : Σ_t φ(k_t) * v_t
        S_k  = torch.sum(phi_k, dim=3, keepdim=True)        # (B,H,Dh,1,1)
        S_kv = torch.sum(phi_k * v, dim=3, keepdim=True)    # (B,H,Dh,1,1)

        # 3) 对每个时间步用当前的 φ(q_t) 去“选取”这个全局聚合
        #    分母要在 Dh 上再汇总一次，这样 phi_q 就真的起作用了
        den = torch.sum(phi_q * S_k, dim=2, keepdim=True) + 1e-6   # (B,H,1,T,1)
        out = (phi_q * S_kv) / den                                 # (B,H,Dh,T,1)

        # 4) 合并 heads
        out = out.reshape(B, C, T, 1)
        out = out / torch.sqrt(torch.tensor(Dh, dtype=out.dtype, device=out.device))
        return out
    
    def _cross_attention_linear(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, C, Tq, _ = q.shape
        _, _, Tk, _ = k.shape
        H = self.n_head
        Dh = C // H
        
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
            padding=(0, 0)
        )
        self.conv2 = nn.Conv2d(
            in_channels=n_state,
            out_channels=n_state,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(0, 0)
        )
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor, pos_emb: Optional[Tensor] = None) -> Tensor:
        # 使用静态 F.pad 替代 conv 的对称 padding，避免 Hailo 自动拆 padding
        # conv1: kernel=(3,1), 期望对称 pad 1 → 使用 F.pad pads=(0,0,1,1)
        # 输入是 B x 80 x 1000 x 1
        x = F.pad(x, (0, 0, 1, 1))
        x = F.silu(self.conv1(x)) # B x 80 x 1000 x 1 -> B x 768 x 1000 x 1

        # conv2: kernel=(3,1), stride=(2,1), 对称 pad 1
        x = F.pad(x, (0, 0, 1, 1))
        x = F.silu(self.conv2(x))  # B x 768 x 1000 x 1 -> B x 768 x 500 x 1

        if pos_emb is not None:
            # 截到同样的时间长度
            x = (x + pos_emb[..., :x.shape[2], :]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)
    
class MambaLikeBlock4D(nn.Module):
    """
    NPU / Hailo 友好的 Mamba 风格块。
    关键改动：使用 F.pad(静态 int) 代替 nn.ConstantPad2d，避免 ONNX 导出产生 Pad→Cast
    链导致 Hailo 解析 KeyError (如 Cast_output_0_value 缺失)。
    """
    def __init__(self, n_state: int, d_conv: int = 3, ssm_kernel: int = 24):
        super().__init__()
        self.n_state = n_state
        # 确保为静态 int，避免 ONNX 导出时插入 Cast
        self.d_conv_ks = int(d_conv)
        self.ln = RMSNorm4D(n_state)
        self.in_proj = Linear(n_state, 2 * n_state)

        # 使用静态整数记录所需的因果 padding（顶端填充），在 forward 中用 F.pad 实现
        # 说明：使用 nn.ConstantPad2d 在某些 PyTorch→ONNX 导出路径下会生成 Pad→Cast 链，
        # Hailo 解析器无法识别中间 Cast，导致 KeyError。改用 F.pad 可避免该问题。
        self.pad_top = int(self.d_conv_ks - 1)

        # depthwise conv 本身不带 padding
        self.d_conv = nn.Conv2d(
            in_channels=2 * n_state,
            out_channels=2 * n_state,
            kernel_size=(d_conv, 1),
            padding=(0, 0),
            groups=2 * n_state,
        )

        self.ssm = SSM4D(n_state, kernel_size=ssm_kernel)
        self.out_proj = Linear(n_state, n_state)

        self.register_parameter(
            "alpha",
            nn.Parameter(torch.full((n_state,), 0.9, dtype=torch.float32))
        )
        self.register_parameter(
            "beta",
            nn.Parameter(torch.full((n_state,), 0.5, dtype=torch.float32))
        )
        theta_init = torch.linspace(0, math.pi / 4, n_state, dtype=torch.float32)
        self.register_parameter("theta", nn.Parameter(theta_init))

    def set_ssm_parameters(self, alpha: Tensor, beta: Tensor, theta: Tensor) -> None:
        with torch.no_grad():
            self.alpha.copy_(alpha.to(self.alpha.device, dtype=self.alpha.dtype))
            self.beta.copy_(beta.to(self.beta.device, dtype=self.beta.dtype))
            self.theta.copy_(theta.to(self.theta.device, dtype=self.theta.dtype))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, 1)
        B, C, T, _ = x.shape

        # ---- 前层标准处理 ----
        h = self.ln(x)
        h = self.in_proj(h)  # (B,2C,T,1)

        # 因果 pad：静态常数，避免动态 If/Pad
        if self.pad_top > 0:
            h = F.pad(h, (0, 0, int(self.pad_top), 0))  # (B,2C,T+pad_top,1)
        h = self.d_conv(h)  # (B,2C,T,1)

        # ---- 拆两半 ----
        h_content = h[:, :C, :, :]
        h_gate    = h[:, C:, :, :]

        # ---- SSM 路径 ----
        h_content = F.silu(h_content)
        alpha = self.alpha.to(dtype=h_content.dtype, device=h_content.device)
        beta  = self.beta .to(dtype=h_content.dtype, device=h_content.device)
        theta = self.theta.to(dtype=h_content.dtype, device=h_content.device)
        h_ssm = self.ssm(h_content, alpha, beta, theta)  # (B,C,T,1)

        # ---- Gate 路径 ----
        gate = torch.sigmoid(h_gate).to(dtype=h_ssm.dtype)  # (B,C,T,1)

        # ==== 🔒 关键部分 ====

        # 1. 用 Add(0) “锚定”两条路径，防止 Hailo 把 Unsqueeze / Shape 折叠掉
        h_ssm = h_ssm + x * 1e-9
        gate  = gate  + x * 1e-9
        h_sel = torch.mul(h_ssm.clone(), gate.clone())

        # ---- 输出投影 + 残差 ----
        h_out = self.out_proj(h_sel)
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
        self.cross_ln = LayerNorm4D(n_state)   # ← 用我们刚写的这个

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.SiLU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm4D(n_state)

    def set_ssm_parameters(self, alpha: Tensor, beta: Tensor, theta: Tensor) -> None:
        self.mamba.set_ssm_parameters(alpha, beta, theta)

    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        # Mamba 自身路径（内部自带残差）
        x = self.mamba(x)
        x = x + x * 1e-9
        # 交叉注意力 + 残差
        x = x + self.cross_attn(self.cross_ln(x), xa)
        x = x + x * 1e-9
        # MLP + 残差
        x = x + self.mlp(self.mlp_ln(x))
        x = x + x * 1e-9
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
        self.n_vocab = n_vocab

        # 训练时的 token embedding
        self.token_emb = nn.Embedding(n_vocab, n_state)

        # 训练时的可学习位置向量
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

        # 训练时的输出到 vocab（部署的时候可以只在 CPU 用）
        self.out_proj = Conv1x1NoUnsqueeze(n_state, n_vocab, kernel_size=1, bias=False)

        self.blocks = nn.ModuleList([
            ResidualMambaCrossBlock(n_state, n_head, d_conv=d_conv, ssm_kernel=ssm_kernel)
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm4D(n_state)

        # 固定 4D shape（给 Hailo encoder 用的，不影响 decoder_core）
        self.fixed_B = 1
        self.fixed_C = n_state
        self.fixed_T = n_ctx
        self.register_buffer(
            "text_shape_4d",
            torch.tensor([self.fixed_B, self.fixed_C, self.fixed_T, 1], dtype=torch.int64),
            persistent=False
        )

    def set_ssm_parameters(self, alphas: Tensor, betas: Tensor, thetas: Tensor) -> None:
        if alphas.shape[0] != len(self.blocks):
            raise ValueError("alphas length must match number of decoder blocks")
        if betas.shape[0] != len(self.blocks) or thetas.shape[0] != len(self.blocks):
            raise ValueError("betas/thetas length must match number of decoder blocks")
        for i, block in enumerate(self.blocks):
            block.set_ssm_parameters(alphas[i], betas[i], thetas[i])

    def forward(self, token_ids: Tensor, xa: Tensor) -> Tensor:
        """
        训练/纯 PyTorch 推理时使用：
        token_ids: (B, T)  int64
        xa        : encoder features (B, C, T_a, 1)
        """
        if token_ids.dim() != 2:
            raise ValueError("token_ids should be (B, T)")

        B, Tt = token_ids.shape

        # 1) token embedding
        x = self.token_emb(token_ids)             # (B, T, C)
        x = x.permute(0, 2, 1).unsqueeze(-1)      # (B, C, T, 1)

        # 2) 加可学习位置向量
        pos = self.positional_embedding[:Tt, :].T[None, :, :].unsqueeze(-1)
        pos = pos.to(dtype=x.dtype, device=x.device)
        x = x + pos

        # 3) Mamba + CrossAttn + MLP blocks
        for block in self.blocks:
            x = block(x, xa)

        x = self.ln(x)

        # 4) 训练时直接输出 logits
        logits = self.out_proj(x)                 # (B, vocab, T, 1)
        return logits

# class DecoderCore(nn.Module):
#     """
#     部署到 Hailo 的精简版 Decoder：
#     - 不做 token embedding
#     - 不做位置向量
#     - 不做 out_proj 到 vocab
#     只做:
#     - 多层 (Mamba + CrossAttn + MLP)
#     - 最后 LayerNorm
#     输入 x 必须已经是 (embedding + pos) 之后的特征。
#     """
#     def __init__(self, decoder: Decoder):
#         super().__init__()
#         self.blocks = decoder.blocks
#         self.ln = decoder.ln

#     def forward(self, x: Tensor, xa: Tensor) -> Tensor:
#         """
#         x : (B, C, T, 1)  已经加好 embedding + pos 的特征
#         xa: (B, C, T_a, 1) encoder 输出
#         """
#         if x.dim() != 4:
#             raise ValueError("x should be (B,C,T,1)")
#         if xa.dim() != 4:
#             raise ValueError("xa should be (B,C,T_a,1)")

#         for block in self.blocks:
#             x = block(x, xa)

#         x = self.ln(x)

#         # 小 hack 防止 Hailo 折叠掉
#         x = x.clone()
#         x = x + 0
#         return x
    
# class WhisperHailoModel(nn.Module):
    # def __init__(self, dims):
    #     super().__init__()
    #     self.dims = dims

    #     # 语音编码器
    #     self.encoder = AudioEncoder(
    #         n_mels=dims.n_mels,
    #         n_ctx=dims.n_audio_ctx,
    #         n_state=dims.n_audio_state,
    #         n_head=dims.n_audio_head,
    #         n_layer=dims.n_audio_layer,
    #     )

    #     # 文本解码器（Mamba + CrossAttention）
    #     self.decoder = Decoder(
    #         n_vocab=dims.n_vocab,
    #         n_ctx=dims.n_text_ctx,
    #         n_state=dims.n_text_state,
    #         n_head=dims.n_text_head,
    #         n_layer=dims.n_text_layer,
    #     )

    # def embed_audio(self, mel: torch.Tensor) -> torch.Tensor:
    #     """编码音频特征 -> (B, C, T_audio, 1)"""
    #     return self.encoder(mel)

    # def logits(self, onehot: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
    #     """解码 logits"""
    #     return self.decoder(onehot, audio_features)

    # def forward(self, mel: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    #     audio_features = self.encoder(mel)
    #     logits = self.decoder(onehot, audio_features)
    #     return logits