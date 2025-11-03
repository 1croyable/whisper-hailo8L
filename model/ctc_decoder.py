# ctc_decoder.py
from typing import List, Sequence, Optional
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass

import kenlm

from model.tokenizer import get_tokenizer

def log_sum_exp(a: float, b: float) -> float:
    """
    用于在 log 空间中稳定地计算 log(exp(a) + exp(b))
    """
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def load_french_tokenizer():
    return get_tokenizer(multilingual=True, language="fr", task="transcribe")

class LMBase:
    """语言模型接口：所有 LM 都应实现 log_prob(prefix, token) 方法"""
    def log_prob(self, prefix: Sequence[int], token: int) -> float:
        return 0.0

@dataclass
class BeamEntry:
    p_blank: float       # 前缀以 blank 结尾的 log 概率
    p_non_blank: float   # 前缀以非 blank 结尾的 log 概率
    lm_score: float      # 已累积的语言模型 log 概率

def ctc_prefix_beam_search(
    logits: torch.Tensor, # (B, V, T) or (B, V, T, 1)
    *, # 强制使用关键字参数
    beam_size: int = 8, # 束宽
    alpha: float = 0.6, # 语言模型权重
    beta: float = -0.3, # 长度惩罚权重
    blank_id: int = 0,
    topk_per_timestep: int = 20, # 每帧取 top-k 候选
    tokenizer=None, # 用于解码，ID映射到文本
) -> List[str]: # 返回: List[str]: 每个 batch 对应的解码文本
    # 语言模型加载
    lm = KenLMAdapter("/model/assets/fr.arpa.bin", tokenizer=tokenizer)

    # 保证维度正确
    if logits.dim() == 4 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)  # (B, V, T)

    # 维度调整为 (B, T, V)
    logits = logits.permute(0, 2, 1)
    # 在 CPU 上计算 log softmax（防止溢出）
    log_probs = F.log_softmax(logits, dim=-1).cpu()

    batch_size, T, V = log_probs.shape
    results: List[str] = []

    # 对每个 batch 独立解码
    for b in range(batch_size):
        lp = log_probs[b]  # 当前样本的 (T, V)
        # 初始化 beam：空前缀
        beam = {(): BeamEntry(p_blank=0.0, p_non_blank=-math.inf, lm_score=0.0)}

        # 遍历时间步
        for t in range(T):
            time_probs = lp[t]
            # 每帧取 top-k 候选，减少计算量
            topk_vals, topk_idx = torch.topk(time_probs, min(topk_per_timestep, V))
            next_beam = {}

            # 遍历当前所有前缀
            for prefix, entry in beam.items():
                # 当前前缀的总 log 概率
                prefix_total = log_sum_exp(entry.p_blank, entry.p_non_blank)

                # blank：保持前缀不变，只更新 p_blank
                p_blank_t = time_probs[blank_id].item()
                val_blank = prefix_total + p_blank_t
                nb = next_beam.get(prefix)
                if nb is None:
                    next_beam[prefix] = BeamEntry(
                        p_blank=val_blank,
                        p_non_blank=-math.inf,
                        lm_score=entry.lm_score,
                    )
                else:
                    nb.p_blank = log_sum_exp(nb.p_blank, val_blank)

                # 对每个可能的 token 扩展前缀
                for k_idx in topk_idx.tolist():
                    if k_idx == blank_id:
                        continue

                    p_token = time_probs[k_idx].item()
                    new_prefix = prefix + (k_idx,)

                    # 计算语言模型增量得分
                    try:
                        lm_inc = lm.log_prob(list(prefix), k_idx)
                    except Exception:
                        lm_inc = 0.0

                    # 重复 token 特殊规则（CTC collapsing）
                    if len(prefix) > 0 and prefix[-1] == k_idx:
                        val_non_blank = entry.p_blank + p_token
                    else:
                        val_non_blank = prefix_total + p_token

                    new_lm_score = entry.lm_score + lm_inc
                    existing = next_beam.get(new_prefix)

                    if existing is None:
                        next_beam[new_prefix] = BeamEntry(
                            p_blank=-math.inf,
                            p_non_blank=val_non_blank,
                            lm_score=new_lm_score,
                        )
                    else:
                        existing.p_non_blank = log_sum_exp(existing.p_non_blank, val_non_blank)
                        existing.lm_score = max(existing.lm_score, new_lm_score)

            # beam 剪枝：仅保留最高得分的 beam_size 个候选
            scored = []
            for prefix, e in next_beam.items():
                total_score = log_sum_exp(e.p_blank, e.p_non_blank) + alpha * e.lm_score + beta * len(prefix)
                scored.append((total_score, prefix, e))

            scored.sort(key=lambda x: x[0], reverse=True)
            beam = {p: e for _, p, e in scored[:beam_size]}

        # 全部时间步结束后，选出最终得分最高的候选
        final_candidates = []
        for prefix, e in beam.items():
            total_score = log_sum_exp(e.p_blank, e.p_non_blank) + alpha * e.lm_score + beta * len(prefix)
            final_candidates.append((total_score, prefix))

        final_candidates.sort(key=lambda x: x[0], reverse=True)
        best_prefix = final_candidates[0][1] if final_candidates else ()

        # 解码为文本
        if tokenizer is not None:
            valid_ids = [i for i in best_prefix if 0 <= i < tokenizer.encoding.n_vocab]
            text = tokenizer.decode(valid_ids) if len(valid_ids) else ""
        else:
            text = " ".join(str(i) for i in best_prefix)

        results.append(text)

    return results

class KenLMAdapter(LMBase):
    """
    使用 KenLM 语言模型的适配器。
    将 tokenizer 的子词序列解码为文本，再用 KenLM 计算增量得分。
    """
    def __init__(self, model_path: str, tokenizer: Optional[object] = None):
        self.model = kenlm.Model(model_path)
        self.tokenizer = tokenizer

    def log_prob(self, prefix: Sequence[int], token: int) -> float:
        if self.tokenizer is None:
            return 0.0
        try:
            prev_text = self.tokenizer.decode(list(prefix)) if prefix else ""
            new_text = self.tokenizer.decode(list(prefix) + [token])
            prev_score = self.model.score(prev_text, bos=False, eos=False)
            new_score = self.model.score(new_text, bos=False, eos=False)
            # kenlm.score 默认是以 log10 为底的，需要转为自然对数
            return (new_score - prev_score) * math.log(10)
        except Exception:
            return 0.0
