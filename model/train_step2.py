import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from model.ctc_decoder import ctc_prefix_beam_search
from model import ensure_mel_4d, load_audio
from model import AudioEncoder, EncoderCTC
from model import load_french_tokenizer
from model.mel import HOP_LENGTH

def evaluate_once(model, audio_path, tokenizer, device):
    model.eval()
    mel_4d = ensure_mel_4d(audio_path, n_mels=80, target_frames=3000, normalize=True).to(device)
    B, C, T, _ = mel_4d.shape
    T_out = T // model.encoder.conv2.stride[0]
    pos = sinusoids(T_out, 512).T.unsqueeze(0).unsqueeze(-1).to(torch.float32).to(device)
    
    with torch.no_grad():
        s_enc = model.encoder(mel_4d, pos)
        logits = model.ctc_linear(s_enc).squeeze(-1)
    
    results = ctc_prefix_beam_search(
        logits, beam_size=8, alpha=0.6, beta=-0.3,
        blank_id=tokenizer.encoding.n_vocab, tokenizer=tokenizer
    )
    print("\n=== Decoded Text ===")
    for i, r in enumerate(results):
        print(f"[{i}] {r}")
    model.train()

def sinusoids(length: int, channels: int, max_timescale: int = 10000):
    assert channels % 2 == 0
    log_timescale_increment = torch.log(torch.tensor(max_timescale)) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).unsqueeze(1) * inv_timescales.unsqueeze(0)
    out = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    return out


class AudioTextDataset(Dataset):
    """
    reading a TSV manifest: audio_path and transcript
    here each line contains:
    audio_path <TAB> transcript
    """
    def __init__(self, manifest_path: str):
        self.samples: List[Tuple[str, str]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")

                self.samples.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch, tokenizer, n_mels=80, target_frames: int = 3000):
    """
    Produces batch suitable for CTCLoss:
    - mel_4d: (B, n_mels, T, 1)
    - targets: 1D concatenation of token ids
    - target_lengths: lengths per sample
    - raw_waveforms: list of numpy arrays (for teacher input)
    """
    mels = []
    target_tensors = []
    target_lens = []
    waveforms = []

    for audio_path, transcript in batch:
        # 尝试安全加载 mel，若失败则跳过该样本
        try:
            mel_4d = ensure_mel_4d(audio_path, n_mels=n_mels, target_frames=target_frames, normalize=True)
        except Exception as e:
            continue

        # ensure shape (1, n_mels, T, 1)
        try:
            mels.append(mel_4d.squeeze(0))
        except Exception as e:
            continue

        # encode transcript and filter out timestamp/special tokens >= timestamp_begin
        try:
            toks = tokenizer.encode(transcript)
            toks = [t for t in toks if t < tokenizer.encoding.n_vocab]
        except Exception as e:
            mels.pop()
            continue

        target_tensors.extend(toks)
        target_lens.append(len(toks))

        # also provide raw waveform for teacher model if needed
        try:
            wav = load_audio(audio_path)
        except Exception:
            wav = None

        # pad/trim raw waveform to match mel target_frames so teacher processor
        # produces input_features with expected time length (e.g. 3000 frames)
        if wav is not None and target_frames is not None:
            target_samples = int(target_frames * HOP_LENGTH)
            if wav.shape[0] < target_samples:
                wav = np.pad(wav, (0, target_samples - wav.shape[0]))
            elif wav.shape[0] > target_samples:
                wav = wav[:target_samples]

        waveforms.append(wav)

    # 如果本 batch 内没有有效样本，返回 None（训练循环会跳过）
    if len(mels) == 0:
        return None

    # stack mels into (B, n_mels, T, 1)
    mels = torch.stack([m.unsqueeze(0) for m in mels], dim=0).squeeze(1)

    if len(target_tensors) == 0:
        targets = torch.empty(0, dtype=torch.long)
    else:
        targets = torch.tensor(target_tensors, dtype=torch.long)

    target_lens = torch.tensor(target_lens, dtype=torch.long)

    return {
        "mel": mels,  # (B, n_mels, T, 1)
        "targets": targets,
        "target_lens": target_lens,
        "waveforms": waveforms,
    }


def build_model_and_optimizer(tokenizer, device, n_mels=80, n_state=512, n_head=8, n_layer=12, n_ctx=3000, lr: float = 5e-5):
    # derive vocab size from tokenizer to stay consistent with blank id
    vocab_size_teacher = tokenizer.encoding.n_vocab
    vocab_size_student = vocab_size_teacher + 1  # reserve last id for CTC blank

    audio_encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
    student = EncoderCTC(audio_encoder, vocab_size_student, n_state).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    return student, optimizer, vocab_size_student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True,
                        help="TSV manifest: audio_path<TAB>transcript per line")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="student_ctc_kd.pth")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # === 1. Tokenizer ===
    tokenizer = load_french_tokenizer()

    # === 2. Dataset & collate factory ===
    ds = AudioTextDataset(args.manifest)
    collate = lambda b: collate_fn(b, tokenizer, n_mels=80, target_frames=3000)

    # === 3. Build student model & optimizer ===
    student, optimizer, vocab_size_student = build_model_and_optimizer(
        tokenizer, device,
        n_mels=80, n_state=512, n_head=8, n_layer=12, n_ctx=3000,
        lr=args.lr,
    )
    ctc_loss_fn = nn.CTCLoss(blank=tokenizer.encoding.n_vocab, zero_infinity=True)

    # === 4. Load Whisper teacher encoder (feature-level KD) ===
    from transformers import WhisperProcessor, WhisperModel
    print("[INFO] Loading Whisper teacher encoder (openai/whisper-base)...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    teacher = WhisperModel.from_pretrained("openai/whisper-base").to(device)
    teacher.eval()

    # KD parameters (feature-level distillation)
    lambda_kd = 0.3
    kd_loss_fn = nn.MSELoss()
    teacher_hidden = teacher.config.d_model
    # projector: map teacher hidden dim -> student n_state (512)
    projector = nn.Linear(teacher_hidden, 512).to(device)
    # add projector params to optimizer so projector gets updated
    optimizer.add_param_group({'params': projector.parameters()})

    # === 修改点 ===
    # 可自定义的路径
    resume_ckpt = os.path.join(os.path.dirname(args.save), "student_ctc_kd.pth.epoch2")

    # 训练超参数
    sample_fraction = 0.05  # 使用数据集的10%进行训练
    use_amp = False  # 混合精度

    # === 加载 checkpoint ===
    if os.path.exists(resume_ckpt):
        print(f"[INFO] Loading checkpoint: {resume_ckpt}")
        sd = torch.load(resume_ckpt, map_location=device)
        student.load_state_dict(sd, strict=False)
    else:
        print("[INFO] No checkpoint found, training from scratch.")

    # === 数据集加载 ===
    total_len = len(ds)
    subset_len = int(total_len * sample_fraction)

    # 只选择 manifest 中磁盘上实际存在的音频路径，尽量满足期望的样本数量
    available_indices = [i for i, (p, _) in enumerate(ds.samples) if os.path.exists(p)]
    if len(available_indices) >= subset_len:
        subset_indices = random.sample(available_indices, subset_len)
    else:
        subset_indices = available_indices

    train_dataset = torch.utils.data.Subset(ds, subset_indices)
    loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=8, pin_memory=True)

    print(f"[INFO] Using {len(subset_indices)}/{total_len} samples ({(len(subset_indices)/total_len)*100:.1f}% of dataset).")

    # === 混合精度训练支持 ===
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # === 训练循环 ===
    for epoch in range(3, args.epochs):  # 从第3轮开始
        # 动态KD：指数衰减
        base_kd = 0.1
        decay = 0.9 ** (epoch)
        lambda_kd_epoch = max(0.05, base_kd * decay)
        print(f"[INFO] Epoch {epoch} KD weight = {lambda_kd_epoch:.3f}")
        # 用来统计这一轮的blank平均值
        epoch_blank_accum = 0.0
        epoch_step = 0

        student.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", mininterval=20)

        for batch in pbar:
            # collate_fn may return None when all samples in this batch were invalid
            if batch is None:
                # simply skip this batch
                pbar.set_postfix({"skipped": "empty_batch"})
                continue

            mel = batch["mel"].to(device)  # (B, n_mels, T, 1)
            targets = batch["targets"].to(device)
            target_lens = batch["target_lens"].to(device)
            B, _, T, _ = mel.shape

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                # positional embedding: compute T_out from student's encoder conv2 stride
                try:
                    stride_time = student.encoder.conv2.stride[0]
                    T_out = T // stride_time
                except Exception:
                    T_out = T // 2
                pos = sinusoids(T_out, 512)
                pos_emb = pos.T.unsqueeze(0).unsqueeze(-1).to(torch.float32).to(device)

                # === Student forward ===
                # get encoder features from student, then apply ctc linear to get logits
                s_enc = student.encoder(mel, pos_emb)  # expect (B, n_state, T_out, 1)
                if not torch.isfinite(s_enc).all():
                    print(f"[WARN] NaN detected in encoder output! max={s_enc.max().item()} min={s_enc.min().item()}")
                    break

                student_logits = student.ctc_linear(s_enc).squeeze(-1)  # (B, V, T_out)
                # 检查数值
                if not torch.isfinite(student_logits).all():
                    print("⚠️ NaN detected in logits! max/min:", student_logits.max().item(), student_logits.min().item())
                    break

                # --- ① 计算 blank 概率 ---
                blank_index = tokenizer.encoding.n_vocab  # 通常 CTC 的 blank 是最后一个 id
                log_probs = F.log_softmax(student_logits.squeeze(-1), dim=1)  # (B, vocab_size, T)
                blank_prob = torch.exp(log_probs[:, blank_index, :])  # (B, T)

                # 计算平均 blank 概率
                blank_ratio = blank_prob.mean()

                # 动态目标比例
                target_blank = 0.99 - 0.05 * (epoch / args.epochs)

                # 惩罚项（可导）
                blank_penalty = torch.clamp(blank_ratio - target_blank, min=0)

                # 空白惩罚 loss
                loss_blank = 2.0 * blank_penalty

                # --- ② 原来的 CTC 部分保持 ---
                log_probs = F.log_softmax(student_logits.permute(2, 0, 1), dim=-1)
                input_lengths = torch.full((B,), T_out, dtype=torch.long, device=device)

                # === CTC loss ===
                if targets.numel() == 0:
                    # no labeled targets in this batch -> zero CTC loss
                    loss_ctc = torch.tensor(0.0, device=device)
                else:
                    loss_ctc = ctc_loss_fn(log_probs, targets, input_lengths, target_lens)

                # === Teacher encoder feature distillation (MSE) ===
                raw_wavs = batch.get("waveforms", None)
                loss_kd = torch.tensor(0.0, device=device)

                if raw_wavs is not None and any([w is not None for w in raw_wavs]):
                    valid_wavs = [w for w in raw_wavs if w is not None]
                    with torch.no_grad():
                        proc_inputs = processor(valid_wavs, sampling_rate=16000, return_tensors="pt", padding=True)
                        input_features = proc_inputs.input_features.to(device)
                        teacher_out = teacher.encoder(input_features=input_features, return_dict=True)
                        t_enc = teacher_out.last_hidden_state  # (B, L_t, teacher_hidden)

                    # interpolate teacher encoder time axis to student T_out
                    t_enc = t_enc.permute(0, 2, 1)  # (B, teacher_hidden, L_t)
                    if t_enc.shape[-1] != T_out:
                        t_enc = F.interpolate(t_enc, size=T_out, mode="linear", align_corners=False)
                    t_enc = t_enc.permute(0, 2, 1)  # (B, T_out, teacher_hidden)

                    # project teacher features to student dim and compute MSE with student encoder outputs
                    t_proj = projector(t_enc)  # (B, T_out, n_state)
                    # normalize student encoder features to (B, T_out, n_state)
                    if s_enc is not None:
                        if s_enc.dim() == 4:
                            s_feats = s_enc.squeeze(-1).permute(0, 2, 1)  # (B, T_out, n_state)
                        else:
                            s_feats = s_enc.permute(0, 2, 1)
                        loss_kd = kd_loss_fn(s_feats, t_proj)

            # --- ④ 总损失（用epoch内的KD权重）---
            loss = loss_ctc + lambda_kd_epoch * loss_kd + loss_blank

            # 检查 loss 是否为正常数值
            if not torch.isfinite(loss):
                print(f"[WARN] NaN loss detected at epoch {epoch}, step {pbar.n}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # === 自动检测数值异常 ===
            with torch.autograd.detect_anomaly():
                scaler.scale(loss).backward()

            # === 混合精度梯度缩放 ===
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

            # === 进行优化器更新 ===
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({
                "total": f"{loss.item():.4f}",
                "ctc": f"{loss_ctc.item():.4f}",
                "kd": f"{loss_kd:.4f}" if isinstance(loss_kd, float) else f"{loss_kd.item():.4f}",
                "blank": f"{blank_ratio.item():.3f}"
            })

        save_path = f"{args.save}.epoch{epoch}"
        torch.save(student.state_dict(), save_path)
        print(f"[INFO] Saved checkpoint: {save_path}")

        # 评估模型
        eval_audio_path = "test_audio.mp3"
        if os.path.exists(eval_audio_path):
            evaluate_once(student, eval_audio_path, tokenizer, device)
        else:
            print("[WARN] eval audio not found, skip evaluation.")

        if epoch_step > 0:
            avg_blank = epoch_blank_accum / epoch_step
            print(f"[INFO] Epoch {epoch} avg blank ratio: {avg_blank:.3f}")

        epoch_blank_accum += blank_ratio.item()
        epoch_step += 1

    torch.save(student.state_dict(), args.save)
    print(f"[INFO] Training complete. Saved model to {args.save}")


if __name__ == "__main__":
    main()
