import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import whisper
import os
import random
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

# 确保你的 model/whisper_hailo_model.py 已经是那个 CNN (Conv2d) 版本
from model.whisper_hailo_model import AudioEncoder as StudentAudioEncoder
from model.mel import log_mel_spectrogram, pad_or_trim

class SpecAugment(nn.Module):
    """
    对 Mel 谱图进行时间和频率的随机掩码。
    """
    def __init__(self, freq_mask_param=15, time_mask_param=70, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, mel):
        # mel shape: (B, n_mels, T)
        mel_aug = mel.clone()
        B, n_mels, T = mel_aug.shape
        
        # 频率掩码
        for _ in range(self.n_freq_masks):
            f = int(torch.rand(1) * self.freq_mask_param)
            f0 = int(torch.rand(1) * (n_mels - f))
            mel_aug[:, f0:f0+f, :] = 0

        # 时间掩码
        for _ in range(self.n_time_masks):
            t = int(torch.rand(1) * self.time_mask_param)
            t0 = int(torch.rand(1) * (T - t))
            mel_aug[:, :, t0:t0+t] = 0
            
        return mel_aug

class FolderSpeechDataset(Dataset):
    def __init__(
        self,
        audio_root: str,
        n_mels: int = 80,
        target_frames: int = 3000,
        max_samples: int = 100000,
    ):
        self.audio_root = Path(audio_root)
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.max_samples = max_samples

        if not self.audio_root.exists():
            raise FileNotFoundError(f"Audio root not found: {self.audio_root}")

        print(f"[DATA] Scanning audio files in {self.audio_root}...")
        # 使用 os.scandir 加速文件扫描，避免 glob 卡死
        self.all_files = []
        extensions = {'.mp3', '.wav', '.flac'}
        # 递归扫描
        for root, dirs, files in os.walk(self.audio_root):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    self.all_files.append(os.path.join(root, file))
                    # 如果只需要大概数量，可以扫够了就停，防止几十万文件扫太久
                    # if len(self.all_files) > max_samples * 2: break 
        
        if not self.all_files:
            raise FileNotFoundError(f"No audio files found in: {self.audio_root}")

        print(f"[DATA] Total audio files found: {len(self.all_files)}")

        # 初始化子集
        self.shuffle_subset()

    def shuffle_subset(self):
        """随机选择一部分文件作为训练集"""
        # 如果文件总数少于 max_samples，就用全部
        current_max = min(len(self.all_files), self.max_samples)
        
        print(f"[DATA] Sampling {current_max} files for this epoch...")
        self.files = random.sample(self.all_files, current_max)
        print(f"[DATA] Subset ready.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        attempts = 0
        while attempts < 5:
            try:
                audio_path = self.files[idx]

                # 1. 提取 Mel 谱图
                mel = log_mel_spectrogram(audio_path, n_mels=self.n_mels, padding=0)

                # 2. 统一长度
                mel = pad_or_trim(mel, length=self.target_frames)  # (80, 3000)

                # 3. 转换为 4D 张量 (B, C, T, 1) 适配 Conv2d
                mel_tensor = mel.unsqueeze(-1)

                return mel, mel_tensor

            except Exception as e:
                # print(f"[WARN] Error loading {self.files[idx]}: {e}") # 调试时可打开
                attempts += 1
                idx = random.randint(0, len(self.files) - 1)

        # 5次失败返回全0，避免崩溃
        return torch.zeros((80, self.target_frames)), torch.zeros((80, self.target_frames, 1))

# 训练主流程
def train_kd():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-iter", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default="train/checkpoints_medium/student")
    parser.add_argument("--teacher-model", default="base")
    parser.add_argument("--audio-root", default="/workspace/train/cv-corpus-22.0-2025-06-20/fr/clips/")
    parser.add_argument("--max-samples", type=int, default=10000)
    args = parser.parse_args()

    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[INFO] Device: {device}")

    # --- 1. 加载 Teacher (Whisper) ---
    print(f"[INFO] Loading Teacher: Whisper-{args.teacher_model}")
    teacher = whisper.load_model(args.teacher_model, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_dims = teacher.dims.n_audio_state
    print(f"[INFO] Teacher dimension: {teacher_dims}")

    # --- 2. 配置 Student (1024维, 24层) ---
    N_MELS = 80
    N_STATE = 512
    N_LAYER = 6

    print(f"[INFO] Building Student: Layers={N_LAYER}, State={N_STATE}")
    student = StudentAudioEncoder(
        n_mels=N_MELS,
        n_state=N_STATE,
        n_layer=N_LAYER
    ).to(device)

    # === [新增] 加载 epoch 1 的权重 ===
    # 注意：根据你的 ls 输出，路径应该是相对当前脚本的
    # ckpt_path = "train/checkpoints_medium/student_ep1.pth"
    # if os.path.exists(ckpt_path):
    #     print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    #     student.load_state_dict(torch.load(ckpt_path, map_location=device))
    # else:
    #     print(f"[WARN] Checkpoint {ckpt_path} not found! Starting from random init.")

    need_projector = (N_STATE != teacher_dims)
    projector = None
    if need_projector:
        print(f"[INFO] Adding Projector: {N_STATE} -> {teacher_dims}")
        projector = nn.Conv1d(N_STATE, teacher_dims, kernel_size=1).to(device)
        trainable_params = list(student.parameters()) + list(projector.parameters())
    else:
        print(f"[INFO] Dimensions match! No Projector needed.")
        trainable_params = student.parameters()

    optimizer = optim.AdamW(trainable_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler()

    augmenter = SpecAugment(freq_mask_param=15, time_mask_param=70, n_freq_masks=2, n_time_masks=2).to(device)

    dataset = FolderSpeechDataset(
        audio_root=args.audio_root,
        n_mels=N_MELS,
        target_frames=3000,
        max_samples=args.max_samples,
    )

    # --- 6. 训练循环 ---
    start_epoch = 0
    target_epoch = 5
    
    for epoch in range(start_epoch, target_epoch):
        print(f"\n===== EPOCH {epoch + 1}/{target_epoch} =====")
        # dataset.shuffle_subset()

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        student.train()
        if projector: projector.train()

        epoch_loss = 0
        total_batches = 0

        optimizer.zero_grad()  # 确保开始前梯度清零

        pbar = tqdm(loader, unit="batch")
        for batch_idx, (mel_teacher_batch, mel_student_4d) in enumerate(pbar):
            mel_clean = mel_teacher_batch.to(device)
            mel_student_in = mel_student_4d.to(device)

            with torch.no_grad():
                teacher_out = teacher.encoder(mel_clean)

            # mel_masked = augmenter(mel_student_in.squeeze(-1)).unsqueeze(-1)
            mel_masked = mel_student_in

            with torch.amp.autocast(device_type="cuda"):
                student_out_4d = student(mel_masked)
                student_out = student_out_4d.squeeze(-1)

                if projector:
                    student_out_proj = projector(student_out)
                    student_final = student_out_proj.permute(0, 2, 1)
                else:
                    student_final = student_out.permute(0, 2, 1)

                loss = criterion(student_final, teacher_out)
                loss = loss / args.accum_iter  # Loss 要除以累积步数

            scaler.scale(loss).backward()  # 只反向传播，不立即 step

            if ((batch_idx + 1) % args.accum_iter == 0) or (batch_idx + 1 == len(loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # 更新完才清零

            epoch_loss += loss.item() * args.accum_iter  # 还原数值用于统计
            total_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * args.accum_iter:.4f}")

        avg_loss = epoch_loss / total_batches if total_batches > 0 else 0
        print(f"[EPOCH {epoch+1}] Avg Loss = {avg_loss:.5f}")

        scheduler.step(avg_loss)

        # 保存权重
        torch.save(student.state_dict(), f"{args.save_path}_ep{epoch+1}.pth")

    print("[INFO] Training Finished!")

if __name__ == "__main__":
    train_kd()