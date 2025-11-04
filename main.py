import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

from model.mel import ensure_mel_4d
from model.whisper_hailo_model import AudioEncoder, EncoderCTC
from model.ctc_decoder import ctc_prefix_beam_search, load_french_tokenizer

def sinusoids(length: int, channels: int, max_timescale: int = 10000):
    """生成正弦位置编码，返回 shape (length, channels)"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    out = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--duration", type=float, default=None, help="Pad/trim audio to this many seconds (optional)")
    parser.add_argument("--weights", type=str, default=None, help="Optional model weights (.pth)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        sys.exit(1)

    # hyperparameters
    n_mels = 80
    n_state = 512
    n_head = 8
    n_layer = 12
    n_ctx = 3000
    vocab_size = 51865 + 1 # CTC blank token

    # 2) mel
    print("[2/5] Preprocessing audio -> mel")
    mel_4d = ensure_mel_4d(
        str(audio_path),
        duration=args.duration,
        sample_rate=16000,
        n_mels=n_mels,
        padding=0,
        target_frames=n_ctx,
        normalize=True,
    )

    # 3) 构建模型
    print(f"[3/5] Building model with n_ctx={n_ctx}")
    audio_encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
    model = EncoderCTC(audio_encoder, vocab_size, n_state)
    model.eval()

    # 加载权重
    if args.weights:
        wpath = Path(args.weights)
        if wpath.exists():
            print(f"[INFO] Loading weights from {wpath}")
            sd = torch.load(str(wpath), map_location="cpu")
            model.load_state_dict(sd, strict=False)
        else:
            print(f"[WARN] Weights file not found: {wpath}")

    # 4) 位置编码
    print("[4/5] Generating positional embedding")
    with torch.no_grad():
        # 经过两次卷积层后，时间维缩小4倍
        T_out = mel_4d.shape[2] // (audio_encoder.conv_stride ** 2)
    pos = sinusoids(T_out, n_state)
    pos_emb = pos.T.unsqueeze(0).unsqueeze(-1).to(torch.float32)

    # 5) 前向
    print("[5/5] Running model forward and decoding")
    with torch.no_grad():
        logits = model(mel_4d, pos_emb)  # (B, V, T, 1)

    # 解码输出
    tokenizer = load_french_tokenizer()
    blank_id = tokenizer.encoding.n_vocab
    outputs = ctc_prefix_beam_search(
    logits,
    beam_size=8,
    alpha=0.6,
    beta=-0.3,
    blank_id=blank_id,
    tokenizer=tokenizer)

    print("\n--- Decoded output ---")
    for i, out in enumerate(outputs):
        print(f"[{i}] {out}")


if __name__ == "__main__":
    main()
