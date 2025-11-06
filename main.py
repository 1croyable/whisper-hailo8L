import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

# 和 train.py 一样的导入方式
from model import ensure_mel_4d, AudioEncoder, EncoderCTC, load_french_tokenizer
from model.ctc_decoder import ctc_prefix_beam_search
from model.mel import HOP_LENGTH  # 不一定用得上，但留着没坏处


def sinusoids(length: int, channels: int, max_timescale: int = 10000):
    """
    和 train.py 里的一模一样：生成正弦位置编码
    """
    assert channels % 2 == 0
    log_timescale_increment = torch.log(torch.tensor(max_timescale)) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length).unsqueeze(1) * inv_timescales.unsqueeze(0)
    out = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    return out


def build_student_from_ckpt(weights_path: Path,
                            tokenizer,
                            device,
                            n_mels=80,
                            n_state=512,
                            n_head=8,
                            n_layer=12,
                            n_ctx=3000):
    """
    跟训练时的构建方式保持一致
    vocab = whisper-vocab + 1(blank)
    """
    vocab_size_student = tokenizer.encoding.n_vocab + 1

    audio_encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
    student = EncoderCTC(audio_encoder, vocab_size_student, n_state).to(device)
    student.eval()

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        sys.exit(1)

    sd = torch.load(str(weights_path), map_location=device)
    # 训练时是 student.state_dict() 存的，所以这里可以 strict=False 防一下
    student.load_state_dict(sd, strict=False)
    print(f"[INFO] Loaded weights from {weights_path}")
    return student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="C:\\Users\\jerem\\Downloads\\cv-corpus-22.0-2025-06-20-fr\\cv-corpus-22.0-2025-06-20\\fr\\clips\\common_voice_fr_39577833.mp3")
    parser.add_argument("--weights", default="model/student_ctc_kd.pth.epoch4", help=".pth from training")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beam", type=int, default=8)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"[INFO] using device: {device}")

    # 1) tokenizer —— 一定要跟训练时的一样
    tokenizer = load_french_tokenizer()
    blank_id = tokenizer.encoding.n_vocab  # 训练里就是这么定义的

    # 2) 构建模型并加载权重
    student = build_student_from_ckpt(
        Path(args.weights),
        tokenizer,
        device,
        n_mels=80,
        n_state=512,
        n_head=8,
        n_layer=12,
        n_ctx=3000,
    )

    # 3) 做和训练时一致的 mel 前处理
    #    训练里 collate_fn 调的是：
    #    ensure_mel_4d(audio_path, n_mels=80, target_frames=3000, normalize=True)
    print("[INFO] computing mel...")
    mel_4d = ensure_mel_4d(
        str(audio_path),
        n_mels=80,
        target_frames=3000,
        normalize=True,
    )  # -> (1, 80, T, 1)
    mel_4d = mel_4d.to(device)

    # 4) 按训练代码的方式，算 student encoder 的 time 维度，然后做位置编码
    B, C, T, _ = mel_4d.shape
    # train.py 里是这么写的：
    # stride_time = student.encoder.conv2.stride[0]; T_out = T // stride_time
    try:
        stride_time = student.encoder.conv2.stride[0]
        T_out = T // stride_time
    except Exception:
        # 防御一下
        T_out = T // 2

    pos = sinusoids(T_out, 512)              # (T_out, 512)
    pos_emb = pos.T.unsqueeze(0).unsqueeze(-1).to(torch.float32).to(device)  # (1, 512, T_out, 1)

    # 5) 前向 + CTC 解码
    with torch.no_grad():
        # 训练里是：
        # s_enc = student.encoder(mel, pos_emb)  # (B, n_state, T_out, 1)
        # student_logits = student.ctc_linear(s_enc).squeeze(-1)  # (B, V, T_out)
        s_enc = student.encoder(mel_4d, pos_emb)               # (B, 512, T_out, 1)
        logits = student.ctc_linear(s_enc).squeeze(-1)         # (B, V, T_out)

    # logits 现在是 (1, V, T_out)
    # 你的 ctc_prefix_beam_search 函数应该能直接吃这个；如果它要 (T, V) 就转一下
    print("[INFO] decoding with CTC prefix beam search...")
    results = ctc_prefix_beam_search(
        logits,                  # (B, V, T)
        beam_size=args.beam,
        alpha=0.6,
        beta=-0.3,
        blank_id=blank_id,
        tokenizer=tokenizer,
    )

    print("\n=== decoded text ===")
    # 我不知道你的 ctc_prefix_beam_search 返回的是 list[str] 还是 list[list[int]]
    # 按你之前 main 的写法，应该已经是字符串了
    if isinstance(results, (list, tuple)):
        for i, r in enumerate(results):
            print(f"[{i}] {r}")
    else:
        print(results)


if __name__ == "__main__":
    main()
