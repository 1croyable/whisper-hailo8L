import torch
from model import AudioEncoder, EncoderCTC
import numpy as np

n_mels = 80
n_ctx = 3000
n_state = 128
n_head = 8
n_layer = 2
vocab_size = 1000

def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

# 初始化模型
audio_encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
model = EncoderCTC(audio_encoder, vocab_size, n_state)
model.eval()  # 🔹重要！

# 创建输入
batch_size = 1
mel = torch.randn(batch_size, n_mels, n_ctx)
pos_emb = sinusoids(n_ctx // 2, n_state).unsqueeze(0).to(torch.float32)

# 导出
torch.onnx.export(
    model, 
    (mel, pos_emb),
    "whisper-hailo8l.onnx",
    input_names=["mel", "pos_emb"],
    output_names=["logits"],
    dynamic_axes=None,
    opset_version=13,
    do_constant_folding=True
)

print("✅ 模型已成功导出为 whisper-hailo8l.onnx")
