import torch
from model import AudioEncoder, EncoderCTC


# === 模型超参数（要与训练时一致）===
n_mels = 80
n_state = 512
n_head = 8
n_layer = 12
n_ctx = 3000
vocab_size = 51865 + 1  # vocab + blank


# === 1. 构建模型结构 ===
audio_encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
model = EncoderCTC(audio_encoder, vocab_size, n_state)
model.eval()


# === 2. 加载训练好的权重 ===
weights_path = "model/student_ctc_kd.pth.epoch0"
sd = torch.load(weights_path, map_location="cpu")
model.load_state_dict(sd, strict=False)
print(f"[INFO] Loaded weights from {weights_path}")


# === 3. 创建一个示例输入 ===
dummy_mel = torch.randn(1, n_mels, n_ctx, 1, dtype=torch.float32)
pos_emb = torch.randn(1, n_state, n_ctx // 2, 1, dtype=torch.float32)  # 模拟位置编码


# === 4. 导出为 ONNX ===
onnx_path = "whisper-hailo8l-trained0.onnx"
torch.onnx.export(
    model,
    (dummy_mel, pos_emb),
    onnx_path,
    input_names=["mel", "pos_emb"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes=None,
)
print(f"[SUCCESS] Exported to {onnx_path}")
