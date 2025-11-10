import torch
import torch.nn.functional as F
from model.whisper_hailo_model import WhisperHailoModel, ModelDimensions

# === 1. 构建模型结构 ===
dem = ModelDimensions()
dem.n_mels = 80
dem.n_audio_ctx = 3000
dem.n_audio_state = 512
dem.n_audio_head = 8
dem.n_audio_layer = 12
dem.n_vocab = 51865
dem.n_text_ctx = 448
dem.n_text_state = 512
dem.n_text_head = 8
dem.n_text_layer = 12

model = WhisperHailoModel(dem)
model.eval()

# === 2. （可选）加载权重 ===
# weights_path = "model/student_ctc_kd.pth.epoch0"
# sd = torch.load(weights_path, map_location="cpu")
# model.load_state_dict(sd, strict=False)
# print(f"[INFO] Loaded weights from {weights_path}")

# === 3. 创建示例输入 ===
dummy_mel = torch.randn(1, dem.n_mels, dem.n_audio_ctx, 1, dtype=torch.float32)

# tokens -> onehot
dummy_tokens = torch.randint(0, dem.n_vocab, (1, dem.n_text_ctx), dtype=torch.long)
dummy_onehot = F.one_hot(dummy_tokens, num_classes=dem.n_vocab).float()  # (1,T,V)
dummy_onehot = dummy_onehot.transpose(1, 2).unsqueeze(-1)  # (1,V,T,1)

# === 4. 导出为 ONNX ===
onnx_path = "whisper-hailo8l-initial.onnx"
torch.onnx.export(
    model,
    (dummy_mel, dummy_onehot),
    onnx_path,
    input_names=["mel", "onehot"],
    output_names=["logits"],
    dynamic_axes=None,
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
    opset_version=12
)

print(f"[SUCCESS] Exported to {onnx_path}")
