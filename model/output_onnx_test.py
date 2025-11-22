import torch
import torch.nn.functional as F
import onnx
import numpy as np
from model.whisper_hailo_model import WhisperHailoModel, ModelDimensions

dem = ModelDimensions()
dem.n_mels = 80
dem.n_audio_ctx = 1000
dem.n_audio_state = 768
dem.n_audio_head = 8
dem.n_audio_layer = 12
dem.n_vocab = 51865
dem.n_text_ctx = 500
dem.n_text_state = 768
dem.n_text_head = 8
dem.n_text_layer = 6

model = WhisperHailoModel(dem)
model.eval()

dummy_mel = torch.randn(1, dem.n_mels, dem.n_audio_ctx, 1, dtype=torch.float32)
# 调整 onehot 的形状以匹配 Conv2d 的输入要求
dummy_tokens = torch.randint(0, dem.n_vocab, (1, dem.n_text_ctx), dtype=torch.long)
dummy_onehot = F.one_hot(dummy_tokens, num_classes=dem.n_vocab).float()  # (1, T, V)
dummy_onehot = dummy_onehot.permute(0, 2, 1).unsqueeze(-1)  # (1, V, T, 1)

n_layer = dem.n_text_layer
n_state = dem.n_text_state
alphas = torch.full((n_layer, n_state), 0.9, dtype=torch.float32)
betas  = torch.full((n_layer, n_state), 0.5, dtype=torch.float32)
thetas = torch.linspace(0, np.pi / 4, n_state, dtype=torch.float32).repeat(n_layer, 1)

model.decoder.set_ssm_parameters(alphas, betas, thetas)

encoder_onnx_path = "whisper-hailo8l-encoder.onnx"
torch.onnx.export(
    model.encoder,
    dummy_mel,
    encoder_onnx_path,
    input_names=["mel"],
    output_names=["audio_features"],
    dynamic_axes=None,
    do_constant_folding=True,
    keep_initializers_as_inputs=False,
    opset_version=12
)
print(f"[SUCCESS] Exported encoder to {encoder_onnx_path}")

# decoder_onnx_path = "whisper-hailo8l-decoder.onnx"
# torch.onnx.export(
#     model.decoder,
#     (dummy_onehot, model.encoder(dummy_mel)),
#     decoder_onnx_path,
#     input_names=["onehot", "xa"],
#     output_names=["decoder_output"],  # 用 graph.output 名
#     dynamic_axes=None,
#     do_constant_folding=True,
#     keep_initializers_as_inputs=False,
#     opset_version=14,
#     training=torch.onnx.TrainingMode.PRESERVE
# )
# print("[SUCCESS] Exported decoder to", decoder_onnx_path)