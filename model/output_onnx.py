import torch
from model import AudioEncoder, EncoderCTC, Linear, LayerNorm, ResidualAttentionBlock
import numpy as np
import torch.nn.functional as F
import onnx
import onnx.numpy_helper as numpy_helper


n_mels = 80
n_ctx = 3000
n_state = 128
n_head = 8
n_layer = 2
vocab_size = 1000


def sinusoids(length, channels, max_timescale=10000):
    """生成正弦位置编码，输出 (channels, length)"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# 初始化模型
audio_encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)
model = EncoderCTC(audio_encoder, vocab_size, n_state)
model.eval()

# 创建输入
batch_size = 1
mel = torch.randn(batch_size, n_mels, n_ctx, 1)

# 计算经过conv层后的时间维长度（T_out）
with torch.no_grad():
    x_conv = F.silu(audio_encoder.conv1(mel))
    x_conv = F.silu(audio_encoder.conv2(x_conv))
    T_out = x_conv.shape[2]

# 创建与 encoder 输出匹配的 4D 位置编码，shape: (1, n_state, T_out, 1)
pos_emb = sinusoids(T_out, n_state).T.unsqueeze(0).unsqueeze(-1).to(torch.float32)

# 导出
torch.onnx.export(
    model,
    (mel, pos_emb),
    "whisper-hailo8l.onnx",
    input_names=["mel", "pos_emb"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes=None,
    do_constant_folding=True,
)

print("✅ 模型已成功导出为 whisper-hailo8l.onnx")

# 尝试将任何 external data 内嵌回单文件 ONNX
try:
    print('检查并内嵌 external data 到单一 ONNX 文件...')
    m = onnx.load('whisper-hailo8l.onnx', load_external_data=True)
    new_inits = []
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        new_inits.append(numpy_helper.from_array(arr, name=init.name))

    # 将 pos_emb 作为 initializer 内嵌进 ONNX
    try:
        pos_arr = pos_emb.cpu().numpy()
        pos_init = numpy_helper.from_array(pos_arr, name='pos_emb')
        new_inits = [n for n in new_inits if n.name != 'pos_emb']
        new_inits.append(pos_init)

        # 从 graph 输入中移除 pos_emb
        inputs = [i for i in m.graph.input if i.name != 'pos_emb']
        m.graph.ClearField('input')
        m.graph.input.extend(inputs)

        print('已将 pos_emb 嵌入为 initializer 并从 graph.input 中移除')
    except Exception as ex:
        print('将 pos_emb 内嵌为 initializer 失败:', ex)

    m.graph.ClearField('initializer')
    m.graph.initializer.extend(new_inits)
    onnx.save(m, 'whisper-hailo8l.onnx')
    print('已将 external data 内嵌，输出为单一文件 whisper-hailo8l.onnx')

except Exception as e:
    print('无法将 external data 内嵌为单文件 ONNX：', e)
