# model-test.py
import torch
import whisper
from model.whisper_hailo_model import AudioEncoder
from model.mel import log_mel_spectrogram, pad_or_trim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 加载你刚刚训练好的第5轮权重
CHECKPOINT_PATH = "train/checkpoints_medium/student_ep5.pth" 
AUDIO_PATH = "train/cv-corpus-22.0-2025-06-20/fr/clips/common_voice_fr_19812198.mp3"

# 1. 初始化 Student (512维, 12层)
student = AudioEncoder(n_mels=80, n_state=512, n_layer=12).to(DEVICE)
student.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
student.eval()

# 2. 加载 Base Decoder (对应 512 维)
decoder_model = whisper.load_model("base", device=DEVICE)
decoder = decoder_model.decoder

# 3. 预处理
mel = log_mel_spectrogram(AUDIO_PATH, n_mels=80, padding=0)
mel = pad_or_trim(mel, length=3000).unsqueeze(0).unsqueeze(-1).to(DEVICE)

# 4. 推理
with torch.no_grad():
    audio_features = student(mel).squeeze(-1).permute(0, 2, 1) # (1, 1500, 512)
    
    # 打印前10个数值，看看是不是乱码
    print(f"Features mean: {audio_features.mean().item():.4f}, std: {audio_features.std().item():.4f}")
    
    # 解码
    options = whisper.DecodingOptions(language="fr", without_timestamps=True)
    result = decoder_model.decode(audio_features, options)

print(f"\n识别结果: {result}")