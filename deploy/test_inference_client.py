import grpc
import numpy as np
import soundfile as sf

from model.mel import log_mel_spectrogram, pad_or_trim
from deploy import encoder_pb2
from deploy import encoder_pb2_grpc

import torch

SERVER_ADDR = "192.168.16.108:50051"
N_MELS = 80
N_FRAMES = 1000

def load_audio_to_mel(path: str):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    wav = wav.astype(np.float32)

    mel = log_mel_spectrogram(
        wav,
        n_mels=N_MELS,
        padding=0
    )

    mel = pad_or_trim(mel, length=N_FRAMES, axis=-1)
    mel = mel.to(torch.float32)
    mel_4d = mel[..., None]
    return mel_4d


# gRPC Client
def grpc_infer(mel4d: np.ndarray):
    # mel4d shape: (80,1000,1)
    mel_bytes = mel4d.numpy().tobytes()
    channel = grpc.insecure_channel(SERVER_ADDR)
    stub = encoder_pb2_grpc.EncoderStub(channel)

    request = encoder_pb2.DecodeRequest(
        mel=mel_bytes,
        batch_size=1
    )

    response = stub.EncodeAndDecode(request)

    return response.text

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str)
    args = parser.parse_args()

    print(f"Loading audio: {args.audio_path}")
    mel4d = load_audio_to_mel(args.audio_path)
    print(f"mel4d shape = {mel4d.shape}")

    print("Sending to Raspberry Pi encoder+decoder service...")
    text = grpc_infer(mel4d)

    print("\n==========================")
    print("   TRANSCRIPTION RESULT")
    print("==========================\n")
    print(text)
    print("\n==========================\n")
