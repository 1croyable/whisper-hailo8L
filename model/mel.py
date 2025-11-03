import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# minimal, hard-coded audio hyperparameters used by the repo
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary. Requires the ffmpeg CLI in PATH.
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int, *, axis: int = -1):
    """
    Pad or trim an array (numpy or torch tensor) along `axis` to `length`.
    Simple utility used to fix waveform length when required.
    """
    if torch.is_tensor(array):
        cur = array.shape[axis]
        if cur > length:
            return array.index_select(dim=axis, index=torch.arange(length, device=array.device))
        if cur < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - cur)
            return F.pad(array, [p for sizes in pad_widths[::-1] for p in sizes])
        return array

    # numpy
    cur = array.shape[axis]
    if cur > length:
        return array.take(indices=range(length), axis=axis)
    if cur < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - cur)
        return np.pad(array, pad_widths)
    return array


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        # always keep filters on CPU (this repo only targets CPU)
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(torch.float32)


@lru_cache(maxsize=1)
def hann_window_cached() -> torch.Tensor:
    return torch.hann_window(N_FFT)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    # accept file path / numpy array / tensor
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if padding > 0:
        audio = F.pad(audio, (0, padding))

    # windowed STFT -> magnitude (all on CPU)
    window = hann_window_cached()
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    # follow the original clipping & scaling used by upstream
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def ensure_mel_4d(
    mel_input: Union[str, np.ndarray, torch.Tensor],
    *,
    duration: Optional[float] = None,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = 80,
    padding: int = 0,
    target_frames: Optional[int] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    将输入统一为模型期望的 4D Tensor: (B, n_mels, T, 1)

    支持输入类型：
      - str: 音频文件路径（使用 load_audio -> 可选 pad_or_trim -> log_mel_spectrogram）
      - np.ndarray / torch.Tensor:
          * waveform: 1D (samples,) 或 (1, samples) -> 转为 log-mel
          * mel: (n_mels, T) 或 (B, n_mels, T) -> 直接扩维

    参数:
      duration: 如果指定（秒），对 waveform 做 pad_or_trim 到该时长（以 sample_rate 计算）
      padding: 转 log-mel 时右侧额外 zero-padding（传给 log_mel_spectrogram）
      device: 若需要，将音频或 mel 移到指定 device
      normalize: 如果 True，对 mel 做简单的 z-score 标准化 (全局)

    返回:
      torch.FloatTensor, shape = (B, n_mels, T, 1)
    """
    # helper: ensure torch tensor on CPU/device
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    # load from file path -> numpy waveform
    if isinstance(mel_input, str):
        audio = load_audio(mel_input, sr=sample_rate)
        audio = to_tensor(audio)
        # pad/trim to duration if requested
        if duration is not None:
            target_samples = int(duration * sample_rate)
            audio = pad_or_trim(audio, length=target_samples, axis=-1)
        mel = log_mel_spectrogram(audio, n_mels=n_mels, padding=padding)
    else:
        # numpy -> torch
        if isinstance(mel_input, np.ndarray):
            x = torch.from_numpy(mel_input)
        elif torch.is_tensor(mel_input):
            x = mel_input
        else:
            raise TypeError(f"Unsupported mel_input type: {type(mel_input)}")

        # waveform detection: 1D or (1, samples) or (samples,) etc.
        if x.dim() == 1 or (x.dim() == 2 and x.shape[0] == 1 and x.shape[1] != n_mels):
            # ensure shape (samples,)
            wav = x if x.dim() == 1 else x.squeeze(0)
            if duration is not None:
                target_samples = int(duration * sample_rate)
                wav = pad_or_trim(wav, length=target_samples, axis=-1)
            mel = log_mel_spectrogram(wav, n_mels=n_mels, padding=padding)
        # already mel: (n_mels, T) or (B, n_mels, T)
        elif x.dim() == 2 and x.shape[0] == n_mels:
            mel = x
        elif x.dim() == 3 and x.shape[1] == n_mels:
            # (B, n_mels, T)
            mel = x
            # will expand later
        else:
            raise ValueError(f"Unsupported tensor shape for mel_input: {x.shape}")

    # now mel is either (n_mels, T) or (B, n_mels, T)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # -> (1, n_mels, T)

    # optionally enforce fixed frame length (useful for fixed-shape inference)
    if target_frames is not None:
        mel = pad_or_trim(mel, length=target_frames, axis=-1)

    # optional normalization (global)
    if normalize:
        mean = mel.mean()
        std = mel.std(unbiased=False)
        mel = (mel - mean) / (std + 1e-6)

    # ensure float32 and add final channel dim -> (B, n_mels, T, 1)
    mel_4d = mel.to(torch.float32).unsqueeze(-1)
    return mel_4d