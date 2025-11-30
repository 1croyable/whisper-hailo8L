#!/usr/bin/env python3
import numpy as np
import grpc
from concurrent import futures
import threading
import time
import onnxruntime as ort
from tokenizer import get_tokenizer
import torch
from whisper_hailo_model import Decoder

from hailo_platform.pyhailort.pyhailort import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InputVStreamParams,
    OutputVStreamParams,
    InferVStreams,
)

from deploy import encoder_pb2
from deploy import encoder_pb2_grpc

HEF_PATH = "/mnt/external/workspace/encoder.hef"
GRPC_PORT = 50051

# ========= 初始化 Hailo 相关：持久化 VDevice + pipeline =========

print("Loading HEF from:", HEF_PATH)
hef = HEF(HEF_PATH)

in_info = hef.get_input_vstream_infos()[0]
out_info = hef.get_output_vstream_infos()[0]

print("Creating VDevice and configuring network_group ...")
vdev = VDevice()
cfg = ConfigureParams.create_from_hef(hef, HailoStreamInterface.PCIe)
network_group = vdev.configure(hef, cfg)[0]
ng_params = network_group.create_params()

print("Creating persistent InferVStreams pipeline ...")
in_vs_params = InputVStreamParams.make_from_network_group(network_group, [in_info])
out_vs_params = OutputVStreamParams.make_from_network_group(network_group, [out_info])

_pipeline_ctx = InferVStreams(network_group, in_vs_params, out_vs_params)
pipeline = _pipeline_ctx.__enter__()

_ng_ctx = network_group.activate(ng_params)
_ng_ctx.__enter__()

infer_lock = threading.Lock()

PER_SAMPLE_IN = int(np.prod(in_info.shape))
PER_SAMPLE_OUT = 500 * 1 * 768

# 初始化 PyTorch 解码器模型
print("Loading PyTorch Decoder model ...")

N_VOCAB = 51865
N_TEXT_CTX = 500
N_STATE = 768
N_HEAD = 8
N_LAYER = 6

decoder_pt = Decoder(
    n_vocab=N_VOCAB,
    n_ctx=N_TEXT_CTX,
    n_state=N_STATE,
    n_head=N_HEAD,
    n_layer=N_LAYER,
)

ckpt_path = "/mnt/external/workspace/decoder_epoch5.pth"
state = torch.load(ckpt_path, map_location="cpu")
decoder_pt.load_state_dict(state, strict=True)
decoder_pt.eval()

# Whisper tokenizer
print("Loading Whisper tokenizer ...")
tokenizer = get_tokenizer(
    multilingual=True,
    language="fr",
    task="transcribe",
)
BOS = tokenizer.sot
EOT = PAD = tokenizer.eot

class EncoderService(encoder_pb2_grpc.EncoderServicer):
    def Encode(self, request, context):
        try:
            B = request.batch_size
            if B <= 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("batch_size must be > 0")
                return encoder_pb2.EncodeResponse()

            expected_bytes = B * PER_SAMPLE_IN * 4
            if len(request.mel) != expected_bytes:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    f"mel bytes size {len(request.mel)} != expected {expected_bytes}"
                )
                return encoder_pb2.EncodeResponse()

            mel_batch = np.frombuffer(request.mel, dtype=np.float32)
            mel_batch = mel_batch.reshape(B, *in_info.shape)
            mel_batch = np.ascontiguousarray(mel_batch)


            start_time = time.time()

            with infer_lock:
                outputs = pipeline.infer({in_info.name: mel_batch})

            end_time = time.time()
            infer_time = end_time - start_time
            print(f"[INFO] Hailo inference time for batch size {B}: {infer_time:.4f} seconds")

            xa = outputs[out_info.name]  # (B,500,1,768)
            xa = np.asarray(xa, dtype=np.float32)

            if xa.size != B * PER_SAMPLE_OUT:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(
                    f"xa size {xa.size} != B * PER_SAMPLE_OUT = {B * PER_SAMPLE_OUT}"
                )
                return encoder_pb2.EncodeResponse()

            xa_bytes = xa.tobytes()
            return encoder_pb2.EncodeResponse(xa=xa_bytes)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Exception: {repr(e)}")
            return encoder_pb2.EncodeResponse()

    def Decode(self, xa, context):
        try:
            print("[DEBUG] Decode() called, xa shape:", xa.shape)
            # xa: numpy array (1,768,500,1)
            xa_torch = torch.tensor(xa, dtype=torch.float32)

            # 构造和训练时一致风格的 tokens: [sot_sequence] + pad(eot)
            tokens = torch.full((1, N_TEXT_CTX), tokenizer.eot, dtype=torch.long)
            sot_seq = tokenizer.sot_sequence  # 例如: [sot, <|fr|>, <|transcribe|>, <|notimestamps|> ...]
            L = min(len(sot_seq), N_TEXT_CTX)
            tokens[0, :L] = torch.tensor(sot_seq[:L], dtype=torch.long)

            with torch.no_grad():
                logits_4d = decoder_pt(tokens, xa_torch)  # (1, vocab, 500, 1)

            logits = logits_4d.squeeze(-1).permute(0, 2, 1)  # (1,500,vocab)
            print("[DEBUG] logits shape =", logits.shape)

            # 对每个位置取 argmax → 得到一整条 token 序列
            token_ids = logits.argmax(dim=-1)[0].tolist()
            print("[DEBUG] token_ids (first 50) =", token_ids[:50])

            # ❌ 不要再在第一个 EOT 截断
            # if EOT in token_ids:
            #     token_ids = token_ids[: token_ids.index(EOT)]

            return token_ids

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Exception in Decode: {repr(e)}")
            return []

    def EncodeAndDecode(self, request, context):
        try:
            encode_response = self.Encode(request, context)

            # 如果没有 xa，则说明 Encode 出错
            if not encode_response.xa:
                return encoder_pb2.DecodeResponse(text="[Encoder Failed]")

            # Decode 正常开始
            xa = np.frombuffer(encode_response.xa, dtype=np.float32)
            xa = xa.reshape(1, 500, 1, 768)
            xa = xa.transpose(0, 3, 1, 2)  # (1,768,500,1)

            token_ids = self.Decode(xa, context)

            text = tokenizer.decode(token_ids)
            return encoder_pb2.DecodeResponse(text=text)

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Exception in EncodeAndDecode: {repr(e)}")
            return encoder_pb2.DecodeResponse(text="[Internal Error]")

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_send_message_length', 200 * 1024 * 1024),
            ('grpc.max_receive_message_length', 200 * 1024 * 1024),
        ]
    )
    encoder_pb2_grpc.add_EncoderServicer_to_server(EncoderService(), server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    print(f"[START] gRPC Encoder server listening on 0.0.0.0:{GRPC_PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
