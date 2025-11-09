import onnx
from onnx import helper, shape_inference

model_path = "whisper-hailo8l-initial.onnx"
fixed_path = "whisper-hailo8l-initial.onnx"

# 1. 读取模型并进行 shape 推理
model = onnx.load(model_path)
model = shape_inference.infer_shapes(model)

# 2. 手动覆盖输出维度（静态）
for output in model.graph.output:
    if output.name == "logits":
        output.type.tensor_type.shape.dim[0].dim_value = 1      # batch
        output.type.tensor_type.shape.dim[1].dim_value = 448    # T_text
        output.type.tensor_type.shape.dim[2].dim_value = 51865  # vocab
        output.type.tensor_type.shape.dim[3].dim_value = 1      # width

onnx.save(model, fixed_path)
print(f"[SUCCESS] Saved static-shaped ONNX to {fixed_path}")
