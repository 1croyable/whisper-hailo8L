import onnx

# 加载 ONNX 模型
m = onnx.load("whisper-hailo8l-initial.onnx")

# 打印输入形状
print("Inputs:")
for inp in m.graph.input:
    dims = [d.dim_value if d.dim_value > 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {dims}")

# 打印输出形状
print("\nOutputs:")
for out in m.graph.output:
    dims = [d.dim_value if d.dim_value > 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {dims}")