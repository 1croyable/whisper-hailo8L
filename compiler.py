from hailo_sdk_client import ClientRunner
import os

# 定义相关变量
chosen_hw_arch = "hailo8l"  # 使用的 Hailo 硬件架构
onnx_model_name = "whisper-hailo8l"  # 模型的名字
onnx_path = "whisper-hailo8l.onnx"
hailo_model_har_path = f"{onnx_model_name}_hailo_model.har"  # 转换后模型的保存路径

# 实例化 ClientRunner 类
runner = ClientRunner(hw_arch=chosen_hw_arch)

# 转换 ONNX 模型为 HAR 格式
print("开始转换 ONNX 模型为 HAR 格式...")
hn, npz = runner.translate_onnx_model(model=onnx_path, net_name=onnx_model_name)
runner.save_har(hailo_model_har_path)
print(f"转换完成, HAR 模型已保存到: {hailo_model_har_path}")