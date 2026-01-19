# Whisper→Hailo8L 项目重启规划

## 0. 项目目标与约束

### 目标

- 场景：法语播客转录
- 设备：Raspberry Pi 5 + Hailo-8L
- 目标优先级：**准确率 > 实时性**（实时要求较宽松）
- 输出质量：达到可产品使用（明显错误可控、专有名词命中率高、整体可读）

### 工程约束

- Hailo 侧运行的网络必须是 **编译后语义稳定** 的静态图（HEF）
- 避免“训练图 ≠ 编译图”导致的失效：所有改动都要以“导出 ONNX → DFC → HEF 后仍对齐”为硬指标
- Decoder 自回归状态（kv_cache）优先放 host CPU（Pi5），不强求 NPU 端 cache

------

## 1) 目标架构（第一阶段的“正确形态”）

### MVP

- **Encoder：Hailo-8L（HEF） whisper base**
- **Decoder：原版 Whisper base decoder（CPU 上跑，带 kv_cache）**
- 句子切分：你现有“逐句蹦出”机制（或 VAD/标点规则）
- 解码：greedy 或小 beam（先稳，再做精修）

------

## 2) 质量

你需要两层指标：**模型正确性对齐** + **任务精度指标**。

### A. 编译正确性对齐

对一批固定 mel 输入（N=50~200）：

- **Encoder 输出相似度**：
  - cosine similarity（逐 token 平均）
  - MSE / MAE（统计）
  - per-channel mean/std drift（量化漂移）
- 对齐对象：
  1. PyTorch FP32 encoder
  2. ONNXRuntime FP32 encoder
  3. Hailo HEF encoder（dequant 后或 runner 输出）

**门槛建议**（经验值，可调整）：

- 平均 cosine ≥ 0.98（越高越好）
- 关键统计量漂移小且稳定（不出现“整段崩坏/塌缩”）

### B. 端到端转录指标

- **WER**（Word Error Rate）对法语

## 3) 里程碑

### M0：环境与工具链

产物：

- 固定版本：Hailo SDK/DFC/HailoRT、whisper 版本、onnxruntime
- 一键脚本：导出 → 转换 → 编译 → 跑对齐测试
- 日志 & 产物目录规范（每次实验可复盘）

验收：

- base encoder ONNX 能稳定导出（固定 input length）
- DFC 能跑到 har/hef（哪怕先不追求精度）

------

### M1：复现 base encoder→HEF + 对齐测试

目标：

- 按 hailo-whisper 的思路：patch + .alls + calib set
- 在 Hailo-8L 上拿到 **数值稳定的 base encoder HEF**

关键动作：

- 校准集生成
- 量化策略

验收：

- 通过“对齐测试 A”（PyTorch/ORT/HEF 一致性达到门槛）
- encoder 输出没有“塌缩/全零/异常分布”

------

### M2：端到端 MVP（base encoder Hailo + base decoder CPU）

目标：

- 端到端能出可用法语字幕（允许稍慢/滞后）
- 建立端到端评测脚本（WER/CER）

关键动作：

- CPU decoder：使用原版 Whisper base decoder（kv_cache）
- 输入策略：5s/10s window + overlap（根据你句子机制适配）
- 解码策略先保守：greedy 或 beam=1

验收：

- 在你的播客样本上达到“可读”水平
- WER/CER 有可复现的 baseline 数值（作为后续改进对照）

------

### M3：精度增强（不换模型先增精度）

目标：让 base 系统尽可能逼近你 GPU 上的理想效果

优先手段：

1. **解码增强**：beam、temperature fallback、长度惩罚、no-speech 阈值
2. **领域热词/提示词**：每个播客频道的词表（嘉宾/地名/栏目名）
3. **句级后处理**：法语拼写/性数一致纠错（轻量 LM 或规则）
4. **低置信度触发重解码**：只对“困难句”加大 beam 或重跑

验收：

- WER/CER 明显下降（相对 M2 有统计显著提升）
- 专有名词命中率上升

------

### M4：研究 small（优先 small encoder 上 Hailo）

目标：提升精度上限，同时保持可编译性

策略：

- 不急着做 small 全模型上 Hailo
- **先做 small encoder→HEF**，decoder 仍 CPU（small encoder + small decoder）

验收：

- small encoder 对齐测试通过
- 端到端 WER/CER 优于 base baseline（至少在你的播客数据上）

------

### M5：medium 精修路线

现实判断：medium 很难全上 Hailo-8L，但你目标是精度，所以建议产品化策略：

- **在线实时**：small/base（Hailo encoder + CPU decoder）
- **句子稳定后精修**：用 medium 在更强算力上重跑（本地 PC/GPU、家用服务器、或云端可选）

验收：

- 用户看到的最终文本接近 GPU medium 水平
- 成本可控（只对关键句或低置信度句精修）

------

## 4) 风险清单与应对策略

### 风险 R1：编译器 rewrite 造成语义漂移

应对：

- 强制对齐测试作为 gating
- 固定输入 shape/固定导出脚本
- 变更 patch/.alls 时必须重新跑对齐测试

### 风险 R2：decoder 性能成为瓶颈

应对：

- kv_cache 必须启用
- 句子级处理 + overlap 控制
- 解码策略分层：实时轻解码 + 精修重解码

### 风险 R3：法语专有名词错误影响观感

应对：

- 热词提示 + RAG（从节目标题简介、历史转录抽词）
- 低置信度触发重解码/精修