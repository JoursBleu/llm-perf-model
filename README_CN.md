# ⚡ LLM 性能建模工具

基于 Roofline 模型的大语言模型推理性能估算器，在浏览器中运行，无需后端。

**在线体验**: [https://joursbleu.github.io/llm-perf-model/](https://joursbleu.github.io/llm-perf-model/)

## 功能

- **Prefill 延迟估算** — 基于计算密集型分析，估算首 Token 时间 (TTFT)
- **Decode 吞吐估算** — 基于显存带宽瓶颈分析，估算每秒生成 Token 数
- **显存用量分析** — 模型权重 + KV Cache + 激活内存的可视化分解
- **Roofline 图表** — 展示 Prefill/Decode 在 Roofline 模型上的位置
- **多设备横向对比** — 所有兼容设备的 Decode 吞吐对比柱状图
- **量化支持** — FP16、INT8/FP8、INT4 (GPTQ/AWQ)、3-bit、2-bit
- **张量并行** — 1-8 卡并行的性能缩放估算

## 支持的模型

| 模型 | 参数量 | 架构 |
|------|--------|------|
| LLaMA 3 8B / 70B / 405B | 8B - 405B | Dense |
| Qwen 2.5 7B / 32B / 72B | 7.6B - 72.7B | Dense |
| Mistral 7B | 7.3B | Dense |
| Mixtral 8x7B / 8x22B | 46.7B - 141B | MoE |
| DeepSeek-V3 / R1 | 671B (活跃 37B) | MoE + MLA |
| Phi-3 Mini | 3.8B | Dense |
| Gemma 2 9B / 27B | 9.2B - 27.2B | Dense |

## 支持的设备

| 类别 | 设备 |
|------|------|
| NVIDIA 数据中心 | H200 SXM, H100 SXM, A100 80GB/40GB, L40S |
| NVIDIA 消费级 | RTX 4090, RTX 4080, RTX 3090 |
| AMD 数据中心 | MI300X, MI250X |
| AMD 消费/专业级 | Radeon PRO W7900, RX 7900 XTX |
| Apple Silicon | M4 Max, M4 Pro, M2 Ultra |

## 建模方法

### Prefill（计算密集型）

Prefill 阶段需要并行处理整个 prompt，主要瓶颈是计算能力：

```
Prefill Time = (2 × 活跃参数量 × Prompt 长度) / (有效 TFLOPS)
```

- 每个 token 经过模型需要约 2×P 次浮点运算（乘法 + 加法）
- MoE 模型只计算活跃参数（如 DeepSeek-V3 用 37B 而非 671B）

### Decode（显存带宽瓶颈）

Decode 阶段逐 token 生成，每生成一个 token 需要从显存读取全部模型权重：

```
每 Token 时间 = 模型大小 (字节) / (有效显存带宽)
```

### KV Cache

```
KV Cache = 2 × 层数 × KV Head 数 × Head 维度 × 序列长度 × 元素字节数 × Batch Size
```

DeepSeek 的 MLA（Multi-head Latent Attention）使用压缩的 KV 表示，大幅降低 KV Cache 占用。

### 利用率假设

| 参数 | 值 | 说明 |
|------|-----|------|
| 计算利用率 | 55% | 包含 attention、layernorm 等非矩阵乘开销 |
| 带宽利用率 | 70% | 包含内存访问模式损失 |

> 这些是理论估算值。实际性能取决于推理框架（vLLM、TensorRT-LLM、llama.cpp 等）、调度策略和系统配置。

## 使用方法

直接用浏览器打开 `index.html`，无需任何构建步骤或服务器。

## License

MIT
