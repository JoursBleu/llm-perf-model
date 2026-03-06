# LLM Performance Model

[中文文档](README_CN.md)

⚡ A browser-based tool for estimating LLM inference performance across different models and hardware.

## Features

- **Prefill Latency** — Estimates time-to-first-token based on compute-bound analysis
- **Decode Throughput** — Estimates tokens/sec based on memory-bandwidth-bound analysis  
- **Memory Breakdown** — Model weights + KV cache + activation memory visualization
- **Roofline Analysis** — Visual roofline chart showing where prefill/decode operations fall
- **Multi-Device Comparison** — Side-by-side bar chart of all compatible devices
- **Quantization Support** — FP16, INT8/FP8, INT4 (GPTQ/AWQ), 3-bit, 2-bit
- **Tensor Parallel** — Multi-GPU scaling estimation

## Supported Models

| Model | Params | Architecture |
|-------|--------|-------------|
| LLaMA 3 8B / 70B / 405B | 8B-405B | Dense |
| Qwen 2.5 7B / 32B / 72B | 7.6B-72.7B | Dense |
| Mistral 7B | 7.3B | Dense |
| Mixtral 8x7B / 8x22B | 46.7B-141B | MoE |
| DeepSeek-V3 / R1 | 671B (37B active) | MoE + MLA |
| Phi-3 Mini 3.8B | 3.8B | Dense |
| Gemma 2 9B / 27B | 9.2B-27.2B | Dense |

## Supported Devices

- **NVIDIA**: H200, H100, A100 (40/80GB), L40S, RTX 4090/4080/3090
- **AMD**: MI300X, MI250X, W7900, RX 7900 XTX
- **Apple**: M4 Max, M4 Pro, M2 Ultra

## Methodology

**Prefill (Compute-Bound)**:
```
Prefill Time = (2 × Active Params × Prompt Length) / (Effective TFLOPS)
```

**Decode (Memory-Bandwidth-Bound)**:
```
Time per Token = Model Size (bytes) / (Effective Memory Bandwidth)
```

**Utilization Assumptions**:
- Compute utilization: 55% (accounts for attention, layernorm, non-matmul overhead)
- Bandwidth utilization: 70% (accounts for memory access pattern overhead)

These are theoretical estimates. Real-world performance depends on the inference framework (vLLM, TensorRT-LLM, llama.cpp, etc.), batching strategy, and system configuration.

## Usage

Just open `index.html` in a browser. No build step, no server required.

Or visit the live version: [https://joursbleu.github.io/llm-perf-model/](https://joursbleu.github.io/llm-perf-model/)

## License

MIT
