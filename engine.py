"""
LLM Inference Performance Model Engine

Roofline-based performance estimator for LLM prefill and decode phases.
Accounts for:
  - Linear layer FLOPs (2 * params * seq_len)
  - Attention FLOPs (O(n^2) for prefill, O(n) per decode step)
  - KV Cache memory (GQA / MLA aware)
  - Memory-bound vs Compute-bound analysis
  - MoE active parameter routing
  - Tensor Parallel scaling with communication overhead
"""

from dataclasses import dataclass, field
from typing import Optional
import math


# ─── Model Definitions ───────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    params: float           # total parameters
    active_params: float    # active parameters (= params for dense, < params for MoE)
    layers: int
    heads: int              # num attention heads
    kv_heads: int           # num KV heads (GQA)
    hidden_dim: int
    head_dim: int
    inter_dim: int          # FFN intermediate dimension
    vocab_size: int
    max_ctx: int
    moe: bool = False
    num_experts: int = 0
    top_k: int = 0
    arch: str = ""
    mla_compressed_dim: Optional[int] = None  # DeepSeek MLA
    softmax_attn_layers: Optional[int] = None  # None=all softmax; N=only N layers use O(n²) softmax, rest use O(n) linear attn
    note: str = ""


MODELS = {
    # LLaMA family
    "llama-3.2-1b": ModelConfig("LLaMA 3.2 1B", 1.24e9, 1.24e9, 16, 32, 8, 2048, 64, 8192, 128256, 131072, arch="LLaMA"),
    "llama-3.2-3b": ModelConfig("LLaMA 3.2 3B", 3.21e9, 3.21e9, 28, 24, 8, 3072, 128, 8192, 128256, 131072, arch="LLaMA"),
    "llama-3-8b": ModelConfig("LLaMA 3 8B", 8e9, 8e9, 32, 32, 8, 4096, 128, 14336, 128256, 8192, arch="LLaMA"),
    "llama-3-70b": ModelConfig("LLaMA 3 70B", 70.6e9, 70.6e9, 80, 64, 8, 8192, 128, 28672, 128256, 8192, arch="LLaMA"),
    "llama-3.1-405b": ModelConfig("LLaMA 3.1 405B", 405e9, 405e9, 126, 128, 8, 16384, 128, 53248, 128256, 131072, arch="LLaMA"),

    # Qwen 2.5 family
    "qwen2.5-0.5b": ModelConfig("Qwen 2.5 0.5B", 0.49e9, 0.49e9, 24, 14, 2, 896, 64, 4864, 151936, 32768, arch="Qwen2"),
    "qwen2.5-1.5b": ModelConfig("Qwen 2.5 1.5B", 1.54e9, 1.54e9, 28, 12, 2, 1536, 128, 8960, 151936, 32768, arch="Qwen2"),
    "qwen2.5-3b": ModelConfig("Qwen 2.5 3B", 3.09e9, 3.09e9, 36, 16, 2, 2048, 128, 11008, 151936, 32768, arch="Qwen2"),
    "qwen2.5-7b": ModelConfig("Qwen 2.5 7B", 7.6e9, 7.6e9, 28, 28, 4, 3584, 128, 18944, 152064, 131072, arch="Qwen2"),
    "qwen2.5-14b": ModelConfig("Qwen 2.5 14B", 14.7e9, 14.7e9, 48, 40, 8, 5120, 128, 13824, 152064, 131072, arch="Qwen2"),
    "qwen2.5-32b": ModelConfig("Qwen 2.5 32B", 32.5e9, 32.5e9, 64, 40, 8, 5120, 128, 27648, 152064, 131072, arch="Qwen2"),
    "qwen2.5-72b": ModelConfig("Qwen 2.5 72B", 72.7e9, 72.7e9, 80, 64, 8, 8192, 128, 29568, 152064, 131072, arch="Qwen2"),

    # Mistral / Mixtral
    "mistral-7b": ModelConfig("Mistral 7B", 7.3e9, 7.3e9, 32, 32, 8, 4096, 128, 14336, 32000, 32768, arch="Mistral"),
    "mistral-small-24b": ModelConfig("Mistral Small 24B", 24e9, 24e9, 40, 32, 8, 5120, 128, 14336, 32768, 32768, arch="Mistral"),
    "mixtral-8x7b": ModelConfig("Mixtral 8x7B", 46.7e9, 12.9e9, 32, 32, 8, 4096, 128, 14336, 32000, 32768, True, 8, 2, arch="Mixtral"),
    "mixtral-8x22b": ModelConfig("Mixtral 8x22B", 141e9, 39e9, 56, 48, 8, 6144, 128, 16384, 32768, 65536, True, 8, 2, arch="Mixtral"),

    # DeepSeek
    "deepseek-v2-lite": ModelConfig("DeepSeek-V2-Lite", 15.7e9, 2.4e9, 27, 16, 16, 2048, 128, 10944, 102400, 32768, True, 64, 6, arch="DeepSeek", mla_compressed_dim=512),
    "deepseek-v2": ModelConfig("DeepSeek-V2", 236e9, 21e9, 60, 128, 128, 5120, 128, 12288, 102400, 131072, True, 160, 6, arch="DeepSeek", mla_compressed_dim=512, note="MLA + shared experts"),
    "deepseek-v3": ModelConfig("DeepSeek-V3", 671e9, 37e9, 61, 128, 128, 7168, 128, 18432, 129280, 131072, True, 256, 8, arch="DeepSeek", mla_compressed_dim=512, note="MLA + shared experts"),
    "deepseek-r1": ModelConfig("DeepSeek-R1", 671e9, 37e9, 61, 128, 128, 7168, 128, 18432, 129280, 131072, True, 256, 8, arch="DeepSeek", mla_compressed_dim=512, note="MLA + shared experts"),
    "deepseek-r1-7b": ModelConfig("DeepSeek-R1-Distill 7B", 7.6e9, 7.6e9, 28, 28, 4, 3584, 128, 18944, 152064, 131072, arch="Qwen2", note="Distilled from Qwen 2.5 7B"),
    "deepseek-r1-14b": ModelConfig("DeepSeek-R1-Distill 14B", 14.7e9, 14.7e9, 48, 40, 8, 5120, 128, 13824, 152064, 131072, arch="Qwen2", note="Distilled from Qwen 2.5 14B"),
    "deepseek-r1-32b": ModelConfig("DeepSeek-R1-Distill 32B", 32.5e9, 32.5e9, 64, 40, 8, 5120, 128, 27648, 152064, 131072, arch="Qwen2", note="Distilled from Qwen 2.5 32B"),
    "deepseek-r1-70b": ModelConfig("DeepSeek-R1-Distill 70B", 70.6e9, 70.6e9, 80, 64, 8, 8192, 128, 28672, 128256, 131072, arch="LLaMA", note="Distilled from LLaMA 3 70B"),

    # Phi
    "phi-3-mini": ModelConfig("Phi-3 Mini 3.8B", 3.8e9, 3.8e9, 32, 32, 32, 3072, 96, 8192, 32064, 128000, arch="Phi"),
    "phi-3-small": ModelConfig("Phi-3 Small 7B", 7.4e9, 7.4e9, 32, 32, 8, 4096, 128, 14336, 32064, 128000, arch="Phi"),
    "phi-3-medium": ModelConfig("Phi-3 Medium 14B", 14e9, 14e9, 40, 40, 10, 5120, 128, 17920, 32064, 128000, arch="Phi"),
    "phi-4": ModelConfig("Phi-4 14B", 14e9, 14e9, 40, 40, 10, 5120, 128, 17920, 100352, 16384, arch="Phi"),

    # MiniMax
    "minimax-text-01": ModelConfig("MiniMax-Text-01", 456e9, 45.9e9, 80, 64, 8, 6144, 128, 9216, 200064, 4096000, True, 32, 2, arch="MiniMax", softmax_attn_layers=10, note="Lightning Attention + MoE"),
    "minimax-m2.1": ModelConfig("MiniMax-M2.1", 229e9, 10.5e9, 62, 48, 8, 3072, 128, 1536, 200064, 196608, True, 256, 8, arch="MiniMax", softmax_attn_layers=0, note="Lightning Attention + MoE + MTP"),

    # Gemma
    "gemma-2-2b": ModelConfig("Gemma 2 2B", 2.6e9, 2.6e9, 26, 8, 4, 2304, 256, 9216, 256000, 8192, arch="Gemma"),
    "gemma-2-9b": ModelConfig("Gemma 2 9B", 9.2e9, 9.2e9, 42, 16, 8, 3584, 256, 14336, 256000, 8192, arch="Gemma"),
    "gemma-2-27b": ModelConfig("Gemma 2 27B", 27.2e9, 27.2e9, 46, 32, 16, 4608, 128, 36864, 256000, 8192, arch="Gemma"),
}


# ─── Device Definitions ──────────────────────────────────────────────────────

@dataclass
class DeviceConfig:
    name: str
    vram: float         # GB
    bw: float           # GB/s memory bandwidth
    fp16_tflops: float
    fp8_tflops: float
    int8_tflops: float
    vendor: str


DEVICES = {
    # NVIDIA Data Center
    "h200-sxm": DeviceConfig("NVIDIA H200 SXM", 141, 4800, 989, 1979, 1979, "nvidia"),
    "h100-sxm": DeviceConfig("NVIDIA H100 SXM", 80, 3350, 989, 1979, 1979, "nvidia"),
    "a100-sxm-80": DeviceConfig("NVIDIA A100 80GB", 80, 2039, 312, 312, 624, "nvidia"),
    "a100-sxm-40": DeviceConfig("NVIDIA A100 40GB", 40, 1555, 312, 312, 624, "nvidia"),
    "l40s": DeviceConfig("NVIDIA L40S", 48, 864, 362, 733, 733, "nvidia"),
    # NVIDIA Consumer / Edge
    "rtx-4090": DeviceConfig("RTX 4090", 24, 1008, 165, 330, 330, "nvidia"),
    "rtx-4080": DeviceConfig("RTX 4080", 16, 717, 97, 194, 194, "nvidia"),
    "rtx-3090": DeviceConfig("RTX 3090", 24, 936, 71, 71, 142, "nvidia"),
    "dgx-spark": DeviceConfig("NVIDIA DGX Spark (GB10)", 128, 273, 209, 418, 418, "nvidia"),
    # AMD Data Center
    "mi300x": DeviceConfig("AMD MI300X", 192, 5300, 1307, 2614, 2614, "amd"),
    "mi250x": DeviceConfig("AMD MI250X", 128, 3277, 383, 383, 383, "amd"),
    # AMD Consumer / Pro
    "rx-9700xt": DeviceConfig("AMD RX 9700 XT", 24, 864, 56, 112, 112, "amd"),
    "rx-9070xt": DeviceConfig("AMD RX 9070 XT", 16, 640, 49, 98, 98, "amd"),
    "w7900": DeviceConfig("AMD W7900", 48, 864, 61, 61, 122, "amd"),
    "rx-7900xtx": DeviceConfig("AMD RX 7900 XTX", 24, 960, 123, 123, 123, "amd"),
    # AMD APU
    "strix-halo": DeviceConfig("AMD Ryzen AI Max 395 (Strix Halo)", 128, 256, 30, 30, 60, "amd"),
    # Apple Silicon
    "m4-max": DeviceConfig("Apple M4 Max", 128, 546, 54, 54, 54, "apple"),
    "m4-pro": DeviceConfig("Apple M4 Pro", 48, 273, 22, 22, 22, "apple"),
    "m2-ultra": DeviceConfig("Apple M2 Ultra", 192, 800, 27, 27, 27, "apple"),
}


# ─── Performance Engine ──────────────────────────────────────────────────────

@dataclass
class PerfConfig:
    """Tunable utilization parameters."""
    compute_util: float = 0.55      # matmul compute utilization
    bw_util: float = 0.70           # memory bandwidth utilization
    tp_comm_overhead: float = 0.05  # per-GPU communication overhead for TP > 1
    flash_attn: bool = True         # assume FlashAttention (reduces memory, IO-aware)


@dataclass
class PerfResult:
    # Identifiers
    model_name: str
    device_name: str
    quant_bits: int
    prompt_len: int
    output_len: int
    batch_size: int
    tp: int

    # Prefill
    prefill_linear_flops: float     # FLOPs for linear layers during prefill
    prefill_attn_flops: float       # FLOPs for attention during prefill
    prefill_total_flops: float
    prefill_time_ms: float
    prefill_tps: float              # tokens/sec during prefill

    # Decode
    decode_linear_flops: float      # per-step linear FLOPs
    decode_attn_flops: float        # per-step attention FLOPs (depends on current seq pos)
    decode_mem_bound_ms: float      # memory-bound time per token
    decode_compute_bound_ms: float  # compute-bound time per token
    decode_time_per_token_ms: float
    decode_tps: float               # tokens/sec decode
    decode_bottleneck: str          # "memory" or "compute"

    # Total
    total_time_ms: float
    total_output_tps: float         # output_len / total_time

    # Memory
    model_size_gb: float            # full model weights
    kv_cache_gb: float
    activation_gb: float
    total_memory_gb: float
    device_memory_gb: float
    fits_in_memory: bool

    # Roofline
    ridge_point: float              # FLOPs/Byte where compute = memory bound
    prefill_arith_intensity: float
    decode_arith_intensity: float

    # Device effective specs
    effective_tflops: float
    effective_bw_gbs: float

    # FlashAttention
    flash_attn: bool
    attn_score_memory_gb: float     # N×N attention matrix memory (0 with FlashAttn)


def get_device_tflops(device: DeviceConfig, quant_bits: int) -> float:
    """Select appropriate TFLOPS based on quantization and device capabilities."""
    if quant_bits <= 8:
        best = max(device.fp8_tflops, device.int8_tflops)
        if best > device.fp16_tflops:
            return best
    return device.fp16_tflops


def calc_kv_cache_gb(model: ModelConfig, seq_len: int, batch_size: int, quant_bits: int) -> float:
    """Calculate KV cache memory in GB."""
    if model.mla_compressed_dim:
        # DeepSeek MLA: compressed KV representation
        # layers * compressed_dim * seq_len * 2 bytes (FP16) * batch
        return (model.layers * model.mla_compressed_dim * seq_len * 2 * batch_size) / 1e9

    # How many layers use standard softmax attention vs linear attention
    softmax_layers = model.softmax_attn_layers if model.softmax_attn_layers is not None else model.layers
    linear_layers = model.layers - softmax_layers

    # Standard GQA KV cache: only softmax attention layers need per-token KV storage
    kv_bytes = max(quant_bits, 16) / 8  # KV cache usually FP16 minimum
    kv_cache = (2 * softmax_layers * model.kv_heads * model.head_dim * seq_len * kv_bytes * batch_size) / 1e9

    # Linear attention layers maintain a fixed-size state: heads * head_dim² * 2 bytes per layer
    linear_state = (linear_layers * model.heads * model.head_dim * model.head_dim * 2 * batch_size) / 1e9

    return kv_cache + linear_state


def calc_attention_flops_prefill(model: ModelConfig, seq_len: int, batch_size: int) -> float:
    """
    Attention FLOPs for prefill phase.
    Softmax attention (standard): O(n²) per layer — 4 * B * H * S² * D
    Linear attention (Lightning): O(n·d) per layer — 4 * B * H * S * D²
      (computes K^T@V then Q@state, each O(n*d²) per head)
    """
    softmax_layers = model.softmax_attn_layers if model.softmax_attn_layers is not None else model.layers
    linear_layers = model.layers - softmax_layers

    softmax_flops = 4 * batch_size * model.heads * seq_len * seq_len * model.head_dim * softmax_layers
    linear_flops = 4 * batch_size * model.heads * seq_len * model.head_dim * model.head_dim * linear_layers

    return softmax_flops + linear_flops


def calc_attention_flops_decode_step(model: ModelConfig, current_pos: int, batch_size: int) -> float:
    """
    Attention FLOPs for one decode step.
    Softmax: O(pos) per layer — 4 * B * H * pos * D (scan full KV cache)
    Linear:  O(d²) per layer  — 4 * B * H * D² (fixed-size state lookup)
    """
    softmax_layers = model.softmax_attn_layers if model.softmax_attn_layers is not None else model.layers
    linear_layers = model.layers - softmax_layers

    softmax_flops = 4 * batch_size * model.heads * current_pos * model.head_dim * softmax_layers
    linear_flops = 4 * batch_size * model.heads * model.head_dim * model.head_dim * linear_layers

    return softmax_flops + linear_flops


def estimate_performance(
    model_key: str,
    device_key: str,
    quant_bits: int = 4,
    prompt_len: int = 512,
    output_len: int = 256,
    batch_size: int = 1,
    tp: int = 1,
    config: Optional[PerfConfig] = None,
) -> PerfResult:
    """
    Main performance estimation function.

    Returns detailed PerfResult with prefill, decode, memory, and roofline analysis.
    """
    if config is None:
        config = PerfConfig()

    model = MODELS[model_key]
    device = DEVICES[device_key]

    # ── Effective device specs with TP ──
    tp_efficiency = 1.0 - config.tp_comm_overhead * max(0, tp - 1)  # diminishing returns
    tp_efficiency = max(tp_efficiency, 0.5)  # floor at 50%

    raw_tflops = get_device_tflops(device, quant_bits)
    effective_tflops = raw_tflops * tp * tp_efficiency * config.compute_util  # TFLOPS
    effective_bw = device.bw * tp * tp_efficiency * config.bw_util  # GB/s

    # ── Memory ──
    model_size_gb = (model.params * quant_bits / 8) / 1e9
    total_seq_len = prompt_len + output_len
    kv_cache_gb = calc_kv_cache_gb(model, total_seq_len, batch_size, quant_bits)
    # Activation memory: hidden states + residual
    activation_gb = (model.hidden_dim * prompt_len * batch_size * 4 * 2) / 1e9
    # Without FlashAttention: must materialize N×N attention score matrix in HBM
    # Peak per layer: batch * heads * seq_len^2 * 4 bytes (FP32 softmax)
    if config.flash_attn:
        attn_score_memory_gb = 0.0  # tiled in SRAM, never materialized
    else:
        attn_score_memory_gb = (batch_size * model.heads * prompt_len * prompt_len * 4) / 1e9
    activation_gb += attn_score_memory_gb
    total_memory_gb = model_size_gb + kv_cache_gb + activation_gb
    device_memory_gb = device.vram * tp
    fits_in_memory = total_memory_gb <= device_memory_gb

    # ── Prefill Phase ──
    # Linear layers: 2 * active_params * prompt_len * batch_size
    prefill_linear_flops = 2 * model.active_params * prompt_len * batch_size
    # Attention: O(n^2) per layer
    prefill_attn_flops = calc_attention_flops_prefill(model, prompt_len, batch_size)
    prefill_total_flops = prefill_linear_flops + prefill_attn_flops

    # Without FlashAttention, attention is IO-bound due to N×N HBM read/write.
    # Model as: linear layers run at full compute_util, attention portion has
    # reduced effective throughput due to HBM IO overhead.
    if config.flash_attn:
        # FlashAttention: fused kernel, attention runs at ~same utilization as matmul
        prefill_time_s = prefill_total_flops / (effective_tflops * 1e12)
    else:
        # Without FlashAttention: attention has extra HBM IO (read Q,K → write S → read S,V → write O)
        # Attention effective throughput ≈ 40% of peak (IO overhead dominates)
        attn_util_factor = 0.40 / config.compute_util  # ratio vs normal utilization
        prefill_linear_time_s = prefill_linear_flops / (effective_tflops * 1e12)
        prefill_attn_time_s = prefill_attn_flops / (effective_tflops * 1e12 * attn_util_factor)
        prefill_time_s = prefill_linear_time_s + prefill_attn_time_s

    prefill_time_ms = prefill_time_s * 1000
    prefill_tps = (prompt_len * batch_size) / prefill_time_s if prefill_time_s > 0 else 0

    # ── Decode Phase ──
    # Linear layers per decode step: 2 * active_params * batch_size
    decode_linear_flops = 2 * model.active_params * batch_size

    # For decode, attention cost varies with position. Use average position.
    avg_decode_pos = prompt_len + output_len / 2  # average KV length during decode
    decode_attn_flops = calc_attention_flops_decode_step(model, avg_decode_pos, batch_size)
    decode_total_flops_per_step = decode_linear_flops + decode_attn_flops

    # Memory-bound: read all active weights once per step
    active_model_bytes = model.active_params * quant_bits / 8
    # Also need to read KV cache for attention (proportional to avg_decode_pos)
    kv_read_bytes = calc_kv_cache_gb(model, int(avg_decode_pos), batch_size, quant_bits) * 1e9
    # Without FlashAttention: extra HBM traffic for attention score materialization
    if not config.flash_attn:
        # Must write and re-read attention scores: 2 * batch * heads * avg_pos * 4 bytes per layer
        attn_io_bytes = 2 * batch_size * model.heads * avg_decode_pos * 4 * model.layers
        kv_read_bytes += attn_io_bytes
    total_read_bytes = active_model_bytes + kv_read_bytes

    decode_mem_bound_s = total_read_bytes / (effective_bw * 1e9)
    decode_compute_bound_s = decode_total_flops_per_step / (effective_tflops * 1e12)

    decode_time_per_step_s = max(decode_mem_bound_s, decode_compute_bound_s)
    decode_time_per_token_ms = (decode_time_per_step_s / batch_size) * 1000
    decode_tps = 1000 / decode_time_per_token_ms if decode_time_per_token_ms > 0 else 0
    decode_bottleneck = "memory" if decode_mem_bound_s >= decode_compute_bound_s else "compute"

    # ── Total ──
    total_decode_ms = decode_time_per_token_ms * output_len
    total_time_ms = prefill_time_ms + total_decode_ms
    total_output_tps = output_len / (total_time_ms / 1000) if total_time_ms > 0 else 0

    # ── Roofline ──
    ridge_point = (effective_tflops * 1e12) / (effective_bw * 1e9)  # FLOPs/Byte
    # Prefill arithmetic intensity: total_flops / bytes_read
    prefill_bytes = active_model_bytes  # weights read once (assuming layers streamed)
    prefill_ai = prefill_total_flops / prefill_bytes if prefill_bytes > 0 else 0
    # Decode arithmetic intensity
    decode_ai = decode_total_flops_per_step / total_read_bytes if total_read_bytes > 0 else 0

    return PerfResult(
        model_name=model.name,
        device_name=device.name,
        quant_bits=quant_bits,
        prompt_len=prompt_len,
        output_len=output_len,
        batch_size=batch_size,
        tp=tp,
        # Prefill
        prefill_linear_flops=prefill_linear_flops,
        prefill_attn_flops=prefill_attn_flops,
        prefill_total_flops=prefill_total_flops,
        prefill_time_ms=prefill_time_ms,
        prefill_tps=prefill_tps,
        # Decode
        decode_linear_flops=decode_linear_flops,
        decode_attn_flops=decode_attn_flops,
        decode_mem_bound_ms=decode_mem_bound_s * 1000,
        decode_compute_bound_ms=decode_compute_bound_s * 1000,
        decode_time_per_token_ms=decode_time_per_token_ms,
        decode_tps=decode_tps,
        decode_bottleneck=decode_bottleneck,
        # Total
        total_time_ms=total_time_ms,
        total_output_tps=total_output_tps,
        # Memory
        model_size_gb=model_size_gb,
        kv_cache_gb=kv_cache_gb,
        activation_gb=activation_gb,
        total_memory_gb=total_memory_gb,
        device_memory_gb=device_memory_gb,
        fits_in_memory=fits_in_memory,
        # Roofline
        ridge_point=ridge_point,
        prefill_arith_intensity=prefill_ai,
        decode_arith_intensity=decode_ai,
        # Device
        effective_tflops=effective_tflops,
        effective_bw_gbs=effective_bw,
        # FlashAttention
        flash_attn=config.flash_attn,
        attn_score_memory_gb=attn_score_memory_gb,
    )


def compare_devices(
    model_key: str,
    quant_bits: int = 4,
    prompt_len: int = 512,
    output_len: int = 256,
    batch_size: int = 1,
    tp: int = 1,
) -> list[dict]:
    """Compare performance across all devices for a given model config."""
    results = []
    for device_key in DEVICES:
        try:
            r = estimate_performance(model_key, device_key, quant_bits, prompt_len, output_len, batch_size, tp)
            results.append({
                "device": r.device_name,
                "device_key": device_key,
                "fits": r.fits_in_memory,
                "prefill_ms": round(r.prefill_time_ms, 2),
                "decode_tps": round(r.decode_tps, 1),
                "total_ms": round(r.total_time_ms, 2),
                "memory_gb": round(r.total_memory_gb, 1),
                "bottleneck": r.decode_bottleneck,
            })
        except Exception:
            pass
    results.sort(key=lambda x: x["decode_tps"], reverse=True)
    return results


def compare_models(
    device_key: str,
    quant_bits: int = 4,
    prompt_len: int = 512,
    output_len: int = 256,
    batch_size: int = 1,
    tp: int = 1,
) -> list[dict]:
    """Compare all models on a given device."""
    results = []
    for model_key in MODELS:
        try:
            r = estimate_performance(model_key, device_key, quant_bits, prompt_len, output_len, batch_size, tp)
            results.append({
                "model": r.model_name,
                "model_key": model_key,
                "fits": r.fits_in_memory,
                "prefill_ms": round(r.prefill_time_ms, 2),
                "decode_tps": round(r.decode_tps, 1),
                "total_ms": round(r.total_time_ms, 2),
                "memory_gb": round(r.total_memory_gb, 1),
                "bottleneck": r.decode_bottleneck,
            })
        except Exception:
            pass
    results.sort(key=lambda x: x["decode_tps"], reverse=True)
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _format_num(n: float) -> str:
    if n >= 1e15:
        return f"{n/1e15:.1f} PFLOP"
    if n >= 1e12:
        return f"{n/1e12:.1f} TFLOP"
    if n >= 1e9:
        return f"{n/1e9:.1f} GFLOP"
    return f"{n:.0f} FLOP"


def _format_time(ms: float) -> str:
    if ms < 1:
        return f"{ms*1000:.0f} μs"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms/1000:.2f} s"


def _format_gb(gb: float) -> str:
    if gb < 1:
        return f"{gb*1024:.0f} MB"
    return f"{gb:.1f} GB"


def print_result(r: PerfResult):
    """Pretty-print a PerfResult to console."""
    print(f"\n{'='*60}")
    print(f"  {r.model_name}  on  {r.device_name}")
    print(f"  {r.quant_bits}-bit | Prompt {r.prompt_len} | Output {r.output_len} | BS {r.batch_size} | TP {r.tp}")
    print(f"{'='*60}")

    print(f"\n  📊 Effective Device Specs")
    print(f"     Compute:  {r.effective_tflops:.0f} TFLOPS")
    print(f"     Memory BW: {r.effective_bw_gbs:.0f} GB/s")
    print(f"     Ridge Point: {r.ridge_point:.0f} FLOPs/Byte")

    print(f"\n  ⚡ Prefill (TTFT)")
    print(f"     Linear FLOPs:    {_format_num(r.prefill_linear_flops)}")
    print(f"     Attention FLOPs: {_format_num(r.prefill_attn_flops)} ({r.prefill_attn_flops/r.prefill_total_flops*100:.1f}%)")
    print(f"     Total FLOPs:     {_format_num(r.prefill_total_flops)}")
    print(f"     Latency:         {_format_time(r.prefill_time_ms)}")
    print(f"     Throughput:      {r.prefill_tps:.0f} tok/s")
    print(f"     Arith Intensity: {r.prefill_arith_intensity:.0f} FLOPs/Byte")

    print(f"\n  🔄 Decode")
    print(f"     Mem-bound:       {_format_time(r.decode_mem_bound_ms)}/step")
    print(f"     Compute-bound:   {_format_time(r.decode_compute_bound_ms)}/step")
    print(f"     Bottleneck:      {'📦 Memory' if r.decode_bottleneck == 'memory' else '⚡ Compute'}")
    print(f"     Per Token:       {_format_time(r.decode_time_per_token_ms)}")
    print(f"     Throughput:      {r.decode_tps:.1f} tok/s")
    print(f"     Arith Intensity: {r.decode_arith_intensity:.1f} FLOPs/Byte")

    print(f"\n  ⏱️  Total")
    print(f"     Prefill + Decode: {_format_time(r.total_time_ms)}")
    print(f"     Output Throughput: {r.total_output_tps:.1f} tok/s")

    print(f"\n  💾 Memory")
    print(f"     Weights:     {_format_gb(r.model_size_gb)}")
    print(f"     KV Cache:    {_format_gb(r.kv_cache_gb)}")
    print(f"     Activation:  {_format_gb(r.activation_gb)}")
    print(f"     Total:       {_format_gb(r.total_memory_gb)} / {_format_gb(r.device_memory_gb)}")
    print(f"     Status:      {'✅ Fits' if r.fits_in_memory else '❌ OOM'}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Inference Performance Estimator")
    parser.add_argument("-m", "--model", default="llama-3-8b", choices=list(MODELS.keys()), help="Model key")
    parser.add_argument("-d", "--device", default="rtx-4090", choices=list(DEVICES.keys()), help="Device key")
    parser.add_argument("-q", "--quant", type=int, default=4, choices=[2, 3, 4, 8, 16], help="Quantization bits")
    parser.add_argument("-p", "--prompt", type=int, default=512, help="Prompt length")
    parser.add_argument("-o", "--output", type=int, default=256, help="Output length")
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size")
    parser.add_argument("-t", "--tp", type=int, default=1, help="Tensor parallel GPUs")
    parser.add_argument("--compare-devices", action="store_true", help="Compare all devices")
    parser.add_argument("--compare-models", action="store_true", help="Compare all models")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-devices", action="store_true", help="List available devices")

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable Models:")
        for k, m in MODELS.items():
            moe_str = f" (MoE {m.num_experts}x top-{m.top_k}, active {m.active_params/1e9:.1f}B)" if m.moe else ""
            print(f"  {k:25s}  {m.name:30s}  {m.params/1e9:.1f}B{moe_str}")
        exit()

    if args.list_devices:
        print("\nAvailable Devices:")
        for k, d in DEVICES.items():
            print(f"  {k:18s}  {d.name:38s}  {d.vram}GB  BW:{d.bw}GB/s  FP16:{d.fp16_tflops}T")
        exit()

    if args.compare_devices:
        results = compare_devices(args.model, args.quant, args.prompt, args.output, args.batch, args.tp)
        print(f"\n{'Device':<42} {'Fits':>5} {'Prefill':>10} {'Decode':>10} {'Total':>10} {'Mem':>8} {'Bottleneck':>12}")
        print("-" * 100)
        for r in results:
            fit = "✅" if r["fits"] else "❌"
            print(f"  {r['device']:<40} {fit:>5} {_format_time(r['prefill_ms']):>10} {r['decode_tps']:>8.1f}t/s {_format_time(r['total_ms']):>10} {r['memory_gb']:>6.1f}GB {r['bottleneck']:>12}")
        exit()

    if args.compare_models:
        results = compare_models(args.device, args.quant, args.prompt, args.output, args.batch, args.tp)
        print(f"\n{'Model':<35} {'Fits':>5} {'Prefill':>10} {'Decode':>10} {'Total':>10} {'Mem':>8} {'Bottleneck':>12}")
        print("-" * 95)
        for r in results:
            fit = "✅" if r["fits"] else "❌"
            print(f"  {r['model']:<33} {fit:>5} {_format_time(r['prefill_ms']):>10} {r['decode_tps']:>8.1f}t/s {_format_time(r['total_ms']):>10} {r['memory_gb']:>6.1f}GB {r['bottleneck']:>12}")
        exit()

    r = estimate_performance(args.model, args.device, args.quant, args.prompt, args.output, args.batch, args.tp)
    print_result(r)
