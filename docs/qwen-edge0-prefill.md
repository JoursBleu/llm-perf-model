# Qwen3.6-35B-A3B and Edge0 K4 prefill arithmetic

[Interactive calculator](../prefill.html) · [中文](qwen-edge0-prefill.zh.md)

This is a text-only, forward-pass **logical operation model**, not a hardware latency prediction. The comparison uses the same recurrent GDN reference algorithm and useful causal-attention pairs for both models. It includes all major projections, routed/shared FFNs, depthwise convolution, GDN state updates, an explicit LM-head policy, and the pinned Edge0 unmerged LoRA profile. Small scalar operations are listed separately. No vision encoder or optional MTP pass is included.

The existing generic estimator treats its per-op view as one ordinary Transformer layer multiplied by the layer count. These hybrid profiles therefore have a dedicated analysis page rather than inheriting that approximation or claiming an SSD-offload speed estimate. The Python hardware API is unchanged.

## Dimensions and variables

| Symbol | Meaning | Value |
|---|---|---:|
| L, La, Lg | Total / full-attention / GDN layers | 40, 10, 30 |
| H | Hidden width | 2048 |
| I, Is | Routed/shared expert intermediate width | 512, 512 |
| E | Routed experts per layer | 256 |
| K | Routed experts executed per token per layer | Qwen 8; Edge0 4 |
| nq, nkv, d | Attention query heads, KV heads, head width | 16, 2, 256 |
| Q, R | nq*d and nkv*d | 4096, 512 |
| nk, nv, dk, dv | GDN key/value heads and head widths | 16, 32, 128, 128 |
| Dk, Dv | nk*dk and nv*dv | 2048, 4096 |
| c, V | Convolution kernel width and vocabulary size | 4, 248320 |

- T: newly processed tokens per sequence. P: already cached prefix tokens, with valid KV and GDN states.
- B: equal-length sequences. Ragged batches must be summed per sequence, not evaluated at a mean length.
- M: positions evaluated by the LM head per sequence. Last-only generation uses 1; all-token logits uses T; the historical Edge0 path uses one position per prefill chunk.
- K is a runtime choice. The Edge0 checkpoint config still says 8, while the compared runtime profile overrides it to 4. All 256 routed experts remain stored.

## Major arithmetic (MAC = 2 FLOPs)

Let `A = B [TP + T(T+1)/2]` be the number of useful query/key pairs, including batch.

```text
MoE + router + shared gate:
F_MoE = LBT [6H(KI + Is) + 2HE + 2H]

Full attention:
F_FA = La [2BTH(3Q + 2R) + 4QA]

GDN:
F_GDN = Lg BT [2H(2Dk + 3Dv + 2nv)
              + 2c(2Dk + Dv)
              + nv*dv*(7dk + 2)]

LM head:
F_head = 2BMHV

Unmerged Edge0 LoRA:
F_LoRA = Σ 2BT r_j(in_j + out_j) = 42,332,160 BT
```

Attention's Q projection produces **both queries and an output gate**: its output width is 2Q. K/V use R and the output projection uses Q. Q is 4096, not H=2048. GDN separately projects qkv, z, a, b and its output. Its recurrent count comes from state decay (dk), k·state (2dk), delta (2), the rank-one update (2dk), and q·state (2dk), for each value head and channel.

The Edge0 LoRA term represents 310 rank-16 adapters: 30 GDN layers with 8 targets and 10 attention layers with 7 targets. In a GDN layer the targets are qkv, z, a, b, out, and shared-expert gate/up/down; in an attention layer they are q, k, v, o, and shared-expert gate/up/down. This gives 1,099,776 and 933,888 FLOPs/token/layer respectively. Routed experts are not included in these adapter targets. The analyzed prefill path uses the ordinary router; decode prerouter work is not added to prefill.

Substituting the dimensions yields:

```text
Qwen = B [4,985,733,120 T + 81,920 T(2P+T+1) + 1,017,118,720 M]
Edge0 = B [4,021,432,320 T + 81,920 T(2P+T+1) + 1,017,118,720 M]
```

Edge0 includes unmerged LoRA. Excluding its independent matrix operations changes the linear coefficient to 3,979,100,160. That toggle is an arithmetic scenario, not a claim that merging LoRA into quantized weights preserves numerics.

## Reference results

B=1, P=0, M=1; Edge0 includes unmerged LoRA. Values below exclude the separate scalar ledger.

| New tokens | Qwen TFLOPs | Edge0 TFLOPs | Reduction |
|---:|---:|---:|---:|
| 2048 | 10.55556370432 | 8.58067566592 | 18.71% |
| 8192 | 46.34237206528 | 38.44281991168 | 17.05% |
| 32768 | 251.33713457152 | 219.73892595712 | 12.57% |
| 131072 | 2060.87664959488 | 1934.48381513728 | 6.13% |

At 2048 tokens, GDN contributes 4.36929 TFLOPs and full attention 1.46046 TFLOPs to both profiles. Reducing routed expert count does not reduce these components. At longer contexts the common quadratic attention term further reduces the percentage saved. The difference for equal T/P/B/M is 964,300,800 BT FLOPs when Edge0 LoRA is enabled; it is not a predicted speedup.

The retained 7253-token, four-logits-position Edge0 audit is reproduced exactly: **33,481,595,514,880 FLOPs**. Regression tests also check the 2048-token GDN-layer audit, cached-prefix partitioning, batch scaling, logits policies, LoRA toggling, and invalid inputs.

## Scalar ledger and execution boundaries

`prefill-model.js` returns a separate `scalar` ledger with explicit formulas for outer/final RMS norms, router softmax and selected-score normalization, SwiGLU, expert mixing, attention Q/K norms, partial RoPE application, attention score scaling/softmax, attention output gating, GDN normalization/gating, convolution activation, decay/beta transforms, and LoRA scaling/addition. The page exposes every row and exports it in JSON.

For that ledger, arithmetic operations and exp/log/rsqrt each count as one **scalar operation**, not equal-cost hardware FLOPs. Top-k/max comparisons, indexing, quantization unpack/dequantization, dtype conversions, RoPE-table construction, allocation, copies and scheduling are excluded. The state-decay constant transform is counted once, not once per chunk. These conventions are suitable for an explicit arithmetic inventory, not an exact kernel-instruction tally.

Useful causal attention uses the lower triangle. A full-square convention instead gives 4LaQBT² when P=0, nearly twice the pair work. Real kernels may perform masked tile work. Chunking retains the prior chunks as a prefix and must not reset P to zero. Chunkwise matrix GDN kernels can have different arithmetic from the recurrent reference used here. Quantization changes storage and kernel execution, not the logical matrix MAC count.

## Primary sources and version scope

- [Qwen model config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json): architecture dimensions and K8.
- [Pinned Edge0 config](https://huggingface.co/Edge0/Edge0-35B-A3B-preview/blob/3fe15cbf2bd5bbcdfd611035e1ac88971ff1cdab/config.json) and [LoRA artifact](https://huggingface.co/Edge0/Edge0-35B-A3B-preview/blob/3fe15cbf2bd5bbcdfd611035e1ac88971ff1cdab/lora_edge0_35b.safetensors): the historical experimental checkpoint profile.
- [Edge0 runtime profile](https://github.com/Edge0-AI/Edge0/blob/main/docs/models/edge0-35b.md): K4; later runtime releases may change execution details.
- [MLX-LM v0.31.0 GDN projections](https://github.com/ml-explore/mlx-lm/blob/v0.31.0/mlx_lm/models/qwen3_5.py), [gated attention and MoE](https://github.com/ml-explore/mlx-lm/blob/v0.31.0/mlx_lm/models/qwen3_next.py), and [recurrent reference](https://github.com/ml-explore/mlx-lm/blob/v0.31.0/mlx_lm/models/gated_delta.py).

The model does not claim all original-Qwen backends execute this particular GDN kernel. The common reference isolates architecture-level work from backend choices.
