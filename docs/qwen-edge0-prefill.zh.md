# Qwen3.6-35B-A3B 与 Edge0 K=4：prefill 算术核算

2026-11-09。只读配置、代码和保留核算结果；没有运行模型实验。[交互计算器](../prefill.html) · [可执行公式](../prefill-model.js) · [回归检查](../tests/prefill-model.test.js) · [English](qwen-edge0-prefill.md)。

## 计数范围

以下 F_main 覆盖文本模型的主要矩阵乘法、depthwise convolution、GDN recurrent state update 和 LM head。一次 MAC=2 FLOPs。矩阵量化不改变逻辑乘加数量。GDN 统一使用递推参考算法，attention 计因果掩码内的有效 pair；不同后端的 chunkwise GDN、masked tile、量化解包及重计算不保证执行同样数量的指令。

归一化、softmax、激活、门控、残差及 LoRA scale/add 另有明确 scalar-operation 账目，见计算器展开表和 prefill-model.js。exp/log/rsqrt 各计1次只是列账约定，不能当作同成本 FLOP 或直接用于运行时间估计。完整执行成本还包括 top-k/max 比较、索引、dtype 转换、RoPE 表生成、量化解包和反量化、调度及数据搬运；这些没有跨后端统一的 FLOPs 常数。

范围不含视觉编码器、可选 MTP、采样、训练反向。词嵌入为查表，不按词表 dense 矩阵乘计算。

## 参数

T 为每个序列新处理的 tokens；P 为已经完成 prefill 且 KV/GDN 状态有效的缓存前缀；B 为等长序列数。M 为每个序列实际送入 LM head 的位置数：生成式最小输出 M=1，每 chunk 末尾 M=ceil(T/C)，全 token logits 为 M=T。不同请求长度不能简单用一个平均长度替代平方项。

| 符号 | 含义 | 数值 |
|---|---|---:|
| L / La / Lg | 总层 / full-attention 层 / GDN 层 | 40 / 10 / 30 |
| H | hidden size | 2048 |
| I / Is | routed / shared expert intermediate | 512 / 512 |
| E | 路由专家总数 | 256 |
| K | 每 token 路由专家数 | Qwen 8；Edge0 4 |
| nq / nkv / d | full-attention query heads / KV heads / head dim | 16 / 2 / 256 |
| Q / R | nq*d / nkv*d | 4096 / 512 |
| nk / nv | GDN key heads / value heads | 16 / 32 |
| dk / dv | GDN key / value head dim | 128 / 128 |
| Dk / Dv | nk*dk / nv*dv | 2048 / 4096 |
| c | GDN depthwise-convolution kernel width | 4 |
| V | 词表大小 | 248320 |

注意 Q=4096 不等于 H=2048；不能用 hidden size / head count 推导配置已经明确给出的 head_dim。

## 全模型主体公式

定义包含 batch 的有效 attention pair 数：

    A = B [T P + T(T+1)/2]

MoE 模块（每层都有，包含路由和共享门控投影）：

    F_MoE = L B T [6 H (K I + Is) + 2 H E + 2 H]

Full attention 的 Q 投影同时产生 query 和输出 gate，输出宽度为 2Q：

    F_FA = La {2 B T H (3Q + 2R) + 4 Q A}

其中投影项为 Q+gate:4BTHQ，K/V:4BTHR，输出:2BTHQ。QKᵀ 与 AV 共为 4QA。

GDN 包含 qkv、z、a/b、输出投影、卷积和递推：

    F_GDN = Lg B T {
        2H(2Dk + 3Dv + 2nv)
        + 2c(2Dk + Dv)
        + nv*dv*(7dk + 2)
    }

GDN recurrent reference 每个 value head、value channel：状态衰减 dk；k·state 为2dk；delta=(v-dot)*beta 为2；rank-one update 为2dk；输出 q·state 为2dk。共7dk+2。使用MAC=2近似，若逐项扣除 dot 的初始加法，数字略有变化。

LM head：

    F_head = 2 B M H V

Edge0 未合并 rank-16 LoRA：

    F_LoRA = sum_j 2 B T r_j (in_j + out_j)
           = 42,332,160 B T

该 LoRA 系数来自固定实验 checkpoint 的310个target/header核算；不是对任何未来 Edge0 版本的假设。原版 Qwen 不加这一项。保留实验路径的 prefill 使用原 router；decode prerouter 不加到 prefill。

    F_main,Qwen = F_MoE(K=8) + F_FA + F_GDN + F_head
    F_main,Edge0 = F_MoE(K=4) + F_FA + F_GDN + F_head + F_LoRA

代入后（单位 FLOPs）：

    F_main,Qwen = B [4,985,733,120 T
                       + 81,920 T(2P+T+1) + 1,017,118,720 M]

    F_main,Edge0 = B [4,021,432,320 T
                        + 81,920 T(2P+T+1) + 1,017,118,720 M]

若 Edge0 不执行独立 LoRA 矩阵，将其线性系数换成3,979,100,160。这里是算术路径假设，不承诺将LoRA合并到量化权重保持同样数值。

在相同T、P、B、M条件下，差值为964,300,800 B T FLOPs。单看K变化，路由FFN节省1,006,632,960 B T；Edge0的独立LoRA抵消其中42,332,160 B T。

## 数值对比

下表B=1，P=0，M=1，Edge0包含独立LoRA，单位TFLOPs。仅逻辑主体，非实测；scalar-operation账目没有混入。

| 新tokens | Qwen K8 | Edge0 K4 | 主体计算减少 |
|---:|---:|---:|---:|
| 2048 | 10.5556 | 8.5807 | 18.71% |
| 8192 | 46.3424 | 38.4428 | 17.05% |
| 32768 | 251.3371 | 219.7389 | 12.57% |
| 131072 | 2060.8766 | 1934.4838 | 6.13% |

上下文越长，不随K缩减的 full-attention 二次项越大，整模型节省比例越小。GDN投影也是大项。因此不能从路由专家减半推断整模型计算减半，更不能据此推断速度翻倍。

如果像参考网页那样统计完整 attention 方阵，P=0时 attention QKᵀ+AV 为4LaQBT²；本核算有效因果工作为2LaQBT(T+1)。两者不是同一计数口径，前者也不等于所有实际 kernel 的执行量。分chunk后，旧chunk成为后续chunk的前缀，不能把每块P重置为0。

LM head若对所有tokens输出logits，每token新增约1.017 GFLOPs。这里默认M=1是明确的生成式对比假设，不宣称每个后端都自动只计算末位置。

## 核验与来源

- 对比保留 Edge0 审计 T=7253 / P=0 / M=4：33,481,595,514,880 FLOPs，完全一致。
- 逐算子求和与封闭公式在多种T/P下吻合；分块并保留前缀的attention总量吻合。
- [官方Qwen模型配置](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
- [Edge0 K4运行配置](https://github.com/Edge0-AI/Edge0/blob/main/docs/models/edge0-35b.md)
- [MLX-LM v0.31.0 GDN层与投影](https://github.com/ml-explore/mlx-lm/blob/v0.31.0/mlx_lm/models/qwen3_5.py)
- [MLX-LM v0.31.0 gated full attention与MoE](https://github.com/ml-explore/mlx-lm/blob/v0.31.0/mlx_lm/models/qwen3_next.py)
- [MLX-LM v0.31.0 GDN recurrence参考](https://github.com/ml-explore/mlx-lm/blob/v0.31.0/mlx_lm/models/gated_delta.py)
- [参考页面](https://zihaopu.cn/llm-perf-model/ops.html) 与其 shared.js?v=17：读取于2026-11-09。其computeLayerOps按普通attention单层乘总层数，不能直接表示这里30/10混合层。
