// ============ SHARED DATA & UTILITIES ============
// Used by both index.html (overall performance) and ops.html (per-op breakdown)

const MODELS = {
    // LLaMA family
    'llama-3.2-1b': { name:'LLaMA 3.2 1B', params:1.24e9, activeParams:1.24e9, layers:16, heads:32, kvHeads:8, hiddenDim:2048, headDim:64, interDim:8192, vocabSize:128256, maxCtx:131072, moe:false, arch:'LLaMA' },
    'llama-3.2-3b': { name:'LLaMA 3.2 3B', params:3.21e9, activeParams:3.21e9, layers:28, heads:24, kvHeads:8, hiddenDim:3072, headDim:128, interDim:8192, vocabSize:128256, maxCtx:131072, moe:false, arch:'LLaMA' },
    'llama-3-8b': { name:'LLaMA 3 8B', params:8e9, activeParams:8e9, layers:32, heads:32, kvHeads:8, hiddenDim:4096, headDim:128, interDim:14336, vocabSize:128256, maxCtx:8192, moe:false, arch:'LLaMA' },
    'llama-3.1-8b': { name:'LLaMA 3.1 8B', params:8e9, activeParams:8e9, layers:32, heads:32, kvHeads:8, hiddenDim:4096, headDim:128, interDim:14336, vocabSize:128256, maxCtx:131072, moe:false, arch:'LLaMA' },
    'llama-3-70b': { name:'LLaMA 3 70B', params:70.6e9, activeParams:70.6e9, layers:80, heads:64, kvHeads:8, hiddenDim:8192, headDim:128, interDim:28672, vocabSize:128256, maxCtx:8192, moe:false, arch:'LLaMA' },
    'llama-3.1-70b': { name:'LLaMA 3.1 70B', params:70.6e9, activeParams:70.6e9, layers:80, heads:64, kvHeads:8, hiddenDim:8192, headDim:128, interDim:28672, vocabSize:128256, maxCtx:131072, moe:false, arch:'LLaMA' },
    'llama-3.3-70b': { name:'LLaMA 3.3 70B', params:70.6e9, activeParams:70.6e9, layers:80, heads:64, kvHeads:8, hiddenDim:8192, headDim:128, interDim:28672, vocabSize:128256, maxCtx:131072, moe:false, arch:'LLaMA' },
    'llama-3.1-405b': { name:'LLaMA 3.1 405B', params:405e9, activeParams:405e9, layers:126, heads:128, kvHeads:8, hiddenDim:16384, headDim:128, interDim:53248, vocabSize:128256, maxCtx:131072, moe:false, arch:'LLaMA' },
    // Qwen 3 family
    'qwen3-0.6b': { name:'Qwen3 0.6B', params:0.6e9, activeParams:0.6e9, layers:28, heads:16, kvHeads:8, hiddenDim:1024, headDim:128, interDim:3072, vocabSize:151936, maxCtx:32768, moe:false, arch:'Qwen3' },
    'qwen3-1.7b': { name:'Qwen3 1.7B', params:1.7e9, activeParams:1.7e9, layers:28, heads:16, kvHeads:8, hiddenDim:2048, headDim:128, interDim:6144, vocabSize:151936, maxCtx:32768, moe:false, arch:'Qwen3' },
    'qwen3-4b': { name:'Qwen3 4B', params:4e9, activeParams:4e9, layers:36, heads:32, kvHeads:8, hiddenDim:2560, headDim:128, interDim:9728, vocabSize:151936, maxCtx:32768, moe:false, arch:'Qwen3' },
    'qwen3-8b': { name:'Qwen3 8B', params:8.2e9, activeParams:8.2e9, layers:36, heads:32, kvHeads:8, hiddenDim:4096, headDim:128, interDim:12288, vocabSize:151936, maxCtx:131072, moe:false, arch:'Qwen3' },
    'qwen3-14b': { name:'Qwen3 14B', params:14.8e9, activeParams:14.8e9, layers:40, heads:40, kvHeads:8, hiddenDim:5120, headDim:128, interDim:17408, vocabSize:151936, maxCtx:131072, moe:false, arch:'Qwen3' },
    'qwen3-32b': { name:'Qwen3 32B', params:32.8e9, activeParams:32.8e9, layers:64, heads:64, kvHeads:8, hiddenDim:5120, headDim:128, interDim:25600, vocabSize:151936, maxCtx:131072, moe:false, arch:'Qwen3' },
    'qwen3-30b-a3b': { name:'Qwen3-30B-A3B (MoE)', params:30.5e9, activeParams:3.3e9, layers:48, heads:32, kvHeads:4, hiddenDim:2048, headDim:128, interDim:768, vocabSize:151936, maxCtx:131072, moe:true, numExperts:128, topK:8, arch:'Qwen3' },
    'qwen3-235b-a22b': { name:'Qwen3-235B-A22B (MoE)', params:235e9, activeParams:22e9, layers:94, heads:64, kvHeads:4, hiddenDim:4096, headDim:128, interDim:1536, vocabSize:151936, maxCtx:131072, moe:true, numExperts:128, topK:8, arch:'Qwen3' },
    // Qwen 2.5 family
    'qwen2.5-0.5b': { name:'Qwen 2.5 0.5B', params:0.49e9, activeParams:0.49e9, layers:24, heads:14, kvHeads:2, hiddenDim:896, headDim:64, interDim:4864, vocabSize:151936, maxCtx:32768, moe:false, arch:'Qwen2' },
    'qwen2.5-1.5b': { name:'Qwen 2.5 1.5B', params:1.54e9, activeParams:1.54e9, layers:28, heads:12, kvHeads:2, hiddenDim:1536, headDim:128, interDim:8960, vocabSize:151936, maxCtx:32768, moe:false, arch:'Qwen2' },
    'qwen2.5-3b': { name:'Qwen 2.5 3B', params:3.09e9, activeParams:3.09e9, layers:36, heads:16, kvHeads:2, hiddenDim:2048, headDim:128, interDim:11008, vocabSize:151936, maxCtx:32768, moe:false, arch:'Qwen2' },
    'qwen2.5-7b': { name:'Qwen 2.5 7B', params:7.6e9, activeParams:7.6e9, layers:28, heads:28, kvHeads:4, hiddenDim:3584, headDim:128, interDim:18944, vocabSize:152064, maxCtx:131072, moe:false, arch:'Qwen2' },
    'qwen2.5-14b': { name:'Qwen 2.5 14B', params:14.7e9, activeParams:14.7e9, layers:48, heads:40, kvHeads:8, hiddenDim:5120, headDim:128, interDim:13824, vocabSize:152064, maxCtx:131072, moe:false, arch:'Qwen2' },
    'qwen2.5-32b': { name:'Qwen 2.5 32B', params:32.5e9, activeParams:32.5e9, layers:64, heads:40, kvHeads:8, hiddenDim:5120, headDim:128, interDim:27648, vocabSize:152064, maxCtx:131072, moe:false, arch:'Qwen2' },
    'qwen2.5-72b': { name:'Qwen 2.5 72B', params:72.7e9, activeParams:72.7e9, layers:80, heads:64, kvHeads:8, hiddenDim:8192, headDim:128, interDim:29568, vocabSize:152064, maxCtx:131072, moe:false, arch:'Qwen2' },
    // Mistral / Mixtral
    'mistral-7b': { name:'Mistral 7B', params:7.3e9, activeParams:7.3e9, layers:32, heads:32, kvHeads:8, hiddenDim:4096, headDim:128, interDim:14336, vocabSize:32000, maxCtx:32768, moe:false, arch:'Mistral' },
    'mistral-small-24b': { name:'Mistral Small 24B', params:24e9, activeParams:24e9, layers:40, heads:32, kvHeads:8, hiddenDim:5120, headDim:128, interDim:14336, vocabSize:32768, maxCtx:32768, moe:false, arch:'Mistral' },
    'mixtral-8x7b': { name:'Mixtral 8x7B', params:46.7e9, activeParams:12.9e9, layers:32, heads:32, kvHeads:8, hiddenDim:4096, headDim:128, interDim:14336, vocabSize:32000, maxCtx:32768, moe:true, numExperts:8, topK:2, arch:'Mixtral' },
    'mixtral-8x22b': { name:'Mixtral 8x22B', params:141e9, activeParams:39e9, layers:56, heads:48, kvHeads:8, hiddenDim:6144, headDim:128, interDim:16384, vocabSize:32768, maxCtx:65536, moe:true, numExperts:8, topK:2, arch:'Mixtral' },
    // DeepSeek
    'deepseek-v2-lite': { name:'DeepSeek-V2-Lite', params:15.7e9, activeParams:2.4e9, layers:27, heads:16, kvHeads:16, hiddenDim:2048, headDim:128, interDim:10944, vocabSize:102400, maxCtx:32768, moe:true, numExperts:64, topK:6, arch:'DeepSeek', mlaCompressedDim:512 },
    'deepseek-v2': { name:'DeepSeek-V2', params:236e9, activeParams:21e9, layers:60, heads:128, kvHeads:128, hiddenDim:5120, headDim:128, interDim:12288, vocabSize:102400, maxCtx:131072, moe:true, numExperts:160, topK:6, arch:'DeepSeek', mlaCompressedDim:512, note:'MLA + shared experts' },
    'deepseek-v3': { name:'DeepSeek-V3', params:671e9, activeParams:37e9, layers:61, heads:128, kvHeads:128, hiddenDim:7168, headDim:128, interDim:18432, vocabSize:129280, maxCtx:131072, moe:true, numExperts:256, topK:8, arch:'DeepSeek', mlaCompressedDim:512, note:'MLA + shared experts' },
    'deepseek-r1': { name:'DeepSeek-R1', params:671e9, activeParams:37e9, layers:61, heads:128, kvHeads:128, hiddenDim:7168, headDim:128, interDim:18432, vocabSize:129280, maxCtx:131072, moe:true, numExperts:256, topK:8, arch:'DeepSeek', mlaCompressedDim:512, note:'MLA + shared experts' },
    'deepseek-r1-7b': { name:'DeepSeek-R1-Distill 7B', params:7.6e9, activeParams:7.6e9, layers:28, heads:28, kvHeads:4, hiddenDim:3584, headDim:128, interDim:18944, vocabSize:152064, maxCtx:131072, moe:false, arch:'Qwen2', note:'Distilled from Qwen 2.5 7B' },
    'deepseek-r1-14b': { name:'DeepSeek-R1-Distill 14B', params:14.7e9, activeParams:14.7e9, layers:48, heads:40, kvHeads:8, hiddenDim:5120, headDim:128, interDim:13824, vocabSize:152064, maxCtx:131072, moe:false, arch:'Qwen2', note:'Distilled from Qwen 2.5 14B' },
    'deepseek-r1-32b': { name:'DeepSeek-R1-Distill 32B', params:32.5e9, activeParams:32.5e9, layers:64, heads:40, kvHeads:8, hiddenDim:5120, headDim:128, interDim:27648, vocabSize:152064, maxCtx:131072, moe:false, arch:'Qwen2', note:'Distilled from Qwen 2.5 32B' },
    'deepseek-r1-70b': { name:'DeepSeek-R1-Distill 70B', params:70.6e9, activeParams:70.6e9, layers:80, heads:64, kvHeads:8, hiddenDim:8192, headDim:128, interDim:28672, vocabSize:128256, maxCtx:131072, moe:false, arch:'LLaMA', note:'Distilled from LLaMA 3 70B' },
    // Phi
    'phi-3-mini': { name:'Phi-3 Mini 3.8B', params:3.8e9, activeParams:3.8e9, layers:32, heads:32, kvHeads:32, hiddenDim:3072, headDim:96, interDim:8192, vocabSize:32064, maxCtx:128000, moe:false, arch:'Phi' },
    'phi-3-small': { name:'Phi-3 Small 7B', params:7.4e9, activeParams:7.4e9, layers:32, heads:32, kvHeads:8, hiddenDim:4096, headDim:128, interDim:14336, vocabSize:32064, maxCtx:128000, moe:false, arch:'Phi' },
    'phi-3-medium': { name:'Phi-3 Medium 14B', params:14e9, activeParams:14e9, layers:40, heads:40, kvHeads:10, hiddenDim:5120, headDim:128, interDim:17920, vocabSize:32064, maxCtx:128000, moe:false, arch:'Phi' },
    'phi-4': { name:'Phi-4 14B', params:14e9, activeParams:14e9, layers:40, heads:40, kvHeads:10, hiddenDim:5120, headDim:128, interDim:17920, vocabSize:100352, maxCtx:16384, moe:false, arch:'Phi' },
    // Gemma
    'gemma-2-2b': { name:'Gemma 2 2B', params:2.6e9, activeParams:2.6e9, layers:26, heads:8, kvHeads:4, hiddenDim:2304, headDim:256, interDim:9216, vocabSize:256000, maxCtx:8192, moe:false, arch:'Gemma' },
    'gemma-2-9b': { name:'Gemma 2 9B', params:9.2e9, activeParams:9.2e9, layers:42, heads:16, kvHeads:8, hiddenDim:3584, headDim:256, interDim:14336, vocabSize:256000, maxCtx:8192, moe:false, arch:'Gemma' },
    'gemma-2-27b': { name:'Gemma 2 27B', params:27.2e9, activeParams:27.2e9, layers:46, heads:32, kvHeads:16, hiddenDim:4608, headDim:128, interDim:36864, vocabSize:256000, maxCtx:8192, moe:false, arch:'Gemma' }
};

const DEVICES = {
    // interconnectBW: unidirectional per-GPU link bandwidth (GB/s) for AllReduce
    // interconnectLatUs: per-AllReduce latency (μs) incl. software overhead
    // maxGPUs: max realistic multi-GPU count for this device
    'h200-sxm': { name:'NVIDIA H200 SXM', vram:141, bw:4800, fp16Tflops:989, fp8Tflops:1979, int8Tflops:1979, vendor:'nvidia', interconnectBW:450, interconnectLatUs:5, maxGPUs:8 },
    'b200-sxm': { name:'NVIDIA B200 SXM', vram:192, bw:8000, fp16Tflops:2250, fp8Tflops:4500, int8Tflops:4500, vendor:'nvidia', interconnectBW:900, interconnectLatUs:3, maxGPUs:8 },
    'h100-sxm': { name:'NVIDIA H100 SXM', vram:80, bw:3350, fp16Tflops:989, fp8Tflops:1979, int8Tflops:1979, vendor:'nvidia', interconnectBW:450, interconnectLatUs:5, maxGPUs:8 },
    'a100-sxm-80': { name:'NVIDIA A100 80GB', vram:80, bw:2039, fp16Tflops:312, fp8Tflops:312, int8Tflops:624, vendor:'nvidia', interconnectBW:300, interconnectLatUs:5, maxGPUs:8 },
    'a100-sxm-40': { name:'NVIDIA A100 40GB', vram:40, bw:1555, fp16Tflops:312, fp8Tflops:312, int8Tflops:624, vendor:'nvidia', interconnectBW:300, interconnectLatUs:5, maxGPUs:8 },
    'l40s': { name:'NVIDIA L40S', vram:48, bw:864, fp16Tflops:362, fp8Tflops:733, int8Tflops:733, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:8 },
    'l4': { name:'NVIDIA L4', vram:24, bw:300, fp16Tflops:121, fp8Tflops:242, int8Tflops:242, vendor:'nvidia', interconnectBW:0, interconnectLatUs:0, maxGPUs:1 },
    // ── GeForce RTX 50 (Blackwell) ──
    'rtx-5090': { name:'RTX 5090', vram:32, bw:1792, fp16Tflops:209, fp8Tflops:419, int8Tflops:419, vendor:'nvidia', interconnectBW:64, interconnectLatUs:10, maxGPUs:2 },
    'rtx-5080': { name:'RTX 5080', vram:16, bw:960, fp16Tflops:113, fp8Tflops:225, int8Tflops:225, vendor:'nvidia', interconnectBW:64, interconnectLatUs:10, maxGPUs:2 },
    'rtx-5070ti': { name:'RTX 5070 Ti', vram:16, bw:896, fp16Tflops:88, fp8Tflops:176, int8Tflops:176, vendor:'nvidia', interconnectBW:64, interconnectLatUs:10, maxGPUs:2 },
    'rtx-5070': { name:'RTX 5070', vram:12, bw:672, fp16Tflops:62, fp8Tflops:123, int8Tflops:123, vendor:'nvidia', interconnectBW:64, interconnectLatUs:10, maxGPUs:2 },
    // ── GeForce RTX 40 (Ada Lovelace) ──
    'rtx-4090': { name:'RTX 4090', vram:24, bw:1008, fp16Tflops:165, fp8Tflops:330, int8Tflops:330, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4080s': { name:'RTX 4080 SUPER', vram:16, bw:736, fp16Tflops:104, fp8Tflops:209, int8Tflops:209, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4080': { name:'RTX 4080', vram:16, bw:717, fp16Tflops:97, fp8Tflops:194, int8Tflops:194, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4070tis': { name:'RTX 4070 Ti SUPER', vram:16, bw:672, fp16Tflops:88, fp8Tflops:176, int8Tflops:176, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4070ti': { name:'RTX 4070 Ti', vram:12, bw:504, fp16Tflops:80, fp8Tflops:160, int8Tflops:160, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4070s': { name:'RTX 4070 SUPER', vram:12, bw:504, fp16Tflops:71, fp8Tflops:142, int8Tflops:142, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4070': { name:'RTX 4070', vram:12, bw:504, fp16Tflops:58, fp8Tflops:116, int8Tflops:116, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4060ti-16': { name:'RTX 4060 Ti 16GB', vram:16, bw:288, fp16Tflops:44, fp8Tflops:88, int8Tflops:88, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4060ti': { name:'RTX 4060 Ti', vram:8, bw:288, fp16Tflops:44, fp8Tflops:88, int8Tflops:88, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-4060': { name:'RTX 4060', vram:8, bw:272, fp16Tflops:30, fp8Tflops:61, int8Tflops:61, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    // ── GeForce RTX 30 (Ampere — no FP8) ──
    'rtx-3090': { name:'RTX 3090', vram:24, bw:936, fp16Tflops:71, fp8Tflops:71, int8Tflops:142, vendor:'nvidia', interconnectBW:56, interconnectLatUs:8, maxGPUs:2 },
    'rtx-3080ti': { name:'RTX 3080 Ti', vram:12, bw:912, fp16Tflops:68, fp8Tflops:68, int8Tflops:136, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-3080': { name:'RTX 3080', vram:10, bw:760, fp16Tflops:60, fp8Tflops:60, int8Tflops:119, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-3070ti': { name:'RTX 3070 Ti', vram:8, bw:608, fp16Tflops:44, fp8Tflops:44, int8Tflops:87, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-3070': { name:'RTX 3070', vram:8, bw:448, fp16Tflops:41, fp8Tflops:41, int8Tflops:81, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-3060ti': { name:'RTX 3060 Ti', vram:8, bw:448, fp16Tflops:32, fp8Tflops:32, int8Tflops:65, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rtx-3060': { name:'RTX 3060', vram:12, bw:360, fp16Tflops:25, fp8Tflops:25, int8Tflops:51, vendor:'nvidia', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    // ── NVIDIA Edge ──
    'dgx-spark': { name:'NVIDIA DGX Spark (GB10)', vram:128, bw:273, fp16Tflops:209, fp8Tflops:418, int8Tflops:418, vendor:'nvidia', interconnectBW:0, interconnectLatUs:0, maxGPUs:1 },
    'mi300x': { name:'AMD MI300X', vram:192, bw:5300, fp16Tflops:1307, fp8Tflops:2614, int8Tflops:2614, vendor:'amd', interconnectBW:448, interconnectLatUs:5, maxGPUs:8 },
    'mi325x': { name:'AMD MI325X', vram:256, bw:6000, fp16Tflops:1307, fp8Tflops:2614, int8Tflops:2614, vendor:'amd', interconnectBW:448, interconnectLatUs:5, maxGPUs:8 },
    'mi250x': { name:'AMD MI250X', vram:128, bw:3277, fp16Tflops:383, fp8Tflops:383, int8Tflops:383, vendor:'amd', interconnectBW:400, interconnectLatUs:5, maxGPUs:8 },
    'rx-9700xt': { name:'Radeon RX 9700 XT', vram:24, bw:864, fp16Tflops:56, fp8Tflops:112, int8Tflops:112, vendor:'amd', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'rx-9070xt': { name:'Radeon RX 9070 XT', vram:16, bw:640, fp16Tflops:49, fp8Tflops:98, int8Tflops:98, vendor:'amd', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'ai-pro-r9700': { name:'Radeon AI PRO R9700', vram:32, bw:645, fp16Tflops:48, fp8Tflops:96, int8Tflops:96, vendor:'amd', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'w7900': { name:'Radeon PRO W7900', vram:48, bw:864, fp16Tflops:61, fp8Tflops:61, int8Tflops:122, vendor:'amd', interconnectBW:32, interconnectLatUs:15, maxGPUs:4 },
    'rx-7900xtx': { name:'Radeon RX 7900 XTX', vram:24, bw:960, fp16Tflops:123, fp8Tflops:123, int8Tflops:123, vendor:'amd', interconnectBW:32, interconnectLatUs:15, maxGPUs:2 },
    'strix-halo': { name:'AMD Ryzen AI Max 395 (Strix Halo)', vram:128, bw:256, fp16Tflops:30, fp8Tflops:30, int8Tflops:60, vendor:'amd', interconnectBW:0, interconnectLatUs:0, maxGPUs:1 },
    'm4-max': { name:'Apple M4 Max', vram:128, bw:546, fp16Tflops:54, fp8Tflops:54, int8Tflops:54, vendor:'apple', interconnectBW:0, interconnectLatUs:0, maxGPUs:1 },
    'm4-pro': { name:'Apple M4 Pro', vram:48, bw:273, fp16Tflops:22, fp8Tflops:22, int8Tflops:22, vendor:'apple', interconnectBW:0, interconnectLatUs:0, maxGPUs:1 },
    'm2-ultra': { name:'Apple M2 Ultra', vram:192, bw:800, fp16Tflops:27, fp8Tflops:27, int8Tflops:27, vendor:'apple', interconnectBW:0, interconnectLatUs:0, maxGPUs:1 },
    // ── Intel Gaudi ──
    'gaudi2': { name:'Intel Gaudi 2', vram:96, bw:2460, fp16Tflops:420, fp8Tflops:865, int8Tflops:865, vendor:'intel', interconnectBW:150, interconnectLatUs:10, maxGPUs:8 },
    'gaudi3': { name:'Intel Gaudi 3', vram:128, bw:3700, fp16Tflops:900, fp8Tflops:1835, int8Tflops:1835, vendor:'intel', interconnectBW:300, interconnectLatUs:8, maxGPUs:8 },
    // ── Google TPU ──
    'tpu-v5p': { name:'Google TPU v5p', vram:95, bw:2765, fp16Tflops:459, fp8Tflops:459, int8Tflops:918, vendor:'google', interconnectBW:400, interconnectLatUs:5, maxGPUs:8 },
    'tpu-v6e': { name:'Google TPU v6e (Trillium)', vram:32, bw:1640, fp16Tflops:918, fp8Tflops:918, int8Tflops:1836, vendor:'google', interconnectBW:400, interconnectLatUs:5, maxGPUs:8 }
};

// ============ CONSTANTS ============
const COMPUTE_UTIL = 0.75;
const BW_UTIL = 0.80;
const NET_BW_UTIL = 0.80;
const TP_COMM_OVERHEAD = 0.05;

function parsePctInput(el, fallback) {
    if (!el) return fallback;
    const v = parseFloat(el.value);
    if (!isFinite(v)) return fallback;
    return Math.min(1, Math.max(0.01, v / 100));
}

function getUtilizationConfig() {
    const computeUtil = parsePctInput(document.getElementById('flops-util'), COMPUTE_UTIL);
    const bwUtil = parsePctInput(document.getElementById('mem-util'), BW_UTIL);
    const netBwUtil = parsePctInput(document.getElementById('net-util'), NET_BW_UTIL);
    return { computeUtil, bwUtil, netBwUtil };
}

// ============ UTILITY FUNCTIONS ============
function getDeviceTflops(device, quantBits) {
    if (quantBits <= 8) {
        const best = Math.max(device.fp8Tflops, device.int8Tflops);
        if (best > device.fp16Tflops) return best;
    }
    return device.fp16Tflops;
}

function calcModelSizeGB(model, quantBits) {
    return (model.params * quantBits / 8) / 1e9;
}

function calcActiveModelSizeGB(model, quantBits) {
    return (model.activeParams * quantBits / 8) / 1e9;
}

function calcKVCacheGB(model, seqLen, batchSize, quantBits, kvQuantBits) {
    if (model.mlaCompressedDim) {
        return (model.layers * model.mlaCompressedDim * seqLen * 2 * batchSize) / 1e9;
    }
    // kvQuantBits: explicit KV cache precision; if omitted, auto from weight quant
    const kvBits = kvQuantBits != null ? kvQuantBits : (quantBits >= 16 ? 16 : 8);
    const bytesPerElem = kvBits / 8;
    return (2 * model.layers * model.kvHeads * model.headDim * seqLen * bytesPerElem * batchSize) / 1e9;
}

function calcAttentionFlopsPrefill(model, seqLen, batchSize) {
    return 4 * batchSize * model.heads * seqLen * seqLen * model.headDim * model.layers;
}

function calcAttentionFlopsDecodeStep(model, currentPos, batchSize) {
    return 4 * batchSize * model.heads * currentPos * model.headDim * model.layers;
}

function fmtFlops(f) {
    if (f >= 1e15) return (f/1e15).toFixed(1) + ' PFLOP';
    if (f >= 1e12) return (f/1e12).toFixed(1) + ' TFLOP';
    if (f >= 1e9) return (f/1e9).toFixed(1) + ' GFLOP';
    return f.toFixed(0) + ' FLOP';
}

function fmtBytes(bytes) {
    if (bytes >= 1e12) return (bytes / 1e12).toFixed(1) + ' TB';
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
    return bytes.toFixed(0) + ' B';
}

function formatMs(ms) {
    if (ms < 1) return (ms * 1000).toFixed(0) + ' μs';
    if (ms < 1000) return ms.toFixed(1) + ' ms';
    return (ms / 1000).toFixed(2) + ' s';
}

function formatGB(gb) {
    if (gb < 1) return (gb * 1024).toFixed(0) + ' MB';
    return gb.toFixed(1) + ' GB';
}

// ============ SHARED CONFIG READER ============
function readConfig() {
    const modelKey = document.getElementById('model-select').value;
    const deviceKey = document.getElementById('device-select').value;
    const quantBits = parseInt(document.getElementById('quant').value);
    const kvQuantEl = document.getElementById('kv-quant');
    const kvBits = kvQuantEl ? parseInt(kvQuantEl.value) : (quantBits >= 16 ? 16 : 8);
    const promptLen = parseInt(document.getElementById('prompt-len').value);
    const outputLen = parseInt(document.getElementById('output-len').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);
    const tp = parseInt(document.getElementById('tp-size').value);
    const flashAttn = document.getElementById('flash-attn').checked;
    const model = MODELS[modelKey];
    const device = DEVICES[deviceKey];

    const { computeUtil, bwUtil, netBwUtil } = getUtilizationConfig();

    let tpEff = Math.max(1.0 - TP_COMM_OVERHEAD * Math.max(0, tp - 1), 0.5);
    const rawTflops = getDeviceTflops(device, quantBits);
    const effectiveTflops = rawTflops * tp * tpEff * computeUtil;
    const effectiveBW = device.bw * tp * tpEff * bwUtil;

    return { modelKey, deviceKey, quantBits, kvBits, promptLen, outputLen, batchSize, tp, flashAttn,
             model, device, effectiveTflops, effectiveBW, computeUtil, bwUtil, netBwUtil };
}

// ============ PER-OP COMPUTATION (shared) ============
function computeLayerOps(model, effTflops, effBW, quantBits, tokens, batchSize, flashAttn, phase, kvPos) {
    const B = batchSize, S = tokens;
    const H = model.hiddenDim, D = model.headDim;
    const Nh = model.heads, Nkv = model.kvHeads;
    const I = model.interDim;
    const qB = quantBits / 8;
    const aB = 2;
    const ops = [];

    function pushOp(name, flops, readBytes, writeBytes, cat) {
        const totalBytes = readBytes + writeBytes;
        const ai = totalBytes > 0 ? flops / totalBytes : 0;
        const cMs = flops / (effTflops * 1e12) * 1000;
        const mMs = totalBytes / (effBW * 1e9) * 1000;
        ops.push({ name, flops, readBytes, writeBytes, totalBytes, ai, computeMs: cMs, memMs: mMs,
                   timeMs: Math.max(cMs, mMs), bound: cMs >= mMs ? 'compute' : 'memory', category: cat });
    }

    function linear(name, M, N, cat) {
        pushOp(name, 2 * B * S * M * N, M * N * qB + B * S * M * aB, B * S * N * aB, cat);
    }

    function elemwise(name, n, fpe, cat) {
        pushOp(name, n * fpe, n * aB, n * aB, cat);
    }

    // ── Pre-attention Norm ──
    elemwise('RMSNorm₁', B * S * H, 5, 'norm');

    // ── QKV Projections ──
    linear('Q Proj', H, Nh * D, 'attn');
    linear('K Proj', H, Nkv * D, 'attn');
    linear('V Proj', H, Nkv * D, 'attn');

    // ── Attention (QK^T + Softmax + AV) ──
    {
        const Skv = phase === 'prefill' ? S : kvPos;
        const flops = 4 * B * Nh * S * Skv * D;
        let rB, wB;
        if (flashAttn) {
            rB = B * (Nh * S + 2 * Nkv * Skv) * D * aB;
            wB = B * Nh * S * D * aB;
        } else {
            const scoreSz = B * Nh * S * Skv * 4;
            rB = B * (Nh * S + 2 * Nkv * Skv) * D * aB + scoreSz;
            wB = scoreSz + B * Nh * S * D * aB;
        }
        pushOp('Attention (QK^T+AV)', flops, rB, wB, 'attn');
    }

    // ── Output Projection ──
    linear('O Proj', Nh * D, H, 'attn');

    // ── Residual + Pre-FFN Norm ──
    elemwise('Residual₁ + RMSNorm₂', B * S * H, 6, 'norm');

    // ── FFN / MoE ──
    if (model.moe) {
        linear('Router', H, model.numExperts || 8, 'ffn');
        const k = model.topK || 2;
        linear('Gate ×' + k + ' experts', H, I * k, 'ffn');
        linear('Up ×' + k + ' experts', H, I * k, 'ffn');
        elemwise('SiLU·Gate', B * S * I * k, 2, 'ffn');
        linear('Down ×' + k + ' experts', I * k, H, 'ffn');
    } else {
        linear('FFN Gate (W₁)', H, I, 'ffn');
        linear('FFN Up (W₃)', H, I, 'ffn');
        elemwise('SiLU·Gate', B * S * I, 2, 'ffn');
        linear('FFN Down (W₂)', I, H, 'ffn');
    }

    // ── Final Residual ──
    elemwise('Residual₂', B * S * H, 1, 'norm');

    return ops;
}

// ============ MULTI-GPU SCALING ============
function calcMultiGPUScaling(model, device, quantBits, promptLen, outputLen, batchSize, flashAttn, kvBits, netBwUtil = NET_BW_UTIL, computeUtil = COMPUTE_UTIL, bwUtil = BW_UTIL) {
    const maxGPUs = device.maxGPUs || 1;
    const icBW = (device.interconnectBW || 0) * 1e9;
    const icLat = (device.interconnectLatUs || 0) * 1e-6;
    const rawTflops = getDeviceTflops(device, quantBits);
    const H = model.hiddenDim;
    const L = model.layers;
    const results = [];

    // Graph mode: all ops captured in CUDA/HIP graph, per-kernel launch latency = 0
    const bw = (icBW > 0 ? icBW : 32e9) * netBwUtil;

    // Point-to-point send between PP stages (pure bandwidth)
    function p2pTime(tokens) {
        return batchSize * tokens * H * 2 / bw;
    }

    // TP communication per layer: 2 × (ReduceScatter + AllGather)
    // Post-attn and post-FFN each need one RS+AG pair
    function tpCommPerLayer(tp, tokens) {
        if (tp <= 1) return 0;
        const msg = batchSize * tokens * H * 2;           // activation size (bytes)
        const rsTime = (tp - 1) / tp * msg / bw;          // ReduceScatter
        const agTime = (tp - 1) / tp * msg / bw;          // AllGather
        return 2 * (rsTime + agTime);                      // ×2 ops per layer
    }

    for (const numGPUs of [1, 2, 4, 8]) {
        if (numGPUs > maxGPUs) {
            results.push({ numGPUs, tp: numGPUs, pp: 1, supported: false });
            continue;
        }

        // Enumerate all TP×PP combos where TP×PP = numGPUs
        for (let tp = numGPUs; tp >= 1; tp--) {
            if (numGPUs % tp !== 0) continue;
            const pp = numGPUs / tp;

            // Memory per GPU: weights & KV split by total GPUs, activations per stage
            const modelSizeGB = calcModelSizeGB(model, quantBits);
            const kvGB = calcKVCacheGB(model, promptLen + outputLen, batchSize, quantBits, kvBits);
            let actGB = (H * promptLen * batchSize * 4 * 2) / 1e9;
            if (!flashAttn) actGB += (batchSize * Math.ceil(model.heads / tp) * promptLen * promptLen * 4) / 1e9;
            const memPerGPU = modelSizeGB / numGPUs + kvGB / numGPUs + actGB;

            if (memPerGPU > device.vram) {
                results.push({ numGPUs, tp, pp, supported: true, fitsInMemory: false, memPerGPU });
                continue;
            }

            // ── Prefill ──
            // Compute split by TP only; PP stages sequential → same total compute
            const pfLinear = 2 * model.activeParams * promptLen * batchSize;
            const pfAttn = calcAttentionFlopsPrefill(model, promptLen, batchSize);
            let pfComputeS;
            if (flashAttn) {
                pfComputeS = (pfLinear + pfAttn) / (rawTflops * tp * computeUtil * 1e12);
            } else {
                pfComputeS = pfLinear / (rawTflops * tp * computeUtil * 1e12)
                           + pfAttn / (rawTflops * tp * computeUtil * 1e12 * (0.40 / computeUtil));
            }
            const pfTPComm = tpCommPerLayer(tp, promptLen) * L;
            const pfPPComm = (pp > 1) ? p2pTime(promptLen) * (pp - 1) : 0;
            const prefillMs = (pfComputeS + pfTPComm + pfPPComm) * 1000;

            // ── Decode ──
            // Memory-bound: PP doesn't help (stages serial, total read same)
            // Read per GPU per stage = (weights/pp)/tp + (kv/pp)/tp
            // Total = pp stages × time_per_stage = (weights + kv) / tp / BW
            const activeBytes = model.activeParams * quantBits / 8;
            const avgPos = promptLen + outputLen / 2;
            let kvRead = calcKVCacheGB(model, Math.floor(avgPos), batchSize, quantBits, kvBits) * 1e9;
            if (!flashAttn) kvRead += 2 * batchSize * model.heads * avgPos * 4 * L;
            const totalRead = activeBytes + kvRead;
            const decMemS = (totalRead / tp) / (device.bw * bwUtil * 1e9);
            const decFlops = 2 * model.activeParams * batchSize + calcAttentionFlopsDecodeStep(model, avgPos, batchSize);
            const decCompS = decFlops / (rawTflops * tp * computeUtil * 1e12);
            const decTPComm = tpCommPerLayer(tp, 1) * L;
            const decPPComm = (pp > 1) ? p2pTime(1) * (pp - 1) : 0;
            const decStepS = Math.max(decMemS, decCompS) + decTPComm + decPPComm;
            const decTokMs = (decStepS / batchSize) * 1000;
            const decodeTPS = decTokMs > 0 ? 1000 / decTokMs : 0;
            const totalMs = prefillMs + decTokMs * outputLen;

            results.push({ numGPUs, tp, pp, supported: true, fitsInMemory: true, memPerGPU,
                           prefillMs, decodeTPS, decodeTokenMs: decTokMs, totalMs });
        }
    }

    // Speedup & efficiency relative to smallest fitting config
    const base = results.find(r => r.supported !== false && r.fitsInMemory);
    for (const r of results) {
        if (r.supported === false || !r.fitsInMemory) continue;
        if (base) {
            r.decodeSpeedup = base.decodeTokenMs / r.decodeTokenMs;
            r.prefillSpeedup = base.prefillMs > 0 ? base.prefillMs / r.prefillMs : 1;
            r.efficiency = (r.decodeSpeedup / (r.numGPUs / (base.numGPUs || 1))) * 100;
        } else { r.decodeSpeedup = null; r.prefillSpeedup = null; r.efficiency = null; }
    }
    return results;
}

// ============ SHARED UI HELPERS ============
function populateModelSelect(selectEl) {
    const groups = {};
    for (const [k, m] of Object.entries(MODELS)) {
        let g = m.arch;
        if (m.arch === 'LLaMA') g = 'LLaMA Family';
        else if (m.arch === 'Qwen3' || m.arch === 'Qwen2') g = 'Qwen Family';
        else if (m.arch === 'Mistral' || m.arch === 'Mixtral') g = 'Mistral / Mixtral';
        else if (m.arch === 'DeepSeek') g = 'DeepSeek';
        else if (m.arch === 'Phi') g = 'Phi';
        else if (m.arch === 'Gemma') g = 'Gemma';
        if (!groups[g]) groups[g] = [];
        groups[g].push({ key: k, name: m.name });
    }
    // Keep order consistent
    selectEl.innerHTML = '';
    for (const g of ['LLaMA Family', 'Qwen Family', 'Mistral / Mixtral', 'DeepSeek', 'Phi', 'Gemma']) {
        if (!groups[g]) continue;
        const og = document.createElement('optgroup');
        og.label = g;
        for (const m of groups[g]) {
            const opt = document.createElement('option');
            opt.value = m.key;
            opt.textContent = m.name;
            og.appendChild(opt);
        }
        selectEl.appendChild(og);
    }
}

// Vendor → device groups mapping
const VENDOR_DEVICE_GROUPS = {
    'amd': {
        label: 'AMD',
        groups: {
            'Consumer / Pro': ['rx-9700xt','rx-9070xt','rx-7900xtx','ai-pro-r9700','w7900'],
            'APU': ['strix-halo'],
            'Data Center': ['mi325x','mi300x','mi250x'],
        }
    },
    'nvidia': {
        label: 'NVIDIA',
        groups: {
            'GeForce RTX 50': ['rtx-5090','rtx-5080','rtx-5070ti','rtx-5070'],
            'GeForce RTX 40': ['rtx-4090','rtx-4080s','rtx-4080','rtx-4070tis','rtx-4070ti','rtx-4070s','rtx-4070','rtx-4060ti-16','rtx-4060ti','rtx-4060'],
            'GeForce RTX 30': ['rtx-3090','rtx-3080ti','rtx-3080','rtx-3070ti','rtx-3070','rtx-3060ti','rtx-3060'],
            'Edge / Arm': ['dgx-spark'],
            'Data Center': ['b200-sxm','h200-sxm','h100-sxm','a100-sxm-80','a100-sxm-40','l40s','l4'],
        }
    },
    'apple': {
        label: 'Apple',
        groups: {
            'Apple Silicon': ['m4-max','m4-pro','m2-ultra'],
        }
    },
    'intel': {
        label: 'Intel',
        groups: {
            'Gaudi Accelerator': ['gaudi3','gaudi2'],
        }
    },
    'google': {
        label: 'Google',
        groups: {
            'Cloud TPU': ['tpu-v6e','tpu-v5p'],
        }
    },
};

function populateVendorSelect(vendorEl) {
    vendorEl.innerHTML = '';
    for (const [vk, v] of Object.entries(VENDOR_DEVICE_GROUPS)) {
        const opt = document.createElement('option');
        opt.value = vk;
        opt.textContent = v.label;
        vendorEl.appendChild(opt);
    }
}

function populateDeviceSelectByVendor(deviceEl, vendor) {
    const vg = VENDOR_DEVICE_GROUPS[vendor];
    if (!vg) return;
    deviceEl.innerHTML = '';
    for (const [g, keys] of Object.entries(vg.groups)) {
        const og = document.createElement('optgroup');
        og.label = g;
        for (const k of keys) {
            if (!DEVICES[k]) continue;
            const opt = document.createElement('option');
            opt.value = k;
            opt.textContent = DEVICES[k].name + ` (${DEVICES[k].vram}GB)`;
            og.appendChild(opt);
        }
        deviceEl.appendChild(og);
    }
}

/** Legacy wrapper — still used if a page hasn't switched to vendor+device yet */
function populateDeviceSelect(selectEl) {
    selectEl.innerHTML = '';
    for (const [vk, v] of Object.entries(VENDOR_DEVICE_GROUPS)) {
        for (const [g, keys] of Object.entries(v.groups)) {
            const og = document.createElement('optgroup');
            og.label = v.label + ' ' + g;
            for (const k of keys) {
                if (!DEVICES[k]) continue;
                const opt = document.createElement('option');
                opt.value = k;
                opt.textContent = DEVICES[k].name + ` (${DEVICES[k].vram}GB)`;
                og.appendChild(opt);
            }
            selectEl.appendChild(og);
        }
    }
}

/** Helper: given a device key, find which vendor it belongs to */
function getVendorForDevice(deviceKey) {
    for (const [vk, v] of Object.entries(VENDOR_DEVICE_GROUPS)) {
        for (const keys of Object.values(v.groups)) {
            if (keys.includes(deviceKey)) return vk;
        }
    }
    return Object.keys(VENDOR_DEVICE_GROUPS)[0];
}

/** Setup vendor+device two-level dropdowns. Returns {vendorEl, deviceEl} */
function initVendorDeviceSelects(vendorEl, deviceEl, onChangeCb) {
    populateVendorSelect(vendorEl);
    populateDeviceSelectByVendor(deviceEl, vendorEl.value);
    vendorEl.addEventListener('change', function() {
        populateDeviceSelectByVendor(deviceEl, vendorEl.value);
        if (onChangeCb) onChangeCb();
    });
    deviceEl.addEventListener('change', function() {
        if (onChangeCb) onChangeCb();
    });
}

function formatModelInfoLine(model) {
    return `${(model.params/1e9).toFixed(1)}B${model.moe?` (active ${(model.activeParams/1e9).toFixed(1)}B)`:''} | ${model.layers}L | ${model.heads}H | ${model.hiddenDim}D${model.moe?` | MoE ${model.numExperts}×top${model.topK}`:''}`;
}

function formatDeviceInfoLine(device) {
    return `${device.vram}GB | ${device.bw} GB/s | FP16: ${device.fp16Tflops}T`;
}

// Sync config between pages via URL params
function saveConfigToURL() {
    const params = new URLSearchParams();
    params.set('m', document.getElementById('model-select').value);
    params.set('d', document.getElementById('device-select').value);
    params.set('q', document.getElementById('quant').value);
    params.set('p', document.getElementById('prompt-len').value);
    params.set('o', document.getElementById('output-len').value);
    params.set('b', document.getElementById('batch-size').value);
    params.set('tp', document.getElementById('tp-size').value);
    params.set('fa', document.getElementById('flash-attn').checked ? '1' : '0');
    const fu = document.getElementById('flops-util');
    const mu = document.getElementById('mem-util');
    const nu = document.getElementById('net-util');
    if (fu) params.set('fu', fu.value);
    if (mu) params.set('mu', mu.value);
    if (nu) params.set('nu', nu.value);
    return params.toString();
}

function loadConfigFromURL() {
    const params = new URLSearchParams(window.location.search);
    // Restore vendor+device — set vendor first so device list is populated
    if (params.has('d')) {
        const dk = params.get('d');
        const vendorEl = document.getElementById('vendor-select');
        const deviceEl = document.getElementById('device-select');
        if (vendorEl && deviceEl) {
            const vk = getVendorForDevice(dk);
            vendorEl.value = vk;
            populateDeviceSelectByVendor(deviceEl, vk);
            deviceEl.value = dk;
        } else if (deviceEl) {
            deviceEl.value = dk;
        }
    }
    if (params.has('m')) document.getElementById('model-select').value = params.get('m');
    if (params.has('q')) document.getElementById('quant').value = params.get('q');
    if (params.has('p')) {
        document.getElementById('prompt-len').value = params.get('p');
        const pv = document.getElementById('prompt-val');
        if (pv) pv.textContent = params.get('p');
    }
    if (params.has('o')) {
        document.getElementById('output-len').value = params.get('o');
        const ov = document.getElementById('output-val');
        if (ov) ov.textContent = params.get('o');
    }
    if (params.has('b')) {
        document.getElementById('batch-size').value = params.get('b');
        const bv = document.getElementById('batch-val');
        if (bv) bv.textContent = params.get('b');
    }
    if (params.has('tp')) {
        document.getElementById('tp-size').value = params.get('tp');
        const tv = document.getElementById('tp-val');
        if (tv) tv.textContent = params.get('tp');
    }
    if (params.has('fa')) document.getElementById('flash-attn').checked = params.get('fa') === '1';
    if (params.has('fu')) {
        const fu = document.getElementById('flops-util');
        if (fu) fu.value = params.get('fu');
    }
    if (params.has('mu')) {
        const mu = document.getElementById('mem-util');
        if (mu) mu.value = params.get('mu');
    }
    if (params.has('nu')) {
        const nu = document.getElementById('net-util');
        if (nu) nu.value = params.get('nu');
    }
}

// ============ I18N ============
const I18N = {
    en: {
        // index.html hero
        'hero.title': 'LLM Inference Performance Estimator',
        'hero.desc': 'Estimate prefill latency, decode throughput, memory usage, and TTFT for LLMs on various GPUs using Roofline analysis.',
        // index.html config panel
        'cfg.model': 'Model',
        'cfg.vendor': 'Vendor',
        'cfg.device': 'Device',
        'cfg.params': 'Parameters',
        'cfg.runtime': 'Runtime Configuration',
        'cfg.quant': 'Quantization',
        'cfg.util': 'Device Utilization',
        'cfg.prompt': 'Prompt Length (prefill)',
        'cfg.output': 'Output Length (decode)',
        'cfg.batch': 'Batch Size',
        'cfg.tp': 'Tensor Parallel (GPUs)',
        'cfg.flops.util': 'FLOPS Utilization (%)',
        'cfg.mem.util': 'Memory Utilization (%)',
        'cfg.net.util': 'Network BW Utilization (%)',
        'cfg.flash': 'FlashAttention',
        'cfg.flash.note': 'IO-aware tiling',
        // index.html result cards
        'res.prefill': 'Prefill Latency',
        'res.prefill.sub': 'Time to First Token',
        'res.decode': 'Decode Speed',
        'res.decode.unit': 'tokens/sec',
        'res.total': 'Total Time',
        'res.total.sub': 'Prefill + Decode',
        'res.mem': 'Model Memory',
        'res.mem.sub': 'Weights + KV Cache',
        // index.html sections
        'sec.breakdown': 'Performance Breakdown',
        'sec.roofline': 'Roofline Analysis',
        'sec.compare': 'Multi-Device Comparison',
        'sec.compare.desc': 'Compare current model + settings across all devices (only showing devices with enough VRAM)',
        'sec.memory': 'Memory Breakdown',
        'mem.weights': 'Model Weights',
        'mem.kv': 'KV Cache (per request)',
        'mem.act': 'Activation Memory (est.)',
        'mem.usage': 'Memory Usage',
        // index.html breakdown table keys
        'tbl.model': 'Model',
        'tbl.arch': 'Architecture',
        'tbl.params': 'Total / Active Params',
        'tbl.quant': 'Quantization',
        'tbl.device': 'Device',
        'tbl.compute': 'Compute (effective)',
        'tbl.bw': 'Bandwidth (effective)',
        'tbl.ridge': 'Ridge Point',
        'tbl.prefill.linear': 'Prefill Linear FLOPs',
        'tbl.prefill.attn': 'Prefill Attn FLOPs',
        'tbl.prefill.total': 'Prefill Total FLOPs',
        'tbl.prefill.ai': 'Prefill AI',
        'tbl.prefill.lat': 'Prefill Latency',
        'tbl.decode.mem': 'Decode Mem-Bound',
        'tbl.decode.comp': 'Decode Compute-Bound',
        'tbl.decode.bottleneck': 'Decode Bottleneck',
        'tbl.decode.ai': 'Decode AI',
        'tbl.decode.speed': 'Decode Speed',
        'tbl.total.lat': 'Total Latency',
        'tbl.fits': 'Fits in Memory',
        'tbl.flash': 'FlashAttention',
        'tbl.fits.yes': 'Yes',
        'tbl.fits.no': 'OOM — increase TP or use quantization',
        'tbl.flash.on': 'Enabled (IO-aware tiling)',
        'tbl.flash.off': 'Disabled',
        'tbl.mem.bound': 'Memory-bound',
        'tbl.comp.bound': 'Compute-bound',
        'tbl.compute.bound.ok': 'compute-bound',
        // CTA
        'cta.ops': 'Want per-operation breakdown?',
        'cta.ops.btn': 'Per-Op Layer Breakdown →',
        'cta.index': 'See overall performance estimates',
        'cta.index.btn': '← Overall Performance',
        // formulas
        'formulas.title': 'Modeling Methodology & Formulas',
        'formulas.prefill.title': 'Prefill (Compute-Bound):',
        'formulas.prefill.desc': 'Processing the entire prompt is compute-intensive. FLOPs ≈ 2 × Params × SeqLen + Attention O(n²). For MoE models, only active parameters participate.',
        'formulas.decode.title': 'Decode (Memory-Bandwidth-Bound):',
        'formulas.decode.desc': 'Each token reads all weights + KV cache from VRAM. Time = max(compute, memory).',
        'formulas.kv.title': 'KV Cache:',
        'formulas.kv.desc': '2 × layers × kv_heads × head_dim × seq_len × bytes. GQA/MLA significantly reduces KV Cache.',
        'formulas.ai.title': 'Arithmetic Intensity:',
        'formulas.ai.desc': "FLOPs/Byte — determines compute-bound vs memory-bound. Prefill AI ≈ SeqLen (high), Decode AI ≈ 1 (low).",
        'formulas.flash.title': 'FlashAttention:',
        'formulas.flash.desc': 'IO-aware tiling keeps N×N scores in SRAM. Without it, O(N²) HBM traffic + lower utilization (~40%).',
        // footer
        'footer.index': 'LLM Perf Model — All calculations run in your browser. Estimates are theoretical upper bounds.',
        'footer.ops': 'LLM Perf Model — Per-operation Roofline analysis runs in your browser. Estimates are theoretical upper bounds.',
        // compare chart
        'chart.decode.tps': 'Tokens/sec (decode)',
        'chart.decode': 'Decode (tok/s)',
        'chart.no.vram': 'No device has enough VRAM.',
        'chart.ai.label': 'Arithmetic Intensity (FLOPs/Byte)',
        'chart.perf.label': 'Performance (TFLOPS)',
        // ops.html
        'ops.title': 'Per-Op Layer Breakdown',
        'ops.desc': 'Per-operation analysis of a single Transformer layer — FLOPs, IO bytes, arithmetic intensity, and bottleneck type.',
        'ops.prefill': 'Prefill',
        'ops.decode': 'Decode (1 step)',
        'ops.col.op': 'Operation',
        'ops.col.flops': 'FLOPs',
        'ops.col.bytes': 'Bytes (R+W)',
        'ops.col.ai': 'AI',
        'ops.col.bound': 'Bound',
        'ops.col.time': 'Time',
        'ops.col.pct': '%',
        'ops.single.layer': 'Single Layer',
        'ops.cat.attn': 'Attention',
        'ops.cat.attn.sub': 'Q/K/V Proj + Attn + O Proj',
        'ops.cat.ffn': 'FFN / MLP',
        'ops.cat.ffn.sub': 'Gate, Up, Down + Activation',
        'ops.cat.norm': 'Norm + Residual',
        'ops.cat.norm.sub': 'RMSNorm, Residual connections',
        'ops.chart.time': 'Time per layer (ms)',
        'ops.chart.compute': 'Compute-bound',
        'ops.chart.memory': 'Memory-bound',
        'ops.n.layers': '× {0} layers',
        'ops.summary': '1 layer: {0} | × {1} layers: {2}',
        // Multi-GPU scaling
        'sec.multigpu': 'Multi-GPU Scaling',
        'sec.multigpu.desc': 'TP (Tensor Parallel) & PP (Pipeline Parallel) scaling with interconnect-aware communication modeling. PP splits layers across GPUs — helps VRAM but not single-request latency.',
        'mg.config': 'Config',
        'mg.decode': 'Decode (tok/s)',
        'mg.prefill': 'Prefill',
        'mg.speedup': 'Speedup',
        'mg.efficiency': 'Efficiency',
        'mg.vram': 'VRAM/GPU',
        'mg.oom': 'OOM',
        'mg.na': 'N/A (single device)',
        'mg.unsupported': 'Not supported',
        'mg.interconnect': 'Interconnect',
        'mg.chart.decode': 'Decode (tok/s)',
        'mg.chart.efficiency': 'Scaling Efficiency (%)',
        'mg.chart.ideal': 'Ideal (linear)',
        // Navigation
        'nav.overall': 'Overall',
        'nav.ops': 'Per-Op',
        'nav.robot': 'Bot',
        // Bot page
        'robot.title': '🤖 AI Bot Sizer',
        'robot.desc': 'Can your device run an LLM chatbot (like OpenClaw) smoothly? Select your model and device to find out.',
        'robot.scenario': 'Scenario',
        'robot.control': 'Chat Params',
        'robot.ctx': 'Context Length',
        'robot.resp': 'Response Length',
        'robot.concurrent': 'Concurrent Users',
        'robot.options': 'Runtime Configuration',
        'robot.reading.speed': 'Human reading speed',
        'robot.comfort.speed': 'Barely usable speed',
        'robot.kv.dtype': 'KV Cache Precision',
        'robot.res.ttft': 'TTFT',
        'robot.res.ttft.sub': 'Time to First Token',
        'robot.res.tps': 'Decode',
        'robot.res.total': 'Total',
        'robot.res.total.sub': 'Full response',
        'robot.res.mem': 'Memory',
        'robot.res.conc': 'Max Conc.',
        'robot.res.conc.sub': 'conversations',
        'robot.mem.title': 'Memory Breakdown',
        'robot.cmp.models': 'All Models on This Device',
        'robot.cmp.devices': 'All Devices for This Model',
        'robot.chart.tps': 'Decode Speed Comparison',
        'robot.method.title': 'Methodology',
        'robot.method.ttft.title': 'TTFT (Time To First Token):',
        'robot.method.ttft.desc': 'Prefill time — processing the full context in one forward pass. Compute-bound. TTFT = FLOPs / (device_TFLOPS × utilization).',
        'robot.method.tps.title': 'Decode Speed (TPS):',
        'robot.method.tps.desc': 'Tokens per second during generation. Memory-bandwidth-bound — each token reads all weights + KV cache.',
        'robot.method.mem.title': 'Memory:',
        'robot.method.mem.desc': 'Model weights + KV cache × concurrent_users + activation memory. Quantization reduces both weights and KV cache.',
        'robot.method.conc.title': 'Concurrent Users:',
        'robot.method.conc.desc': 'Max conversations = floor((VRAM - weights - activations) / KV_cache_per_user).',
        'robot.method.note.title': 'Note:',
        'robot.method.note.desc': 'Estimates are theoretical upper bounds. Real performance is typically 60-80% of estimates due to framework overhead.',
        'robot.footer': 'LLM Perf Model — AI Bot Sizer. All calculations run in your browser.',
    },
    zh: {
        'hero.title': 'LLM 推理性能估算器',
        'hero.desc': '基于 Roofline 模型估算大语言模型在不同 GPU 上的 Prefill 延迟、Decode 吞吐、显存占用和首 Token 时间。',
        'cfg.model': '模型',
        'cfg.vendor': '厂商',
        'cfg.device': '设备',
        'cfg.params': '参数',
        'cfg.runtime': '运行时配置',
        'cfg.quant': '量化',
        'cfg.util': '设备利用率',
        'cfg.prompt': 'Prompt 长度（预填充）',
        'cfg.output': '输出长度（解码）',
        'cfg.batch': '批大小',
        'cfg.tp': '张量并行（GPU 数）',
        'cfg.flops.util': 'FLOPS 利用率（%）',
        'cfg.mem.util': '显存带宽利用率（%）',
        'cfg.net.util': '网络带宽利用率（%）',
        'cfg.flash': 'FlashAttention',
        'cfg.flash.note': 'IO 感知分块',
        'res.prefill': '预填充延迟',
        'res.prefill.sub': '首 Token 时间',
        'res.decode': '解码速度',
        'res.decode.unit': 'tokens/秒',
        'res.total': '总时间',
        'res.total.sub': '预填充 + 解码',
        'res.mem': '模型显存',
        'res.mem.sub': '权重 + KV Cache',
        'sec.breakdown': '性能详情',
        'sec.roofline': 'Roofline 分析',
        'sec.compare': '多设备对比',
        'sec.compare.desc': '当前模型 + 当前参数在所有设备上的对比（仅显示显存足够的设备）',
        'sec.memory': '显存占用',
        'mem.weights': '模型权重',
        'mem.kv': 'KV Cache（单请求）',
        'mem.act': '激活显存（估算）',
        'mem.usage': '显存使用率',
        'tbl.model': '模型',
        'tbl.arch': '架构',
        'tbl.params': '总参数 / 活跃参数',
        'tbl.quant': '量化',
        'tbl.device': '设备',
        'tbl.compute': '算力（有效）',
        'tbl.bw': '带宽（有效）',
        'tbl.ridge': 'Ridge Point',
        'tbl.prefill.linear': 'Prefill 线性层 FLOPs',
        'tbl.prefill.attn': 'Prefill 注意力 FLOPs',
        'tbl.prefill.total': 'Prefill 总 FLOPs',
        'tbl.prefill.ai': 'Prefill 算术强度',
        'tbl.prefill.lat': 'Prefill 延迟',
        'tbl.decode.mem': 'Decode 访存瓶颈',
        'tbl.decode.comp': 'Decode 算力瓶颈',
        'tbl.decode.bottleneck': 'Decode 瓶颈类型',
        'tbl.decode.ai': 'Decode 算术强度',
        'tbl.decode.speed': 'Decode 速度',
        'tbl.total.lat': '总延迟',
        'tbl.fits': '显存是否足够',
        'tbl.flash': 'FlashAttention',
        'tbl.fits.yes': '足够',
        'tbl.fits.no': '显存不足 — 增大 TP 或使用量化',
        'tbl.flash.on': '已启用（IO 感知分块）',
        'tbl.flash.off': '已禁用',
        'tbl.mem.bound': '访存瓶颈',
        'tbl.comp.bound': '算力瓶颈',
        'tbl.compute.bound.ok': '算力瓶颈',
        'cta.ops': '想看逐算子分析？',
        'cta.ops.btn': '🔬 逐 Op 层分析 →',
        'cta.index': '查看整体性能估算',
        'cta.index.btn': '← 📊 整体性能',
        'formulas.title': '建模方法与公式',
        'formulas.prefill.title': 'Prefill（计算密集）：',
        'formulas.prefill.desc': '处理 Prompt 是计算密集型。FLOPs ≈ 2 × 参数量 × 序列长度 + 注意力 O(n²)。MoE 模型只有活跃参数参与。',
        'formulas.decode.title': 'Decode（访存密集）：',
        'formulas.decode.desc': '每 Token 需从显存读取全部权重 + KV Cache。耗时 = max(计算, 访存)。',
        'formulas.kv.title': 'KV Cache：',
        'formulas.kv.desc': '2 × 层数 × kv_heads × head_dim × 序列长度 × 字节。GQA/MLA 大幅减少 KV Cache。',
        'formulas.ai.title': '算术强度：',
        'formulas.ai.desc': 'FLOPs/Byte — 决定操作是计算密集还是访存密集。Prefill AI ≈ SeqLen（高）, Decode AI ≈ 1（低）。',
        'formulas.flash.title': 'FlashAttention：',
        'formulas.flash.desc': 'IO 感知分块将 N×N 分数保存在 SRAM 中。关闭后，O(N²) HBM 读写 + 利用率降至约 40%。',
        'footer.index': 'LLM Perf Model — 所有计算均在浏览器中完成。估算值为理论上界。',
        'footer.ops': 'LLM Perf Model — 逐算子 Roofline 分析在浏览器中完成。估算值为理论上界。',
        'chart.decode.tps': '解码速度（tokens/秒）',
        'chart.decode': '解码（tok/s）',
        'chart.no.vram': '没有设备有足够的显存。',
        'chart.ai.label': '算术强度（FLOPs/Byte）',
        'chart.perf.label': '性能（TFLOPS）',
        'ops.title': '逐 Op 层分析',
        'ops.desc': '单个 Transformer 层内每个操作的 FLOPs、IO 字节、算术强度和瓶颈类型。',
        'ops.prefill': 'Prefill',
        'ops.decode': 'Decode（单步）',
        'ops.col.op': '操作',
        'ops.col.flops': 'FLOPs',
        'ops.col.bytes': '字节（读+写）',
        'ops.col.ai': 'AI',
        'ops.col.bound': '瓶颈',
        'ops.col.time': '耗时',
        'ops.col.pct': '%',
        'ops.single.layer': '单层',
        'ops.cat.attn': '注意力',
        'ops.cat.attn.sub': 'Q/K/V 投影 + 注意力 + O 投影',
        'ops.cat.ffn': 'FFN / MLP',
        'ops.cat.ffn.sub': 'Gate, Up, Down + 激活',
        'ops.cat.norm': '归一化 + 残差',
        'ops.cat.norm.sub': 'RMSNorm, 残差连接',
        'ops.chart.time': '每层耗时 (ms)',
        'ops.chart.compute': '算力瓶颈',
        'ops.chart.memory': '访存瓶颈',
        'ops.n.layers': '× {0} 层',
        'ops.summary': '单层: {0} | × {1} 层: {2}',
        // 多卡扩展
        'sec.multigpu': '多卡扩展分析',
        'sec.multigpu.desc': 'TP（张量并行）与 PP（流水线并行）多卡性能扩展预测。PP 按层切分模型——减少显存但不改善单请求延迟。',
        'mg.config': '配置',
        'mg.decode': '解码（tok/s）',
        'mg.prefill': 'Prefill',
        'mg.speedup': '加速比',
        'mg.efficiency': '效率',
        'mg.vram': '显存/卡',
        'mg.oom': '显存不足',
        'mg.na': 'N/A（单设备）',
        'mg.unsupported': '不支持',
        'mg.interconnect': '互联类型',
        'mg.chart.decode': '解码（tok/s）',
        'mg.chart.efficiency': '扩展效率（%）',
        'mg.chart.ideal': '理想（线性扩展）',
        // 导航
        'nav.overall': '首页',
        'nav.ops': '逐算子',
        'nav.robot': 'Bot 选型',
        // Bot 页面
        'robot.title': '🤖 AI Bot 选型',
        'robot.desc': '你的设备能否流畅运行 LLM 聊天机器人（如 OpenClaw）？选择模型和设备，立即评估。',
        'robot.scenario': '场景',
        'robot.control': '对话参数',
        'robot.ctx': '上下文长度',
        'robot.resp': '回复长度',
        'robot.concurrent': '并发用户数',
        'robot.options': '运行时配置',
        'robot.reading.speed': '人类阅读速度',
        'robot.comfort.speed': '勉强可用速度',
        'robot.kv.dtype': 'KV Cache 精度',
        'robot.res.ttft': 'TTFT',
        'robot.res.ttft.sub': '首 Token 延迟',
        'robot.res.tps': '解码',
        'robot.res.total': '总时间',
        'robot.res.total.sub': '完整回复',
        'robot.res.mem': '显存',
        'robot.res.conc': '最大并发',
        'robot.res.conc.sub': '个会话',
        'robot.mem.title': '显存占用明细',
        'robot.cmp.models': '该设备上所有模型',
        'robot.cmp.devices': '该模型在所有设备',
        'robot.chart.tps': '解码速度对比',
        'robot.method.title': '建模方法',
        'robot.method.ttft.title': 'TTFT（首 Token 延迟）：',
        'robot.method.ttft.desc': '预填充时间 — 一次前向传播处理完整上下文。计算密集型。TTFT = FLOPs / (设备算力 × 利用率)。',
        'robot.method.tps.title': '解码速度（TPS）：',
        'robot.method.tps.desc': '生成时每秒 token 数。访存密集型 — 每个 token 需读取全部权重 + KV cache。',
        'robot.method.mem.title': '显存：',
        'robot.method.mem.desc': '模型权重 + KV cache × 并发用户数 + 激活显存。量化同时减少权重和 KV cache。',
        'robot.method.conc.title': '并发用户数：',
        'robot.method.conc.desc': '最大会话数 = floor((显存 - 权重 - 激活) / 每用户 KV cache)。',
        'robot.method.note.title': '注意：',
        'robot.method.note.desc': '估算为理论上界。实际性能通常为估算值的 60-80%（框架开销等）。',
        'robot.footer': 'LLM Perf Model — AI Bot 选型。所有计算均在浏览器中完成。',
    }
};

function getLang() { return localStorage.getItem('llm-perf-lang') || 'en'; }

function t(key) { return (I18N[getLang()] || I18N.en)[key] || (I18N.en)[key] || key; }

function setLang(lang) {
    localStorage.setItem('llm-perf-lang', lang);
    const sel = document.getElementById('lang-select');
    if (sel) sel.value = lang;
    applyI18n();
    // 刷新模型/设备详情面板的语言（先执行，它们内部会触发 recalc）
    try { if (typeof onModelChange === 'function') onModelChange(); } catch(e) { console.warn('setLang onModelChange:', e); }
    try { if (typeof onDeviceChange === 'function') onDeviceChange(); } catch(e) { console.warn('setLang onDeviceChange:', e); }
    // 如果页面没有 onModelChange/onDeviceChange（如 robot.html），回退到直接 recalc
    if (typeof onModelChange !== 'function' && typeof onDeviceChange !== 'function') {
        try {
            if (typeof calculate === 'function') calculate();
            else if (typeof renderOps === 'function') renderOps();
            else if (typeof evaluate === 'function') evaluate();
        } catch(e) { console.warn('setLang recalc:', e); }
    }
}

function onLangChange(sel) {
    setLang(sel.value);
}

function initLang() {
    const sel = document.getElementById('lang-select');
    if (sel) sel.value = getLang();
    applyI18n();
}

function applyI18n() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        const val = t(key);
        if (el.tagName === 'INPUT') el.placeholder = val;
        else el.textContent = val;
    });
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
        el.innerHTML = t(el.getAttribute('data-i18n-html'));
    });
}

// ============ THEME ============
function chartThemeColors() {
    const isLight = document.documentElement.classList.contains('light');
    return {
        grid: isLight ? '#e5e7eb' : '#1e293b',
        tick: isLight ? '#64748b' : '#9ca3af',
        legend: isLight ? '#374151' : '#e2e8f0',
        barBg: isLight ? '#e2e8f0' : '#334155',
        barBorder: isLight ? '#cbd5e1' : '#475569',
    };
}

function initTheme() {
    const saved = localStorage.getItem('llm-perf-theme') || 'auto';
    applyTheme(saved);
}

function applyTheme(theme) {
    const html = document.documentElement;
    html.classList.remove('light');
    if (theme === 'light') {
        html.classList.add('light');
    } else if (theme === 'auto') {
        if (window.matchMedia('(prefers-color-scheme: light)').matches) {
            html.classList.add('light');
        }
    }
    localStorage.setItem('llm-perf-theme', theme);
    updateThemeButton(theme);
    const isLight = html.classList.contains('light');
    if (typeof Chart !== 'undefined') {
        Chart.defaults.color = isLight ? '#64748b' : '#9ca3af';
        Chart.defaults.borderColor = isLight ? '#e5e7eb' : '#1e293b';
    }
    if (typeof calculate === 'function') calculate();
    else if (typeof renderOps === 'function') renderOps();
    else if (typeof evaluate === 'function') evaluate();
}

function cycleTheme() {
    const cur = localStorage.getItem('llm-perf-theme') || 'auto';
    const next = cur === 'auto' ? 'light' : cur === 'light' ? 'dark' : 'auto';
    applyTheme(next);
}

function updateThemeButton(theme) {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const icons = { auto: '\u{1F317}', light: '\u2600\uFE0F', dark: '\u{1F319}' };
    const labels = { auto: 'Auto', light: 'Light', dark: 'Dark' };
    btn.textContent = icons[theme] + ' ' + labels[theme];
    btn.title = 'Theme: ' + labels[theme];
}

window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
    if ((localStorage.getItem('llm-perf-theme') || 'auto') === 'auto') {
        applyTheme('auto');
    }
});
