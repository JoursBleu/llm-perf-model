/* Logical arithmetic model, MAC = 2. No model execution or hardware timing. */
(function (root) {
  const cfg = {
    L: 40,
    La: 10,
    Lg: 30,
    H: 2048,
    I: 512,
    Is: 512,
    E: 256,
    V: 248320,
    nq: 16,
    nkv: 2,
    d: 256,
    nk: 16,
    nv: 32,
    dk: 128,
    dv: 128,
    c: 4,
  };
  function calculate({ T = 2048, P = 0, B = 1, M = 1, lora = true } = {}) {
    for (const [k, v] of Object.entries({ T, P, B, M }))
      if (!Number.isSafeInteger(v) || v < 0)
        throw Error(k + " must be a nonnegative integer");
    if (T < 1 || B < 1 || B > 1024 || M > T)
      throw Error("Require T ≥ 1, 1 ≤ B ≤ 1024, 0 ≤ M ≤ T");
    if (typeof lora !== "boolean") throw Error("lora must be a boolean");
    if (T + P > 262144) throw Error("T + P must be ≤ 262144");
    const { L, La, Lg, H, I, Is, E, V, nq, nkv, d, nk, nv, dk, dv, c } = cfg;
    const X = B * T,
      Q = nq * d,
      R = nkv * d,
      Dk = nk * dk,
      Dv = nv * dv,
      Dc = 2 * Dk + Dv,
      A = B * (T * P + (T * (T + 1)) / 2);
    const rows = [];
    const add = (name, formula, q, e = q, group = "common") =>
      rows.push({ name, formula, q, e, group });
    add(
      "Routed FFN · gate/up/down",
      "6 L B T H K I",
      6 * L * X * H * 8 * I,
      6 * L * X * H * 4 * I,
      "ffn",
    );
    add(
      "Shared FFN · gate/up/down",
      "6 L B T H Is",
      6 * L * X * H * Is,
      undefined,
      "ffn",
    );
    add(
      "Router projection",
      "2 L B T H E",
      2 * L * X * H * E,
      undefined,
      "ffn",
    );
    add(
      "Shared-expert gate projection",
      "2 L B T H",
      2 * L * X * H,
      undefined,
      "ffn",
    );
    add(
      "Attention · Q + output gate",
      "4 La B T H Q",
      4 * La * X * H * Q,
      undefined,
      "attention",
    );
    add(
      "Attention · K + V",
      "4 La B T H R",
      4 * La * X * H * R,
      undefined,
      "attention",
    );
    add(
      "Attention · output projection",
      "2 La B T H Q",
      2 * La * X * H * Q,
      undefined,
      "attention",
    );
    add(
      "Attention · causal QKᵀ + AV",
      "4 La Q A; A=B[TP+T(T+1)/2]",
      4 * La * Q * A,
      undefined,
      "attention",
    );
    add(
      "GDN · Q/K/V projections",
      "2 Lg B T H (2Dk+Dv)",
      2 * Lg * X * H * Dc,
      undefined,
      "gdn",
    );
    add(
      "GDN · z projection",
      "2 Lg B T H Dv",
      2 * Lg * X * H * Dv,
      undefined,
      "gdn",
    );
    add(
      "GDN · a/b projections",
      "4 Lg B T H nv",
      4 * Lg * X * H * nv,
      undefined,
      "gdn",
    );
    add(
      "GDN · output projection",
      "2 Lg B T H Dv",
      2 * Lg * X * H * Dv,
      undefined,
      "gdn",
    );
    add(
      "GDN · depthwise convolution",
      "2 Lg B T (2Dk+Dv)c",
      2 * Lg * X * Dc * c,
      undefined,
      "gdn",
    );
    add(
      "GDN · recurrence reference",
      "Lg B T nv dv (7dk+2)",
      Lg * X * nv * dv * (7 * dk + 2),
      undefined,
      "gdn",
    );
    add(
      "Edge0 · unmerged LoRA",
      "Σ 2 B T r(in+out) = 42,332,160 B T",
      0,
      lora ? 42332160 * X : 0,
      "lora",
    );
    add("LM head", "2 B M H V", 2 * B * M * H * V, undefined, "head");
    const totals = {
      q: rows.reduce((s, r) => s + r.q, 0),
      e: rows.reduce((s, r) => s + r.e, 0),
    };
    // Optional scalar-operation convention: +,-,*,/,exp,log,rsqrt = one each.
    // Comparisons, indexing, quantization and RoPE-table generation excluded.
    const scalar = [];
    const sa = (name, formula, q, e = q) =>
      scalar.push({ name, formula, q, e });
    sa("Outer RMS norms + residuals", "LBT(10H+4)", L * X * (10 * H + 4));
    sa("Final RMS norm", "BT(4H+2)", X * (4 * H + 2));
    sa(
      "Router softmax + top-k renormalization",
      "LBT(4E+2K-2)",
      L * X * (4 * E + 16 - 2),
      L * X * (4 * E + 8 - 2),
    );
    sa(
      "Expert SwiGLU",
      "6LBT(KI+Is)",
      6 * L * X * (8 * I + Is),
      6 * L * X * (4 * I + Is),
    );
    sa(
      "Expert weighted mix + shared sigmoid/mix",
      "LBT[(2K+1)H+4]",
      L * X * (17 * H + 4),
      L * X * (9 * H + 4),
    );
    sa(
      "Attention Q/K RMS norms",
      "La BT (nq+nkv)(4d+2)",
      La * X * (nq + nkv) * (4 * d + 2),
    );
    sa(
      "Partial RoPE application",
      "3 La BT (nq+nkv)(d/4)",
      3 * La * X * (nq + nkv) * (d / 4),
    );
    sa(
      "Attention score scale + softmax",
      "La nq (5A−BT)",
      La * nq * (5 * A - X),
    );
    sa("Attention output sigmoid + gate", "5 La BT Q", 5 * La * X * Q);
    sa(
      "GDN Q/K normalization + scale",
      "2 Lg BT nk(4dk+2)",
      2 * Lg * X * nk * (4 * dk + 2),
    );
    sa(
      "GDN output RMS norm + SiLU(z) gate",
      "Lg BT nv(10dv+2)",
      Lg * X * nv * (10 * dv + 2),
    );
    sa("GDN convolution SiLU", "5 Lg BT (2Dk+Dv)", 5 * Lg * X * Dc);
    sa(
      "GDN decay / beta transform",
      "10 Lg BT nv + 2 Lg nv",
      10 * Lg * X * nv + 2 * Lg * nv,
    );
    sa(
      "LoRA scale and add",
      "2BT Σ adapter_out = 1,335,040 BT",
      0,
      lora ? 1335040 * X : 0,
    );
    const scalarTotals = {
      q: scalar.reduce((s, r) => s + r.q, 0),
      e: scalar.reduce((s, r) => s + r.e, 0),
    };
    return {
      methodology: {
        macFlops: 2,
        attention: "causal useful pairs",
        gdn: "recurrent logical reference",
        logits: "M positions per sequence",
        scalar:
          "Separate approximate operation ledger; not equal-cost hardware FLOPs",
        scope:
          "Text-only forward pass; no vision encoder, MTP, quantization overhead, data movement or measured runtime",
      },
      config: { ...cfg },
      workload: { T, P, B, M, lora },
      rows,
      totals,
      scalar,
      scalarTotals,
    };
  }
  root.PrefillModel = { calculate, cfg };
  if (typeof module !== "undefined") module.exports = root.PrefillModel;
})(typeof globalThis !== "undefined" ? globalThis : this);
