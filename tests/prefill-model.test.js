const test = require("node:test");
const assert = require("node:assert/strict");
const { calculate } = require("../prefill-model.js");

test("2048-token model totals reproduce the documented reference", () => {
  const { totals } = calculate();
  assert.equal(totals.q, 10555563704320);
  assert.equal(totals.e, 8580675665920);
});

test("retained 7253-token, four-logits Edge0 audit is reproduced", () => {
  assert.equal(calculate({ T: 7253, M: 4 }).totals.e, 33481595514880);
});

test("one 2048-token Edge0 GDN layer reproduces the independent L0 audit", () => {
  const { rows } = calculate();
  const gdn =
    rows.filter((r) => r.group === "gdn").reduce((s, r) => s + r.e, 0) / 30;
  const moe =
    rows.filter((r) => r.group === "ffn").reduce((s, r) => s + r.e, 0) / 40;
  // Eight rank-16 GDN/shared-expert adapters from the audited tensor headers.
  assert.equal(gdn + moe + 2252341248, 214475735040);
});

test("partitioning a prompt preserves attention history and total useful work", () => {
  const whole = calculate({ T: 8192, P: 100 });
  const first = calculate({ T: 2048, P: 100, M: 0 });
  const rest = calculate({ T: 6144, P: 2148, M: 1 });
  for (const key of ["q", "e"])
    assert.equal(whole.totals[key], first.totals[key] + rest.totals[key]);
  assert.ok(rest.totals.q > calculate({ T: 6144, P: 0 }).totals.q);
});

test("batch scales per-sequence major work; logits policy is explicit", () => {
  const one = calculate({ T: 8192, P: 4096 });
  const batched = calculate({ T: 8192, P: 4096, B: 3 });
  for (const key of ["q", "e"])
    assert.equal(batched.totals[key], 3 * one.totals[key]);
  const all = calculate({ T: 2048, M: 2048 });
  const last = calculate({ T: 2048, M: 1 });
  assert.equal(all.totals.q - last.totals.q, 2047 * 2 * 2048 * 248320);
});

test("K reduction affects routed FFNs, not attention, GDN or shared FFNs", () => {
  const r = calculate();
  assert.equal(r.rows[0].q, 2 * r.rows[0].e);
  for (const row of r.rows.filter((x) =>
    ["attention", "gdn", "head"].includes(x.group),
  ))
    assert.equal(row.q, row.e);
  assert.equal(
    r.rows.find((x) => x.name.startsWith("Shared FFN")).q,
    r.rows.find((x) => x.name.startsWith("Shared FFN")).e,
  );
  const noAdapter = calculate({ lora: false });
  assert.equal(r.totals.q, noAdapter.totals.q);
  assert.equal(r.totals.e - noAdapter.totals.e, 42332160 * 2048);
});

test("scalar operations are explicitly separate from major FLOPs", () => {
  const r = calculate();
  assert.equal(r.scalarTotals.q, 15511213952);
  assert.equal(r.scalarTotals.e, 15895910272);
  assert.equal(
    r.rows.reduce((s, row) => s + row.e, 0),
    r.totals.e,
  );
  assert.match(r.methodology.scalar, /Separate/);
});

test("long context golden results and domain boundaries", () => {
  const r = calculate({ T: 131072 });
  assert.equal(r.totals.q, 2060876649594880);
  assert.equal(r.totals.e, 1934483815137280);
  assert.ok(Number.isFinite(calculate({ T: 262144, B: 1024 }).totals.q));
  for (const args of [
    { T: 0 },
    { T: 2.5 },
    { T: NaN },
    { T: "2048" },
    { P: -1 },
    { B: 0 },
    { B: 1025 },
    { T: 262144, P: 1 },
    { M: 2049 },
    { M: -1 },
    { lora: "false" },
  ])
    assert.throws(() => calculate(args));
});
