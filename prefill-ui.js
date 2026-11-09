const $ = (id) => document.getElementById(id);
const colors = {
  ffn: "#285d51",
  attention: "#659b99",
  gdn: "#acb591",
  lora: "#c26b38",
  head: "#d6b777",
};
const names = {
  ffn: "MoE + router",
  attention: "Full attention",
  gdn: "GDN",
  lora: "LoRA",
  head: "LM head",
};
let current;
const fmt = (n) =>
  (n / 1e9).toLocaleString("en-US", {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  });
function render() {
  try {
    const T = Number($("tokens").value),
      P = Number($("prefix").value),
      B = Number($("batch").value),
      mode = $("head").value,
      M =
        mode === "last"
          ? 1
          : mode === "all"
            ? T
            : mode === "chunks"
              ? Math.ceil(T / 2048)
              : 0;
    current = PrefillModel.calculate({ T, P, B, M, lora: $("lora").checked });
    $("error").textContent = "";
    const r = current;
    $("qtotal").innerHTML =
      (r.totals.q / 1e12).toFixed(3) + '<span class="unit">TFLOPs</span>';
    $("etotal").innerHTML =
      (r.totals.e / 1e12).toFixed(3) + '<span class="unit">TFLOPs</span>';
    $("saving").textContent =
      ((1 - r.totals.e / r.totals.q) * 100).toFixed(2) + "%";
    $("qtoken").textContent =
      (r.totals.q / (B * T) / 1e9).toFixed(3) + " GFLOPs / new token";
    $("etoken").textContent =
      (r.totals.e / (B * T) / 1e9).toFixed(3) + " GFLOPs / new token";
    for (const key of ["q", "e"]) {
      const groups = {};
      r.rows.forEach(
        (x) => (groups[x.group] = (groups[x.group] || 0) + x[key]),
      );
      $(key + "bar").innerHTML = Object.entries(colors)
        .map(
          ([g, color]) =>
            '<span title="' +
            names[g] +
            ": " +
            fmt(groups[g] || 0) +
            ' GFLOPs" style="width:' +
            ((groups[g] || 0) / r.totals.q) * 100 +
            "%;background:" +
            color +
            '"></span>',
        )
        .join("");
    }
    $("legend").innerHTML = Object.entries(colors)
      .map(
        ([g, c]) =>
          '<span><i style="background:' + c + '"></i>' + names[g] + "</span>",
      )
      .join("");
    const row = (x) =>
      "<tr><td>" +
      x.name +
      "</td><td>" +
      x.formula +
      "</td><td>" +
      fmt(x.q) +
      "</td><td>" +
      fmt(x.e) +
      "</td></tr>";
    $("rows").innerHTML = r.rows.map(row).join("");
    $("scalar").innerHTML = r.scalar.map(row).join("");
    $("totals").innerHTML = row({
      name: "主要计算合计",
      formula: "",
      ...r.totals,
    });
    $("scalarTotals").innerHTML = row({
      name: "近似 scalar 合计",
      formula: "",
      ...r.scalarTotals,
    });
  } catch (e) {
    current = null;
    $("error").textContent = e.message;
    for (const id of [
      "qtotal",
      "etotal",
      "saving",
      "qtoken",
      "etoken",
      "rows",
      "scalar",
      "totals",
      "scalarTotals",
      "qbar",
      "ebar",
    ])
      $(id).textContent = "";
  }
}
for (const id of ["tokens", "prefix", "batch", "head", "lora"])
  $(id).addEventListener("input", render);
$("download").onclick = () => {
  if (!current) return;
  const blob = new Blob([JSON.stringify(current, null, 2)], {
      type: "application/json",
    }),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = "prefill-calculation.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
render();
