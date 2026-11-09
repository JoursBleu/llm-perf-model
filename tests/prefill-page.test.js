const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "prefill.html"), "utf8");

test("static page dependencies and relative document links exist", () => {
  for (const match of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
    if (/^https?:/.test(match[1])) continue;
    assert.ok(fs.existsSync(path.join(root, match[1].split("?")[0])), match[1]);
  }
  for (const page of ["index.html", "ops.html", "robot.html"])
    assert.match(
      fs.readFileSync(path.join(root, page), "utf8"),
      /href="prefill.html"/,
    );
  assert.equal(
    [...html.matchAll(/id="([^"]+)"/g)].length,
    new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1])).size,
  );
});

test("page rendering script handles controls and invalid input without stale results", () => {
  // Minimal document adapter tests the script contract, not browser layout.
  const elements = {};
  for (const m of html.matchAll(/id="([^"]+)"/g)) {
    let content = "";
    elements[m[1]] = {
      value: "",
      checked: false,
      events: {},
      set innerHTML(v) {
        content = v;
      },
      get innerHTML() {
        return content;
      },
      set textContent(v) {
        content = v;
      },
      get textContent() {
        return content;
      },
      addEventListener(name, handler) {
        this.events[name] = handler;
      },
    };
  }
  elements.tokens.value = "2048";
  elements.prefix.value = "0";
  elements.batch.value = "1";
  elements.head.value = "last";
  elements.lora.checked = true;
  const ctx = vm.createContext({
    document: { getElementById: (id) => elements[id] },
    setTimeout,
  });
  vm.runInContext(
    fs.readFileSync(path.join(root, "prefill-model.js"), "utf8"),
    ctx,
  );
  vm.runInContext(
    fs.readFileSync(path.join(root, "prefill-ui.js"), "utf8"),
    ctx,
  );
  assert.match(elements.qtotal.innerHTML, /^10.556/);
  assert.equal(elements.saving.textContent, "18.71%");
  elements.tokens.value = "32768";
  elements.tokens.events.input();
  assert.match(elements.qtotal.innerHTML, /^251.337/);
  elements.head.value = "all";
  elements.head.events.input();
  assert.match(elements.qtotal.innerHTML, /^284.665/);
  elements.tokens.value = "7253";
  elements.head.value = "chunks";
  elements.head.events.input();
  assert.match(elements.etotal.innerHTML, /^33.482/);
  elements.tokens.value = "0";
  elements.tokens.events.input();
  assert.ok(elements.error.textContent);
  assert.equal(elements.qtotal.innerHTML, "");
  assert.equal(elements.rows.innerHTML, "");
});
