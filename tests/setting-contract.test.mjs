import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

import {
  ENGINES,
  INTERPOLATION_MODELS,
  INTERPOLATION_RES_MODES,
  MODES,
  UPSCALE_POLICIES,
  UPSCALE_POLICIES_BY_ENGINE,
  upscalePoliciesForEngine,
} from "../src/core/fsrcnnx-setting-contract.js";
import { POLICY_OPTIONS } from "../src/popup.js";

// The neural "native" policy was offered by the popup and accepted by the runtime
// while the content-script gate rejected it, so a shipped feature was unreachable
// through the UI. Three independent copies of these enums existed and nothing
// asserted they agreed. fsrcnnx-main.js and popup.js now import the contract
// directly; the content script cannot (it is not an ES module, and it must
// validate commands that arrive before the pipeline module loads), so its copy is
// pinned here instead.

// Executes the real COMMANDS table out of src/content.js without running the rest
// of the content script, so the accepted sets are read from shipped code rather
// than restated.
async function loadCommandGate() {
  const source = await readFile(new URL("../src/content.js", import.meta.url), "utf8");
  const start = source.indexOf("function invalidInput(");
  const end = source.indexOf("function normalizeCommandResponse(");
  assert.ok(start > 0 && end > start, "content.js command-gate section markers must match");
  const context = vm.createContext({});
  vm.runInContext(
    `${source.slice(start, end)}\nglobalThis.__commands = COMMANDS;`,
    context,
  );
  return context.__commands;
}

// The gate stores its allowlist in a closure, so probe it behaviorally: a value is
// accepted when validate() returns null for a well-formed payload.
function acceptedValues(command, field, candidates) {
  return candidates.filter((value) => command.validate({ type: "x", [field]: value }) === null);
}

test("the content-script gate accepts exactly the contract's values", async () => {
  const commands = await loadCommandGate();
  const cases = [
    ["FSRCNNX_SETMODE", "mode", MODES],
    ["FSRCNNX_SETENGINE", "engine", ENGINES],
    ["FSRCNNX_SETINTERPRES", "mode", INTERPOLATION_RES_MODES],
    ["FSRCNNX_SETINTERPMODEL", "key", INTERPOLATION_MODELS],
    ["FSRCNNX_SETPOLICY", "policy", UPSCALE_POLICIES],
  ];
  for (const [type, field, expected] of cases) {
    const command = commands[type];
    assert.ok(command, `${type} must exist`);
    assert.deepEqual(
      acceptedValues(command, field, expected).sort(),
      [...expected].sort(),
      `${type} must accept every contract value for ${field}`,
    );
    for (const rejected of ["", "nope", "__proto__", "DISPLAY"]) {
      assert.notEqual(
        command.validate({ type: "x", [field]: rejected }),
        null,
        `${type} must still reject ${JSON.stringify(rejected)}`,
      );
    }
  }
});

test("the popup offers exactly the contract's policies for every engine", () => {
  assert.deepEqual(
    Object.keys(POLICY_OPTIONS).sort(),
    [...ENGINES].sort(),
    "every engine needs a popup policy list",
  );
  for (const engine of ENGINES) {
    assert.deepEqual(
      POLICY_OPTIONS[engine].map(([value]) => value).sort(),
      [...upscalePoliciesForEngine(engine)].sort(),
      `popup policy values for ${engine} must match the contract`,
    );
  }
});

test("the policy union is derived from the per-engine sets", () => {
  const union = new Set(Object.values(UPSCALE_POLICIES_BY_ENGINE).flat());
  assert.deepEqual([...UPSCALE_POLICIES].sort(), [...union].sort());
  // force3 belongs to the standard engine only; force8 to the fixed-x2 engines.
  assert.ok(upscalePoliciesForEngine("fsrcnnx").includes("force3"));
  assert.ok(!upscalePoliciesForEngine("fsrcnnx").includes("force8"));
  for (const engine of ["fsrcnnx-hi", "artcnn"]) {
    assert.ok(upscalePoliciesForEngine(engine).includes("force8"), engine);
    assert.ok(!upscalePoliciesForEngine(engine).includes("force3"), engine);
  }
  // The regression that motivated this module.
  assert.ok(upscalePoliciesForEngine("neural").includes("native"));
  assert.ok(UPSCALE_POLICIES.includes("native"));
});

test("an unknown engine falls back to a real policy set rather than undefined", () => {
  assert.deepEqual(upscalePoliciesForEngine("not-an-engine"), UPSCALE_POLICIES_BY_ENGINE.fsrcnnx);
});
