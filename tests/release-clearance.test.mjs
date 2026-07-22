import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { inspectReleaseClearance } from "../tools/check-release-clearance.mjs";

const root = resolve(import.meta.dirname, "..");
const highX2Artifacts = new Map([
  [
    "shaders/FSRCNNX_x2_56-16-4-1.glsl",
    "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
  ],
  [
    "model/FSRCNNX_x2_56-16-4-1.wgsl",
    "267ba203867483a467c535fd03c36c62ff9428116111d4d258dc5c295ef8e0d7",
  ],
  [
    "model/FSRCNNX_x2_56-16-4-1.passes.json",
    "57395ac668b4cbebea69938a9089c9bea0029ce785f7cc6dad239c4be31d43e7",
  ],
]);
const highX2Mirror = "https://github.com/resc863/Project/blob/0e6bdb96f2845d883ec0131af8598c438c68e30a/mpv-config/FSRCNNX_x2_56-16-4-1.glsl";

test("current release-clearance record is structurally valid and explicitly blocked", () => {
  const result = inspectReleaseClearance({ rootDir: root });
  assert.deepEqual(result.errors, []);
  assert.ok(result.blocked.length > 0);

  const fp16Gate = result.ledger.gates.find((gate) => gate.id === "unproven-rife-fp16-conversion");
  const fp16Artifact = fp16Gate.artifacts.find(
    (artifact) => artifact.path === "model/rife_v4.26_fp16.onnx",
  );
  assert.equal(fp16Gate.status, "cleared");
  assert.match(readFileSync(join(root, "MODEL_PROVENANCE.md"), "utf8"),
    new RegExp(fp16Artifact.sha256));

  for (const id of ["unidentified-rife-model", "unreproducible-span-smoke-model"]) {
    const gate = result.ledger.gates.find((entry) => entry.id === id);
    assert.equal(gate.status, "blocked", `${id} must continue to block private-history publication`);
    assert.ok(gate.artifacts.every((artifact) => artifact.disposition === "removed"));
  }

  const highGate = result.ledger.gates.find((gate) => gate.id === "unknown-high-x2-shader-origin");
  assert.equal(highGate.status, "blocked");
  assert.deepEqual(
    highGate.artifacts.map(({ path, sha256, disposition }) => ({ path, sha256, disposition })),
    [...highX2Artifacts].map(([path, sha256]) => ({ path, sha256, disposition: "present" })),
  );
  assert.ok(highGate.evidence.includes(highX2Mirror));
  for (const document of ["MODEL_PROVENANCE.md", "shaders/README.md", "THIRD_PARTY_NOTICES.md"]) {
    const record = readFileSync(join(root, document), "utf8");
    assert.match(record, /0e6bdb96f2845d883ec0131af8598c438c68e30a/);
    assert.match(record, /authoritative/i);
  }

  const lgplGate = result.ledger.gates.find((gate) => gate.id === "lgpl-compliance-review");
  const ortGate = result.ledger.gates.find((gate) => gate.id === "onnx-runtime-third-party-review");
  assert.equal(lgplGate.status, "blocked");
  assert.equal(ortGate.status, "blocked");
  assert.ok(lgplGate.evidence.includes("LGPL_REBUILDING.md"));
  assert.ok(lgplGate.artifacts.every((artifact) => /^[0-9a-f]{64}$/.test(artifact.sha256)));
  assert.ok(ortGate.evidence.includes("vendor/ort/LICENSE"));
  assert.ok(ortGate.artifacts.every((artifact) => /^[0-9a-f]{64}$/.test(artifact.sha256)));
});

test("restored high x2 assets remain an explicit current-boundary blocker", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-high-x2-"));
  try {
    mkdirSync(join(fixture, "model"));
    mkdirSync(join(fixture, "shaders"));
    for (const path of highX2Artifacts.keys()) {
      copyFileSync(join(root, path), join(fixture, path));
    }

    const makeLedger = ({ status = "blocked", disposition = "present" } = {}) => ({
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "unknown-high-x2-shader-origin",
        status,
        artifacts: [...highX2Artifacts].map(([path, sha256]) => ({
          path,
          sha256,
          disposition,
        })),
        evidence: [highX2Mirror],
        resolution: "Obtain authoritative origin and license evidence.",
      }],
    });

    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify(makeLedger()));
    const valid = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["unknown-high-x2-shader-origin"],
    });
    assert.deepEqual(valid.errors, []);
    assert.equal(valid.blocked.length, 1);

    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify(makeLedger({ status: "cleared" })));
    const accidentallyCleared = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["unknown-high-x2-shader-origin"],
    });
    assert.ok(accidentallyCleared.errors.some((error) =>
      error.includes("must remain blocked pending authoritative origin and license evidence")));

    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify(makeLedger({ disposition: "removed" })));
    const falselyRemoved = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["unknown-high-x2-shader-origin"],
    });
    assert.ok(falselyRemoved.errors.some((error) => error.includes("must be explicitly present")));
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("release-clearance validation detects artifact drift", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-"));
  try {
    writeFileSync(join(fixture, "artifact.bin"), "changed");
    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify({
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "fixture-gate",
        status: "blocked",
        artifacts: [{
          path: "artifact.bin",
          sha256: createHash("sha256").update("original").digest("hex"),
        }],
        resolution: "Replace the fixture.",
      }],
    }));

    const result = inspectReleaseClearance({ rootDir: fixture, requiredGateIds: ["fixture-gate"] });
    assert.equal(result.errors.length, 1);
    assert.match(result.errors[0], /artifact\.bin hash is/);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("release-clearance validation rejects a deleted gate inventory", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-empty-"));
  try {
    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify({
      schemaVersion: 1,
      scope: "test",
      gates: [],
    }));

    const result = inspectReleaseClearance({ rootDir: fixture, requiredGateIds: ["required-gate"] });
    assert.deepEqual(result.errors, ["release clearance: missing required gate required-gate"]);
    assert.deepEqual(result.blocked, []);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("cleared gates require meaningful evidence references", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-evidence-"));
  try {
    writeFileSync(join(fixture, "artifact.bin"), "fixture");
    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify({
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "required-gate",
        status: "cleared",
        artifacts: [{ path: "artifact.bin" }],
        evidence: [null],
        resolution: "Retain the evidence.",
      }],
    }));

    const result = inspectReleaseClearance({ rootDir: fixture, requiredGateIds: ["required-gate"] });
    assert.deepEqual(result.errors, [
      "release clearance gate 1: evidence must contain only non-empty string references",
    ]);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("blocked and cleared gates require repository-local evidence files to exist", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-missing-evidence-"));
  try {
    writeFileSync(join(fixture, "artifact.bin"), "fixture");
    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify({
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "required-gate",
        status: "blocked",
        artifacts: [{ path: "artifact.bin" }],
        evidence: ["missing-evidence.md"],
        resolution: "Retain the evidence.",
      }],
    }));

    const result = inspectReleaseClearance({ rootDir: fixture, requiredGateIds: ["required-gate"] });
    assert.deepEqual(result.errors, [
      "release clearance gate 1 evidence 1: missing missing-evidence.md (ENOENT)",
    ]);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("cleared FP16 gate requires its artifact hash in the provenance record", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-fp16-evidence-"));
  try {
    mkdirSync(join(fixture, "model"));
    const bytes = "reproduced FP16 fixture";
    const hash = createHash("sha256").update(bytes).digest("hex");
    writeFileSync(join(fixture, "model", "rife_v4.26_fp16.onnx"), bytes);
    writeFileSync(join(fixture, "MODEL_PROVENANCE.md"), "# Missing the artifact digest\n");
    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify({
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "unproven-rife-fp16-conversion",
        status: "cleared",
        artifacts: [{ path: "model/rife_v4.26_fp16.onnx", sha256: hash }],
        evidence: ["MODEL_PROVENANCE.md"],
        resolution: "Retain deterministic reproduction evidence.",
      }],
    }));

    const result = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["unproven-rife-fp16-conversion"],
    });
    assert.deepEqual(result.errors, [
      "release clearance: cleared FP16 artifact hash is absent from MODEL_PROVENANCE.md",
    ]);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("cleared removal records retain historical hashes and enforce absence", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-removed-"));
  try {
    const ledger = {
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "removed-gate",
        status: "cleared",
        artifacts: [{
          path: "removed.bin",
          sha256: createHash("sha256").update("historical bytes").digest("hex"),
          disposition: "removed",
        }],
        evidence: ["evidence.txt"],
        resolution: "Keep the artifact absent.",
      }],
    };
    writeFileSync(join(fixture, "evidence.txt"), "The artifact was removed from the release boundary.\n");
    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify(ledger));

    const absent = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["removed-gate"],
    });
    assert.deepEqual(absent.errors, []);
    assert.deepEqual(absent.blocked, []);

    writeFileSync(join(fixture, "removed.bin"), "historical bytes");
    const restored = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["removed-gate"],
    });
    assert.deepEqual(restored.errors, [
      "release clearance gate 1 artifact 1: removed artifact removed.bin must remain absent",
    ]);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});
