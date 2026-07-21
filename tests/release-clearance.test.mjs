import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { inspectReleaseClearance } from "../tools/check-release-clearance.mjs";

const root = resolve(import.meta.dirname, "..");

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

test("cleared gates require repository-local evidence files to exist", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-missing-evidence-"));
  try {
    writeFileSync(join(fixture, "artifact.bin"), "fixture");
    writeFileSync(join(fixture, "release-clearance.json"), JSON.stringify({
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "required-gate",
        status: "cleared",
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
