import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { inspectReleaseClearance } from "../tools/check-release-clearance.mjs";

const root = resolve(import.meta.dirname, "..");
const provenanceFile = "docs/compliance/MODEL_PROVENANCE.md";
const neuralModelArtifacts = new Map([
  [
    "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
    "4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2",
  ],
  [
    "model/neural/realesrganv2_animevideo_xsx2.fp16.onnx",
    "f674a410b528aec55bb9f9f594cb1aaea580237adb29abd9dc32296d34b690a0",
  ],
]);
const neuralModelEvidence = [
  provenanceFile,
  "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
  "tools/neural-export/README.md",
  "tools/neural-export/export.py",
  "tools/neural-export/requirements.txt",
  "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.3.0",
  "https://github.com/xinntao/Real-ESRGAN/blob/f07aaffda04c7e69f11e6bfaf8023a6435471459/LICENSE",
];
const neuralModelProvenanceMarkers = [
  "27985aa2198711ecd72f9bb274ec7b164e018fc9ce2933daaa7c7ab36a2bd3fe",
  "f674a410b528aec55bb9f9f594cb1aaea580237adb29abd9dc32296d34b690a0",
  "f07aaffda04c7e69f11e6bfaf8023a6435471459",
  "4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2",
];
const highX2Artifacts = new Map([
  [
    "shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl",
    "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
  ],
  [
    "model/FSRCNNX_x2_56-16-4-1.wgsl",
    "19a5327c8f96b7cb0593512f846f75ef266a3d857a84532c4dc5a374296e3d11",
  ],
  [
    "model/FSRCNNX_x2_56-16-4-1.passes.json",
    "4b7512ca17fd9788f4876f2681207fa8fb3b10c46d314ea2b3ce684864fb4d70",
  ],
]);
const highX2Evidence = [
  provenanceFile,
  "shaders/README.md",
  "docs/compliance/LGPL_REBUILDING.md",
  "https://web.archive.org/web/20190330194401/https://github.com/igv/FSRCNN-TensorFlow/releases",
  "https://web.archive.org/web/20201011050553id_/https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/checkpoints_params.7z",
  "https://github.com/igv/FSRCNN-TensorFlow/blob/1aa11ab0e1fc12741fdb84cef31da5619a478670/gen.py",
];
const highX2ProvenanceMarkers = [
  "28167f74341256054c790e94c30a10964818f6bdbe7aedb97c6507208123fc10",
  "a27f732e1609a0d26e768d63447a42b04acd71918386026e1ca18a937ceea290",
  "aa99254fd8001f2d0ac99e93a71f7225d78227e282b727b9c4bf7e5901e601ca",
  "b507e0ec6c0d9ab22d440736677cd2ccb8a8b5441e190889ca7ec762d53ca063",
  "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
];
const lgplLicenseArtifacts = new Map([
  [
    "LICENSES/GPL-3.0.txt",
    "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
  ],
  [
    "LICENSES/LGPL-3.0.txt",
    "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118",
  ],
]);
const debandVersions = Object.freeze({
  initial: Object.freeze({
    bytes: 3932,
    sha256: "05495befe6af578baa3fb84b69983a0dddc943d60b249411cc81c7eca1bbbbaf",
  }),
  later: Object.freeze({
    bytes: 4160,
    sha256: "56155c7bd5a15b5524ec1b44baeb4b5cb368e57f9adaf5ff8635bd1a2dba3f84",
  }),
});

function writeComplianceFile(rootDir, name, contents) {
  const directory = join(rootDir, "docs", "compliance");
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, name), contents);
}

function writeLedger(rootDir, ledger) {
  writeComplianceFile(rootDir, "release-clearance.json", JSON.stringify(ledger));
}

test("current release-clearance record is structurally valid and explicitly blocked", () => {
  const result = inspectReleaseClearance({ rootDir: root });
  assert.deepEqual(result.errors, []);
  assert.ok(result.blocked.length > 0);

  const fp16Gate = result.ledger.gates.find((gate) => gate.id === "unproven-rife-fp16-conversion");
  const fp16Artifact = fp16Gate.artifacts.find(
    (artifact) => artifact.path === "model/rife_v4.26_fp16.onnx",
  );
  assert.equal(fp16Gate.status, "cleared");
  assert.match(readFileSync(join(root, provenanceFile), "utf8"),
    new RegExp(fp16Artifact.sha256));

  for (const id of ["unidentified-rife-model", "unreproducible-span-smoke-model"]) {
    const gate = result.ledger.gates.find((entry) => entry.id === id);
    assert.equal(gate.status, "blocked", `${id} must continue to block private-history publication`);
    assert.ok(gate.artifacts.every((artifact) => artifact.disposition === "removed"));
  }

  const highGate = result.ledger.gates.find((gate) => gate.id === "unknown-high-x2-shader-origin");
  assert.equal(highGate.status, "cleared");
  assert.deepEqual(
    highGate.artifacts.map(({ path, sha256, disposition }) => ({ path, sha256, disposition })),
    [...highX2Artifacts].map(([path, sha256]) => ({ path, sha256, disposition: "present" })),
  );
  for (const reference of highX2Evidence) assert.ok(highGate.evidence.includes(reference));
  const provenance = readFileSync(join(root, provenanceFile), "utf8");
  for (const marker of highX2ProvenanceMarkers) assert.ok(provenance.includes(marker));

  const neuralGate = result.ledger.gates.find((gate) => gate.id === "neural-model-provenance");
  assert.equal(neuralGate.status, "cleared");
  assert.deepEqual(
    neuralGate.artifacts.map(({ path, sha256, disposition }) => ({ path, sha256, disposition })),
    [...neuralModelArtifacts].map(([path, sha256]) => ({ path, sha256, disposition: "present" })),
  );
  for (const reference of neuralModelEvidence) assert.ok(neuralGate.evidence.includes(reference));
  for (const marker of neuralModelProvenanceMarkers) assert.ok(provenance.includes(marker));

  const debandGate = result.ledger.gates.find(
    (gate) => gate.id === "unresolved-deband-port-origin",
  );
  assert.deepEqual(debandGate.artifacts, [{
    path: "fsrcnnx-deband.js",
    sha256: debandVersions.later.sha256,
    historicalSha256: [debandVersions.initial.sha256],
    disposition: "removed",
  }]);
  assert.ok(provenance.includes(
    `Initial ${debandVersions.initial.bytes.toLocaleString("en-US")}-byte revision: ` +
    `\`${debandVersions.initial.sha256}\``,
  ));
  assert.ok(provenance.includes(
    `later ${debandVersions.later.bytes.toLocaleString("en-US")}-byte revision: ` +
    `\`${debandVersions.later.sha256}\``,
  ));

  const lgplGate = result.ledger.gates.find((gate) => gate.id === "lgpl-compliance-review");
  const ortGate = result.ledger.gates.find((gate) => gate.id === "onnx-runtime-third-party-review");
  assert.equal(lgplGate.status, "blocked");
  assert.equal(ortGate.status, "blocked");
  assert.ok(lgplGate.evidence.includes("docs/compliance/LGPL_REBUILDING.md"));
  assert.deepEqual(
    lgplGate.artifacts
      .filter((artifact) => lgplLicenseArtifacts.has(artifact.path))
      .map(({ path, sha256 }) => [path, sha256]),
    [...lgplLicenseArtifacts],
  );
  for (const path of lgplLicenseArtifacts.keys()) {
    assert.ok(lgplGate.evidence.includes(path));
  }
  for (const [path, sha256] of highX2Artifacts) {
    assert.ok(lgplGate.artifacts.some((artifact) =>
      artifact.path === path && artifact.sha256 === sha256));
  }
  assert.ok(lgplGate.artifacts.every((artifact) => /^[0-9a-f]{64}$/.test(artifact.sha256)));
  assert.ok(ortGate.evidence.includes("vendor/ort/LICENSE"));
  assert.ok(ortGate.artifacts.every((artifact) => /^[0-9a-f]{64}$/.test(artifact.sha256)));
});

test("cleared high x2 provenance retains official evidence and exact LGPL inventory", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-high-x2-"));
  try {
    mkdirSync(join(fixture, "model"), { recursive: true });
    mkdirSync(join(fixture, "shaders", "upstream"), { recursive: true });
    for (const path of highX2Artifacts.keys()) {
      copyFileSync(join(root, path), join(fixture, path));
    }
    for (const document of [
      provenanceFile,
      "shaders/README.md",
      "docs/compliance/LGPL_REBUILDING.md",
    ]) {
      const target = join(fixture, document);
      mkdirSync(resolve(target, ".."), { recursive: true });
      writeFileSync(target, highX2ProvenanceMarkers.join("\n"));
    }

    const makeLedger = ({
      status = "cleared",
      disposition = "present",
      evidence = highX2Evidence,
    } = {}) => ({
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
        evidence,
        resolution: "Retain the official release and exact reproduction evidence.",
      }],
    });

    writeLedger(fixture, makeLedger());
    const valid = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["unknown-high-x2-shader-origin"],
    });
    assert.deepEqual(valid.errors, []);
    assert.equal(valid.blocked.length, 0);

    writeLedger(fixture, makeLedger({ status: "blocked" }));
    const accidentallyBlocked = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["unknown-high-x2-shader-origin"],
    });
    assert.ok(accidentallyBlocked.errors.some((error) =>
      error.includes("must remain cleared by the official release and reproduction evidence")));

    writeLedger(fixture, makeLedger({ evidence: highX2Evidence.slice(1) }));
    const missingEvidence = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["unknown-high-x2-shader-origin"],
    });
    assert.ok(missingEvidence.errors.some((error) =>
      error.includes(`is missing authoritative evidence ${provenanceFile}`)));

    writeLedger(fixture, makeLedger({ disposition: "removed" }));
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
    writeLedger(fixture, {
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
    });

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
    writeLedger(fixture, {
      schemaVersion: 1,
      scope: "test",
      gates: [],
    });

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
    writeLedger(fixture, {
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "required-gate",
        status: "cleared",
        artifacts: [{ path: "artifact.bin" }],
        evidence: [null],
        resolution: "Retain the evidence.",
      }],
    });

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
    writeLedger(fixture, {
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "required-gate",
        status: "blocked",
        artifacts: [{ path: "artifact.bin" }],
        evidence: ["missing-evidence.md"],
        resolution: "Retain the evidence.",
      }],
    });

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
    writeComplianceFile(fixture, "MODEL_PROVENANCE.md", "# Missing the artifact digest\n");
    writeLedger(fixture, {
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "unproven-rife-fp16-conversion",
        status: "cleared",
        artifacts: [{ path: "model/rife_v4.26_fp16.onnx", sha256: hash }],
        evidence: [provenanceFile],
        resolution: "Retain deterministic reproduction evidence.",
      }],
    });

    const result = inspectReleaseClearance({
      rootDir: fixture,
      requiredGateIds: ["unproven-rife-fp16-conversion"],
    });
    assert.deepEqual(result.errors, [
      `release clearance: cleared FP16 artifact hash is absent from ${provenanceFile}`,
    ]);
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});

test("cleared removal records retain historical hashes and enforce absence", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-removed-"));
  try {
    const primaryHash = createHash("sha256").update("historical bytes").digest("hex");
    const olderHash = createHash("sha256").update("older historical bytes").digest("hex");
    const ledger = {
      schemaVersion: 1,
      scope: "test",
      gates: [{
        id: "removed-gate",
        status: "cleared",
        artifacts: [{
          path: "removed.bin",
          sha256: primaryHash,
          historicalSha256: [olderHash],
          disposition: "removed",
        }],
        evidence: ["evidence.txt"],
        resolution: "Keep the artifact absent.",
      }],
    };
    writeFileSync(join(fixture, "evidence.txt"), "The artifact was removed from the release boundary.\n");
    writeLedger(fixture, ledger);

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

test("historicalSha256 accepts only unique additional hashes for removed artifacts", () => {
  const fixture = mkdtempSync(join(tmpdir(), "fsrcnnx-release-clearance-history-"));
  try {
    const primaryBytes = "primary bytes";
    const primaryHash = createHash("sha256").update(primaryBytes).digest("hex");
    const olderHash = createHash("sha256").update("older bytes").digest("hex");
    writeFileSync(join(fixture, "evidence.txt"), "Historical artifact inventory.\n");

    const inspectArtifact = (artifact) => {
      writeLedger(fixture, {
        schemaVersion: 1,
        scope: "test",
        gates: [{
          id: "history-gate",
          status: "blocked",
          artifacts: [artifact],
          evidence: ["evidence.txt"],
          resolution: "Retain every historical byte identity.",
        }],
      });
      return inspectReleaseClearance({
        rootDir: fixture,
        requiredGateIds: ["history-gate"],
      }).errors;
    };
    const removedArtifact = {
      path: "removed.bin",
      sha256: primaryHash,
      disposition: "removed",
    };

    for (const { historicalSha256, expected } of [
      {
        historicalSha256: [],
        expected: "historicalSha256 must be a non-empty array when present",
      },
      {
        historicalSha256: olderHash,
        expected: "historicalSha256 must be a non-empty array when present",
      },
      {
        historicalSha256: [olderHash.toUpperCase()],
        expected: "historicalSha256 1: hash must be 64 lowercase hexadecimal characters",
      },
      {
        historicalSha256: [olderHash, olderHash],
        expected: `historicalSha256 2: duplicate historical hash ${olderHash}`,
      },
      {
        historicalSha256: [primaryHash],
        expected: "historicalSha256 1: hash must differ from primary sha256",
      },
    ]) {
      assert.deepEqual(
        inspectArtifact({ ...removedArtifact, historicalSha256 }),
        [`release clearance gate 1 artifact 1: ${expected}`],
      );
    }

    writeFileSync(join(fixture, "present.bin"), primaryBytes);
    assert.deepEqual(
      inspectArtifact({
        path: "present.bin",
        sha256: primaryHash,
        historicalSha256: [olderHash],
        disposition: "present",
      }),
      [
        "release clearance gate 1 artifact 1: " +
        "historicalSha256 is allowed only for removed artifacts",
      ],
    );
  } finally {
    rmSync(fixture, { recursive: true, force: true });
  }
});
