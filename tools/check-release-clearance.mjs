import { createHash } from "node:crypto";
import { lstatSync, readFileSync } from "node:fs";
import { isAbsolute, normalize, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(import.meta.dirname, "..");
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const STATUS_VALUES = new Set(["blocked", "cleared"]);
const ARTIFACT_DISPOSITIONS = new Set(["present", "removed"]);
export const REQUIRED_RELEASE_GATE_IDS = Object.freeze([
  "unidentified-rife-model",
  "unproven-rife-fp16-conversion",
  "unknown-high-x2-shader-origin",
  "missing-x3-x4-shader-sources",
  "unreproducible-span-smoke-model",
  "unresolved-deband-port-origin",
  "lgpl-compliance-review",
  "onnx-runtime-third-party-review",
]);
const HIGH_X2_GATE_ID = "unknown-high-x2-shader-origin";
const HIGH_X2_ARTIFACTS = Object.freeze(new Map([
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
]));

function safeRelativePath(path) {
  return typeof path === "string"
    && path.length > 0
    && !isAbsolute(path)
    && !path.includes("\\")
    && normalize(path) === path
    && !path.split("/").includes("..");
}

function inspectRegularFile(rootDir, path, label, errors) {
  const absolute = resolve(rootDir, path);
  const escaped = relative(rootDir, absolute).split(/[\\/]/).includes("..");
  if (escaped) {
    errors.push(`${label}: path escapes the repository`);
    return null;
  }

  let metadata;
  try {
    metadata = lstatSync(absolute);
  } catch (error) {
    errors.push(`${label}: missing ${path} (${error.code || error.message})`);
    return null;
  }
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    errors.push(`${label}: ${path} must be a regular, non-symlink file`);
    return null;
  }
  return absolute;
}

function isExternalEvidenceReference(reference) {
  try {
    const url = new URL(reference);
    return url.protocol === "https:";
  } catch {
    return false;
  }
}

export function inspectReleaseClearance({
  rootDir = root,
  ledgerFile = "release-clearance.json",
  requiredGateIds = REQUIRED_RELEASE_GATE_IDS,
} = {}) {
  const errors = [];
  let ledger;
  try {
    ledger = JSON.parse(readFileSync(resolve(rootDir, ledgerFile), "utf8"));
  } catch (error) {
    return { errors: [`${ledgerFile}: ${error.message}`], blocked: [], ledger: null };
  }

  if (ledger?.schemaVersion !== 1) errors.push("release clearance: schemaVersion must be 1");
  if (typeof ledger?.scope !== "string" || !ledger.scope.trim()) {
    errors.push("release clearance: scope must be a non-empty string");
  }
  if (!Array.isArray(ledger?.gates)) {
    errors.push("release clearance: gates must be an array");
    return { errors, blocked: [], ledger };
  }
  if (!Array.isArray(requiredGateIds) || requiredGateIds.length === 0 ||
      requiredGateIds.some((id) => typeof id !== "string" || !id)) {
    throw new TypeError("requiredGateIds must be a non-empty string array");
  }

  const ids = new Set();
  for (const [index, gate] of ledger.gates.entries()) {
    const label = `release clearance gate ${index + 1}`;
    if (typeof gate?.id !== "string" || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(gate.id)) {
      errors.push(`${label}: id must be a kebab-case string`);
    } else if (ids.has(gate.id)) {
      errors.push(`${label}: duplicate id ${gate.id}`);
    } else {
      ids.add(gate.id);
    }

    if (!STATUS_VALUES.has(gate?.status)) {
      errors.push(`${label}: status must be blocked or cleared`);
    }
    if (typeof gate?.resolution !== "string" || !gate.resolution.trim()) {
      errors.push(`${label}: resolution must be a non-empty string`);
    }
    if (!Array.isArray(gate?.artifacts) || gate.artifacts.length === 0) {
      errors.push(`${label}: artifacts must be a non-empty array`);
      continue;
    }

    const paths = new Set();
    for (const [artifactIndex, artifact] of gate.artifacts.entries()) {
      const artifactLabel = `${label} artifact ${artifactIndex + 1}`;
      if (!safeRelativePath(artifact?.path)) {
        errors.push(`${artifactLabel}: path must be a normalized repository-relative path`);
        continue;
      }
      if (paths.has(artifact.path)) errors.push(`${artifactLabel}: duplicate path ${artifact.path}`);
      paths.add(artifact.path);

      const disposition = artifact.disposition ?? "present";
      if (!ARTIFACT_DISPOSITIONS.has(disposition)) {
        errors.push(`${artifactLabel}: disposition must be present or removed`);
        continue;
      }

      if (artifact.sha256 !== undefined &&
          (typeof artifact.sha256 !== "string" || !SHA256_PATTERN.test(artifact.sha256))) {
        errors.push(`${artifactLabel}: sha256 must be 64 lowercase hexadecimal characters`);
        continue;
      }

      const absolute = resolve(rootDir, artifact.path);

      if (disposition === "removed") {
        if (artifact.sha256 === undefined) {
          errors.push(`${artifactLabel}: a removed artifact must retain its historical sha256`);
          continue;
        }
        try {
          lstatSync(absolute);
          errors.push(`${artifactLabel}: removed artifact ${artifact.path} must remain absent`);
        } catch (error) {
          if (error.code !== "ENOENT") {
            errors.push(`${artifactLabel}: cannot confirm removal of ${artifact.path} (${error.code || error.message})`);
          }
        }
        continue;
      }

      const checkedPath = inspectRegularFile(rootDir, artifact.path, artifactLabel, errors);
      if (checkedPath === null) continue;

      if (artifact.sha256 !== undefined) {
        const actual = createHash("sha256").update(readFileSync(checkedPath)).digest("hex");
        if (actual !== artifact.sha256) {
          errors.push(`${artifactLabel}: ${artifact.path} hash is ${actual}, expected ${artifact.sha256}`);
        }
      }
    }

    const validEvidenceInventory = gate.evidence === undefined || (Array.isArray(gate.evidence) &&
      gate.evidence.every((reference) => typeof reference === "string" && reference.trim()));
    if (!validEvidenceInventory) {
      errors.push(`${label}: evidence must contain only non-empty string references`);
    }
    if (gate.status === "cleared" && (!Array.isArray(gate.evidence) || gate.evidence.length === 0)) {
      errors.push(`${label}: a cleared gate must retain at least one evidence reference`);
    }
    // Local evidence must remain inspectable even while a gate is blocked. A
    // blocked record is still an auditable claim about the current boundary;
    // accepting missing evidence until clearance would let its support rot.
    if (validEvidenceInventory && Array.isArray(gate.evidence)) {
      for (const [evidenceIndex, reference] of gate.evidence.entries()) {
        if (isExternalEvidenceReference(reference)) continue;
        const evidenceLabel = `${label} evidence ${evidenceIndex + 1}`;
        if (!safeRelativePath(reference)) {
          errors.push(`${evidenceLabel}: evidence must be an HTTPS URL or normalized repository-relative path`);
          continue;
        }
        inspectRegularFile(rootDir, reference, evidenceLabel, errors);
      }
    }
  }

  const requiredIds = new Set(requiredGateIds);
  for (const id of requiredIds) {
    if (!ids.has(id)) errors.push(`release clearance: missing required gate ${id}`);
  }
  for (const id of ids) {
    if (!requiredIds.has(id)) errors.push(`release clearance: unexpected gate ${id}`);
  }

  // The restored high x2 model is intentionally a current-boundary blocker.
  // Keep this invariant explicit so a missing disposition, hash drift, or an
  // accidental status flip cannot silently turn byte identity into a license
  // claim. Once authoritative origin and permission are established, updating
  // this guard must be part of the deliberate gate-clearance change.
  if (requiredIds.has(HIGH_X2_GATE_ID)) {
    const highGate = ledger.gates.find((gate) => gate?.id === HIGH_X2_GATE_ID);
    if (highGate?.status !== "blocked") {
      errors.push(`release clearance: ${HIGH_X2_GATE_ID} must remain blocked pending authoritative origin and license evidence`);
    }

    const highArtifacts = Array.isArray(highGate?.artifacts) ? highGate.artifacts : [];
    if (highArtifacts.length !== HIGH_X2_ARTIFACTS.size) {
      errors.push(`release clearance: ${HIGH_X2_GATE_ID} must inventory exactly ${HIGH_X2_ARTIFACTS.size} artifacts`);
    }
    for (const [path, sha256] of HIGH_X2_ARTIFACTS) {
      const artifact = highArtifacts.find((entry) => entry?.path === path);
      if (!artifact) {
        errors.push(`release clearance: ${HIGH_X2_GATE_ID} is missing ${path}`);
        continue;
      }
      if (artifact.disposition !== "present") {
        errors.push(`release clearance: ${HIGH_X2_GATE_ID} artifact ${path} must be explicitly present`);
      }
      if (artifact.sha256 !== sha256) {
        errors.push(`release clearance: ${HIGH_X2_GATE_ID} artifact ${path} must retain SHA-256 ${sha256}`);
      }
    }
  }

  const fp16Gate = ledger.gates.find((gate) => gate?.id === "unproven-rife-fp16-conversion");
  if (fp16Gate?.status === "cleared") {
    const fp16Artifact = fp16Gate.artifacts?.find(
      (artifact) => artifact?.path === "model/rife_v4.26_fp16.onnx",
    );
    if (!SHA256_PATTERN.test(fp16Artifact?.sha256 ?? "")) {
      errors.push("release clearance: cleared FP16 gate must retain the artifact SHA-256");
    } else {
      let provenance = "";
      try {
        provenance = readFileSync(resolve(rootDir, "MODEL_PROVENANCE.md"), "utf8");
      } catch (error) {
        errors.push(`release clearance: cannot read MODEL_PROVENANCE.md (${error.code || error.message})`);
      }
      if (provenance && !provenance.includes(fp16Artifact.sha256)) {
        errors.push("release clearance: cleared FP16 artifact hash is absent from MODEL_PROVENANCE.md");
      }
    }
  }

  return {
    errors,
    blocked: ledger.gates.filter((gate) => gate?.status === "blocked"),
    ledger,
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const recordOnly = process.argv.includes("--record-only");
  const result = inspectReleaseClearance();
  const historyOnly = result.blocked.filter((gate) =>
    Array.isArray(gate.artifacts) && gate.artifacts.length > 0 &&
      gate.artifacts.every((artifact) => artifact.disposition === "removed"));
  const currentBoundary = result.blocked.filter((gate) =>
    Array.isArray(gate.artifacts) &&
      gate.artifacts.some((artifact) => artifact.disposition !== "removed"));
  if (result.errors.length) {
    for (const error of result.errors) console.error(error);
    process.exitCode = 1;
  } else if (result.blocked.length && !recordOnly) {
    console.error(`Public distribution blocked by ${result.blocked.length} unresolved gate(s):`);
    if (currentBoundary.length) {
      console.error(`Current tree or extension package (${currentBoundary.length}):`);
      for (const gate of currentBoundary) console.error(`- ${gate.id}: ${gate.resolution}`);
    }
    if (historyOnly.length) {
      console.error(`Repository-publication history only (${historyOnly.length}):`);
      for (const gate of historyOnly) console.error(`- ${gate.id}: ${gate.resolution}`);
    }
    process.exitCode = 1;
  } else {
    const suffix = result.blocked.length
      ? `; ${currentBoundary.length} current-boundary and ${historyOnly.length} history-only gate(s) remain blocked`
      : "";
    console.log(`Release-clearance record: ok (${result.ledger.gates.length} gates${suffix})`);
  }
}
