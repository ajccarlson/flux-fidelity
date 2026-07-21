import { createHash } from "node:crypto";
import { lstatSync, readFileSync } from "node:fs";
import { isAbsolute, normalize, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(import.meta.dirname, "..");
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const STATUS_VALUES = new Set(["blocked", "cleared"]);
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

function safeRelativePath(path) {
  return typeof path === "string"
    && path.length > 0
    && !isAbsolute(path)
    && !path.includes("\\")
    && normalize(path) === path
    && !path.split("/").includes("..");
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

      const absolute = resolve(rootDir, artifact.path);
      const escaped = relative(rootDir, absolute).split(/[\\/]/).includes("..");
      if (escaped) {
        errors.push(`${artifactLabel}: path escapes the repository`);
        continue;
      }
      let metadata;
      try {
        metadata = lstatSync(absolute);
      } catch (error) {
        errors.push(`${artifactLabel}: missing ${artifact.path} (${error.code || error.message})`);
        continue;
      }
      if (!metadata.isFile() || metadata.isSymbolicLink()) {
        errors.push(`${artifactLabel}: ${artifact.path} must be a regular, non-symlink file`);
        continue;
      }

      if (artifact.sha256 !== undefined) {
        if (typeof artifact.sha256 !== "string" || !SHA256_PATTERN.test(artifact.sha256)) {
          errors.push(`${artifactLabel}: sha256 must be 64 lowercase hexadecimal characters`);
          continue;
        }
        const actual = createHash("sha256").update(readFileSync(absolute)).digest("hex");
        if (actual !== artifact.sha256) {
          errors.push(`${artifactLabel}: ${artifact.path} hash is ${actual}, expected ${artifact.sha256}`);
        }
      }
    }

    if (gate.evidence !== undefined && (!Array.isArray(gate.evidence) ||
        gate.evidence.some((reference) => typeof reference !== "string" || !reference.trim()))) {
      errors.push(`${label}: evidence must contain only non-empty string references`);
    }
    if (gate.status === "cleared" && (!Array.isArray(gate.evidence) || gate.evidence.length === 0)) {
      errors.push(`${label}: a cleared gate must retain at least one evidence reference`);
    }
  }

  const requiredIds = new Set(requiredGateIds);
  for (const id of requiredIds) {
    if (!ids.has(id)) errors.push(`release clearance: missing required gate ${id}`);
  }
  for (const id of ids) {
    if (!requiredIds.has(id)) errors.push(`release clearance: unexpected gate ${id}`);
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
  if (result.errors.length) {
    for (const error of result.errors) console.error(error);
    process.exitCode = 1;
  } else if (result.blocked.length && !recordOnly) {
    console.error(`Public release blocked by ${result.blocked.length} unresolved gate(s):`);
    for (const gate of result.blocked) console.error(`- ${gate.id}: ${gate.resolution}`);
    process.exitCode = 1;
  } else {
    const suffix = result.blocked.length ? `; ${result.blocked.length} remain blocked` : "";
    console.log(`Release-clearance record: ok (${result.ledger.gates.length} gates${suffix})`);
  }
}
