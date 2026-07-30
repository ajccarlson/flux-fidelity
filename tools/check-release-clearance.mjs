import { createHash } from "node:crypto";
import { lstatSync, readFileSync } from "node:fs";
import { isAbsolute, posix, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(import.meta.dirname, "..");
const DEFAULT_LEDGER_FILE = "docs/compliance/release-clearance.json";
const PROVENANCE_FILE = "docs/compliance/MODEL_PROVENANCE.md";
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const STATUS_VALUES = new Set(["blocked", "cleared", "accepted-risk"]);
const ARTIFACT_DISPOSITIONS = new Set(["present", "removed"]);
export const REQUIRED_RELEASE_GATE_IDS = Object.freeze([
  "unidentified-rife-model",
  "unproven-rife-fp16-conversion",
  "neural-model-provenance",
  "cda-vsr-license-review",
  "unknown-high-x2-shader-origin",
  "missing-x3-x4-shader-sources",
  "unreproducible-span-smoke-model",
  "unresolved-deband-port-origin",
  "lgpl-compliance-review",
  "onnx-runtime-third-party-review",
]);
const HIGH_X2_GATE_ID = "unknown-high-x2-shader-origin";
const LGPL_GATE_ID = "lgpl-compliance-review";
const CDA_VSR_GATE_ID = "cda-vsr-license-review";
const ACCEPTED_HISTORY_RISK_GATE_IDS = new Set([
  "unidentified-rife-model",
  "missing-x3-x4-shader-sources",
  "unreproducible-span-smoke-model",
  "unresolved-deband-port-origin",
]);
const ACCEPTED_REVIEW_RISK_GATE_IDS = new Set([
  CDA_VSR_GATE_ID,
  LGPL_GATE_ID,
  "onnx-runtime-third-party-review",
]);
const ACCEPTED_RISK_GATE_IDS = new Set([
  ...ACCEPTED_HISTORY_RISK_GATE_IDS,
  ...ACCEPTED_REVIEW_RISK_GATE_IDS,
]);
const NEURAL_MODEL_GATE_ID = "neural-model-provenance";
const NEURAL_MODEL_ARTIFACTS = Object.freeze(new Map([
  [
    "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
    "4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2",
  ],
  [
    "model/neural/realesrganv2_animevideo_xsx2.fp16.onnx",
    "f674a410b528aec55bb9f9f594cb1aaea580237adb29abd9dc32296d34b690a0",
  ],
]));
const NEURAL_MODEL_EVIDENCE = Object.freeze([
  PROVENANCE_FILE,
  "LICENSES/Real-ESRGAN-BSD-3-Clause.txt",
  "tools/neural-export/README.md",
  "tools/neural-export/export.py",
  "tools/neural-export/requirements.txt",
  "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.3.0",
  "https://github.com/xinntao/Real-ESRGAN/blob/f07aaffda04c7e69f11e6bfaf8023a6435471459/LICENSE",
]);
const NEURAL_MODEL_PROVENANCE_MARKERS = Object.freeze([
  "27985aa2198711ecd72f9bb274ec7b164e018fc9ce2933daaa7c7ab36a2bd3fe",
  "f674a410b528aec55bb9f9f594cb1aaea580237adb29abd9dc32296d34b690a0",
  "f07aaffda04c7e69f11e6bfaf8023a6435471459",
  "4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2",
]);
const CDA_VSR_ARTIFACTS = Object.freeze(new Map([
  [
    "model/neural/cda-vsr-initializer.onnx",
    "7773490658a7cad663e9b4f7e9cc8269b3d0c7a9a8e5840ec3151e895c1161f1",
  ],
  [
    "model/neural/cda-vsr-recurrent.onnx",
    "442be6f8d356889070ed70acdb49f9d2d77f24b6947e51e823404ca5a6d66a05",
  ],
]));
const CDA_VSR_EVIDENCE = Object.freeze([
  PROVENANCE_FILE,
  "THIRD_PARTY_NOTICES.md",
  "tools/cda-vsr/README.md",
  "tools/cda-vsr/promote_bundled.py",
  "https://github.com/sspBIT/CDA-VSR/tree/5707d997759996f19521c3beaddfb3d1ea965d44",
  "https://github.com/sspBIT/CDA-VSR/blob/5707d997759996f19521c3beaddfb3d1ea965d44/README.md",
  "https://github.com/sspBIT/CDA-VSR/blob/5707d997759996f19521c3beaddfb3d1ea965d44/NOTICE",
]);
const CDA_VSR_PROVENANCE_MARKERS = Object.freeze([
  "5707d997759996f19521c3beaddfb3d1ea965d44",
  "0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b",
  "afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e",
  "7d688658a2acdf249d5224dac7c6d4cdad8764ecc02e34084bc0306cabf3ac0d",
  "c1c69f1163f2d83bfa8af40ed69edc9cfc962d50e86c79c31f2019cee7c7af24",
  "7773490658a7cad663e9b4f7e9cc8269b3d0c7a9a8e5840ec3151e895c1161f1",
  "442be6f8d356889070ed70acdb49f9d2d77f24b6947e51e823404ca5a6d66a05",
  "License not specified by upstream",
]);
const HIGH_X2_ARTIFACTS = Object.freeze(new Map([
  [
    "shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl",
    "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
  ],
  [
    "model/FSRCNNX_x2_56-16-4-1.wgsl",
    "c7075c901a3e1aa25d1279eb360181ddd9d179e59dc4950162471c1fc9e711c9",
  ],
  [
    "model/FSRCNNX_x2_56-16-4-1.passes.json",
    "baee24ca3cbcbb01d73e6da17fc5d6f46fe1e3e66f4f5e4b9491484980bebf5a",
  ],
]));
const HIGH_X2_EVIDENCE = Object.freeze([
  PROVENANCE_FILE,
  "shaders/README.md",
  "docs/compliance/LGPL_REBUILDING.md",
  "https://web.archive.org/web/20190330194401/https://github.com/igv/FSRCNN-TensorFlow/releases",
  "https://web.archive.org/web/20201011050553id_/https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/checkpoints_params.7z",
  "https://github.com/igv/FSRCNN-TensorFlow/blob/1aa11ab0e1fc12741fdb84cef31da5619a478670/gen.py",
]);
const HIGH_X2_PROVENANCE_MARKERS = Object.freeze([
  "28167f74341256054c790e94c30a10964818f6bdbe7aedb97c6507208123fc10",
  "a27f732e1609a0d26e768d63447a42b04acd71918386026e1ca18a937ceea290",
  "aa99254fd8001f2d0ac99e93a71f7225d78227e282b727b9c4bf7e5901e601ca",
  "b507e0ec6c0d9ab22d440736677cd2ccb8a8b5441e190889ca7ec762d53ca063",
  "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
]);

function safeRelativePath(path) {
  return typeof path === "string"
    && path.length > 0
    && !isAbsolute(path)
    && !path.includes("\\")
    && posix.normalize(path) === path
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
  ledgerFile = DEFAULT_LEDGER_FILE,
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
      errors.push(`${label}: status must be blocked, cleared, or accepted-risk`);
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

      const primaryShaValid = artifact.sha256 === undefined ||
        (typeof artifact.sha256 === "string" && SHA256_PATTERN.test(artifact.sha256));
      if (!primaryShaValid) {
        errors.push(`${artifactLabel}: sha256 must be 64 lowercase hexadecimal characters`);
      }

      if (artifact.historicalSha256 !== undefined) {
        if (disposition !== "removed") {
          errors.push(`${artifactLabel}: historicalSha256 is allowed only for removed artifacts`);
        }
        if (!Array.isArray(artifact.historicalSha256) ||
            artifact.historicalSha256.length === 0) {
          errors.push(
            `${artifactLabel}: historicalSha256 must be a non-empty array when present`,
          );
        } else {
          const historicalHashes = new Set();
          for (const [hashIndex, hash] of artifact.historicalSha256.entries()) {
            const hashLabel = `${artifactLabel}: historicalSha256 ${hashIndex + 1}`;
            if (typeof hash !== "string" || !SHA256_PATTERN.test(hash)) {
              errors.push(
                `${hashLabel}: hash must be 64 lowercase hexadecimal characters`,
              );
              continue;
            }
            if (historicalHashes.has(hash)) {
              errors.push(`${hashLabel}: duplicate historical hash ${hash}`);
              continue;
            }
            historicalHashes.add(hash);
            if (hash === artifact.sha256) {
              errors.push(`${hashLabel}: hash must differ from primary sha256`);
            }
          }
        }
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

      if (artifact.sha256 !== undefined && primaryShaValid) {
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
    if (gate.status !== "blocked" &&
        (!Array.isArray(gate.evidence) || gate.evidence.length === 0)) {
      errors.push(`${label}: a nonblocking gate must retain at least one evidence reference`);
    }
    if (gate.status === "accepted-risk") {
      if (!ACCEPTED_RISK_GATE_IDS.has(gate.id)) {
        errors.push(`${label}: accepted-risk is allowed only for a designated release gate`);
      }
      if (ACCEPTED_HISTORY_RISK_GATE_IDS.has(gate.id) &&
          gate.artifacts.some((artifact) => artifact?.disposition !== "removed")) {
        errors.push(`${label}: historical accepted-risk requires every artifact to be removed`);
      }
      if (ACCEPTED_REVIEW_RISK_GATE_IDS.has(gate.id) &&
          gate.artifacts.some((artifact) => artifact?.disposition === "removed")) {
        errors.push(`${label}: review accepted-risk cannot cover a removed historical artifact`);
      }
      if (typeof gate.riskAcceptance?.acceptedBy !== "string" ||
          !gate.riskAcceptance.acceptedBy.trim()) {
        errors.push(`${label}: accepted-risk requires a non-empty riskAcceptance.acceptedBy`);
      }
      if (typeof gate.riskAcceptance?.acceptedOn !== "string" ||
          !/^\d{4}-\d{2}-\d{2}$/.test(gate.riskAcceptance.acceptedOn)) {
        errors.push(`${label}: accepted-risk requires riskAcceptance.acceptedOn in YYYY-MM-DD form`);
      }
      if (ACCEPTED_REVIEW_RISK_GATE_IDS.has(gate.id) &&
          gate.riskAcceptance?.reviewDeferred !== true) {
        errors.push(`${label}: accepted-risk requires riskAcceptance.reviewDeferred to be true`);
      }
      if (ACCEPTED_HISTORY_RISK_GATE_IDS.has(gate.id) &&
          gate.riskAcceptance?.historyRetentionAccepted !== true) {
        errors.push(
          `${label}: historical accepted-risk requires ` +
          "riskAcceptance.historyRetentionAccepted to be true",
        );
      }
      if (typeof gate.riskAcceptance?.rationale !== "string" ||
          !gate.riskAcceptance.rationale.trim()) {
        errors.push(`${label}: accepted-risk requires a non-empty riskAcceptance.rationale`);
      }
    } else if (gate.riskAcceptance !== undefined) {
      errors.push(`${label}: riskAcceptance is allowed only with accepted-risk status`);
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

  // High's origin is cleared by an archived official release and exact
  // reproduction chain. Keep that evidence and its separate LGPL review
  // inventory explicit so provenance clearance cannot imply legal clearance.
  if (requiredIds.has(HIGH_X2_GATE_ID)) {
    const highGate = ledger.gates.find((gate) => gate?.id === HIGH_X2_GATE_ID);
    if (highGate?.status !== "cleared") {
      errors.push(`release clearance: ${HIGH_X2_GATE_ID} must remain cleared by the official release and reproduction evidence`);
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

    const highEvidence = new Set(Array.isArray(highGate?.evidence) ? highGate.evidence : []);
    for (const reference of HIGH_X2_EVIDENCE) {
      if (!highEvidence.has(reference)) {
        errors.push(`release clearance: ${HIGH_X2_GATE_ID} is missing authoritative evidence ${reference}`);
      }
    }

    let provenance = "";
    try {
      provenance = readFileSync(resolve(rootDir, PROVENANCE_FILE), "utf8");
    } catch {
      // The generic evidence check reports the missing file.
    }
    for (const marker of HIGH_X2_PROVENANCE_MARKERS) {
      if (provenance && !provenance.includes(marker)) {
        errors.push(`release clearance: ${HIGH_X2_GATE_ID} provenance is missing ${marker}`);
      }
    }

    if (requiredIds.has(LGPL_GATE_ID)) {
      const lgplGate = ledger.gates.find((gate) => gate?.id === LGPL_GATE_ID);
      const lgplArtifacts = Array.isArray(lgplGate?.artifacts) ? lgplGate.artifacts : [];
      for (const [path, sha256] of HIGH_X2_ARTIFACTS) {
        const artifact = lgplArtifacts.find((entry) => entry?.path === path);
        if (!artifact || artifact.sha256 !== sha256 ||
            (artifact.disposition !== undefined && artifact.disposition !== "present")) {
          errors.push(`release clearance: ${LGPL_GATE_ID} must retain High artifact ${path} at SHA-256 ${sha256}`);
        }
      }
    }
  }

  if (requiredIds.has(NEURAL_MODEL_GATE_ID)) {
    const neuralGate = ledger.gates.find((gate) => gate?.id === NEURAL_MODEL_GATE_ID);
    if (neuralGate?.status !== "cleared") {
      errors.push(`release clearance: ${NEURAL_MODEL_GATE_ID} must remain cleared by the official release, license, and reproduction evidence`);
    }

    const artifacts = Array.isArray(neuralGate?.artifacts) ? neuralGate.artifacts : [];
    if (artifacts.length !== NEURAL_MODEL_ARTIFACTS.size) {
      errors.push(`release clearance: ${NEURAL_MODEL_GATE_ID} must inventory exactly ${NEURAL_MODEL_ARTIFACTS.size} artifacts`);
    }
    for (const [path, sha256] of NEURAL_MODEL_ARTIFACTS) {
      const artifact = artifacts.find((entry) => entry?.path === path);
      if (!artifact) {
        errors.push(`release clearance: ${NEURAL_MODEL_GATE_ID} is missing ${path}`);
        continue;
      }
      if (artifact.disposition !== "present") {
        errors.push(`release clearance: ${NEURAL_MODEL_GATE_ID} artifact ${path} must be explicitly present`);
      }
      if (artifact.sha256 !== sha256) {
        errors.push(`release clearance: ${NEURAL_MODEL_GATE_ID} artifact ${path} must retain SHA-256 ${sha256}`);
      }
    }

    const evidence = new Set(Array.isArray(neuralGate?.evidence) ? neuralGate.evidence : []);
    for (const reference of NEURAL_MODEL_EVIDENCE) {
      if (!evidence.has(reference)) {
        errors.push(`release clearance: ${NEURAL_MODEL_GATE_ID} is missing authoritative evidence ${reference}`);
      }
    }

    let provenance = "";
    try {
      provenance = readFileSync(resolve(rootDir, PROVENANCE_FILE), "utf8");
    } catch {
      // The generic evidence check reports the missing file.
    }
    for (const marker of NEURAL_MODEL_PROVENANCE_MARKERS) {
      if (provenance && !provenance.includes(marker)) {
        errors.push(`release clearance: ${NEURAL_MODEL_GATE_ID} provenance is missing ${marker}`);
      }
    }
  }

  if (requiredIds.has(CDA_VSR_GATE_ID)) {
    const cdaGate = ledger.gates.find((gate) => gate?.id === CDA_VSR_GATE_ID);
    if (cdaGate?.status !== "accepted-risk") {
      errors.push(
        `release clearance: ${CDA_VSR_GATE_ID} must remain an explicit ` +
        "repository-owner accepted risk while upstream licensing is unspecified",
      );
    }

    const artifacts = Array.isArray(cdaGate?.artifacts) ? cdaGate.artifacts : [];
    if (artifacts.length !== CDA_VSR_ARTIFACTS.size) {
      errors.push(
        `release clearance: ${CDA_VSR_GATE_ID} must inventory exactly ` +
        `${CDA_VSR_ARTIFACTS.size} artifacts`,
      );
    }
    for (const [path, sha256] of CDA_VSR_ARTIFACTS) {
      const artifact = artifacts.find((entry) => entry?.path === path);
      if (!artifact) {
        errors.push(`release clearance: ${CDA_VSR_GATE_ID} is missing ${path}`);
        continue;
      }
      if (artifact.disposition !== "present") {
        errors.push(
          `release clearance: ${CDA_VSR_GATE_ID} artifact ${path} must be explicitly present`,
        );
      }
      if (artifact.sha256 !== sha256) {
        errors.push(
          `release clearance: ${CDA_VSR_GATE_ID} artifact ${path} ` +
          `must retain SHA-256 ${sha256}`,
        );
      }
    }

    const evidence = new Set(Array.isArray(cdaGate?.evidence) ? cdaGate.evidence : []);
    for (const reference of CDA_VSR_EVIDENCE) {
      if (!evidence.has(reference)) {
        errors.push(
          `release clearance: ${CDA_VSR_GATE_ID} is missing evidence ${reference}`,
        );
      }
    }

    let provenance = "";
    try {
      provenance = readFileSync(resolve(rootDir, PROVENANCE_FILE), "utf8");
    } catch {
      // The generic evidence check reports the missing file.
    }
    for (const marker of CDA_VSR_PROVENANCE_MARKERS) {
      if (provenance && !provenance.includes(marker)) {
        errors.push(`release clearance: ${CDA_VSR_GATE_ID} provenance is missing ${marker}`);
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
        provenance = readFileSync(resolve(rootDir, PROVENANCE_FILE), "utf8");
      } catch (error) {
        errors.push(`release clearance: cannot read ${PROVENANCE_FILE} (${error.code || error.message})`);
      }
      if (provenance && !provenance.includes(fp16Artifact.sha256)) {
        errors.push(`release clearance: cleared FP16 artifact hash is absent from ${PROVENANCE_FILE}`);
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
