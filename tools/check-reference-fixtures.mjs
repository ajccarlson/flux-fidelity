import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { isAbsolute, relative, resolve, sep } from "node:path";
import {
  encodeReferenceInput,
  REFERENCE_CASES,
  REFERENCE_FIXTURE_SCHEMA_VERSION,
  REFERENCE_GENERATOR_POLICY,
  REFERENCE_INPUTS,
  REFERENCE_INPUT_VERSION,
  REFERENCE_TOOLCHAIN,
} from "../reference-fixtures.js";
import { PACKAGE_FILES } from "./package-files.mjs";

const root = resolve(import.meta.dirname, "..");
const metadataPath = resolve(root, "validation", "reference-fixtures.json");
const errors = [];

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function equal(actual, expected) {
  return JSON.stringify(actual) === JSON.stringify(expected);
}

function commandText(command, args) {
  return [command, ...args].map((value) => (
    /^[A-Za-z0-9_./:=+-]+$/u.test(value) ? value : JSON.stringify(value)
  )).join(" ");
}

function expectedMpvOptions(entry) {
  return entry.id === "filter:ssimds-reference"
    ? [...REFERENCE_GENERATOR_POLICY.ssimOptions]
    : [];
}

function expectedCaptureCommand(entry) {
  const temporaryName = entry.id.replaceAll(":", "-");
  const shader = entry.oracle.kind === "mpv-libplacebo-artcnn-f32"
    ? `<temporary-directory>/${entry.id}.f32.glsl`
    : `<repository>/${entry.source.path}`;
  return commandText("xvfb-run", [
    "-a",
    "mpv",
    ...REFERENCE_GENERATOR_POLICY.mpvBaseOptions,
    ...expectedMpvOptions(entry),
    `--geometry=${entry.output.width}x${entry.output.height}`,
    "--script=<temporary-directory>/reference-capture.lua",
    `--script-opts=reference-output=<temporary-directory>/${temporaryName}.png,reference-delay=0.35`,
    `--glsl-shader=${shader}`,
    `<input:${entry.inputs[0].id}>`,
  ]);
}

function expectedOracle(entry) {
  if (entry.output.kind === "computed") return entry.oracle;
  return {
    ...entry.oracle,
    capture: "window",
    options: expectedMpvOptions(entry),
    command: expectedCaptureCommand(entry),
  };
}

function safeRepositoryPath(path, label) {
  if (typeof path !== "string" || !path || path.includes("\\")) {
    errors.push(`${label}: invalid repository path`);
    return null;
  }
  const absolute = resolve(root, path);
  const local = relative(root, absolute);
  if (isAbsolute(local) || local === ".." || local.startsWith(`..${sep}`)) {
    errors.push(`${label}: path escapes the repository (${path})`);
    return null;
  }
  return absolute;
}

function read(path, label) {
  try {
    return readFileSync(path);
  } catch (error) {
    errors.push(`${label}: cannot read (${error.message})`);
    return null;
  }
}

function expectedInputMetadata(role, id) {
  const spec = REFERENCE_INPUTS[id];
  const encoded = encodeReferenceInput(id);
  return {
    role,
    id,
    width: spec.width,
    height: spec.height,
    channels: spec.channels,
    encoding: spec.encoding,
    formula: spec.formula,
    portableAnymapByteLength: encoded.byteLength,
    portableAnymapSha256: sha256(encoded),
  };
}

function verifyGrayscaleReference(entry, bytes) {
  let mismatchCount = 0;
  let maxChannelDelta = 0;
  for (let offset = 0, pixel = 0; offset < bytes.byteLength; offset += 6, pixel++) {
    const r = bytes.readUInt16LE(offset);
    const g = bytes.readUInt16LE(offset + 2);
    const b = bytes.readUInt16LE(offset + 4);
    const delta = Math.max(r, g, b) - Math.min(r, g, b);
    if (delta) mismatchCount++;
    maxChannelDelta = Math.max(maxChannelDelta, delta);
  }
  if (maxChannelDelta > 2) {
    errors.push(`${entry.id}: grayscale screenshot channel spread is ${maxChannelDelta} units`);
  }
  return { mismatchCount, maxChannelDelta };
}

let metadata = null;
try {
  metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
} catch (error) {
  errors.push(`validation/reference-fixtures.json: ${error.message}`);
}

if (metadata) {
  if (metadata.schemaVersion !== REFERENCE_FIXTURE_SCHEMA_VERSION) {
    errors.push(`metadata schemaVersion must be ${REFERENCE_FIXTURE_SCHEMA_VERSION}`);
  }
  if (metadata.inputVersion !== REFERENCE_INPUT_VERSION) {
    errors.push(`metadata inputVersion must be ${REFERENCE_INPUT_VERSION}`);
  }
  if (!equal(metadata.generator, REFERENCE_GENERATOR_POLICY)) {
    errors.push("metadata generator policy drifted from the audited canonical record");
  }
  if (!equal(metadata.toolchain, REFERENCE_TOOLCHAIN)) {
    errors.push("metadata toolchain drifted from the audited canonical record");
  }

  if (!Array.isArray(metadata.cases)) {
    errors.push("metadata cases must be an array");
  } else {
    const expectedIds = REFERENCE_CASES.map((entry) => entry.id);
    const actualIds = metadata.cases.map((entry) => entry?.id);
    if (!equal(actualIds, expectedIds)) {
      errors.push("metadata cases must match the canonical ordered case inventory");
    }
    if (new Set(actualIds).size !== actualIds.length) errors.push("metadata case IDs must be unique");

    for (const entry of REFERENCE_CASES) {
      const recorded = metadata.cases.find((candidate) => candidate?.id === entry.id);
      if (!recorded) continue;
      if (recorded.label !== entry.label) errors.push(`${entry.id}: label drifted`);
      if (!equal(recorded.source, entry.source)) errors.push(`${entry.id}: source record drifted`);
      if (!equal(recorded.tolerances, entry.tolerances)) errors.push(`${entry.id}: tolerances drifted`);
      if (!equal(
        recorded.inputs,
        entry.inputs.map(({ role, id }) => expectedInputMetadata(role, id)),
      )) {
        errors.push(`${entry.id}: deterministic input record drifted`);
      }
      if (!equal(recorded.oracle, expectedOracle(entry))) errors.push(`${entry.id}: oracle record drifted`);

      if (entry.source) {
        const sourcePath = safeRepositoryPath(entry.source.path, `${entry.id} source`);
        const sourceBytes = sourcePath ? read(sourcePath, `${entry.id} source`) : null;
        if (sourceBytes && sha256(sourceBytes) !== entry.source.sha256) {
          errors.push(`${entry.source.path}: source SHA-256 drifted`);
        }
      }

      if (entry.output.kind === "fixture") {
        for (const field of ["kind", "path", "width", "height", "format", "comparison"]) {
          if (recorded.output?.[field] !== entry.output[field]) {
            errors.push(`${entry.id}: output ${field} drifted`);
          }
        }
        if (recorded.output?.sha256 !== entry.output.sha256) {
          errors.push(`${entry.id}: fixture SHA-256 drifted from the audited canonical digest`);
        }
        if (recorded.output?.channels !== 3) errors.push(`${entry.id}: fixture must have three channels`);
        const expectedLength = entry.output.width * entry.output.height * 3 * 2;
        if (recorded.output?.byteLength !== expectedLength) {
          errors.push(`${entry.id}: recorded fixture length must be ${expectedLength}`);
        }
        if (!PACKAGE_FILES.includes(entry.output.path)) {
          errors.push(`${entry.id}: ${entry.output.path} is outside the package boundary`);
        }
        const fixturePath = safeRepositoryPath(entry.output.path, `${entry.id} fixture`);
        const fixture = fixturePath ? read(fixturePath, `${entry.id} fixture`) : null;
        if (fixture) {
          if (fixture.byteLength !== expectedLength) {
            errors.push(`${entry.id}: fixture has ${fixture.byteLength} bytes; expected ${expectedLength}`);
          }
          const digest = sha256(fixture);
          if (digest !== entry.output.sha256) {
            errors.push(`${entry.id}: fixture SHA-256 does not match the audited canonical digest`);
          }
          if (digest !== recorded.output?.sha256) {
            errors.push(`${entry.id}: fixture SHA-256 does not match metadata`);
          }
          if (/^(?:FSRCNNX|ArtCNN)/u.test(entry.id)) {
            const diagnostics = verifyGrayscaleReference(entry, fixture);
            if (!equal(recorded.output?.channelDiagnostics, diagnostics)) {
              errors.push(`${entry.id}: grayscale channel diagnostics drifted`);
            }
          }
        }
      } else {
        if (!equal(recorded.output, entry.output)) errors.push(`${entry.id}: computed output record drifted`);
        if (recorded.output?.path || recorded.output?.sha256 || recorded.output?.byteLength) {
          errors.push(`${entry.id}: computed output must not name fixture bytes`);
        }
      }

      for (const [metric, limit] of Object.entries(entry.tolerances)) {
        if (!Number.isFinite(limit) || limit <= 0) {
          errors.push(`${entry.id}: tolerance ${metric} must be finite and positive`);
        }
      }
    }
  }
}

for (const path of [
  "reference-fixtures.js",
  "validation/README.md",
  "validation/reference-fixtures.json",
  ...REFERENCE_CASES.filter((entry) => entry.output.kind === "fixture").map((entry) => entry.output.path),
]) {
  if (!PACKAGE_FILES.includes(path)) errors.push(`package boundary is missing ${path}`);
}

if (errors.length) {
  for (const error of errors) console.error(error);
  process.exit(1);
}
console.log(`Reference fixtures: ok (${REFERENCE_CASES.length} cases)`);
