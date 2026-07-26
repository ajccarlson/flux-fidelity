#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants as fsConstants, createReadStream } from "node:fs";
import {
  copyFile,
  lstat,
  mkdtemp,
  readFile,
  realpath,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { validateNeuralManifest } from "../../src/core/fsrcnnx-neural.js";
import { buildPackage } from "../build-package.mjs";

const PROJECT_ROOT = resolve(import.meta.dirname, "../..");
const BROWSER_VALIDATOR = resolve(PROJECT_ROOT, "tools/browser-validation.mjs");
const RECEIPT_FILENAME = "cda-vsr-export.json";
const RECEIPT_FORMAT = 2;
const RECEIPT_TOOL = "FSRCNNX-EXT CDA-VSR conversion toolkit";
const RECEIPT_OPSET = 17;
const PRIOR_PROVIDER = "decoded-cda-v1";
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const GRAPH_FILES = Object.freeze({
  initializer: "cda-vsr-initializer.onnx",
  recurrent: "cda-vsr-recurrent.onnx",
});
const REFERENCE_SOURCE_SHA256 =
  "0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b";
const REFERENCE_CHECKPOINT_SHA256 =
  "afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e";

export const CDA_BROWSER_PROBE_MODEL_KEY = "cda-vsr-4x-local-probe";

function receiptError(detail) {
  throw new Error(`invalid CDA-VSR export receipt: ${detail}`);
}

function requireRecord(value, at) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    receiptError(`${at} must be an object`);
  }
  return value;
}

function requirePositiveInteger(value, at) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    receiptError(`${at} must be a positive safe integer`);
  }
  return value;
}

function requireFiniteNonnegative(value, at) {
  if (!Number.isFinite(value) || value < 0) {
    receiptError(`${at} must be a finite nonnegative number`);
  }
  return value;
}

function requireSha256(value, at) {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
    receiptError(`${at} must be a lowercase SHA-256 digest`);
  }
  return value;
}

function sameStringArray(value, expected) {
  return Array.isArray(value) && value.length === expected.length &&
    value.every((item, index) => item === expected[index]);
}

function validateInputIdentity(receipt) {
  const inputs = requireRecord(receipt.inputs, "inputs");
  const source = requireRecord(inputs.source, "inputs.source");
  const checkpoint = requireRecord(inputs.checkpoint, "inputs.checkpoint");
  const sourceSha256 = requireSha256(source.sha256, "inputs.source.sha256");
  const checkpointSha256 = requireSha256(
    checkpoint.sha256,
    "inputs.checkpoint.sha256",
  );
  requirePositiveInteger(source.bytes, "inputs.source.bytes");
  requirePositiveInteger(checkpoint.bytes, "inputs.checkpoint.bytes");

  const identity = requireRecord(receipt.input_identity, "input_identity");
  const canonical = sourceSha256 === REFERENCE_SOURCE_SHA256 &&
    checkpointSha256 === REFERENCE_CHECKPOINT_SHA256;
  const expectedPolicy = canonical
    ? "canonical-reference"
    : "explicit-unpinned-acknowledgement";
  if (identity.policy !== expectedPolicy) {
    receiptError(`input_identity.policy must be ${expectedPolicy}`);
  }
  if (identity.reference_source_sha256 !== REFERENCE_SOURCE_SHA256 ||
      identity.reference_checkpoint_sha256 !== REFERENCE_CHECKPOINT_SHA256) {
    receiptError("input_identity canonical reference hashes do not match the toolkit");
  }
  if (identity.architecture_execution !== "trusted-python-code") {
    receiptError("input_identity must record trusted architecture execution");
  }
}

function validateParityEvidence(receipt, captureFixture) {
  const policy = requireRecord(receipt.parity_policy, "parity_policy");
  if (policy.skipped !== false || policy.dynamic_shape_runtime_validated !== true) {
    receiptError("browser probing requires completed dynamic-shape parity");
  }
  const frames = requirePositiveInteger(policy.frames, "parity_policy.frames");
  if (frames < 2) receiptError("parity_policy.frames must be at least two");
  const maxAbs = requireFiniteNonnegative(policy.max_abs, "parity_policy.max_abs");
  const maxMean = requireFiniteNonnegative(policy.max_mean, "parity_policy.max_mean");
  if (maxAbs === 0 || maxMean === 0) {
    receiptError("parity limits must be positive");
  }

  const parity = requireRecord(receipt.parity, "parity");
  if (parity.spatial_shape !== "dynamic") {
    receiptError("parity.spatial_shape must be dynamic");
  }
  if (parity.frames_per_shape !== frames ||
      parity.max_abs_limit !== maxAbs ||
      parity.max_mean_limit !== maxMean) {
    receiptError("parity summary does not match parity_policy");
  }
  const worstMax = requireFiniteNonnegative(parity.worst_max_abs, "parity.worst_max_abs");
  const worstMean = requireFiniteNonnegative(parity.worst_mean_abs, "parity.worst_mean_abs");
  if (worstMax > maxAbs || worstMean > maxMean) {
    receiptError("recorded graph parity exceeds its limits");
  }
  if (!Array.isArray(parity.tested_shapes) || parity.tested_shapes.length < 2) {
    receiptError("dynamic graph parity must cover at least two shapes");
  }
  const shapes = parity.tested_shapes.map((value, index) => {
    const shape = requireRecord(value, `parity.tested_shapes[${index}]`);
    return {
      height: requirePositiveInteger(shape.height, `parity.tested_shapes[${index}].height`),
      width: requirePositiveInteger(shape.width, `parity.tested_shapes[${index}].width`),
    };
  });
  if (shapes[0].height !== captureFixture.height ||
      shapes[0].width !== captureFixture.width) {
    receiptError("the first parity shape must match the capture fixture");
  }
  if (!shapes.some(({ height, width }) =>
    height !== captureFixture.height || width !== captureFixture.width)) {
    receiptError("dynamic graph parity must include a distinct shape");
  }
}

function makeProbeEntry(contract) {
  return {
    key: CDA_BROWSER_PROBE_MODEL_KEY,
    label: "CDA-VSR 4x (local browser probe)",
    scale: 4,
    fp16: false,
    arch: "CDA-VSR local export",
    contract: structuredClone(contract),
  };
}

export function validateCdaExportReceipt(receipt) {
  requireRecord(receipt, "receipt");
  if (receipt.format !== RECEIPT_FORMAT) {
    receiptError(`format must be ${RECEIPT_FORMAT}`);
  }
  if (receipt.tool !== RECEIPT_TOOL) receiptError("tool identity is not recognized");
  if (receipt.opset !== RECEIPT_OPSET) {
    receiptError(`opset must be ${RECEIPT_OPSET}`);
  }

  const distribution = requireRecord(receipt.distribution, "distribution");
  if (distribution.architecture_license_status !== "not-established" ||
      distribution.checkpoint_license_status !== "not-established" ||
      distribution.checkpoint_redistribution_clearance !== false ||
      distribution.generated_assets !== "experimental-local-only" ||
      distribution.shipping_catalog !== false) {
    receiptError("distribution must retain the experimental local-only boundary");
  }

  const spatial = requireRecord(receipt.spatial_shape, "spatial_shape");
  const captureFixtureValue = requireRecord(
    spatial.capture_fixture,
    "spatial_shape.capture_fixture",
  );
  const captureFixture = Object.freeze({
    height: requirePositiveInteger(
      captureFixtureValue.height,
      "spatial_shape.capture_fixture.height",
    ),
    width: requirePositiveInteger(
      captureFixtureValue.width,
      "spatial_shape.capture_fixture.width",
    ),
  });
  if (spatial.mode !== "dynamic" || spatial.graph_shape_compatible !== true) {
    receiptError("browser probing requires dynamic, graph-shape-compatible exports");
  }
  if (spatial.source_resolution_ceiling !== null) {
    receiptError("spatial_shape.source_resolution_ceiling must remain null");
  }

  validateInputIdentity(receipt);
  validateParityEvidence(receipt, captureFixture);

  const runtime = requireRecord(receipt.runtime_contract, "runtime_contract");
  if (runtime.prior_provider !== PRIOR_PROVIDER ||
      !sameStringArray(runtime.motion_component_order, ["x", "y"]) ||
      runtime.motion_units !== "low-resolution-pixels" ||
      runtime.catalog_compatible_at_graph_shape_level !== true ||
      runtime.shipping_catalog !== false) {
    receiptError("runtime_contract is not a compatible non-shipping CDA contract");
  }
  const contract = requireRecord(
    runtime.manifest_v2_template,
    "runtime_contract.manifest_v2_template",
  );
  let normalizedEntry;
  try {
    [normalizedEntry] = validateNeuralManifest([makeProbeEntry(contract)]);
  } catch (error) {
    receiptError(`manifest_v2_template is incompatible: ${error.message}`);
  }
  if (normalizedEntry.contract.version !== 2 ||
      normalizedEntry.contract.mode !== "temporal" ||
      normalizedEntry.contract.resetGraph !== "initialize" ||
      normalizedEntry.contract.recurrentGraph !== "recurrent") {
    receiptError("manifest_v2_template must expose the CDA temporal graph pair");
  }

  const graphs = requireRecord(receipt.graphs, "graphs");
  const roles = Object.keys(graphs).sort();
  if (!sameStringArray(roles, Object.keys(GRAPH_FILES).sort())) {
    receiptError("graphs must contain exactly initializer and recurrent");
  }
  const verifiedGraphs = {};
  for (const [role, filename] of Object.entries(GRAPH_FILES)) {
    const graph = requireRecord(graphs[role], `graphs.${role}`);
    if (graph.file !== filename) {
      receiptError(`graphs.${role}.file must be ${filename}`);
    }
    if (graph.spatial_shape !== "dynamic") {
      receiptError(`graphs.${role}.spatial_shape must be dynamic`);
    }
    const contractGraphName = role === "initializer" ? "initialize" : "recurrent";
    if (contract.graphs?.[contractGraphName]?.file !== filename) {
      receiptError(`manifest graph ${contractGraphName} does not match graphs.${role}`);
    }
    verifiedGraphs[role] = Object.freeze({
      file: filename,
      bytes: requirePositiveInteger(graph.bytes, `graphs.${role}.bytes`),
      sha256: requireSha256(graph.sha256, `graphs.${role}.sha256`),
    });
  }

  return Object.freeze({
    contract: structuredClone(contract),
    graphs: Object.freeze(verifiedGraphs),
  });
}

async function sha256File(path) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

async function requireRegularFile(path, label) {
  let metadata;
  try {
    metadata = await lstat(path);
  } catch (error) {
    throw new Error(`${label} was not found: ${path}`, { cause: error });
  }
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    throw new Error(`${label} must be a regular, non-symbolic-link file: ${path}`);
  }
  return metadata;
}

export async function verifyCdaExportDirectory(onnxDir) {
  const requested = resolve(onnxDir);
  let resolvedDirectory;
  try {
    resolvedDirectory = await realpath(requested);
  } catch (error) {
    throw new Error(`CDA-VSR export directory was not found: ${requested}`, { cause: error });
  }
  const directoryMetadata = await lstat(resolvedDirectory);
  if (!directoryMetadata.isDirectory()) {
    throw new Error(`CDA-VSR export path is not a directory: ${resolvedDirectory}`);
  }

  const receiptPath = join(resolvedDirectory, RECEIPT_FILENAME);
  await requireRegularFile(receiptPath, "CDA-VSR export receipt");
  let receipt;
  try {
    receipt = JSON.parse(await readFile(receiptPath, "utf8"));
  } catch (error) {
    throw new Error(`could not parse CDA-VSR export receipt: ${error.message}`, { cause: error });
  }
  const validated = validateCdaExportReceipt(receipt);
  const verifiedGraphs = {};
  for (const [role, expected] of Object.entries(validated.graphs)) {
    const path = join(resolvedDirectory, expected.file);
    const metadata = await requireRegularFile(path, `CDA-VSR ${role} graph`);
    if (metadata.size !== expected.bytes) {
      throw new Error(
        `CDA-VSR ${role} graph byte length differs from receipt ` +
        `(expected ${expected.bytes}, got ${metadata.size})`,
      );
    }
    const sha256 = await sha256File(path);
    if (sha256 !== expected.sha256) {
      throw new Error(`CDA-VSR ${role} graph SHA-256 differs from receipt`);
    }
    verifiedGraphs[role] = Object.freeze({
      ...expected,
      path,
    });
  }
  return Object.freeze({
    directory: resolvedDirectory,
    receipt: Object.freeze({
      path: receiptPath,
      sha256: await sha256File(receiptPath),
    }),
    contract: validated.contract,
    graphs: Object.freeze(verifiedGraphs),
  });
}

export function makeCdaProbeManifest(manifest, contract) {
  const sourceEntries = Array.isArray(manifest) ? manifest : manifest?.models;
  if (!Array.isArray(sourceEntries)) {
    throw new Error("staged neural manifest must be an array or {models: array}");
  }
  const entries = [...structuredClone(sourceEntries), makeProbeEntry(contract)];
  try {
    validateNeuralManifest(entries);
  } catch (error) {
    throw new Error(`CDA-VSR probe manifest is invalid: ${error.message}`, { cause: error });
  }
  return Array.isArray(manifest)
    ? entries
    : { ...structuredClone(manifest), models: entries };
}

function isContainedBy(parent, child) {
  const path = relative(parent, child);
  return path === "" || (!path.startsWith("..") && !isAbsolute(path));
}

export async function stageCdaBrowserProbe({
  onnxDir,
  projectRoot = PROJECT_ROOT,
  temporaryParent = tmpdir(),
  buildPackageImpl = buildPackage,
} = {}) {
  if (!onnxDir) throw new Error("onnxDir is required");
  const verified = await verifyCdaExportDirectory(onnxDir);
  const temporaryRoot = await mkdtemp(join(resolve(temporaryParent), "fsrcnnx-cda-browser-probe-"));
  try {
    const packageResult = await buildPackageImpl({
      rootDir: resolve(projectRoot),
      distDir: temporaryRoot,
    });
    const extensionRoot = await realpath(packageResult.stage);
    if (!isContainedBy(temporaryRoot, extensionRoot)) {
      throw new Error("package builder returned a stage outside the disposable directory");
    }

    const manifestPath = join(extensionRoot, "model/neural/manifest.json");
    await requireRegularFile(manifestPath, "staged neural manifest");
    let manifest;
    try {
      manifest = JSON.parse(await readFile(manifestPath, "utf8"));
    } catch (error) {
      throw new Error(`could not parse staged neural manifest: ${error.message}`, { cause: error });
    }
    const probeManifest = makeCdaProbeManifest(manifest, verified.contract);

    for (const [role, graph] of Object.entries(verified.graphs)) {
      const target = join(extensionRoot, "model/neural", graph.file);
      await copyFile(graph.path, target, fsConstants.COPYFILE_EXCL);
      const copied = await requireRegularFile(target, `staged CDA-VSR ${role} graph`);
      if (copied.size !== graph.bytes || await sha256File(target) !== graph.sha256) {
        throw new Error(`staged CDA-VSR ${role} graph differs from the verified export`);
      }
    }
    await writeFile(manifestPath, `${JSON.stringify(probeManifest, null, 2)}\n`, {
      encoding: "utf8",
      mode: 0o644,
    });
    validateNeuralManifest(JSON.parse(await readFile(manifestPath, "utf8")));

    let cleaned = false;
    return Object.freeze({
      extensionRoot,
      modelKey: CDA_BROWSER_PROBE_MODEL_KEY,
      receipt: verified.receipt,
      graphs: verified.graphs,
      async cleanup() {
        if (cleaned) return;
        cleaned = true;
        await rm(temporaryRoot, {
          recursive: true,
          force: true,
          maxRetries: 3,
          retryDelay: 100,
        });
      },
    });
  } catch (error) {
    await rm(temporaryRoot, {
      recursive: true,
      force: true,
      maxRetries: 3,
      retryDelay: 100,
    });
    throw error;
  }
}

export function parseProbeArguments(argv) {
  let onnxDir = null;
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === "--onnx-dir") {
      const value = argv[++index];
      if (!value || value.startsWith("--")) throw new Error("--onnx-dir requires a path");
      onnxDir = resolve(value);
      continue;
    }
    throw new Error(`unknown CDA browser-probe argument: ${argument}`);
  }
  if (!onnxDir) {
    throw new Error(
      "usage: node tools/cda-vsr/browser_probe.mjs --onnx-dir <export-directory>",
    );
  }
  return Object.freeze({ onnxDir });
}

async function runBrowserValidator(extensionRoot, modelKey, signal) {
  const child = spawn(process.execPath, [
    BROWSER_VALIDATOR,
    "--extension-root",
    extensionRoot,
    "--neural-model-key",
    modelKey,
    "--require-temporal-neural-runs",
  ], {
    cwd: PROJECT_ROOT,
    env: process.env,
    stdio: "inherit",
    windowsHide: true,
  });
  const abort = () => child.kill("SIGTERM");
  signal?.addEventListener("abort", abort, { once: true });
  try {
    const result = await new Promise((resolveExit, rejectExit) => {
      child.once("error", rejectExit);
      child.once("exit", (code, exitSignal) => resolveExit({ code, exitSignal }));
    });
    if (signal?.aborted) {
      throw signal.reason instanceof Error ? signal.reason : new Error("browser probe aborted");
    }
    if (result.code !== 0) {
      throw new Error(
        `browser validation failed ` +
        `(exit ${result.code ?? "none"}${result.exitSignal ? `, signal ${result.exitSignal}` : ""})`,
      );
    }
  } finally {
    signal?.removeEventListener("abort", abort);
  }
}

async function main(options, signal) {
  const staged = await stageCdaBrowserProbe({ onnxDir: options.onnxDir });
  try {
    console.log(`Verified CDA-VSR export receipt: ${staged.receipt.sha256}`);
    for (const [role, graph] of Object.entries(staged.graphs)) {
      console.log(`Verified ${role} graph: ${graph.bytes} bytes, SHA-256 ${graph.sha256}`);
    }
    console.log(`Disposable extension root: ${staged.extensionRoot}`);
    console.log(`Requested neural model: ${staged.modelKey}`);
    await runBrowserValidator(staged.extensionRoot, staged.modelKey, signal);
    console.log("CDA-VSR external-artifact browser probe passed.");
  } finally {
    await staged.cleanup();
    console.log("Removed disposable CDA-VSR extension stage.");
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const abortController = new AbortController();
  let receivedSignal = null;
  const handlers = new Map();
  for (const name of ["SIGINT", "SIGTERM"]) {
    const handler = () => {
      receivedSignal = name;
      abortController.abort(new Error(`received ${name}`));
    };
    handlers.set(name, handler);
    process.once(name, handler);
  }
  try {
    await main(parseProbeArguments(process.argv.slice(2)), abortController.signal);
  } catch (error) {
    console.error(error.message || String(error));
    process.exitCode = receivedSignal === "SIGINT"
      ? 130
      : receivedSignal === "SIGTERM" ? 143 : 1;
  } finally {
    for (const [name, handler] of handlers) process.removeListener(name, handler);
  }
}
