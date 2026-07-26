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
const LEGACY_RECEIPT_FORMAT = 2;
const RECEIPT_FORMAT = 3;
const RECEIPT_TOOL = "FSRCNNX-EXT CDA-VSR conversion toolkit";
const RECEIPT_OPSET = 17;
const PRIOR_PROVIDER = "decoded-cda-v1";
const PARITY_SEED = 20260726;
const MINIMUM_PARITY_FRAMES = 25;
const FP32_PRECISION = "float32";
const MIXED_FP16_PRECISION = "mixed-fp16";
const NATIVE_OUTPUT_SCALE = 4;
const DERIVED_OUTPUT_SCALE = 2;
const DERIVED_X2_CONTRACT = Object.freeze({
  kind: "aligned-subpixel-average-v1",
  source_scale: 4,
  output_scale: 2,
  source_head_channels: 48,
  derived_head_channels: 12,
  phase_reduction: "mean-aligned-2x2",
  residual_base: "bilinear-align-corners-false-x2",
  shipping_catalog: false,
});
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const GRAPH_FILES = Object.freeze({
  initializer: "cda-vsr-initializer.onnx",
  recurrent: "cda-vsr-recurrent.onnx",
});
const STAGED_PROBE_GRAPH_FILES = Object.freeze({
  initializer: "cda-vsr-local-probe-initializer.onnx",
  recurrent: "cda-vsr-local-probe-recurrent.onnx",
});
const GRAPH_GRID_SAMPLE_NODES = Object.freeze({
  initializer: 0,
  recurrent: 5,
});
const DEFAULT_PARITY_LIMITS = Object.freeze({
  [FP32_PRECISION]: Object.freeze({
    output: Object.freeze({ max_abs: 2e-4, max_mean: 2e-5 }),
    state: Object.freeze({ max_abs: 2e-4, max_mean: 2e-5 }),
  }),
  [MIXED_FP16_PRECISION]: Object.freeze({
    output: Object.freeze({ max_abs: 5e-3, max_mean: 5e-4 }),
    state: Object.freeze({ max_abs: 2e-2, max_mean: 2e-3 }),
  }),
});
const REFERENCE_SOURCE_SHA256 =
  "0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b";
const REFERENCE_CHECKPOINT_SHA256 =
  "afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e";

export const CDA_BROWSER_PROBE_MODEL_KEY = "cda-vsr-local-probe";

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

function requireFinitePositive(value, at) {
  if (!Number.isFinite(value) || value <= 0) {
    receiptError(`${at} must be a finite positive number`);
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

function requireExactValue(value, expected, at) {
  if (Array.isArray(expected)) {
    if (!Array.isArray(value) || value.length !== expected.length) {
      receiptError(`${at} does not match the exact expected contract`);
    }
    expected.forEach((item, index) => {
      requireExactValue(value[index], item, `${at}[${index}]`);
    });
    return value;
  }
  if (expected && typeof expected === "object") {
    const record = requireRecord(value, at);
    const actualKeys = Object.keys(record).sort();
    const expectedKeys = Object.keys(expected).sort();
    if (!sameStringArray(actualKeys, expectedKeys)) {
      receiptError(`${at} does not have the exact expected fields`);
    }
    for (const key of expectedKeys) {
      requireExactValue(record[key], expected[key], `${at}.${key}`);
    }
    return record;
  }
  if (value !== expected) {
    receiptError(`${at} must be ${JSON.stringify(expected)}`);
  }
  return value;
}

function precisionContract(profile) {
  if (profile !== FP32_PRECISION && profile !== MIXED_FP16_PRECISION) {
    receiptError(`precision.profile must be ${FP32_PRECISION} or ${MIXED_FP16_PRECISION}`);
  }
  const computeDtype = profile === MIXED_FP16_PRECISION ? "float16" : "float32";
  return {
    profile,
    weight_dtype: computeDtype,
    feature_dtype: computeDtype,
    state_dtype: computeDtype,
    coordinate_dtype: "float32",
    grid_sample_dtype: "float32",
    public_inputs: {
      frame: "float32",
      motion: "float32",
      residual: "float32",
      state_low: computeDtype,
      state_high: computeDtype,
    },
    public_outputs: {
      output: "float32",
      next_state_low: computeDtype,
      next_state_high: computeDtype,
    },
  };
}

function temporalTilingProfile(precision, outputScale = NATIVE_OUTPUT_SCALE) {
  const mixed = precision === MIXED_FP16_PRECISION;
  return {
    kind: "temporal-state-atlas-v1",
    scale: outputScale,
    halo: 64,
    haloDerivation: {
      motionSearchRadius: 8,
      fixedRecurrentRadius: 35,
      minimum: 64,
      alignment: 8,
    },
    largestLogicalBytesPerSourcePixel: mixed ? 512 : 776,
    preferredInputExtent: 512,
    inputAlignment: 8,
    workgroupSize: 8,
    stateAtlas: {
      stateCount: 2,
      channelsPerState: 64,
      arrayLayersPerState: 16,
      textureFormat: mixed ? "rgba16float" : "rgba32float",
    },
  };
}

function graphLogicalMemoryProfile(
  role,
  precision,
  outputScale = NATIVE_OUTPUT_SCALE,
) {
  const mixed = precision === MIXED_FP16_PRECISION;
  const computeDtype = mixed ? "float16" : "float32";
  const computeBytes = mixed ? 2 : 4;
  const channels = role === "recurrent" ? 194 : 64;
  const convBytes = channels * computeBytes;
  const gridChannels = role === "recurrent"
    ? [32, 32, 32, 32, 128]
    : [];
  const gridBytes = gridChannels.length ? 512 : 0;
  const publicBytes = 3 * outputScale * outputScale * 4;
  const candidates = [
    ["conv-input", convBytes],
    ["grid-sample-source", gridBytes],
    ["public-output", publicBytes],
  ];
  const [largestTensorKind, largestBytes] = candidates.reduce(
    (largest, candidate) => candidate[1] > largest[1] ? candidate : largest,
  );
  const conv = {
    channels,
    dtype: computeDtype,
    spatial_scale: 1,
    bytes_per_source_pixel: convBytes,
  };
  return {
    largest_bytes_per_source_pixel: largestBytes,
    largest_tensor_kind: largestTensorKind,
    max_conv_input: conv,
    deform_align_predictor_input: role === "recurrent" ? { ...conv } : null,
    grid_sample_sources: {
      channels: gridChannels,
      dtype: gridChannels.length ? "float32" : null,
      spatial_scale: 1,
      largest_bytes_per_source_pixel: gridBytes,
    },
    public_output: {
      channels: 3,
      dtype: "float32",
      spatial_scale: outputScale,
      bytes_per_source_pixel: publicBytes,
    },
  };
}

function validatePrecision(receipt) {
  const recorded = requireRecord(receipt.precision, "precision");
  const expected = precisionContract(recorded.profile);
  requireExactValue(recorded, expected, "precision");
  return expected;
}

function validateInputIdentity(receipt, { canonicalOnly = false } = {}) {
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
  if (canonicalOnly && !canonical) {
    receiptError("browser probing format 3 requires the canonical source and checkpoint");
  }
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
  return Object.freeze({ sourceSha256, checkpointSha256, canonical });
}

function validateLegacyParityEvidence(receipt, captureFixture) {
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

function dynamicProbeShape({ height, width }) {
  const probe = {
    height: height + (height % 2 === 1 ? 2 : 3),
    width: width + (width % 2 === 1 ? 4 : 5),
  };
  if (probe.height === probe.width) probe.width += 2;
  return probe;
}

function requireSameNumber(value, expected, at) {
  requireFiniteNonnegative(value, at);
  if (value !== expected) {
    receiptError(`${at} does not match its recorded evidence`);
  }
  return value;
}

function validateTensorLimits(value, precision, at) {
  const limits = requireRecord(value, at);
  const classes = Object.keys(limits).sort();
  if (!sameStringArray(classes, ["output", "state"])) {
    receiptError(`${at} must contain exactly output and state`);
  }
  const validated = {};
  for (const tensorClass of ["output", "state"]) {
    const itemAt = `${at}.${tensorClass}`;
    const item = requireRecord(limits[tensorClass], itemAt);
    if (!sameStringArray(Object.keys(item).sort(), ["max_abs", "max_mean"])) {
      receiptError(`${itemAt} must contain exactly max_abs and max_mean`);
    }
    const maxAbs = requireFinitePositive(item.max_abs, `${itemAt}.max_abs`);
    const maxMean = requireFinitePositive(item.max_mean, `${itemAt}.max_mean`);
    const defaults = DEFAULT_PARITY_LIMITS[precision][tensorClass];
    if (maxAbs > defaults.max_abs || maxMean > defaults.max_mean) {
      receiptError(
        `${itemAt} is looser than the ${precision} browser-probe defaults`,
      );
    }
    validated[tensorClass] = { max_abs: maxAbs, max_mean: maxMean };
  }
  return validated;
}

function validateParitySequence(
  value,
  {
    at,
    expectedShape,
    frames,
    motionFixture,
    tensorLimits,
    maxAbs,
    maxMean,
  },
) {
  const result = requireRecord(value, at);
  if (result.height !== expectedShape.height ||
      result.width !== expectedShape.width ||
      result.frames !== frames ||
      result.seed !== PARITY_SEED ||
      result.motion_fixture !== motionFixture) {
    receiptError(`${at} fixture identity is inconsistent`);
  }
  requireSameNumber(result.max_abs_limit, maxAbs, `${at}.max_abs_limit`);
  requireSameNumber(result.max_mean_limit, maxMean, `${at}.max_mean_limit`);
  requireExactValue(result.tensor_limits, tensorLimits, `${at}.tensor_limits`);

  if (!Array.isArray(result.records) || result.records.length !== frames) {
    receiptError(`${at}.records must contain exactly ${frames} frames`);
  }
  const measured = { output: [], state: [] };
  const tensorNames = ["next_state_high", "next_state_low", "output"];
  result.records.forEach((recordValue, frameIndex) => {
    const recordAt = `${at}.records[${frameIndex}]`;
    const record = requireRecord(recordValue, recordAt);
    const expectedRole = frameIndex === 0 ? "initializer" : "recurrent";
    if (record.frame !== frameIndex || record.role !== expectedRole) {
      receiptError(`${recordAt} frame role is inconsistent`);
    }
    const tensors = requireRecord(record.tensors, `${recordAt}.tensors`);
    if (!sameStringArray(Object.keys(tensors).sort(), tensorNames)) {
      receiptError(`${recordAt}.tensors does not match the temporal output ABI`);
    }
    for (const tensorName of tensorNames) {
      const metricAt = `${recordAt}.tensors.${tensorName}`;
      const metrics = requireRecord(tensors[tensorName], metricAt);
      const meanAbs = requireFiniteNonnegative(metrics.mean_abs, `${metricAt}.mean_abs`);
      const p99_9Abs = requireFiniteNonnegative(
        metrics.p99_9_abs,
        `${metricAt}.p99_9_abs`,
      );
      const tensorMaxAbs = requireFiniteNonnegative(
        metrics.max_abs,
        `${metricAt}.max_abs`,
      );
      if (meanAbs > tensorMaxAbs || p99_9Abs > tensorMaxAbs) {
        receiptError(`${metricAt} contains impossible error summaries`);
      }
      const tensorClass = tensorName === "output" ? "output" : "state";
      measured[tensorClass].push({
        max_abs: tensorMaxAbs,
        max_mean: meanAbs,
        p99_9_abs: p99_9Abs,
      });
    }
  });

  const expectedWorst = {};
  for (const tensorClass of ["output", "state"]) {
    const values = measured[tensorClass];
    const summary = {
      worst_max_abs: Math.max(...values.map(({ max_abs: value }) => value)),
      worst_mean_abs: Math.max(...values.map(({ max_mean: value }) => value)),
      worst_p99_9_abs: Math.max(...values.map(({ p99_9_abs: value }) => value)),
    };
    const limit = tensorLimits[tensorClass];
    if (summary.worst_max_abs > limit.max_abs ||
        summary.worst_mean_abs > limit.max_mean) {
      receiptError(`${at} ${tensorClass} evidence exceeds its parity limits`);
    }
    expectedWorst[tensorClass] = summary;
  }

  const recordedWorst = requireRecord(
    result.worst_by_tensor_class,
    `${at}.worst_by_tensor_class`,
  );
  if (!sameStringArray(Object.keys(recordedWorst).sort(), ["output", "state"])) {
    receiptError(`${at}.worst_by_tensor_class has an invalid schema`);
  }
  for (const tensorClass of ["output", "state"]) {
    const item = requireRecord(
      recordedWorst[tensorClass],
      `${at}.worst_by_tensor_class.${tensorClass}`,
    );
    for (const [metric, expected] of Object.entries(expectedWorst[tensorClass])) {
      requireSameNumber(
        item[metric],
        expected,
        `${at}.worst_by_tensor_class.${tensorClass}.${metric}`,
      );
    }
  }

  const summary = {
    worst_max_abs: Math.max(
      expectedWorst.output.worst_max_abs,
      expectedWorst.state.worst_max_abs,
    ),
    worst_mean_abs: Math.max(
      expectedWorst.output.worst_mean_abs,
      expectedWorst.state.worst_mean_abs,
    ),
    worst_p99_9_abs: Math.max(
      expectedWorst.output.worst_p99_9_abs,
      expectedWorst.state.worst_p99_9_abs,
    ),
  };
  for (const [metric, expected] of Object.entries(summary)) {
    requireSameNumber(result[metric], expected, `${at}.${metric}`);
  }
  return summary;
}

function validateParityEvidenceV3(receipt, captureFixture, precision) {
  const policy = requireRecord(receipt.parity_policy, "parity_policy");
  if (policy.skipped !== false || policy.dynamic_shape_runtime_validated !== true) {
    receiptError("browser probing requires completed dynamic-shape parity");
  }
  const frames = requirePositiveInteger(policy.frames, "parity_policy.frames");
  if (frames < MINIMUM_PARITY_FRAMES) {
    receiptError(
      `parity_policy.frames must cover at least ${MINIMUM_PARITY_FRAMES} frames`,
    );
  }
  if (policy.reference_precision !== FP32_PRECISION) {
    receiptError("parity_policy.reference_precision must be float32");
  }
  if (policy.state_chains !== "independent") {
    receiptError("parity_policy.state_chains must be independent");
  }
  if (!sameStringArray(
    policy.motion_fixtures,
    ["decoded-integer", "fractional-stress"],
  )) {
    receiptError(
      "parity_policy.motion_fixtures must cover decoded-integer and fractional-stress",
    );
  }
  const tensorLimits = validateTensorLimits(
    policy.tensor_limits,
    precision,
    "parity_policy.tensor_limits",
  );
  const maxAbs = Math.max(
    tensorLimits.output.max_abs,
    tensorLimits.state.max_abs,
  );
  const maxMean = Math.max(
    tensorLimits.output.max_mean,
    tensorLimits.state.max_mean,
  );
  requireSameNumber(policy.max_abs, maxAbs, "parity_policy.max_abs");
  requireSameNumber(policy.max_mean, maxMean, "parity_policy.max_mean");

  const parity = requireRecord(receipt.parity, "parity");
  if (parity.spatial_shape !== "dynamic" ||
      parity.reference_precision !== FP32_PRECISION ||
      parity.state_chains !== "independent" ||
      parity.primary_motion_fixture !== "decoded-integer") {
    receiptError("parity must record dynamic independent FP32-reference evidence");
  }
  if (parity.frames_per_shape !== frames || parity.seed !== PARITY_SEED) {
    receiptError("parity fixture identity does not match parity_policy");
  }
  requireSameNumber(parity.max_abs_limit, maxAbs, "parity.max_abs_limit");
  requireSameNumber(parity.max_mean_limit, maxMean, "parity.max_mean_limit");
  requireExactValue(parity.tensor_limits, tensorLimits, "parity.tensor_limits");

  const expectedShapes = [
    { ...captureFixture },
    dynamicProbeShape(captureFixture),
  ];
  requireExactValue(parity.tested_shapes, expectedShapes, "parity.tested_shapes");
  if (!Array.isArray(parity.shape_results) ||
      parity.shape_results.length !== expectedShapes.length) {
    receiptError("parity.shape_results must cover both dynamic shapes");
  }

  const summaries = parity.shape_results.map((result, index) =>
    validateParitySequence(result, {
      at: `parity.shape_results[${index}]`,
      expectedShape: expectedShapes[index],
      frames,
      motionFixture: "decoded-integer",
      tensorLimits,
      maxAbs,
      maxMean,
    }));
  summaries.push(validateParitySequence(parity.fractional_motion_stress, {
    at: "parity.fractional_motion_stress",
    expectedShape: expectedShapes[0],
    frames,
    motionFixture: "fractional-stress",
    tensorLimits,
    maxAbs,
    maxMean,
  }));

  const aggregate = {
    worst_max_abs: Math.max(...summaries.map(({ worst_max_abs: value }) => value)),
    worst_mean_abs: Math.max(...summaries.map(({ worst_mean_abs: value }) => value)),
    worst_p99_9_abs: Math.max(
      ...summaries.map(({ worst_p99_9_abs: value }) => value),
    ),
  };
  for (const [metric, expected] of Object.entries(aggregate)) {
    requireSameNumber(parity[metric], expected, `parity.${metric}`);
  }
}

function makeProbeEntry(contract) {
  const stagedContract = structuredClone(contract);
  const outputScale = stagedContract.tiling?.scale ?? NATIVE_OUTPUT_SCALE;
  stagedContract.graphs.initialize.file = STAGED_PROBE_GRAPH_FILES.initializer;
  stagedContract.graphs.recurrent.file = STAGED_PROBE_GRAPH_FILES.recurrent;
  return {
    key: CDA_BROWSER_PROBE_MODEL_KEY,
    label: `CDA-VSR ${outputScale}x (local browser probe)`,
    scale: outputScale,
    fp16: false,
    arch: "CDA-VSR local export",
    contract: stagedContract,
  };
}

function validateManifestPrecision(contract, precision) {
  const expected = precisionContract(precision);
  const initialize = contract.graphs.initialize;
  const recurrent = contract.graphs.recurrent;
  const checks = [
    [initialize.inputs.frame, "float32", "initialize.inputs.frame"],
    [initialize.outputs.output, "float32", "initialize.outputs.output"],
    [
      initialize.outputs.next_state_low,
      expected.state_dtype,
      "initialize.outputs.next_state_low",
    ],
    [
      initialize.outputs.next_state_high,
      expected.state_dtype,
      "initialize.outputs.next_state_high",
    ],
    [recurrent.inputs.frame, "float32", "recurrent.inputs.frame"],
    [recurrent.inputs.motion, "float32", "recurrent.inputs.motion"],
    [recurrent.inputs.residual, "float32", "recurrent.inputs.residual"],
    [recurrent.inputs.state_low, expected.state_dtype, "recurrent.inputs.state_low"],
    [recurrent.inputs.state_high, expected.state_dtype, "recurrent.inputs.state_high"],
    [recurrent.outputs.output, "float32", "recurrent.outputs.output"],
    [
      recurrent.outputs.next_state_low,
      expected.state_dtype,
      "recurrent.outputs.next_state_low",
    ],
    [
      recurrent.outputs.next_state_high,
      expected.state_dtype,
      "recurrent.outputs.next_state_high",
    ],
  ];
  for (const [descriptor, dtype, at] of checks) {
    if (descriptor?.dtype !== dtype) {
      receiptError(`runtime_contract.manifest_v2_template.${at}.dtype must be ${dtype}`);
    }
  }

  for (const [descriptors, role, at] of [
    [initialize.outputs, "state-out", "initialize.outputs"],
    [recurrent.inputs, "state-in", "recurrent.inputs"],
    [recurrent.outputs, "state-out", "recurrent.outputs"],
  ]) {
    const states = Object.values(descriptors)
      .filter((descriptor) => descriptor.role === role);
    if (states.length !== 2 ||
        !sameStringArray(states.map(({ state }) => state).sort(), ["high", "low"]) ||
        states.some(({ channels, dtype }) =>
          channels !== 64 || dtype !== expected.state_dtype)) {
      receiptError(
        `runtime_contract.manifest_v2_template.${at} must expose exactly ` +
        `two 64-channel ${expected.state_dtype} recurrent states`,
      );
    }
  }
}

function validateTemporalTilingProfile(contract, precision, outputScale) {
  requireExactValue(
    contract.tiling,
    temporalTilingProfile(precision, outputScale),
    "runtime_contract.manifest_v2_template.tiling",
  );
}

function expectedGraphShapes(role, outputScale = NATIVE_OUTPUT_SCALE) {
  const spatial = ["height", "width"];
  const inputs = role === "initializer"
    ? { frame: [1, 3, ...spatial] }
    : {
      frame: [1, 3, ...spatial],
      motion: [1, 2, ...spatial],
      residual: [1, 1, ...spatial],
      state_low: [1, 64, ...spatial],
      state_high: [1, 64, ...spatial],
    };
  return {
    inputs,
    outputs: {
      output: [
        1,
        3,
        `output_height_x${outputScale}`,
        `output_width_x${outputScale}`,
      ],
      next_state_low: [1, 64, ...spatial],
      next_state_high: [1, 64, ...spatial],
    },
  };
}

function validateGraphEvidenceV3(
  graph,
  role,
  precision,
  captureFixture,
  outputScale,
) {
  const at = `graphs.${role}`;
  const profile = precisionContract(precision);
  if (graph.precision_profile !== precision) {
    receiptError(`${at}.precision_profile must be ${precision}`);
  }
  requireExactValue(graph.capture_fixture, captureFixture, `${at}.capture_fixture`);
  const shapes = expectedGraphShapes(role, outputScale);
  requireExactValue(graph.inputs, shapes.inputs, `${at}.inputs`);
  requireExactValue(graph.outputs, shapes.outputs, `${at}.outputs`);

  const expectedPublicInputs = role === "initializer"
    ? { frame: profile.public_inputs.frame }
    : profile.public_inputs;
  requireExactValue(
    graph.public_dtypes,
    {
      inputs: expectedPublicInputs,
      outputs: profile.public_outputs,
    },
    `${at}.public_dtypes`,
  );

  const expectedGridSamples = GRAPH_GRID_SAMPLE_NODES[role];
  if (graph.grid_sample_nodes !== expectedGridSamples) {
    receiptError(`${at}.grid_sample_nodes must be ${expectedGridSamples}`);
  }
  requireExactValue(
    graph.precision_islands,
    {
      coordinate_dtype: "float32",
      grid_sample_dtype: "float32",
      grid_sample_nodes: expectedGridSamples,
    },
    `${at}.precision_islands`,
  );
  requireExactValue(
    graph.logical_memory,
    graphLogicalMemoryProfile(role, precision, outputScale),
    `${at}.logical_memory`,
  );

  if (!Array.isArray(graph.operators) ||
      graph.operators.some((operator) => typeof operator !== "string")) {
    receiptError(`${at}.operators must be a string array`);
  }
  const hasGridSample = graph.operators.includes("GridSample");
  if (hasGridSample !== (expectedGridSamples > 0)) {
    receiptError(`${at}.operators is inconsistent with its GridSample count`);
  }
  requirePositiveInteger(graph.nodes, `${at}.nodes`);

  const initializerDtypes = requireRecord(
    graph.initializer_dtypes,
    `${at}.initializer_dtypes`,
  );
  for (const [dtype, count] of Object.entries(initializerDtypes)) {
    requirePositiveInteger(count, `${at}.initializer_dtypes.${dtype}`);
  }
  const initializerCount = requirePositiveInteger(
    initializerDtypes[profile.weight_dtype],
    `${at}.initializer_dtypes.${profile.weight_dtype}`,
  );
  const otherFloatDtype = profile.weight_dtype === "float16" ? "float32" : "float16";
  if (Object.hasOwn(initializerDtypes, otherFloatDtype)) {
    receiptError(`${at}.initializer_dtypes contains ${otherFloatDtype} weights`);
  }
  requireExactValue(
    graph.weight_derivation,
    {
      source_dtype: "float32",
      target_dtype: profile.weight_dtype,
      method: precision === MIXED_FP16_PRECISION
        ? "ieee-754-binary16-round-to-nearest"
        : "identity",
      initializer_count: initializerCount,
    },
    `${at}.weight_derivation`,
  );
  if (outputScale === DERIVED_OUTPUT_SCALE) {
    requireExactValue(
      graph.output_derivation,
      DERIVED_X2_CONTRACT,
      `${at}.output_derivation`,
    );
  } else if (graph.output_derivation != null) {
    receiptError(`${at}.output_derivation is not valid for native 4x output`);
  }

  const castTargets = requireRecord(graph.cast_targets, `${at}.cast_targets`);
  for (const [dtype, count] of Object.entries(castTargets)) {
    requirePositiveInteger(count, `${at}.cast_targets.${dtype}`);
  }
  if (precision === MIXED_FP16_PRECISION) {
    if (!graph.operators.includes("Cast") ||
        (castTargets.float16 ?? 0) < Math.max(1, expectedGridSamples) ||
        (castTargets.float32 ?? 0) < 1) {
      receiptError(
        `${at} must record explicit float16/float32 mixed-precision casts`,
      );
    }
  } else if (Object.hasOwn(castTargets, "float16")) {
    receiptError(`${at}.cast_targets must not introduce float16 in the FP32 profile`);
  }
}

export function validateCdaExportReceipt(receipt) {
  requireRecord(receipt, "receipt");
  const legacy = receipt.format === LEGACY_RECEIPT_FORMAT;
  if (!legacy && receipt.format !== RECEIPT_FORMAT) {
    receiptError(
      `format must be ${LEGACY_RECEIPT_FORMAT} or ${RECEIPT_FORMAT}`,
    );
  }
  if (receipt.tool !== RECEIPT_TOOL) receiptError("tool identity is not recognized");
  if (receipt.opset !== RECEIPT_OPSET) {
    receiptError(`opset must be ${RECEIPT_OPSET}`);
  }
  const precision = legacy ? FP32_PRECISION : validatePrecision(receipt).profile;
  let outputScale = NATIVE_OUTPUT_SCALE;
  if (receipt.output_derivation != null) {
    if (legacy) receiptError("format 2 receipts cannot derive a 2x output head");
    requireExactValue(
      receipt.output_derivation,
      DERIVED_X2_CONTRACT,
      "output_derivation",
    );
    outputScale = DERIVED_OUTPUT_SCALE;
  }
  if (legacy && receipt.precision != null) {
    const legacyPrecision = requireRecord(receipt.precision, "precision");
    if (legacyPrecision.profile !== FP32_PRECISION) {
      receiptError("format 2 receipts are legacy FP32 exports");
    }
  }

  const distribution = requireRecord(receipt.distribution, "distribution");
  const expectedDistribution = {
    architecture_license_status: "not-established",
    checkpoint_license_status: "not-established",
    checkpoint_redistribution_clearance: false,
    generated_assets: "experimental-local-only",
    shipping_catalog: false,
  };
  if (!legacy) {
    requireExactValue(distribution, expectedDistribution, "distribution");
  } else if (distribution.architecture_license_status !== "not-established" ||
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

  validateInputIdentity(receipt, { canonicalOnly: !legacy });
  if (legacy) {
    validateLegacyParityEvidence(receipt, captureFixture);
  } else {
    validateParityEvidenceV3(receipt, captureFixture, precision);
  }

  const runtime = requireRecord(receipt.runtime_contract, "runtime_contract");
  if (runtime.prior_provider !== PRIOR_PROVIDER ||
      !sameStringArray(runtime.motion_component_order, ["x", "y"]) ||
      runtime.motion_units !== "low-resolution-pixels" ||
      runtime.catalog_compatible_at_graph_shape_level !== true ||
      runtime.shipping_catalog !== false) {
    receiptError("runtime_contract is not a compatible non-shipping CDA contract");
  }
  if ((!legacy && runtime.precision_profile !== precision) ||
      (legacy && runtime.precision_profile != null &&
       runtime.precision_profile !== FP32_PRECISION)) {
    receiptError(`runtime_contract.precision_profile must be ${precision}`);
  }
  const contract = requireRecord(
    runtime.manifest_v2_template,
    "runtime_contract.manifest_v2_template",
  );
  if (!legacy) validateTemporalTilingProfile(contract, precision, outputScale);
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
  validateManifestPrecision(normalizedEntry.contract, precision);

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
    if (!legacy) {
      validateGraphEvidenceV3(
        graph,
        role,
        precision,
        captureFixture,
        outputScale,
      );
    } else if (graph.precision_profile != null &&
               graph.precision_profile !== FP32_PRECISION) {
      receiptError(`graphs.${role}.precision_profile must be float32 for format 2`);
    }
    verifiedGraphs[role] = Object.freeze({
      file: filename,
      bytes: requirePositiveInteger(graph.bytes, `graphs.${role}.bytes`),
      sha256: requireSha256(graph.sha256, `graphs.${role}.sha256`),
    });
  }

  return Object.freeze({
    precision,
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
    precision: validated.precision,
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
      const target = join(
        extensionRoot,
        "model/neural",
        STAGED_PROBE_GRAPH_FILES[role],
      );
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
      precision: verified.precision,
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
    console.log(`Verified precision profile: ${staged.precision}`);
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
