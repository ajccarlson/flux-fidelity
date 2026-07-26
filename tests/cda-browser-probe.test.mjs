import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  access,
  mkdtemp,
  mkdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";

import {
  CDA_BROWSER_PROBE_MODEL_KEY,
  makeCdaProbeManifest,
  stageCdaBrowserProbe,
  validateCdaExportReceipt,
  verifyCdaExportDirectory,
} from "../tools/cda-vsr/browser_probe.mjs";

const root = resolve(import.meta.dirname, "..");
const referenceSource =
  "0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b";
const referenceCheckpoint =
  "afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e";
const paritySeed = 20260726;
const fp32Precision = "float32";
const mixedPrecision = "mixed-fp16";
const parityLimits = {
  [fp32Precision]: {
    output: { max_abs: 2e-4, max_mean: 2e-5 },
    state: { max_abs: 2e-4, max_mean: 2e-5 },
  },
  [mixedPrecision]: {
    output: { max_abs: 5e-3, max_mean: 5e-4 },
    state: { max_abs: 2e-2, max_mean: 2e-3 },
  },
};

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function temporalContract(stateDtype = "float32") {
  const rgb = { role: "rgb", dtype: "float32", channels: 3 };
  const stateOut = (state) => ({
    role: "state-out",
    state,
    dtype: stateDtype,
    channels: 64,
  });
  return {
    version: 2,
    mode: "temporal",
    resetGraph: "initialize",
    recurrentGraph: "recurrent",
    graphs: {
      initialize: {
        file: "cda-vsr-initializer.onnx",
        inputs: { frame: { ...rgb } },
        outputs: {
          output: { ...rgb },
          next_state_low: stateOut("low"),
          next_state_high: stateOut("high"),
        },
      },
      recurrent: {
        file: "cda-vsr-recurrent.onnx",
        inputs: {
          frame: { ...rgb },
          motion: {
            role: "motion",
            dtype: "float32",
            channels: 2,
            provider: "decoded-cda-v1",
          },
          residual: {
            role: "residual",
            dtype: "float32",
            channels: 1,
            provider: "decoded-cda-v1",
          },
          state_low: {
            role: "state-in",
            state: "low",
            reset: "required",
            dtype: stateDtype,
            channels: 64,
          },
          state_high: {
            role: "state-in",
            state: "high",
            reset: "required",
            dtype: stateDtype,
            channels: 64,
          },
        },
        outputs: {
          output: { ...rgb },
          next_state_low: stateOut("low"),
          next_state_high: stateOut("high"),
        },
      },
    },
  };
}

function precisionContract(profile) {
  const computeDtype = profile === mixedPrecision ? "float16" : "float32";
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

function graphShapes(role) {
  const spatial = ["height", "width"];
  return {
    inputs: role === "initializer"
      ? { frame: [1, 3, ...spatial] }
      : {
        frame: [1, 3, ...spatial],
        motion: [1, 2, ...spatial],
        residual: [1, 1, ...spatial],
        state_low: [1, 64, ...spatial],
        state_high: [1, 64, ...spatial],
      },
    outputs: {
      output: [1, 3, "output_height_x4", "output_width_x4"],
      next_state_low: [1, 64, ...spatial],
      next_state_high: [1, 64, ...spatial],
    },
  };
}

function paritySequence(
  shape,
  {
    sequenceIndex,
    motionFixture,
    limits,
    frames = 25,
  },
) {
  const records = [];
  for (let frame = 0; frame < frames; frame++) {
    const maximum = (sequenceIndex + 1) * (frame + 1) * 4e-7;
    const metrics = {
      mean_abs: maximum / 2,
      p99_9_abs: maximum * 0.75,
      max_abs: maximum,
    };
    records.push({
      frame,
      role: frame === 0 ? "initializer" : "recurrent",
      tensors: {
        output: { ...metrics },
        next_state_low: { ...metrics },
        next_state_high: { ...metrics },
      },
    });
  }
  const worst = {
    worst_max_abs: Math.max(
      ...records.flatMap(({ tensors }) =>
        Object.values(tensors).map(({ max_abs: value }) => value)),
    ),
    worst_mean_abs: Math.max(
      ...records.flatMap(({ tensors }) =>
        Object.values(tensors).map(({ mean_abs: value }) => value)),
    ),
    worst_p99_9_abs: Math.max(
      ...records.flatMap(({ tensors }) =>
        Object.values(tensors).map(({ p99_9_abs: value }) => value)),
    ),
  };
  const maxAbs = Math.max(limits.output.max_abs, limits.state.max_abs);
  const maxMean = Math.max(limits.output.max_mean, limits.state.max_mean);
  return {
    ...shape,
    frames,
    seed: paritySeed,
    motion_fixture: motionFixture,
    max_abs_limit: maxAbs,
    max_mean_limit: maxMean,
    tensor_limits: structuredClone(limits),
    worst_by_tensor_class: {
      output: { ...worst },
      state: { ...worst },
    },
    ...worst,
    records,
  };
}

function receiptV3For(
  initializer,
  recurrent,
  { precision = mixedPrecision } = {},
) {
  const profile = precisionContract(precision);
  const stateDtype = profile.state_dtype;
  const limits = structuredClone(parityLimits[precision]);
  const capture = { height: 8, width: 8 };
  const shapes = [capture, { height: 11, width: 13 }];
  const shapeResults = shapes.map((shape, index) =>
    paritySequence(shape, {
      sequenceIndex: index,
      motionFixture: "decoded-integer",
      limits,
    }));
  const fractional = paritySequence(capture, {
    sequenceIndex: shapeResults.length,
    motionFixture: "fractional-stress",
    limits,
  });
  const sequences = [...shapeResults, fractional];
  const maxAbs = Math.max(limits.output.max_abs, limits.state.max_abs);
  const maxMean = Math.max(limits.output.max_mean, limits.state.max_mean);
  const contract = temporalContract(stateDtype);

  const graphFor = (role, bytes) => {
    const mixed = precision === mixedPrecision;
    const gridSampleNodes = role === "recurrent" ? 5 : 0;
    const targetDtype = profile.weight_dtype;
    const initializerCount = role === "recurrent" ? 20 : 10;
    const shapesForRole = graphShapes(role);
    return {
      file: role === "initializer"
        ? "cda-vsr-initializer.onnx"
        : "cda-vsr-recurrent.onnx",
      bytes: bytes.length,
      sha256: sha256(bytes),
      precision_profile: precision,
      spatial_shape: "dynamic",
      capture_fixture: { ...capture },
      ...shapesForRole,
      operators: mixed
        ? ["Cast", "Conv", ...(gridSampleNodes ? ["GridSample"] : [])]
        : ["Conv", ...(gridSampleNodes ? ["GridSample"] : [])],
      grid_sample_nodes: gridSampleNodes,
      public_dtypes: {
        inputs: role === "initializer"
          ? { frame: profile.public_inputs.frame }
          : structuredClone(profile.public_inputs),
        outputs: structuredClone(profile.public_outputs),
      },
      initializer_dtypes: { [targetDtype]: initializerCount },
      cast_targets: mixed
        ? {
          float16: Math.max(1, gridSampleNodes),
          float32: Math.max(1, gridSampleNodes),
        }
        : {},
      precision_islands: {
        coordinate_dtype: "float32",
        grid_sample_dtype: "float32",
        grid_sample_nodes: gridSampleNodes,
      },
      weight_derivation: {
        source_dtype: "float32",
        target_dtype: targetDtype,
        method: mixed ? "ieee-754-binary16-round-to-nearest" : "identity",
        initializer_count: initializerCount,
      },
      nodes: role === "recurrent" ? 511 : 238,
    };
  };

  return {
    format: 3,
    tool: "FSRCNNX-EXT CDA-VSR conversion toolkit",
    opset: 17,
    precision: profile,
    distribution: {
      architecture_license_status: "not-established",
      checkpoint_license_status: "not-established",
      checkpoint_redistribution_clearance: false,
      generated_assets: "experimental-local-only",
      shipping_catalog: false,
    },
    spatial_shape: {
      capture_fixture: { ...capture },
      graph_shape_compatible: true,
      mode: "dynamic",
      source_resolution_ceiling: null,
    },
    inputs: {
      source: { name: "cdavsr_arch.py", bytes: 100, sha256: referenceSource },
      checkpoint: { name: "best.pth", bytes: 200, sha256: referenceCheckpoint },
      contract: { architecture: "CDAVSR" },
    },
    input_identity: {
      policy: "canonical-reference",
      reference_source_sha256: referenceSource,
      reference_checkpoint_sha256: referenceCheckpoint,
      architecture_execution: "trusted-python-code",
    },
    runtime_contract: {
      prior_provider: "decoded-cda-v1",
      motion_component_order: ["x", "y"],
      motion_units: "low-resolution-pixels",
      precision_profile: precision,
      catalog_compatible_at_graph_shape_level: true,
      shipping_catalog: false,
      manifest_v2_template: contract,
    },
    parity_policy: {
      frames: 25,
      max_abs: maxAbs,
      max_mean: maxMean,
      tensor_limits: structuredClone(limits),
      reference_precision: "float32",
      state_chains: "independent",
      motion_fixtures: ["decoded-integer", "fractional-stress"],
      skipped: false,
      dynamic_shape_runtime_validated: true,
    },
    parity: {
      spatial_shape: "dynamic",
      reference_precision: "float32",
      state_chains: "independent",
      primary_motion_fixture: "decoded-integer",
      tested_shapes: structuredClone(shapes),
      frames_per_shape: 25,
      seed: paritySeed,
      max_abs_limit: maxAbs,
      max_mean_limit: maxMean,
      tensor_limits: structuredClone(limits),
      worst_max_abs: Math.max(
        ...sequences.map(({ worst_max_abs: value }) => value),
      ),
      worst_mean_abs: Math.max(
        ...sequences.map(({ worst_mean_abs: value }) => value),
      ),
      worst_p99_9_abs: Math.max(
        ...sequences.map(({ worst_p99_9_abs: value }) => value),
      ),
      shape_results: shapeResults,
      fractional_motion_stress: fractional,
    },
    graphs: {
      initializer: graphFor("initializer", initializer),
      recurrent: graphFor("recurrent", recurrent),
    },
  };
}

function receiptFor(initializer, recurrent) {
  const contract = temporalContract();
  return {
    format: 2,
    tool: "FSRCNNX-EXT CDA-VSR conversion toolkit",
    opset: 17,
    distribution: {
      architecture_license_status: "not-established",
      checkpoint_license_status: "not-established",
      checkpoint_redistribution_clearance: false,
      generated_assets: "experimental-local-only",
      shipping_catalog: false,
    },
    spatial_shape: {
      capture_fixture: { height: 8, width: 8 },
      graph_shape_compatible: true,
      mode: "dynamic",
      source_resolution_ceiling: null,
    },
    inputs: {
      source: { name: "cdavsr_arch.py", bytes: 100, sha256: referenceSource },
      checkpoint: { name: "best.pth", bytes: 200, sha256: referenceCheckpoint },
      contract: { architecture: "CDAVSR" },
    },
    input_identity: {
      policy: "canonical-reference",
      reference_source_sha256: referenceSource,
      reference_checkpoint_sha256: referenceCheckpoint,
      architecture_execution: "trusted-python-code",
    },
    runtime_contract: {
      prior_provider: "decoded-cda-v1",
      motion_component_order: ["x", "y"],
      motion_units: "low-resolution-pixels",
      catalog_compatible_at_graph_shape_level: true,
      shipping_catalog: false,
      manifest_v2_template: contract,
    },
    parity_policy: {
      frames: 2,
      max_abs: 2e-4,
      max_mean: 2e-5,
      skipped: false,
      dynamic_shape_runtime_validated: true,
    },
    parity: {
      spatial_shape: "dynamic",
      tested_shapes: [
        { height: 8, width: 8 },
        { height: 11, width: 13 },
      ],
      frames_per_shape: 2,
      max_abs_limit: 2e-4,
      max_mean_limit: 2e-5,
      worst_max_abs: 4e-6,
      worst_mean_abs: 6e-7,
    },
    graphs: {
      initializer: {
        file: "cda-vsr-initializer.onnx",
        bytes: initializer.length,
        sha256: sha256(initializer),
        spatial_shape: "dynamic",
      },
      recurrent: {
        file: "cda-vsr-recurrent.onnx",
        bytes: recurrent.length,
        sha256: sha256(recurrent),
        spatial_shape: "dynamic",
      },
    },
  };
}

async function makeExportFixture(parent) {
  const directory = await mkdtemp(join(parent, "cda-export-"));
  const initializer = Buffer.from("initializer graph fixture");
  const recurrent = Buffer.from("recurrent graph fixture");
  const receipt = receiptFor(initializer, recurrent);
  await Promise.all([
    writeFile(join(directory, "cda-vsr-initializer.onnx"), initializer),
    writeFile(join(directory, "cda-vsr-recurrent.onnx"), recurrent),
    writeFile(
      join(directory, "cda-vsr-export.json"),
      `${JSON.stringify(receipt, null, 2)}\n`,
    ),
  ]);
  return { directory, initializer, recurrent, receipt };
}

test("CDA browser probe preserves legacy format-2 FP32 receipts", async (t) => {
  const parent = await mkdtemp(join(tmpdir(), "fsrcnnx-cda-probe-test-"));
  t.after(() => rm(parent, { recursive: true, force: true }));
  const fixture = await makeExportFixture(parent);

  const before = structuredClone(fixture.receipt);
  const validated = validateCdaExportReceipt(fixture.receipt);
  assert.deepEqual(fixture.receipt, before, "receipt validation must not mutate evidence");
  assert.equal(validated.precision, fp32Precision);
  assert.equal(validated.contract.version, 2);

  const verified = await verifyCdaExportDirectory(fixture.directory);
  assert.equal(verified.precision, fp32Precision);
  assert.equal(verified.graphs.initializer.bytes, fixture.initializer.length);
  assert.equal(verified.graphs.recurrent.bytes, fixture.recurrent.length);
  assert.match(verified.receipt.sha256, /^[0-9a-f]{64}$/);

  await writeFile(join(fixture.directory, "cda-vsr-recurrent.onnx"), "tampered");
  await assert.rejects(
    verifyCdaExportDirectory(fixture.directory),
    /recurrent graph byte length differs from receipt/,
  );
});

test("CDA browser probe accepts complete mixed-fp16 format-3 evidence", async (t) => {
  const initializer = Buffer.from("mixed initializer");
  const recurrent = Buffer.from("mixed recurrent");
  const receipt = receiptV3For(initializer, recurrent);
  const before = structuredClone(receipt);

  const validated = validateCdaExportReceipt(receipt);

  assert.deepEqual(receipt, before, "receipt validation must not mutate evidence");
  assert.equal(validated.precision, mixedPrecision);
  assert.equal(
    validated.contract.graphs.recurrent.inputs.state_low.dtype,
    "float16",
  );
  assert.equal(
    validated.contract.graphs.recurrent.outputs.output.dtype,
    "float32",
  );
  const manifest = makeCdaProbeManifest([], validated.contract);
  assert.equal(manifest[0].fp16, false);

  const parent = await mkdtemp(join(tmpdir(), "fsrcnnx-cda-mixed-probe-test-"));
  t.after(() => rm(parent, { recursive: true, force: true }));
  await Promise.all([
    writeFile(join(parent, "cda-vsr-initializer.onnx"), initializer),
    writeFile(join(parent, "cda-vsr-recurrent.onnx"), recurrent),
    writeFile(
      join(parent, "cda-vsr-export.json"),
      `${JSON.stringify(receipt, null, 2)}\n`,
    ),
  ]);
  const verified = await verifyCdaExportDirectory(parent);
  assert.equal(verified.precision, mixedPrecision);
});

test("CDA browser probe accepts strict format-3 FP32 evidence", () => {
  const receipt = receiptV3For(
    Buffer.from("fp32 initializer"),
    Buffer.from("fp32 recurrent"),
    { precision: fp32Precision },
  );
  const validated = validateCdaExportReceipt(receipt);
  assert.equal(validated.precision, fp32Precision);
  assert.equal(
    validated.contract.graphs.recurrent.inputs.state_high.dtype,
    "float32",
  );
});

test("CDA browser probe rejects incomplete or forged format-3 precision evidence", () => {
  const initializer = Buffer.from("mixed initializer");
  const recurrent = Buffer.from("mixed recurrent");
  const cases = [
    [
      "precision contract",
      (receipt) => { receipt.precision.coordinate_dtype = "float16"; },
      /precision\.coordinate_dtype must be "float32"/,
    ],
    [
      "canonical identity",
      (receipt) => {
        receipt.inputs.checkpoint.sha256 = "1".repeat(64);
        receipt.input_identity.policy = "explicit-unpinned-acknowledgement";
      },
      /requires the canonical source and checkpoint/,
    ],
    [
      "trained horizon",
      (receipt) => { receipt.parity_policy.frames = 24; },
      /must cover at least 25 frames/,
    ],
    [
      "independent state chains",
      (receipt) => { receipt.parity.state_chains = "shared"; },
      /independent FP32-reference evidence/,
    ],
    [
      "motion fixtures",
      (receipt) => {
        receipt.parity_policy.motion_fixtures = ["fractional-stress"];
      },
      /must cover decoded-integer and fractional-stress/,
    ],
    [
      "per-state limit",
      (receipt) => {
        receipt.parity_policy.tensor_limits.state.max_abs = 0.03;
      },
      /looser than the mixed-fp16 browser-probe defaults/,
    ],
    [
      "fractional fixture",
      (receipt) => {
        receipt.parity.fractional_motion_stress.motion_fixture =
          "decoded-integer";
      },
      /fixture identity is inconsistent/,
    ],
    [
      "p99.9 summary",
      (receipt) => {
        receipt.parity.shape_results[0].records[0].tensors.output.p99_9_abs =
          1e-3;
      },
      /impossible error summaries/,
    ],
    [
      "runtime state dtype",
      (receipt) => {
        receipt.runtime_contract.manifest_v2_template.graphs.recurrent
          .inputs.state_low.dtype = "float32";
      },
      /manifest_v2_template is incompatible|state_low\.dtype must be float16/,
    ],
    [
      "public dtypes",
      (receipt) => {
        receipt.graphs.recurrent.public_dtypes.inputs.motion = "float16";
      },
      /public_dtypes\.inputs\.motion must be "float32"/,
    ],
    [
      "GridSample island",
      (receipt) => {
        receipt.graphs.recurrent.precision_islands.grid_sample_dtype =
          "float16";
      },
      /precision_islands\.grid_sample_dtype must be "float32"/,
    ],
    [
      "weight derivation",
      (receipt) => {
        receipt.graphs.initializer.weight_derivation.method = "truncate";
      },
      /weight_derivation\.method must be/,
    ],
    [
      "cast boundary",
      (receipt) => {
        receipt.graphs.recurrent.cast_targets.float16 = 1;
      },
      /explicit float16\/float32 mixed-precision casts/,
    ],
  ];
  for (const [label, mutate, expected] of cases) {
    const receipt = receiptV3For(initializer, recurrent);
    mutate(receipt);
    assert.throws(() => validateCdaExportReceipt(receipt), expected, label);
  }
});

test("CDA browser probe rejects fixed, capped, skipped, shipping, and renamed exports", () => {
  const initializer = Buffer.from("initializer");
  const recurrent = Buffer.from("recurrent");
  const cases = [
    [
      "fixed",
      (receipt) => { receipt.spatial_shape.mode = "fixed"; },
      /requires dynamic/,
    ],
    [
      "ceiling",
      (receipt) => { receipt.spatial_shape.source_resolution_ceiling = { width: 320 }; },
      /source_resolution_ceiling must remain null/,
    ],
    [
      "skipped",
      (receipt) => { receipt.parity_policy.skipped = true; },
      /requires completed dynamic-shape parity/,
    ],
    [
      "shipping",
      (receipt) => { receipt.distribution.shipping_catalog = true; },
      /experimental local-only boundary/,
    ],
    [
      "renamed graph",
      (receipt) => { receipt.graphs.initializer.file = "../initializer.onnx"; },
      /graphs\.initializer\.file must be cda-vsr-initializer\.onnx/,
    ],
    [
      "legacy mixed precision",
      (receipt) => { receipt.precision = { profile: mixedPrecision }; },
      /format 2 receipts are legacy FP32 exports/,
    ],
  ];
  for (const [label, mutate, expected] of cases) {
    const receipt = receiptFor(initializer, recurrent);
    mutate(receipt);
    assert.throws(() => validateCdaExportReceipt(receipt), expected, label);
  }
});

test("CDA browser probe appends a local entry without replacing the shipping smoke model", () => {
  const manifest = [{
    key: "shipping-smoke",
    label: "Shipping smoke",
    file: "shipping-smoke.onnx",
    scale: 2,
    input: "input",
    output: "output",
    fp16: true,
  }];
  const probe = makeCdaProbeManifest(manifest, temporalContract());
  assert.equal(probe.length, 2);
  assert.equal(probe[0].key, "shipping-smoke");
  assert.equal(probe[1].key, CDA_BROWSER_PROBE_MODEL_KEY);
  assert.equal(probe[1].fp16, false);
  assert.equal(probe[1].contract.version, 2);
  assert.equal(probe[1].contract.graphs.initialize.file, "cda-vsr-initializer.onnx");
});

test("CDA browser probe stages and removes only a disposable extension copy", async (t) => {
  const parent = await mkdtemp(join(tmpdir(), "fsrcnnx-cda-stage-test-"));
  t.after(() => rm(parent, { recursive: true, force: true }));
  const fixture = await makeExportFixture(parent);
  const sourceManifestPath = join(root, "model/neural/manifest.json");
  const sourceManifestBefore = await readFile(sourceManifestPath, "utf8");

  const staged = await stageCdaBrowserProbe({
    onnxDir: fixture.directory,
    projectRoot: root,
    temporaryParent: parent,
    buildPackageImpl: async ({ distDir }) => {
      const stage = join(distDir, "fsrcnnx-ext");
      const neuralDirectory = join(stage, "model/neural");
      await mkdir(neuralDirectory, { recursive: true });
      await writeFile(join(neuralDirectory, "manifest.json"), JSON.stringify([{
        key: "shipping-smoke",
        label: "Shipping smoke",
        file: "shipping-smoke.onnx",
        scale: 2,
        input: "input",
        output: "output",
      }]));
      return { stage };
    },
  });
  const extensionRoot = staged.extensionRoot;
  try {
    assert.notEqual(dirname(dirname(extensionRoot)), root);
    assert.deepEqual(
      await readFile(
        join(extensionRoot, "model/neural/cda-vsr-initializer.onnx"),
      ),
      fixture.initializer,
    );
    const stagedManifest = JSON.parse(
      await readFile(join(extensionRoot, "model/neural/manifest.json"), "utf8"),
    );
    assert.equal(stagedManifest[0].key, "shipping-smoke");
    assert.equal(stagedManifest[1].key, CDA_BROWSER_PROBE_MODEL_KEY);
    assert.equal(await readFile(sourceManifestPath, "utf8"), sourceManifestBefore);
  } finally {
    await staged.cleanup();
  }
  await staged.cleanup();
  await assert.rejects(access(extensionRoot));
});
