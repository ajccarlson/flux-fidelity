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

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function temporalContract() {
  const rgb = { role: "rgb", dtype: "float32", channels: 3 };
  const stateOut = (state) => ({
    role: "state-out",
    state,
    dtype: "float32",
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
            dtype: "float32",
            channels: 64,
          },
          state_high: {
            role: "state-in",
            state: "high",
            reset: "required",
            dtype: "float32",
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

test("CDA browser probe verifies the dynamic receipt and exact graph bytes", async (t) => {
  const parent = await mkdtemp(join(tmpdir(), "fsrcnnx-cda-probe-test-"));
  t.after(() => rm(parent, { recursive: true, force: true }));
  const fixture = await makeExportFixture(parent);

  const before = structuredClone(fixture.receipt);
  const validated = validateCdaExportReceipt(fixture.receipt);
  assert.deepEqual(fixture.receipt, before, "receipt validation must not mutate evidence");
  assert.equal(validated.contract.version, 2);

  const verified = await verifyCdaExportDirectory(fixture.directory);
  assert.equal(verified.graphs.initializer.bytes, fixture.initializer.length);
  assert.equal(verified.graphs.recurrent.bytes, fixture.recurrent.length);
  assert.match(verified.receipt.sha256, /^[0-9a-f]{64}$/);

  await writeFile(join(fixture.directory, "cda-vsr-recurrent.onnx"), "tampered");
  await assert.rejects(
    verifyCdaExportDirectory(fixture.directory),
    /recurrent graph byte length differs from receipt/,
  );
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
