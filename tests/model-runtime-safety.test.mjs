import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { ArtCnnModel } from "../fsrcnnx-artcnn-runtime.js";
import {
  allocateModelChain,
  DEFAULT_MODEL_WORKING_SET_BYTES,
  preflightModelDimensions,
  preflightModelChain,
  splitModelEntries,
  validateModelBundle,
} from "../fsrcnnx-model-bundle.js";
import { FsrcnnxModel } from "../fsrcnnx-runtime.js";
import { GENERATED_MODEL_CATALOG } from "../fsrcnnx-model-catalog.js";

globalThis.GPUTextureUsage = {
  STORAGE_BINDING: 1,
  TEXTURE_BINDING: 2,
  COPY_SRC: 4,
};

function loadBundle(name, kind) {
  const suffix = kind === "artcnn" ? "artcnn" : "passes";
  return {
    name,
    kind,
    manifest: JSON.parse(readFileSync(new URL(`../model/${name}.${suffix}.json`, import.meta.url), "utf8")),
    wgsl: readFileSync(new URL(`../model/${name}.${kind === "artcnn" ? "artcnn.wgsl" : "wgsl"}`, import.meta.url), "utf8"),
  };
}

const shippedBundles = GENERATED_MODEL_CATALOG.map(({ name, kind }) => loadBundle(name, kind));

function passSource(index, inputCount) {
  const bindings = [];
  for (let binding = 0; binding < inputCount; binding++) {
    bindings.push(`@group(0) @binding(${binding}) var input${binding}: texture_2d<f32>;`);
  }
  bindings.push(`@group(0) @binding(${inputCount}) var output: texture_storage_2d<rgba16float, write>;`);
  return `//==== ENTRY pass${index} : fixture ====\n${bindings.join("\n")}\n` +
    "@compute @workgroup_size(8,8)\nfn main() {}\n";
}

function fsrcnnxFixture() {
  const manifest = {
    name: "fixture_x2",
    whenThreshold: 1.3,
    passes: [
      { index: 0, desc: "conv", binds: ["LUMA"], save: "A", components: 1,
        widthMul: 1, heightMul: 1, kind: "conv" },
      { index: 1, desc: "shuffle", binds: ["A"], save: null, components: 1,
        widthMul: 2, heightMul: 2, kind: "shuffle" },
    ],
  };
  return { manifest, wgsl: passSource(0, 1) + passSource(1, 1) };
}

function artcnnFixture() {
  const manifest = {
    name: "ArtCNN_fixture",
    scale: 2,
    whenThreshold: 1.3,
    passes: [
      { index: 0, desc: "conv", binds: ["LUMA"], save: "NATIVE", widthMul: 1,
        heightMul: 1, kind: "conv", relu: false, numResults: 1, skipSum: false },
      { index: 1, desc: "d2s", binds: ["NATIVE"], save: null, widthMul: 2,
        heightMul: 2, kind: "d2s", relu: false, numResults: 0, skipSum: false },
    ],
  };
  return { manifest, wgsl: passSource(0, 1) + passSource(1, 1) };
}

function fakeTexture(width, height, owner) {
  return {
    width,
    height,
    destroyed: 0,
    createView() {
      owner.counts.views++;
      if (owner.faults.viewAt === owner.counts.views) throw new Error("injected view failure");
      return { texture: this };
    },
    destroy() { this.destroyed++; },
  };
}

function fakeDevice() {
  const device = {
    limits: {
      maxTextureDimension2D: 32768,
      maxBindingsPerBindGroup: 1000,
      maxSampledTexturesPerShaderStage: 16,
      maxStorageTexturesPerShaderStage: 4,
    },
    faults: { textureAt: -1, viewAt: -1, bindGroupAt: -1, pipelineAt: -1 },
    counts: { textures: 0, views: 0, bindGroups: 0, pipelines: 0 },
    textures: [],
    createShaderModule(descriptor) { return descriptor; },
    createComputePipeline() {
      this.counts.pipelines++;
      if (this.faults.pipelineAt === this.counts.pipelines) throw new Error("injected pipeline failure");
      return { getBindGroupLayout: () => ({}) };
    },
    createTexture(descriptor) {
      this.counts.textures++;
      if (this.faults.textureAt === this.counts.textures) throw new Error("injected texture failure");
      const size = descriptor.size;
      const texture = fakeTexture(size.width, size.height, this);
      this.textures.push(texture);
      return texture;
    },
    createBindGroup(descriptor) {
      this.counts.bindGroups++;
      if (this.faults.bindGroupAt === this.counts.bindGroups) throw new Error("injected bind-group failure");
      return descriptor;
    },
  };
  return device;
}

function luma(width, height, device) {
  return fakeTexture(width, height, device);
}

test("all shipped model bundles pass the shared strict validator", () => {
  for (const bundle of shippedBundles) {
    const result = validateModelBundle(bundle.kind, bundle.manifest, bundle.wgsl, {
      expectedName: bundle.name,
    });
    assert.equal(result.entries.size, bundle.manifest.passes.length, bundle.name);
  }
});

test("working-set estimates describe the runtime's physical texture allocation", () => {
  const expectedBytesPerPixel = new Map([
    ["FSRCNNX_x2_16-0-4-1", 168],
    ["FSRCNNX_x2_56-16-4-1", 392],
    ["FSRCNNX_x3_16-0-4-1", 224],
    ["FSRCNNX_x4_16-0-4-1", 288],
    ["ArtCNN_C4F32", 424],
    ["ArtCNN_C4F32_DN", 424],
    ["ArtCNN_C4F32_DS", 424],
  ]);
  for (const bundle of shippedBundles) {
    const plan = preflightModelDimensions(bundle.kind, bundle.manifest, 1920, 1080, {
      deviceLimits: { maxTextureDimension2D: 32768 },
      maxWorkingSetBytes: Number.MAX_SAFE_INTEGER,
    });
    assert.equal(plan.workingSetBytes, expectedBytesPerPixel.get(bundle.name) * 1920 * 1080,
      bundle.name);
  }
});

test("ArtCNN packed-feature boundary is checked before allocation", () => {
  const bundle = shippedBundles.find(({ name }) => name === "ArtCNN_C4F32");
  const options = {
    deviceLimits: { maxTextureDimension2D: 8192 },
    maxWorkingSetBytes: Number.MAX_SAFE_INTEGER,
  };
  assert.equal(preflightModelDimensions("artcnn", bundle.manifest, 2048, 1, options)
    .resources.find(({ name }) => name === "conv2d").width, 8192);
  assert.throws(
    () => preflightModelDimensions("artcnn", bundle.manifest, 2049, 1, options),
    (error) => error.code === "MODEL_DIMENSION_LIMIT" && /conv2d texture 8196x2/.test(error.message),
  );
  assert.throws(
    () => preflightModelDimensions("artcnn", bundle.manifest, 1920, 1080),
    (error) => error.code === "MODEL_WORKING_SET_LIMIT" &&
      error.message.includes(String(DEFAULT_MODEL_WORKING_SET_BYTES)),
  );
});

test("dimension preflight rejects invalid and unsafe arithmetic", () => {
  const { manifest } = fsrcnnxFixture();
  assert.throws(() => preflightModelDimensions("fsrcnnx", manifest, 0, 10), /positive safe integer/);
  assert.throws(() => preflightModelDimensions("fsrcnnx", manifest, Number.MAX_SAFE_INTEGER, 1, {
    deviceLimits: { maxTextureDimension2D: Number.MAX_SAFE_INTEGER },
    maxWorkingSetBytes: Number.MAX_SAFE_INTEGER,
  }), /safe integer range/);
});

test("malformed manifests and WGSL entry maps are rejected deterministically", () => {
  const cases = [
    ["non-contiguous manifest index", ({ manifest }) => { manifest.passes[1].index = 4; }, /non-contiguous index/],
    ["unsupported kind", ({ manifest }) => { manifest.passes[0].kind = "magic"; }, /unsupported kind/],
    ["future bind", ({ manifest }) => { manifest.passes[0].binds = ["FUTURE"]; }, /unavailable resource/],
    ["reserved save", ({ manifest }) => { manifest.passes[0].save = "LUMA"; }, /reserved resource/],
    ["synthetic output save", ({ manifest }) => { manifest.passes[0].save = "OUTPUT"; }, /reserved resource OUTPUT/],
    ["fractional multiplier", ({ manifest }) => { manifest.passes[0].widthMul = 1.5; }, /positive safe integer/],
    ["non-square shuffle", ({ manifest }) => { manifest.passes[1].heightMul = 3; }, /scale must be square/],
    ["binding declaration mismatch", (bundle) => { bundle.wgsl = bundle.wgsl.replace("@binding(1)", "@binding(4)"); }, /bindings do not match/],
    ["nonzero bind group", (bundle) => {
      bundle.wgsl = bundle.wgsl.replace(
        "@compute",
        "@group(1) @binding(0) var extra: sampler;\n@compute",
      );
    }, /only declare bind group 0/],
    ["non-texture input binding", (bundle) => {
      bundle.wgsl = bundle.wgsl.replace("var input0: texture_2d<f32>;", "var input0: sampler;");
    }, /input binding 0 must be a sampled texture_2d<f32>/],
    ["non-storage output binding", (bundle) => {
      bundle.wgsl = bundle.wgsl.replace(
        "texture_storage_2d<rgba16float, write>",
        "texture_2d<f32>",
      );
    }, /output binding 1 must be a writable rgba16float storage texture/],
    ["missing marker", (bundle) => { bundle.wgsl = "@compute @workgroup_size(8,8) fn main() {}"; }, /missing \/\/==== ENTRY markers/],
    ["duplicate marker", (bundle) => { bundle.wgsl = passSource(0, 1) + passSource(0, 1); }, /duplicate WGSL entry/],
    ["out-of-order markers", (bundle) => { bundle.wgsl = passSource(1, 1) + passSource(0, 1); }, /ordered and contiguous/],
    ["missing exact entry", (bundle) => { bundle.wgsl = passSource(0, 1); }, /passes but WGSL has/],
  ];
  for (const [label, mutate, pattern] of cases) {
    const original = fsrcnnxFixture();
    const bundle = { manifest: structuredClone(original.manifest), wgsl: original.wgsl };
    mutate(bundle);
    assert.throws(() => validateModelBundle("fsrcnnx", bundle.manifest, bundle.wgsl), pattern, label);
  }

  const selfAlias = fsrcnnxFixture();
  selfAlias.manifest.passes.splice(1, 0, {
    index: 1, desc: "alias", binds: ["A"], save: "A", components: 1,
    widthMul: 1, heightMul: 1, kind: "conv",
  });
  selfAlias.manifest.passes[2].index = 2;
  selfAlias.wgsl = passSource(0, 1) + passSource(1, 1) + passSource(2, 1);
  assert.throws(() => validateModelBundle("fsrcnnx", selfAlias.manifest, selfAlias.wgsl), /reads and writes A/);

  const bindingLimit = fsrcnnxFixture();
  bindingLimit.manifest.passes[1].binds = ["A", "LUMA"];
  bindingLimit.wgsl = passSource(0, 1) + passSource(1, 2);
  assert.throws(() => validateModelBundle("fsrcnnx", bindingLimit.manifest, bindingLimit.wgsl, {
    deviceLimits: { maxSampledTexturesPerShaderStage: 1 },
  }), /device limit is 1/);

  const art = artcnnFixture();
  art.manifest.scale = 3;
  assert.throws(() => validateModelBundle("artcnn", art.manifest, art.wgsl), /exactly 2x/);
});

for (const spec of [
  { label: "FSRCNNX", bundle: shippedBundles[0], Model: FsrcnnxModel, lumaKey: "lumaTexture" },
  { label: "ArtCNN", bundle: shippedBundles[4], Model: ArtCnnModel, lumaKey: "lumaTex" },
]) {
  test(`${spec.label} publishes pipelines only after every pass builds`, () => {
    const device = fakeDevice();
    const model = new spec.Model(device, spec.bundle.manifest, spec.bundle.wgsl,
      { maxWorkingSetBytes: Number.MAX_SAFE_INTEGER });
    device.faults.pipelineAt = 2;
    assert.throws(() => model.buildPipelines(), /injected pipeline failure/);
    assert.equal(model.pipelines.length, 0);
    device.faults.pipelineAt = -1;
    model.buildPipelines();
    assert.equal(model.pipelines.length, spec.bundle.manifest.passes.length);
  });

  test(`${spec.label} resize failure preserves the complete old allocation`, () => {
    const device = fakeDevice();
    const model = new spec.Model(device, spec.bundle.manifest, spec.bundle.wgsl,
      { maxWorkingSetBytes: Number.MAX_SAFE_INTEGER });
    const oldLuma = luma(16, 12, device);
    model.allocate(16, 12, oldLuma);
    const oldOutput = model.outputTexture;
    const oldTextures = [...model.textures.values()];
    const before = device.textures.length;
    device.faults.textureAt = device.counts.textures + 2;

    assert.throws(() => model.allocate(20, 14, luma(20, 14, device)), /injected texture failure/);
    assert.equal(model.outputTexture, oldOutput);
    assert.equal(model[spec.lumaKey], oldLuma);
    assert.equal(model.lumaW, 16);
    assert.equal(model.lumaH, 12);
    assert.equal(oldOutput.destroyed, 0);
    assert.ok(oldTextures.every((texture) => texture.destroyed === 0));
    assert.ok(device.textures.slice(before).every((texture) => texture.destroyed === 1));

    device.faults.textureAt = -1;
    model.allocate(20, 14, luma(20, 14, device));
    assert.equal(model.lumaW, 20);
    assert.equal(oldOutput.destroyed, 1);
    assert.ok(oldTextures.every((texture) => texture.destroyed === 1));
  });

  test(`${spec.label} same-size rebind failure retains old input and bind groups`, () => {
    const device = fakeDevice();
    const model = new spec.Model(device, spec.bundle.manifest, spec.bundle.wgsl,
      { maxWorkingSetBytes: Number.MAX_SAFE_INTEGER });
    const oldLuma = luma(12, 10, device);
    model.allocate(12, 10, oldLuma);
    const oldBindGroups = model.bindGroups;
    const owned = [model.outputTexture, ...model.textures.values()];
    device.faults.bindGroupAt = device.counts.bindGroups + 2;
    const nextLuma = luma(12, 10, device);

    assert.throws(() => model.allocate(12, 10, nextLuma), /injected bind-group failure/);
    assert.equal(model[spec.lumaKey], oldLuma);
    assert.equal(model.bindGroups, oldBindGroups);
    assert.ok(owned.every((texture) => texture.destroyed === 0));

    device.faults.bindGroupAt = -1;
    model.allocate(12, 10, nextLuma);
    assert.equal(model[spec.lumaKey], nextLuma);
    assert.notEqual(model.bindGroups, oldBindGroups);
    assert.ok(owned.every((texture) => texture.destroyed === 0));
  });

  test(`${spec.label} view failure destroys candidates without disturbing the old generation`, () => {
    const device = fakeDevice();
    const model = new spec.Model(device, spec.bundle.manifest, spec.bundle.wgsl,
      { maxWorkingSetBytes: Number.MAX_SAFE_INTEGER });
    const oldLuma = luma(10, 8, device);
    model.allocate(10, 8, oldLuma);
    const oldOutput = model.outputTexture;
    const before = device.textures.length;
    device.faults.viewAt = device.counts.views + 2;

    assert.throws(() => model.allocate(14, 9, luma(14, 9, device)), /injected view failure/);
    assert.equal(model.outputTexture, oldOutput);
    assert.equal(model[spec.lumaKey], oldLuma);
    assert.equal(oldOutput.destroyed, 0);
    assert.ok(device.textures.slice(before).every((texture) => texture.destroyed === 1));

    device.faults.viewAt = -1;
    model.allocate(14, 9, luma(14, 9, device));
    assert.equal(model.lumaW, 14);
    assert.equal(oldOutput.destroyed, 1);
  });

  test(`${spec.label} destroy is idempotent, complete, and terminal`, () => {
    const device = fakeDevice();
    const model = new spec.Model(device, spec.bundle.manifest, spec.bundle.wgsl,
      { maxWorkingSetBytes: Number.MAX_SAFE_INTEGER });
    const input = luma(8, 6, device);
    model.allocate(8, 6, input);
    const owned = [model.outputTexture, ...model.textures.values()];
    model.destroy();
    model.destroy();
    assert.ok(owned.every((texture) => texture.destroyed === 1));
    assert.equal(input.destroyed, 0);
    assert.equal(model.outputTexture, null);
    assert.equal(model.bindGroups, null);
    assert.equal(model.lumaW, 0);
    assert.throws(() => model.allocate(8, 6, input), /has been destroyed/);
    assert.throws(() => model.preflight(8, 6), /has been destroyed/);
    assert.throws(() => model.run({}, input), /has been destroyed/);
  });
}

test("splitModelEntries rejects malformed marker syntax rather than slicing from -1", () => {
  assert.throws(() => splitModelEntries("//==== ENTRY nope\n@compute fn main() {}"), /malformed WGSL entry marker/);
});

test("chained allocation clears every stage after a later failure and retries coherently", () => {
  const calls = [];
  const makeStage = (index) => ({
    maxWorkingSetBytes: 1000,
    outputTexture: null,
    fail: index === 1,
    preflight(width, height) {
      return { workingSetBytes: 100, outputWidth: width * 2, outputHeight: height * 2 };
    },
    allocate(width, height, input) {
      calls.push(["allocate", index, width, height, input?.stage ?? "input"]);
      if (this.fail) throw new Error("injected later-stage failure");
      this.outputTexture = { stage: index };
    },
    resetAllocation() {
      calls.push(["reset", index]);
      this.outputTexture = null;
    },
  });
  const stages = [makeStage(0), makeStage(1), makeStage(2)];
  assert.equal(preflightModelChain(stages, 8, 6, "fixture").workingSetBytes, 300);
  assert.throws(() => allocateModelChain(stages, 8, 6, { stage: "input" }, "fixture"),
    /injected later-stage failure/);
  assert.ok(stages.every((stage) => stage.outputTexture === null));
  assert.deepEqual(calls.slice(-3), [["reset", 0], ["reset", 1], ["reset", 2]]);

  stages[1].fail = false;
  const plan = allocateModelChain(stages, 8, 6, { stage: "input" }, "fixture");
  assert.deepEqual([plan.outputWidth, plan.outputHeight], [64, 48]);
  assert.ok(stages.every((stage) => stage.outputTexture));

  stages[0].maxWorkingSetBytes = 250;
  assert.throws(() => preflightModelChain(stages, 8, 6, "fixture"),
    (error) => error.code === "MODEL_WORKING_SET_LIMIT");
});
