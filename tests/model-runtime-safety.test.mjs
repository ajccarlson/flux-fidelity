import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { ArtCnnModel } from "../src/core/fsrcnnx-artcnn-runtime.js";
import {
  allocateModelChain,
  DEFAULT_MODEL_WORKING_SET_BYTES,
  passSaves,
  preflightModelDimensions,
  preflightModelChain,
  splitModelEntries,
  validateModelBundle,
} from "../src/core/fsrcnnx-model-bundle.js";
import { FsrcnnxModel } from "../src/core/fsrcnnx-runtime.js";
import { GENERATED_MODEL_CATALOG } from "../src/core/fsrcnnx-model-catalog.js";

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

function passSource(index, inputCount, outputCount = 1) {
  const bindings = [];
  for (let binding = 0; binding < inputCount; binding++) {
    bindings.push(`@group(0) @binding(${binding}) var input${binding}: texture_2d<f32>;`);
  }
  // A fused pass declares one storage texture per output, immediately after its
  // inputs, which is the binding layout the runtime builds bind groups against.
  for (let offset = 0; offset < outputCount; offset++) {
    bindings.push(
      `@group(0) @binding(${inputCount + offset}) var output${offset}: texture_storage_2d<rgba16float, write>;`,
    );
  }
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
  const defaultPlan = preflightModelDimensions("artcnn", bundle.manifest, 1920, 1080);
  assert.equal(defaultPlan.workingSetBytes, 424 * 1920 * 1080);
  assert.equal(defaultPlan.maxWorkingSetBytes, DEFAULT_MODEL_WORKING_SET_BYTES);
  assert.throws(
    () => preflightModelDimensions("artcnn", bundle.manifest, 1920, 1080, {
      maxWorkingSetBytes: 512 * 1024 * 1024,
    }),
    (error) => error.code === "MODEL_WORKING_SET_LIMIT" &&
      error.message.includes(String(512 * 1024 * 1024)),
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
  {
    label: "FSRCNNX",
    bundle: shippedBundles.find(({ kind }) => kind === "fsrcnnx"),
    Model: FsrcnnxModel,
    lumaKey: "lumaTexture",
  },
  {
    label: "ArtCNN",
    bundle: shippedBundles.find(({ name }) => name === "ArtCNN_C4F32"),
    Model: ArtCnnModel,
    lumaKey: "lumaTex",
  },
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

// The working-set machinery existed and was fully implemented, but production
// never passed a budget, so DEFAULT_MODEL_WORKING_SET_BYTES (MAX_SAFE_INTEGER)
// applied and the guard only caught integer overflow. These pin the real byte
// cost of the shipped manifests so the budget in fsrcnnx-main.js can be reasoned
// about rather than guessed at, and so a transpiler change that multiplies the
// intermediate count fails here rather than at a user's GPU.
const PRODUCTION_BUDGET_BYTES = 2048 * 1024 * 1024;
const limits = { maxTextureDimension2D: 8192, maxStorageTexturesPerShaderStage: 4 };

function workingSetAt(name, width, height) {
  const bundle = loadBundle(name, "fsrcnnx");
  return preflightModelDimensions("fsrcnnx", bundle.manifest, width, height, {
    deviceLimits: limits,
  }).workingSetBytes;
}

test("every intermediate is source-sized, so cost scales with source area", () => {
  // Doubling each dimension quadruples the working set: the intermediates do not
  // shrink as the network deepens, and none of them are aliased or reused.
  const at720 = workingSetAt("FSRCNNX_x2_56-16-4-1", 1280, 720);
  const at1440 = workingSetAt("FSRCNNX_x2_56-16-4-1", 2560, 1440);
  assert.equal(at1440, at720 * 4);

  // FSRCNNX High keeps far more state than standard, which is what makes it the
  // configuration worth degrading away from when memory runs short.
  const high = workingSetAt("FSRCNNX_x2_56-16-4-1", 1920, 1080);
  const standard = workingSetAt("FSRCNNX_x2_16-0-4-1", 1920, 1080);
  assert.ok(high > standard * 2, `High ${high} should dwarf standard ${standard}`);
});

test("the production budget admits every working setup and refuses the rest", () => {
  const fits = (name, w, h) => {
    try {
      preflightModelDimensions("fsrcnnx", loadBundle(name, "fsrcnnx").manifest, w, h, {
        deviceLimits: limits,
        maxWorkingSetBytes: PRODUCTION_BUDGET_BYTES,
      });
      return true;
    } catch (error) {
      assert.equal(error.code, "MODEL_WORKING_SET_LIMIT");
      return false;
    }
  };
  // Configurations that work today must keep working: the budget is a guard
  // against a multi-gigabyte allocation, not a quality reduction. High at 1440p
  // is 1.35 GiB and the deepest working chain is 1.62 GiB, so both must pass.
  assert.equal(fits("FSRCNNX_x2_56-16-4-1", 1920, 1080), true);
  assert.equal(fits("FSRCNNX_x2_56-16-4-1", 2560, 1440), true);
  // A 4K source through High is the case that would otherwise ask for ~3 GiB.
  assert.equal(fits("FSRCNNX_x2_56-16-4-1", 3840, 2160), false);
  // Standard FSRCNNX is the degradation target, so it must still fit there.
  assert.equal(fits("FSRCNNX_x2_16-0-4-1", 3840, 2160), true);
});

test("an over-budget model reports a code the renderer can branch on", () => {
  // The render path retries with a cheaper model for this code alone and rethrows
  // everything else, so an untyped error would turn a recoverable capacity
  // problem into a dropped frame.
  assert.throws(
    () => preflightModelDimensions("fsrcnnx", loadBundle("FSRCNNX_x2_56-16-4-1", "fsrcnnx").manifest,
      3840, 2160, { deviceLimits: limits, maxWorkingSetBytes: PRODUCTION_BUDGET_BYTES }),
    (error) => error.code === "MODEL_WORKING_SET_LIMIT" && /working set/.test(error.message),
  );
});

test("a chain budget is the tightest stage budget and covers every stage together", () => {
  const bundle = loadBundle("FSRCNNX_x2_16-0-4-1", "fsrcnnx");
  const stage = (budget) => new FsrcnnxModel(
    { limits, createTexture: () => ({}), createShaderModule: () => ({}) },
    bundle.manifest, bundle.wgsl,
    { expectedName: "FSRCNNX_x2_16-0-4-1", maxWorkingSetBytes: budget },
  );
  const single = preflightModelChain([stage(PRODUCTION_BUDGET_BYTES)], 1920, 1080);
  const doubled = preflightModelChain(
    [stage(PRODUCTION_BUDGET_BYTES), stage(PRODUCTION_BUDGET_BYTES)], 1920, 1080,
  );
  // The second stage runs at 2x, so a two-deep chain costs five times one stage,
  // not twice. Budgeting per stage rather than per chain would miss that.
  assert.equal(doubled.workingSetBytes, single.workingSetBytes * 5);
  assert.equal(doubled.maxWorkingSetBytes, PRODUCTION_BUDGET_BYTES);

  // The tightest stage wins, so one conservatively-budgeted stage constrains all.
  // Both budgets comfortably fit a 64x64 chain; only which one is reported differs.
  const tighter = 50 * 1024 * 1024;
  const mixed = preflightModelChain([stage(PRODUCTION_BUDGET_BYTES), stage(tighter)], 64, 64);
  assert.equal(mixed.maxWorkingSetBytes, tighter);
  assert.ok(mixed.workingSetBytes < tighter);

  // And a budget below the chain's actual cost is refused with the branchable code.
  assert.throws(
    () => preflightModelChain([stage(PRODUCTION_BUDGET_BYTES), stage(4096)], 64, 64),
    (error) => error.code === "MODEL_WORKING_SET_LIMIT",
  );
});

// createComputePipeline blocks while the driver compiles, and allocate() runs
// inside the frame callback — so the first frame of FSRCNNX High paid for 54
// compiles of a ~380 KB shader at once. Warming moves that off the critical path
// without making it a requirement.
function asyncPipelineDevice({ resolveNow = true } = {}) {
  const device = fakeDevice();
  device.asyncCalls = 0;
  device.pending = [];
  device.createComputePipelineAsync = function (descriptor) {
    this.asyncCalls++;
    const pipeline = { getBindGroupLayout: () => ({}), descriptor };
    if (resolveNow) return Promise.resolve(pipeline);
    return new Promise((resolve) => this.pending.push(() => resolve(pipeline)));
  };
  return device;
}

test("warming compiles through the async form and satisfies later builds", async () => {
  const { manifest, wgsl } = fsrcnnxFixture();
  const device = asyncPipelineDevice();
  const model = new FsrcnnxModel(device, manifest, wgsl, { expectedName: "fixture_x2" });

  const warmed = await model.warmPipelines();
  assert.equal(warmed.length, manifest.passes.length);
  assert.equal(device.asyncCalls, manifest.passes.length);
  // The blocking form must not have been used at all.
  assert.equal(device.counts.pipelines, 0);

  // allocate() still calls buildPipelines(), which must now find the work done
  // rather than recompiling on the frame callback.
  model.allocate(64, 64, luma(64, 64, device));
  assert.equal(device.counts.pipelines, 0);
  assert.equal(model.pipelines, warmed);
});

test("warming is single-flight, so concurrent callers share one compile", async () => {
  const { manifest, wgsl } = fsrcnnxFixture();
  const device = asyncPipelineDevice({ resolveNow: false });
  const model = new FsrcnnxModel(device, manifest, wgsl, { expectedName: "fixture_x2" });

  const first = model.warmPipelines();
  const second = model.warmPipelines();
  assert.equal(device.asyncCalls, manifest.passes.length, "a second call must not recompile");
  for (const resolve of device.pending) resolve();
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a, b);
  // Once settled the promise is released, and a later call is a cheap hit.
  assert.equal(model._warmPromise, null);
  assert.equal(await model.warmPipelines(), a);
  assert.equal(device.asyncCalls, manifest.passes.length);
});

test("a device without the async form still warms, just synchronously", async () => {
  const { manifest, wgsl } = fsrcnnxFixture();
  // The unit-suite fakes implement only createComputePipeline. Warming must
  // degrade rather than become a hard requirement on the device.
  const device = fakeDevice();
  const model = new FsrcnnxModel(device, manifest, wgsl, { expectedName: "fixture_x2" });
  const warmed = await model.warmPipelines();
  assert.equal(warmed.length, manifest.passes.length);
  assert.equal(device.counts.pipelines, manifest.passes.length);
});

test("a warm that lands after destruction publishes nothing", async () => {
  const { manifest, wgsl } = fsrcnnxFixture();
  const device = asyncPipelineDevice({ resolveNow: false });
  const model = new FsrcnnxModel(device, manifest, wgsl, { expectedName: "fixture_x2" });
  const warming = model.warmPipelines();
  model.destroy();
  for (const resolve of device.pending) resolve();
  assert.deepEqual(await warming, []);
  assert.deepEqual(model.pipelines, []);
});

test("a synchronous build that overtakes a warm is not replaced by it", async () => {
  const { manifest, wgsl } = fsrcnnxFixture();
  const device = asyncPipelineDevice({ resolveNow: false });
  const model = new FsrcnnxModel(device, manifest, wgsl, { expectedName: "fixture_x2" });
  const warming = model.warmPipelines();
  // A frame arrived before warming finished and compiled synchronously. Those
  // pipelines are already bound into bind groups, so the late warm must not
  // swap them out from underneath.
  const built = model.buildPipelines();
  for (const resolve of device.pending) resolve();
  await warming;
  assert.equal(model.pipelines, built);
});

test("ArtCNN warms through the same contract", async () => {
  const { manifest, wgsl } = artcnnFixture();
  const device = asyncPipelineDevice();
  const model = new ArtCnnModel(device, manifest, wgsl, { expectedName: "ArtCNN_fixture" });
  const warmed = await model.warmPipelines();
  assert.equal(warmed.length, manifest.passes.length);
  assert.equal(device.asyncCalls, manifest.passes.length);
  assert.equal(device.counts.pipelines, 0);
});

// Upstream writes one hook per output feature map, so the network is full of
// sibling passes reading identical inputs with identical footprints and writing
// different textures. Run separately each re-fetched every input texel. Fusing
// them into one multi-output dispatch is the single largest saving available,
// and these lock the invariants that make it safe.
test("the shipped FSRCNNX models fuse into far fewer dispatches", () => {
  const expectations = new Map([
    ["FSRCNNX_x2_16-0-4-1", { upstream: 26, dispatches: 8 }],
    ["FSRCNNX_x2_56-16-4-1", { upstream: 54, dispatches: 16 }],
  ]);
  for (const [name, { upstream, dispatches }] of expectations) {
    const { manifest } = loadBundle(name, "fsrcnnx");
    assert.equal(manifest.passes.length, dispatches, name);
    // Every upstream hook must still be accounted for — fusion may reorganize
    // dispatches but must never drop a layer.
    const members = manifest.passes.reduce(
      (total, pass) => total + (pass.fusedFrom?.length ?? 1), 0,
    );
    assert.equal(members, upstream, `${name} must retain every upstream pass`);
    const covered = manifest.passes.flatMap((pass) => pass.fusedFrom ?? []);
    const sorted = [...covered].sort((a, b) => a - b);
    assert.deepEqual(sorted, [...new Set(sorted)], "no upstream pass may be fused twice");
  }
});

test("a fused pass never reads what one of its own outputs writes", () => {
  // This is the correctness condition for fusion: siblings run in a single
  // dispatch, so a member reading another member's save would consume a texture
  // that does not exist yet.
  for (const name of ["FSRCNNX_x2_16-0-4-1", "FSRCNNX_x2_56-16-4-1"]) {
    const { manifest } = loadBundle(name, "fsrcnnx");
    for (const pass of manifest.passes) {
      const saves = pass.saves ?? (pass.save ? [pass.save] : []);
      for (const bind of pass.binds) {
        assert.ok(!saves.includes(bind),
          `${name} pass ${pass.index} binds ${bind} while also writing it`);
      }
      assert.equal(new Set(saves).size, saves.length,
        `${name} pass ${pass.index} writes a resource twice`);
    }
  }
});

test("fusion stays inside the portable WebGPU binding limits", () => {
  // maxStorageTexturesPerShaderStage is 4 in the baseline, which is exactly what
  // a four-way group needs — there is no headroom, so this must be enforced.
  for (const name of ["FSRCNNX_x2_16-0-4-1", "FSRCNNX_x2_56-16-4-1"]) {
    const { manifest, wgsl } = loadBundle(name, "fsrcnnx");
    for (const pass of manifest.passes) {
      const saves = pass.saves ?? (pass.save ? [pass.save] : []);
      assert.ok(Math.max(saves.length, 1) <= 4, `${name} pass ${pass.index} storage bindings`);
      assert.ok(pass.binds.length <= 16, `${name} pass ${pass.index} sampled bindings`);
    }
    // And the bundle validates against a device advertising exactly the baseline.
    assert.doesNotThrow(() => validateModelBundle("fsrcnnx", manifest, wgsl, {
      expectedName: name,
      deviceLimits: {
        maxTextureDimension2D: 8192,
        maxBindingsPerBindGroup: 1000,
        maxSampledTexturesPerShaderStage: 16,
        maxStorageTexturesPerShaderStage: 4,
      },
    }));
  }
});

test("a five-output pass is refused on a baseline device", () => {
  const manifest = {
    name: "over_x2",
    whenThreshold: 1.3,
    passes: [
      { index: 0, desc: "too wide", binds: ["LUMA"],
        saves: ["A", "B", "C", "D", "E"], components: 4,
        widthMul: 1, heightMul: 1, kind: "conv" },
      { index: 1, desc: "shuffle", binds: ["A"], save: null, components: 1,
        widthMul: 2, heightMul: 2, kind: "shuffle" },
    ],
  };
  assert.throws(
    () => validateModelBundle("fsrcnnx", manifest, passSource(0, 1) + passSource(1, 1), {
      expectedName: "over_x2",
      deviceLimits: { maxStorageTexturesPerShaderStage: 4 },
    }),
    (error) => error.code === "MODEL_BINDING_LIMIT" && /5 storage textures/.test(error.message),
  );
});

test("a fused pass binds its outputs after its inputs, in manifest order", () => {
  const manifest = {
    name: "fused_x2",
    whenThreshold: 1.3,
    passes: [
      { index: 0, desc: "fused", binds: ["LUMA"], saves: ["A", "B"], components: 4,
        widthMul: 1, heightMul: 1, kind: "conv" },
      { index: 1, desc: "shuffle", binds: ["A"], save: null, components: 1,
        widthMul: 2, heightMul: 2, kind: "shuffle" },
    ],
  };
  // One sampled input at binding 0, two storage outputs at bindings 1 and 2.
  const wgsl = passSource(0, 1, 2) + passSource(1, 1);
  const device = fakeDevice();
  const model = new FsrcnnxModel(device, manifest, wgsl, { expectedName: "fused_x2" });
  const lumaTexture = luma(64, 64, device);
  model.allocate(64, 64, lumaTexture);

  const [fused] = model.bindGroups;
  assert.deepEqual(fused.entries.map(({ binding }) => binding), [0, 1, 2]);
  // Binding 1 must be A's texture and binding 2 must be B's — a swap here would
  // silently transpose two feature maps and no dimension check would notice.
  assert.equal(fused.entries[0].resource.texture, lumaTexture);
  assert.equal(fused.entries[1].resource.texture, model.textures.get("A"));
  assert.equal(fused.entries[2].resource.texture, model.textures.get("B"));

  // The terminal shuffle still writes the model output rather than a save.
  const [, terminal] = model.bindGroups;
  assert.equal(terminal.entries[1].resource.texture, model.outputTexture);
});

test("each fused output still gets its own texture of the right size", () => {
  const { manifest } = loadBundle("FSRCNNX_x2_16-0-4-1", "fsrcnnx");
  const plan = preflightModelDimensions("fsrcnnx", manifest, 320, 180, {
    deviceLimits: { maxTextureDimension2D: 8192 },
  });
  const names = plan.resources.map(({ name }) => name);
  // 17 intermediates plus the output: fusing dispatches must not fuse storage.
  assert.equal(new Set(names).size, names.length);
  assert.equal(names.length, 18);
  for (const resource of plan.resources) {
    if (resource.name === "OUTPUT") {
      assert.deepEqual([resource.width, resource.height], [640, 360]);
    } else {
      assert.deepEqual([resource.width, resource.height], [320, 180]);
    }
  }
});
