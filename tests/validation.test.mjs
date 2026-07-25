import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";

import {
  ARTCNN_MODEL_NAMES,
  FSRCNNX_HIGH_MODEL_NAME,
  FSRCNNX_STANDARD_MODEL_NAMES,
  GENERATED_MODEL_ASSET_PATHS,
  GENERATED_MODEL_CATALOG,
} from "../src/core/fsrcnnx-model-catalog.js";
import {
  acquireValidationDevice,
  alignedBytesPerRow,
  buildCorePipelines,
  createValidationPlan,
  float16ToNumber,
  inspectRgba16Float,
  inspectOrtFloatTensor,
  numberToFloat16,
  ONNX_VALIDATION_CHECKS,
  REFERENCE_VALIDATION_CHECKS,
  summarizeValidation,
  withGpuErrorScopes,
  withTimeout,
} from "../validation/fsrcnnx-validation.js";

test("the shared generated-model catalog is complete, immutable, and backed by files", () => {
  assert.equal(GENERATED_MODEL_CATALOG.length, 5);
  assert.equal(new Set(GENERATED_MODEL_CATALOG.map(({ name }) => name)).size, 5);
  assert.equal(Object.isFrozen(GENERATED_MODEL_CATALOG), true);
  assert.ok(GENERATED_MODEL_CATALOG.every(Object.isFrozen));
  assert.deepEqual(FSRCNNX_STANDARD_MODEL_NAMES, [
    "FSRCNNX_x2_16-0-4-1",
  ]);
  assert.equal(FSRCNNX_HIGH_MODEL_NAME, "FSRCNNX_x2_56-16-4-1");
  assert.deepEqual(ARTCNN_MODEL_NAMES, ["ArtCNN_C4F32", "ArtCNN_C4F32_DN", "ArtCNN_C4F32_DS"]);
  assert.equal(GENERATED_MODEL_ASSET_PATHS.length, 10);
  assert.deepEqual(GENERATED_MODEL_ASSET_PATHS, [...GENERATED_MODEL_ASSET_PATHS].sort());
  for (const path of GENERATED_MODEL_ASSET_PATHS) {
    assert.equal(existsSync(new URL(`../${path}`, import.meta.url)), true, path);
  }
});

test("validation accounting is fixed before execution and treats skips as incompatible", () => {
  const plan = createValidationPlan(GENERATED_MODEL_CATALOG);
  assert.equal(plan.length, 24);
  assert.equal(new Set(plan.map(({ id }) => id)).size, plan.length);
  assert.equal(Object.isFrozen(ONNX_VALIDATION_CHECKS), true);
  assert.ok(ONNX_VALIDATION_CHECKS.every(Object.isFrozen));
  assert.equal(Object.isFrozen(REFERENCE_VALIDATION_CHECKS), true);
  assert.ok(REFERENCE_VALIDATION_CHECKS.every(Object.isFrozen));
  assert.deepEqual(plan.slice(2, 6).map(({ id }) => id), [
    "color:extract-reference",
    "color:recombine-reference",
    "filter:ssimds-reference",
    "filter:sharpen-reference",
  ]);
  assert.deepEqual(plan.slice(7, 9).map(({ id }) => id), [
    "onnx:rife-v4.26-fp16",
    "onnx:rife-v4.26",
  ]);
  const results = new Map();
  assert.deepEqual(summarizeValidation(plan, results), {
    pass: 0, fail: 0, skip: 0, pending: 24, total: 24, complete: false, ok: false,
  });
  for (const check of plan) results.set(check.id, { status: "pass" });
  assert.equal(summarizeValidation(plan, results).ok, true);
  results.set(plan[3].id, { status: "skip" });
  const incompatible = summarizeValidation(plan, results);
  assert.equal(incompatible.complete, true);
  assert.equal(incompatible.skip, 1);
  assert.equal(incompatible.ok, false);
  results.set("unknown", { status: "pass" });
  assert.throws(() => summarizeValidation(plan, results), /does not belong/);
});

test("ORT output inspection reads GPU tensors and rejects shape, dtype, and numeric corruption", async () => {
  const cpu = await inspectOrtFloatTensor({
    type: "float32",
    dims: [1, 1, 1, 3],
    data: new Float32Array([0.1, 0.5, 0.9]),
  }, [1, 1, 1, 3], "CPU fixture");
  assert.equal(cpu.elements, 3);
  assert.ok(Math.abs(cpu.min - 0.1) < 1e-6);
  assert.ok(Math.abs(cpu.max - 0.9) < 1e-6);
  assert.equal(Object.isFrozen(cpu), true);
  assert.equal(Object.isFrozen(cpu.dims), true);

  let readbacks = 0;
  const gpu = await inspectOrtFloatTensor({
    dataType: "float32",
    dims: [1, 1, 2, 2],
    location: "gpu-buffer",
    async getData() {
      readbacks++;
      return new Float32Array([-1, -0.25, 0.25, 1]);
    },
  }, [1, 1, 2, 2], "GPU fixture");
  assert.equal(readbacks, 1);
  assert.deepEqual(gpu.dims, [1, 1, 2, 2]);
  assert.equal(gpu.min, -1);
  assert.equal(gpu.max, 1);

  await assert.rejects(inspectOrtFloatTensor(null, [1], "missing fixture"), /missing fixture is missing/);
  await assert.rejects(inspectOrtFloatTensor({
    type: "float16", dims: [1], data: new Uint16Array([0]),
  }, [1]), /dtype 'float16'/);
  await assert.rejects(inspectOrtFloatTensor({
    type: "float32", dims: [1, 2], data: new Float32Array(2),
  }, [2, 1]), /shape \[1,2\]; expected \[2,1\]/);
  await assert.rejects(inspectOrtFloatTensor({
    type: "float32", dims: [2], data: new Float32Array([0, NaN]),
  }, [2]), /1\/2 non-finite/);
  await assert.rejects(inspectOrtFloatTensor({
    type: "float32", dims: [2], data: new Float32Array([0.5, 0.5]),
  }, [2]), /constant \(0\.5\)/);
  await assert.rejects(inspectOrtFloatTensor({
    type: "float32", dims: [2], location: "gpu-buffer",
  }, [2]), /no getData\(\) readback/);
  await assert.rejects(inspectOrtFloatTensor({
    type: "float32", dims: [Number.MAX_SAFE_INTEGER, 2], data: new Float32Array(1),
  }, [Number.MAX_SAFE_INTEGER, 2]), /element count exceeds the safe integer range/);
});

test("WebGPU acquisition mirrors production adapter preferences and optional features", async () => {
  await assert.rejects(acquireValidationDevice(null, 20), /navigator\.gpu is unavailable/);
  await assert.rejects(acquireValidationDevice({ requestAdapter: async () => null }, 20), /no compatible adapter/);

  const device = {};
  let adapterOptions;
  let deviceOptions;
  const gpu = {
    async requestAdapter(options) {
      adapterOptions = options;
      return {
        features: new Set(["float32-filterable"]),
        info: { vendor: "Test", architecture: "GPU" },
        async requestDevice(options) {
          deviceOptions = options;
          return device;
        },
      };
    },
  };
  const acquired = await acquireValidationDevice(gpu, 20);
  assert.equal(acquired.device, device);
  assert.deepEqual(adapterOptions, { powerPreference: "high-performance" });
  assert.deepEqual(deviceOptions, { requiredFeatures: ["float32-filterable"] });
  assert.match(acquired.detail, /Test GPU; float32-filterable/);
});

test("a device that resolves after acquisition timeout is retired", async () => {
  let resolveDevice;
  const requested = new Promise((resolve) => { resolveDevice = resolve; });
  let destroys = 0;
  const gpu = {
    async requestAdapter() {
      return {
        features: new Set(),
        requestDevice: () => requested,
      };
    },
  };
  await assert.rejects(acquireValidationDevice(gpu, 5), /device request timed out/);
  resolveDevice({ destroy() { destroys++; } });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(destroys, 1);
});

test("timeouts and WebGPU error scopes fail deterministically and unwind every scope", async () => {
  await assert.rejects(withTimeout(new Promise(() => {}), 5, "fixture"), /fixture timed out/);
  const events = [];
  const device = {
    errors: [null, { message: "invalid binding" }, null],
    pushErrorScope(filter) { events.push(`push:${filter}`); },
    async popErrorScope() { events.push("pop"); return this.errors.shift(); },
  };
  await assert.rejects(
    withGpuErrorScopes(device, "pipeline", async () => 42, 20),
    /out-of-memory: invalid binding/,
  );
  assert.deepEqual(events, [
    "push:internal", "push:out-of-memory", "push:validation", "pop", "pop", "pop",
  ]);
});

test("GPU error scopes remain active until a delayed operation has fully settled", async () => {
  const events = [];
  const device = {
    pushErrorScope(filter) { events.push(`push:${filter}`); },
    async popErrorScope() { events.push("pop"); return null; },
  };
  const result = await withGpuErrorScopes(device, "delayed inference", async () => {
    events.push("operation:start");
    await new Promise((resolve) => setTimeout(resolve, 12));
    events.push("operation:cleanup");
    return 42;
  }, 5);
  assert.equal(result, 42);
  assert.equal(events.indexOf("operation:cleanup") < events.indexOf("pop"), true);
});

test("GPU scope diagnostics are retained alongside an operation failure", async () => {
  const device = {
    errors: [{ message: "shader binding mismatch" }, null, null],
    pushErrorScope() {},
    async popErrorScope() { return this.errors.shift(); },
  };
  await assert.rejects(
    withGpuErrorScopes(device, "fixture inference", async () => {
      throw new Error("submission rejected");
    }),
    (error) => /submission rejected/.test(error.message) &&
      /validation: shader binding mismatch/.test(error.message) &&
      error.cause?.message === "submission rejected",
  );
});

test("rgba16float conversion and padded readback inspection reject non-finite output", () => {
  assert.equal(alignedBytesPerRow(32), 256);
  assert.equal(alignedBytesPerRow(37), 512);
  assert.equal(alignedBytesPerRow(74), 768);
  assert.throws(() => alignedBytesPerRow(0), /positive safe integer/);
  assert.throws(() => alignedBytesPerRow(1, 0), /bytes per pixel/);

  for (const value of [-65504, -1, -0, 0, 0.1, 1, 65504]) {
    const decoded = float16ToNumber(numberToFloat16(value));
    const tolerance = Math.max(1e-7, Math.abs(value) * 0.001);
    assert.ok(Math.abs(decoded - value) <= tolerance, `${value} became ${decoded}`);
  }
  assert.equal(float16ToNumber(numberToFloat16(Infinity)), Infinity);
  assert.equal(float16ToNumber(numberToFloat16(-Infinity)), -Infinity);
  assert.equal(Number.isNaN(float16ToNumber(numberToFloat16(NaN))), true);
  assert.equal(numberToFloat16(70_000), 0x7c00, "finite overflow must become infinity, not NaN");
  assert.equal(numberToFloat16(1 + 2 ** -11), 0x3c00, "a midpoint must round to the even lower value");
  assert.equal(numberToFloat16(1 + 3 * 2 ** -11), 0x3c02, "a midpoint must round to the even upper value");

  const words = new Uint16Array(16).fill(0x7e00);
  for (const index of [0, 1, 2, 3, 8, 9, 10, 11]) words[index] = numberToFloat16(index / 12);
  const finite = inspectRgba16Float(words, 1, 2, 8);
  assert.equal(finite.components, 8);
  assert.equal(finite.nonFinite, 0, "row padding must not be inspected as output");
  assert.deepEqual(finite.channelMin, [0, 1 / 12, 2 / 12, 3 / 12].map((value) => (
    float16ToNumber(numberToFloat16(value))
  )));
  words[9] = 0x7c00;
  assert.equal(inspectRgba16Float(words, 1, 2, 8).nonFinite, 1);
});

test("core validation constructs the covered color/filter pipeline variants", () => {
  const descriptors = [];
  const device = {
    createShaderModule(descriptor) {
      assert.equal(typeof descriptor.code, "string");
      assert.equal(descriptor.code.includes("NaN"), false);
      return { code: descriptor.code };
    },
    createComputePipeline(descriptor) {
      descriptors.push(["compute", descriptor]);
      return descriptor;
    },
    createRenderPipeline(descriptor) {
      descriptors.push(["render", descriptor]);
      return descriptor;
    },
  };
  const pipelines = buildCorePipelines(device, "bgra8unorm");
  assert.equal(pipelines.length, 12);
  assert.equal(descriptors.filter(([kind]) => kind === "compute").length, 2);
  assert.equal(descriptors.filter(([kind]) => kind === "render").length, 10);
  assert.ok(descriptors.some(([, descriptor]) => descriptor.fragment?.targets[0].format === "rgba16float"));
});
