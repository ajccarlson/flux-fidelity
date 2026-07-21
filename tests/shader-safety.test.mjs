import assert from "node:assert/strict";
import test from "node:test";

import { buildSharpenShader } from "../fsrcnnx-sharpen.js";
import {
  buildL2Shader,
  buildMeanShader,
  SSIMDS_MR_WGSL,
} from "../fsrcnnx-ssimds.js";
import {
  estimateSsimWork,
  SsimDownscaler,
  SSIMDS_WORK_BUDGET,
} from "../fsrcnnx-ssimds-runtime.js";

const EPS = 1e-6;

function assertFiniteWgsl(source) {
  assert.doesNotMatch(source, /\b(?:NaN|Infinity)\b/);
}

test("sharpen builder normalizes non-finite inputs", () => {
  for (const value of [NaN, Infinity, -Infinity, "invalid", Symbol("invalid")]) {
    const sharpen = buildSharpenShader(value);
    assertFiniteWgsl(sharpen);
    assert.match(sharpen, /const curve_height : f32 = 1\.0000;/);
  }

  assert.match(buildSharpenShader(-5), /const curve_height : f32 = 0\.1000;/);
  assert.match(buildSharpenShader(50), /const curve_height : f32 = 2\.0000;/);
});

test("sharpen's flat-field guards preserve a finite constant field", () => {
  const source = buildSharpenShader(1);
  assert.match(source, /if \(s <= NUM_EPS\) \{ return 0\.0; \}/);
  assert.match(source, /if \(weightsum > NUM_EPS\)/);
  assert.match(source, /neg_laplace = c0_Y;/);
  assert.doesNotMatch(source, /pow\(r, 2\.0\)/);

  const softLimit = (value, limit) => {
    if (limit <= EPS) return 0;
    const ratio = value / limit;
    const ratio2 = ratio * ratio;
    return Math.min(1, Math.abs(ratio) * (27 + ratio2) / (27 + 9 * ratio2)) * limit;
  };
  const weightedMean = (a, b, weight) => weight * Math.abs(a) + Math.abs(1 - weight) * Math.abs(b);
  const flatReference = (center) => {
    const weightSum = 0;
    const laplaceSquares = 0;
    const negativeLaplace = weightSum > EPS
      ? Math.sqrt(Math.max(laplaceSquares, 0) / weightSum)
      : center;
    const sharpDiff = (center - negativeLaplace) * 0.01;
    const minDistance = 0;
    const limited = weightedMean(Math.max(sharpDiff, 0), softLimit(Math.max(sharpDiff, 0), minDistance), 0.167)
      - weightedMean(Math.min(sharpDiff, 0), softLimit(Math.min(sharpDiff, 0), minDistance), 0.25);
    return center + (Math.min(1, Math.max(0, center + limited)) - center);
  };

  assert.equal(Number.isNaN(Math.sqrt(0 / 0)), true); // regression mechanism
  for (const center of [0, 0.18, 0.5, 1]) {
    const output = flatReference(center);
    assert.equal(Number.isFinite(output), true);
    assert.equal(output, center);
  }
});

test("SSim shader builders reject invalid ratios and guard zero weight sums", () => {
  for (const ratio of [0, -1, NaN, Infinity, -Infinity, 1e-8, "invalid", Symbol("invalid")]) {
    assert.throws(() => buildMeanShader(ratio, 1), RangeError);
    assert.throws(() => buildMeanShader(1, ratio), RangeError);
    assert.throws(() => buildL2Shader(0, ratio), RangeError);
  }
  for (const axis of [-1, 2, "1", null]) {
    assert.throws(() => buildL2Shader(axis, 1), RangeError);
  }

  const mean = buildMeanShader(1, 1.05);
  const l2 = buildL2Shader(1, 2);
  assertFiniteWgsl(mean);
  assertFiniteWgsl(l2);
  assert.match(mean, /if \(abs\(W\) <= NUM_EPS\)/);
  assert.match(l2, /if \(abs\(W\) <= NUM_EPS\)/);
  assert.match(SSIMDS_MR_WGSL, /Sh \/ max\(Sl, NUM_EPS\)/);

  // Uniform fields have Sl=Sh=0. The protected select operand remains finite and
  // the structural branch yields the identity ratio R=1.
  const sl = 0, sh = 0, sigmaNoiseSquared = 10 / (255 * 255);
  const varianceRatio = Math.min(1, Math.max(0, sh / Math.max(sl, EPS)));
  const structuralRatio = Math.sqrt((sh + sigmaNoiseSquared) / (sl + sigmaNoiseSquared));
  const ratio = sl > sh ? varianceRatio : structuralRatio;
  assert.equal(Number.isFinite(varianceRatio), true);
  assert.equal(ratio, 1);
});

test("Catmull-Rom weight sums stay nonzero across supported downscale ratios", () => {
  const mn = (x) => {
    if (x < 1) return (1.5 * x - 2.5) * x * x + 1;
    if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
  };
  const weightSum = (ratio, center) => {
    let sum = 0;
    for (let sample = Math.ceil(center - 2 * ratio); sample <= Math.floor(center + 2 * ratio); sample++) {
      sum += mn(Math.abs((sample - center) / ratio));
    }
    return sum;
  };

  for (const ratio of [1, 1.05, 1.5, 2, 4, 8, 16, 32]) {
    for (const fraction of [0, 0.125, 0.25, 0.5, 0.875]) {
      assert.ok(weightSum(ratio, 20 + fraction) > EPS, `ratio=${ratio}, fraction=${fraction}`);
    }
  }
});

test("SSim work estimation allows normal downscales and bypasses extreme ratios", () => {
  const normal = estimateSsimWork(3840, 2160, 1920, 1080);
  assert.equal(normal.allowed, true);
  assert.ok(normal.estimatedTextureSamples < SSIMDS_WORK_BUDGET.maxEstimatedTextureSamples);

  const extreme = estimateSsimWork(7680, 4320, 1, 1);
  assert.equal(extreme.allowed, false);
  assert.ok(extreme.tapsX > SSIMDS_WORK_BUDGET.maxAxisTaps);
  assert.match(extreme.reason, /axis taps|mean taps|estimated texture samples/);

  for (const dimensions of [
    [0, 1080, 1, 1],
    [1920.5, 1080, 1, 1],
    [1920, 1080, 1921, 1080],
    [1920, 1080, 1920, 1081],
  ]) {
    assert.throws(() => estimateSsimWork(...dimensions), RangeError);
  }
});

test("SSim budget bypass records no GPU work and returns the input texture", () => {
  const fakeDevice = {
    limits: { maxTextureDimension2D: 8192 },
    createSampler() { return {}; },
  };
  const input = {
    createView() { throw new Error("budget bypass must not create a texture view"); },
  };
  const scaler = new SsimDownscaler(fakeDevice);

  assert.equal(scaler.prepare(7680, 4320, 1, 1, input), false);
  assert.equal(scaler.bypassed, true);
  assert.equal(scaler.lastPlan.allowed, false);
  assert.equal(scaler.run(null, input), input);
  assert.equal(scaler.textures, null);
  assert.equal(scaler.bindGroups, null);

  assert.throws(() => scaler.prepare(9000, 4320, 1, 1, input), /device limit/);
  assert.throws(() => scaler.prepare(7680, 4320, 1, 1, null), TypeError);
});

test("SSim resize publication is atomic and retryable after pipeline or texture failure", (t) => {
  const previousUsage = globalThis.GPUTextureUsage;
  globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_SRC: 4 };
  t.after(() => {
    if (previousUsage === undefined) delete globalThis.GPUTextureUsage;
    else globalThis.GPUTextureUsage = previousUsage;
  });

  const events = { pipeline: 0, texture: 0, bind: 0, failPipelineAt: -1, failTextureAt: -1, failBindAt: -1 };
  const makeTexture = () => ({
    destroyed: 0,
    createView() { return { texture: this }; },
    destroy() { this.destroyed++; },
  });
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    createSampler() { return {}; },
    createShaderModule() { return {}; },
    createRenderPipeline() {
      events.pipeline++;
      if (events.pipeline === events.failPipelineAt) throw new Error("injected pipeline failure");
      return { getBindGroupLayout() { return {}; } };
    },
    createTexture() {
      events.texture++;
      if (events.texture === events.failTextureAt) throw new Error("injected texture failure");
      return makeTexture();
    },
    createBindGroup(descriptor) {
      events.bind++;
      if (events.bind === events.failBindAt) throw new Error("injected bind-group failure");
      return descriptor;
    },
  };
  const inputA = makeTexture();
  const inputB = makeTexture();
  const scaler = new SsimDownscaler(device);
  assert.equal(scaler.prepare(128, 96, 64, 48, inputA), true);
  const original = {
    textures: scaler.textures,
    pipelines: scaler.pipelines,
    bindGroups: scaler.bindGroups,
  };

  events.failBindAt = events.bind + 2;
  assert.throws(() => scaler.prepare(128, 96, 64, 48, inputB), /injected bind-group failure/);
  assert.equal(scaler.textures, original.textures);
  assert.equal(scaler.bindGroups, original.bindGroups);
  assert.equal(scaler.hiTex, inputA);
  events.failBindAt = -1;
  assert.equal(scaler.prepare(128, 96, 64, 48, inputB), true);
  assert.equal(scaler.hiTex, inputB);
  assert.equal(scaler.textures, original.textures);
  assert.notEqual(scaler.bindGroups, original.bindGroups);

  events.failPipelineAt = events.pipeline + 2;
  assert.throws(() => scaler.prepare(160, 120, 80, 60, inputA), /injected pipeline failure/);
  assert.equal(scaler.textures, original.textures);
  assert.equal(scaler.pipelines, original.pipelines);
  assert.equal(scaler.hiTex, inputB);
  assert.ok(Object.values(original.textures).every((texture) => texture.destroyed === 0));

  events.failPipelineAt = -1;
  assert.equal(scaler.prepare(160, 120, 80, 60, inputA), true);
  assert.ok(Object.values(original.textures).every((texture) => texture.destroyed === 1));
  const secondGeneration = scaler.textures;

  events.failTextureAt = events.texture + 2;
  assert.throws(() => scaler.prepare(192, 144, 96, 72, inputB), /injected texture failure/);
  assert.equal(scaler.textures, secondGeneration);
  assert.equal(scaler.hiTex, inputA);
  assert.ok(Object.values(secondGeneration).every((texture) => texture.destroyed === 0));

  events.failTextureAt = -1;
  assert.equal(scaler.prepare(192, 144, 96, 72, inputB), true);
  assert.ok(Object.values(secondGeneration).every((texture) => texture.destroyed === 1));
  scaler.destroy();
});
