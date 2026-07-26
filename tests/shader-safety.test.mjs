import assert from "node:assert/strict";
import test from "node:test";

import { buildSharpenShader } from "../src/core/fsrcnnx-sharpen.js";
import {
  buildL2Shader,
  buildMeanShader,
  SSIMDS_MR_WGSL,
} from "../src/core/fsrcnnx-ssimds.js";
import {
  estimateSsimWork,
  SsimDownscaler,
  SSIMDS_WORK_BUDGET,
} from "../src/core/fsrcnnx-ssimds-runtime.js";

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
  assert.equal(l2, buildL2Shader(1, 2, true));
  assert.equal(buildL2Shader(0, 2), buildL2Shader(0, 2, false));
  assert.doesNotMatch(buildL2Shader(1, 2, false), /s = s \* s;/);
  assert.throws(() => buildL2Shader(1, 2, "false"), TypeError);
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

test("SSim work estimation keeps total work diagnostic and bounds only individual passes", () => {
  const normal = estimateSsimWork(3840, 2160, 1920, 1080);
  assert.equal(normal.allowed, true);
  assert.equal(normal.path, "direct");
  assert.equal(normal.stages.length, 0);

  // Aggregate work is deliberately not an eligibility gate. This otherwise
  // device-valid 8K -> 4K plan remains on the unchanged direct SSimDS path.
  const fourK = estimateSsimWork(7680, 4320, 3840, 2160);
  assert.equal(fourK.path, "direct");
  assert.ok(fourK.estimatedTextureSamples > 512_000_000);

  const extreme = estimateSsimWork(7680, 4320, 1, 1);
  assert.equal(extreme.allowed, true);
  assert.equal(extreme.path, "multistage");
  assert.ok(extreme.tapsX > SSIMDS_WORK_BUDGET.maxAxisTaps);
  assert.equal(extreme.reason, null);
  assert.ok(extreme.stages.length > 1);
  assert.deepEqual(
    [extreme.stages.at(-1).outputWidth, extreme.stages.at(-1).outputHeight],
    [1, 1],
  );
  for (const stage of extreme.stages) {
    assert.ok(stage.tapsX <= SSIMDS_WORK_BUDGET.maxAxisTaps);
    assert.ok(stage.tapsY <= SSIMDS_WORK_BUDGET.maxAxisTaps);
    assert.ok(stage.outputWidth <= stage.inputWidth);
    assert.ok(stage.outputHeight <= stage.inputHeight);
  }

  for (const dimensions of [
    [0, 1080, 1, 1],
    [1920.5, 1080, 1, 1],
    [1920, 1080, 1921, 1080],
    [1920, 1080, 1920, 1081],
  ]) {
    assert.throws(() => estimateSsimWork(...dimensions), RangeError);
  }
});

test("SSim extreme ratios run bounded two-moment stages and clean resources atomically", (t) => {
  const previousUsage = globalThis.GPUTextureUsage;
  globalThis.GPUTextureUsage = { RENDER_ATTACHMENT: 1, TEXTURE_BINDING: 2, COPY_SRC: 4 };
  t.after(() => {
    if (previousUsage === undefined) delete globalThis.GPUTextureUsage;
    else globalThis.GPUTextureUsage = previousUsage;
  });

  const events = {
    pipeline: 0,
    texture: 0,
    bind: 0,
    failTextureAt: -1,
    textures: [],
    passes: [],
  };
  const makeTexture = (descriptor = null) => ({
    descriptor,
    destroyed: 0,
    createView() { return { texture: this }; },
    destroy() { this.destroyed++; },
  });
  const device = {
    limits: { maxTextureDimension2D: 8192 },
    createSampler() { return {}; },
    createShaderModule({ code }) { return { code }; },
    createRenderPipeline(descriptor) {
      events.pipeline++;
      return {
        code: descriptor.fragment.module.code,
        getBindGroupLayout() { return {}; },
      };
    },
    createTexture(descriptor) {
      events.texture++;
      if (events.texture === events.failTextureAt) throw new Error("injected texture failure");
      const texture = makeTexture(descriptor);
      events.textures.push(texture);
      return texture;
    },
    createBindGroup(descriptor) {
      events.bind++;
      return descriptor;
    },
  };
  const encoder = {
    beginRenderPass(descriptor) {
      const record = { descriptor, pipeline: null, bindGroup: null, draws: 0 };
      events.passes.push(record);
      return {
        setPipeline(pipeline) { record.pipeline = pipeline; },
        setBindGroup(_index, bindGroup) { record.bindGroup = bindGroup; },
        draw() { record.draws++; },
        end() {},
      };
    },
  };
  const inputA = makeTexture();
  const inputB = makeTexture();
  const scaler = new SsimDownscaler(device);

  // A device-valid 8K source has no aggregate-work ceiling: the unchanged
  // direct chain is prepared and applied for an ordinary 2x downscale.
  assert.equal(scaler.prepare(7680, 4320, 3840, 2160, inputA), true);
  assert.equal(scaler.path, "direct");
  const directGeneration = [...events.textures];
  const directOutput = scaler.run(encoder, inputA);
  assert.equal(directOutput, scaler.textures.out);
  assert.notEqual(directOutput, inputA);
  assert.equal(events.passes.length, 5);
  events.passes.length = 0;

  const firstGenerationStart = events.textures.length;
  assert.equal(scaler.prepare(7680, 4320, 1, 1, inputA), true);
  assert.equal(scaler.path, "multistage");
  assert.equal(scaler.lastPlan.allowed, true);
  assert.ok(scaler.lastPlan.stages.length > 1);
  assert.ok(directGeneration.every((texture) => texture.destroyed === 1));
  assert.match(scaler.pipelines.stages[0].verticalL2.code, /s = s \* s;/);
  assert.doesNotMatch(scaler.pipelines.stages[0].verticalMean.code, /s = s \* s;/);
  for (const stage of scaler.pipelines.stages.slice(1)) {
    assert.equal(stage.verticalL2, stage.verticalMean);
    assert.doesNotMatch(stage.verticalL2.code, /s = s \* s;/);
  }

  const firstGeneration = events.textures.slice(firstGenerationStart);
  const output = scaler.run(encoder, inputA);
  assert.equal(output, scaler.textures.out);
  assert.notEqual(output, inputA);
  assert.equal(events.passes.length, scaler.lastPlan.stages.length * 4 + 2);
  assert.ok(events.passes.every((pass) => pass.pipeline && pass.bindGroup && pass.draws === 1));

  const failedGenerationStart = events.textures.length;
  events.failTextureAt = events.texture + 2;
  assert.throws(
    () => scaler.prepare(6400, 3200, 1, 1, inputB),
    /injected texture failure/,
  );
  assert.equal(scaler.hiTex, inputA);
  assert.ok(firstGeneration.every((texture) => texture.destroyed === 0));
  const failedGeneration = events.textures.slice(failedGenerationStart);
  assert.equal(failedGeneration.length, 1);
  assert.equal(failedGeneration[0].destroyed, 1);

  events.failTextureAt = -1;
  const finalGenerationStart = events.textures.length;
  assert.equal(scaler.prepare(6400, 3200, 1, 1, inputB), true);
  assert.ok(firstGeneration.every((texture) => texture.destroyed === 1));
  const finalGeneration = events.textures.slice(finalGenerationStart);
  scaler.destroy();
  assert.ok(finalGeneration.every((texture) => texture.destroyed === 1));

  assert.throws(() => scaler.prepare(9000, 4320, 1, 1, inputA), /device limit/);
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
