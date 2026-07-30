import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// Why there is no shader-f16 arithmetic path for FSRCNNX.
//
// Half-precision arithmetic roughly doubles ALU throughput on hardware that
// supports it, and the model's intermediates are already rgba16float — only the
// math is f32 — so converting the accumulation looks like a free win. It is not.
//
// The reference fixtures allow the *entire* network a maximum deviation of 0.03
// against mpv/libplacebo. These measurements use the real shipped weights and
// show that a single convolution pass consumes most of that budget on its own,
// before any of the other 25 passes contribute. Accumulating in f32 and storing
// in f16 avoids the error but gains nothing, because the storage is f16 already.
//
// There is a second, independent blocker: packaged-Chromium validation runs under
// SwiftShader, which does not advertise shader-f16, so an f16 path would ship
// with no gate able to execute it. These tests are the standing answer, so the
// question is settled by measurement rather than re-argued.

const f16 = Math.f16round;
const REFERENCE_MAX_DEVIATION = 0.03;

function shippedWeights() {
  const wgsl = readFileSync(new URL("../model/FSRCNNX_x2_16-0-4-1.wgsl", import.meta.url), "utf8");
  const vecs = [...wgsl.matchAll(/vec4f\(([-\d.e, ]+)\)/g)]
    .map((match) => match[1].split(",").map(Number))
    .filter((values) => values.length === 4 && values.every(Number.isFinite));
  const mats = [...wgsl.matchAll(/mat4x4f\(([-\d.e, ]+)\)/g)]
    .map((match) => match[1].split(",").map(Number))
    .filter((values) => values.length === 16 && values.every(Number.isFinite));
  return { vecs, mats };
}

// Deterministic so the recorded numbers are reproducible.
function lcg(seed) {
  let state = seed;
  return () => (state = (state * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
}

test("one 25-tap feature pass in f16 already spends most of the whole-network budget", () => {
  const { vecs } = shippedWeights();
  assert.ok(vecs.length > 100, "expected the shipped weights to be parseable");
  const random = lcg(12345);
  let worst = 0;
  for (let trial = 0; trial < 1500; trial++) {
    const weights = vecs[Math.floor(random() * vecs.length)];
    const exact = [0, 0, 0, 0];
    const half = [0, 0, 0, 0];
    for (let tap = 0; tap < 25; tap++) {
      // Luma is in [0,1], which is the best-conditioned input in the network.
      const sample = random();
      const sample16 = f16(sample);
      for (let channel = 0; channel < 4; channel++) {
        exact[channel] += weights[channel] * sample;
        half[channel] = f16(half[channel] + f16(f16(weights[channel]) * sample16));
      }
    }
    for (let channel = 0; channel < 4; channel++) {
      worst = Math.max(worst, Math.abs(exact[channel] - half[channel]));
    }
  }
  // Measured at roughly 2.6e-2 against a 3.0e-2 allowance for all 26 passes.
  assert.ok(worst > REFERENCE_MAX_DEVIATION / 2,
    `a single pass should consume over half the budget; measured ${worst.toExponential(3)}`);
});

function mappingPassError(featureRange, trials = 800) {
  const { mats } = shippedWeights();
  const random = lcg(999);
  let worst = 0;
  for (let trial = 0; trial < trials; trial++) {
    const exact = [0, 0, 0, 0];
    const half = [0, 0, 0, 0];
    for (let term = 0; term < 36; term++) {
      const matrix = mats[Math.floor(random() * mats.length)];
      const vector = [0, 1, 2, 3].map(() => (random() * 2 - 1) * featureRange);
      const vector16 = vector.map(f16);
      for (let row = 0; row < 4; row++) {
        let exactDot = 0;
        let halfDot = 0;
        for (let column = 0; column < 4; column++) {
          exactDot += matrix[column * 4 + row] * vector[column];
          halfDot = f16(halfDot + f16(f16(matrix[column * 4 + row]) * vector16[column]));
        }
        exact[row] += exactDot;
        half[row] = f16(half[row] + halfDot);
      }
    }
    for (let row = 0; row < 4; row++) worst = Math.max(worst, Math.abs(exact[row] - half[row]));
  }
  return worst;
}

test("a 36-term mapping pass in f16 scales its error with the feature range", () => {
  // Feature maps are not bounded to [0,1] the way luma is; their magnitude is
  // whatever the preceding layers produced. The error tracks that magnitude, so
  // there is no input regime where the budget is comfortable — and the model
  // stacks four mapping stages plus residuals plus the sub-pixel convolution.
  const modest = mappingPassError(1);
  const typical = mappingPassError(2);
  const large = mappingPassError(4);
  assert.ok(modest < typical && typical < large, "error must grow with feature magnitude");
  // Measured at roughly 1.4e-2, 2.9e-2 and 5.7e-2 against a 3.0e-2 allowance for
  // the whole network: one pass alone spends about half the budget at the low end
  // and exceeds it outright at the high end.
  assert.ok(typical > REFERENCE_MAX_DEVIATION / 2,
    `a typical mapping pass should spend over half the budget; measured ${typical.toExponential(3)}`);
  assert.ok(large > REFERENCE_MAX_DEVIATION,
    `a high-magnitude mapping pass should exceed it; measured ${large.toExponential(3)}`);
});

test("the weights themselves are representable, so precision loss is in the accumulation", () => {
  // Worth separating: if the weights were the problem, a rescaling could fix it.
  // They are not — nearly all are comfortably inside the f16 normal range, and
  // the error comes from summing 25 to 126 terms into a 10-bit mantissa.
  const { vecs, mats } = shippedWeights();
  const magnitudes = [...vecs.flat(), ...mats.flat()].map(Math.abs).filter((value) => value > 0);
  const F16_MIN_NORMAL = 6.103515625e-5;
  const subnormal = magnitudes.filter((value) => value < F16_MIN_NORMAL).length;
  assert.ok(subnormal / magnitudes.length < 0.001,
    `only a negligible share of weights should be f16-subnormal; got ${subnormal}`);
  assert.ok(Math.max(...magnitudes) < 65504, "no weight may overflow f16");
});
