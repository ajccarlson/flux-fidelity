import assert from "node:assert/strict";
import test from "node:test";

import {
  CDA_TEMPORAL_LOGICAL_BYTES,
  createCdaTemporalTilingProfile,
  deriveCdaTemporalHalo,
  planTemporalNeuralTiling,
} from "../src/core/fsrcnnx-neural-temporal-tiling.js";

const MIB = 1024 * 1024;

function limits(overrides = {}) {
  return {
    maxBufferSize: 256 * MIB,
    maxStorageBufferBindingSize: 128 * MIB,
    maxTextureDimension2D: 8192,
    maxTextureArrayLayers: 256,
    maxComputeWorkgroupsPerDimension: 65_535,
    ...overrides,
  };
}

function cda(bytes = CDA_TEMPORAL_LOGICAL_BYTES.mixed, options = {}) {
  return createCdaTemporalTilingProfile({
    largestLogicalBytesPerSourcePixel: bytes,
    ...options,
  });
}

test("CDA halo covers initializer and bounded recurrent dependency radii", () => {
  assert.equal(deriveCdaTemporalHalo(0), 64);
  assert.equal(deriveCdaTemporalHalo(8), 64);
  assert.equal(deriveCdaTemporalHalo(29), 64);
  assert.equal(deriveCdaTemporalHalo(30), 72);
  assert.equal(deriveCdaTemporalHalo(32), 72);
  assert.throws(() => deriveCdaTemporalHalo(-1), /non-negative safe integer/);
});

test("tiny odd and non-square sources remain one clipped real-edge tile", () => {
  const plan = planTemporalNeuralTiling(13, 7, cda(), limits());
  assert.equal(plan.outputWidth, 52);
  assert.equal(plan.outputHeight, 28);
  assert.equal(plan.tiles.length, 1);
  assert.deepEqual(plan.tiles[0], {
    index: 0,
    row: 0,
    column: 0,
    coreX: 0,
    coreY: 0,
    coreWidth: 13,
    coreHeight: 7,
    inputX: 0,
    inputY: 0,
    inputWidth: 13,
    inputHeight: 7,
    stateCropX: 0,
    stateCropY: 0,
    cropX: 0,
    cropY: 0,
    dstX: 0,
    dstY: 0,
    outWidth: 52,
    outHeight: 28,
    logicalBytes: 46_592,
  });
  assert.equal(Object.isFrozen(plan), true);
  assert.equal(Object.isFrozen(plan.profile), true);
  assert.equal(Object.isFrozen(plan.limits), true);
  assert.equal(Object.isFrozen(plan.tiles), true);
  assert.equal(Object.isFrozen(plan.tiles[0]), true);
});

test("a source that fits the planned input extent is not split unnecessarily", () => {
  const plan = planTemporalNeuralTiling(450, 173, cda(), limits());
  assert.equal(plan.maxInputExtent, 512);
  assert.equal(plan.coreExtent, 450);
  assert.equal(plan.tiles.length, 1);
  assert.deepEqual(
    [plan.tiles[0].inputWidth, plan.tiles[0].inputHeight],
    [450, 173],
  );
});

test("cores own every source pixel exactly once and halos are clipped", () => {
  const width = 1001;
  const height = 777;
  const plan = planTemporalNeuralTiling(width, height, cda(), limits());
  const ownership = new Uint8Array(width * height);

  for (const tile of plan.tiles) {
    assert.ok(tile.inputX >= 0);
    assert.ok(tile.inputY >= 0);
    assert.ok(tile.inputX + tile.inputWidth <= width);
    assert.ok(tile.inputY + tile.inputHeight <= height);
    assert.ok(tile.inputWidth <= plan.maxInputExtent);
    assert.ok(tile.inputHeight <= plan.maxInputExtent);
    assert.equal(tile.cropX, tile.stateCropX * 4);
    assert.equal(tile.cropY, tile.stateCropY * 4);
    assert.equal(tile.dstX, tile.coreX * 4);
    assert.equal(tile.dstY, tile.coreY * 4);
    assert.equal(tile.outWidth, tile.coreWidth * 4);
    assert.equal(tile.outHeight, tile.coreHeight * 4);
    for (let y = tile.coreY; y < tile.coreY + tile.coreHeight; y++) {
      for (let x = tile.coreX; x < tile.coreX + tile.coreWidth; x++) {
        ownership[y * width + x]++;
      }
    }
  }

  assert.equal(
    ownership.every((owners) => owners === 1),
    true,
    "every source pixel has one and only one core owner",
  );
  const interior = plan.tiles.find(({ row, column }) => row === 1 && column === 1);
  assert.ok(interior);
  assert.equal(interior.inputX, interior.coreX - 64);
  assert.equal(interior.inputY, interior.coreY - 64);
  assert.equal(interior.stateCropX, 64);
  assert.equal(interior.stateCropY, 64);
  assert.equal(interior.cropX, 256);
  assert.equal(interior.cropY, 256);
});

test("128 MiB limits select bounded FP32 and mixed input extents", () => {
  const fp32 = planTemporalNeuralTiling(
    1280,
    720,
    cda(CDA_TEMPORAL_LOGICAL_BYTES.fp32),
    limits(),
  );
  const mixed = planTemporalNeuralTiling(
    1280,
    720,
    cda(CDA_TEMPORAL_LOGICAL_BYTES.mixed),
    limits(),
  );

  assert.equal(fp32.hardInputExtent, 415);
  assert.equal(fp32.maxInputExtent, 408);
  assert.equal(fp32.coreExtent, 280);
  assert.ok(fp32.maxTileLogicalBytes <= 128 * MIB);

  assert.equal(mixed.hardInputExtent, 512);
  assert.equal(mixed.maxInputExtent, 512);
  assert.equal(mixed.coreExtent, 384);
  assert.ok(mixed.maxTileLogicalBytes <= 128 * MIB);
  assert.ok(mixed.tiles.length < fp32.tiles.length);
});

test("the smaller of max buffer and binding limits controls the plan", () => {
  const plan = planTemporalNeuralTiling(
    1280,
    720,
    cda(CDA_TEMPORAL_LOGICAL_BYTES.mixed),
    limits({
      maxBufferSize: 64 * MIB,
      maxStorageBufferBindingSize: 128 * MIB,
    }),
  );
  assert.equal(plan.bindingLimit, 64 * MIB);
  assert.equal(plan.hardInputExtent, 362);
  assert.equal(plan.maxInputExtent, 360);
  assert.equal(plan.coreExtent, 232);
  assert.ok(plan.maxTileLogicalBytes <= 64 * MIB);
  assert.ok(plan.maxInputExtent < 512);
});

test("impossible tensor limits fail physically without changing the profile", () => {
  const bytes = 128 * 128 * CDA_TEMPORAL_LOGICAL_BYTES.fp32;
  assert.throws(
    () => planTemporalNeuralTiling(
      256,
      256,
      cda(CDA_TEMPORAL_LOGICAL_BYTES.fp32),
      limits({
        maxBufferSize: bytes,
        maxStorageBufferBindingSize: bytes,
      }),
    ),
    (error) => error?.code === "NEURAL_LIMIT" && /cannot contain both/.test(error.message),
  );
});

test("physical texture, layer, output, and dispatch limits are explicit", () => {
  const profile = cda();
  assert.throws(
    () => planTemporalNeuralTiling(
      8193,
      1,
      profile,
      limits(),
    ),
    (error) => error?.code === "NEURAL_LIMIT" && /source dimensions/.test(error.message),
  );
  assert.throws(
    () => planTemporalNeuralTiling(
      2049,
      1,
      profile,
      limits(),
    ),
    (error) => error?.code === "NEURAL_LIMIT" && /output dimensions/.test(error.message),
  );
  assert.throws(
    () => planTemporalNeuralTiling(
      16,
      16,
      profile,
      limits({ maxTextureArrayLayers: 15 }),
    ),
    (error) => error?.code === "NEURAL_LIMIT" && /requires 16 layers/.test(error.message),
  );
  assert.throws(
    () => planTemporalNeuralTiling(
      17,
      16,
      profile,
      limits({ maxComputeWorkgroupsPerDimension: 2 }),
    ),
    (error) => error?.code === "NEURAL_LIMIT" && /source dispatch/.test(error.message),
  );
});

test("safe-integer and contract validation fail deterministically", () => {
  assert.throws(
    () => planTemporalNeuralTiling(0, 1, cda(), limits()),
    /positive safe integer/,
  );
  assert.throws(
    () => planTemporalNeuralTiling(1.5, 1, cda(), limits()),
    /positive safe integer/,
  );
  assert.throws(
    () => planTemporalNeuralTiling(
      Number.MAX_SAFE_INTEGER,
      1,
      cda(),
      limits({ maxTextureDimension2D: Number.MAX_SAFE_INTEGER }),
    ),
    (error) => error?.code === "NEURAL_LIMIT" && /safe integer range/.test(error.message),
  );
  assert.throws(
    () => planTemporalNeuralTiling(
      16,
      16,
      { ...cda(), preferredInputExtent: 128 },
      limits(),
    ),
    /must leave a positive core/,
  );
});

test("larger sources keep the selected profile and extent and only add tiles", () => {
  const profile = cda();
  const small = planTemporalNeuralTiling(640, 360, profile, limits());
  const large = planTemporalNeuralTiling(1920, 1080, profile, limits());

  assert.deepEqual(large.profile, small.profile);
  assert.equal(large.halo, small.halo);
  assert.equal(large.scale, small.scale);
  assert.equal(large.maxInputExtent, small.maxInputExtent);
  assert.equal(large.coreExtent, small.coreExtent);
  assert.ok(large.tiles.length > small.tiles.length);
});
