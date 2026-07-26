import assert from "node:assert/strict";
import test from "node:test";

import {
  TEMPORAL_ATLAS_UNIFORM_BYTES,
  TEMPORAL_ATLAS_WORKGROUP_SIZE,
  TEMPORAL_AUXILIARY_PACK_WGSL,
  buildTemporalAuxiliaryPackUniform,
  buildTemporalStatePackUniform,
  buildTemporalStateUnpackUniform,
  createTemporalStateAtlasShaders,
  createTemporalStateAtlasSpec,
  normalizeTemporalAtlasTileGeometry,
  temporalAuxiliaryPackAddress,
  temporalPlanarIndex,
  temporalStateChannelAddress,
  temporalStatePackAddress,
  temporalStateUnpackAddress,
} from "../src/core/fsrcnnx-neural-temporal-atlas.js";

const oddTile = Object.freeze({
  sourceWidth: 11,
  sourceHeight: 9,
  inputX: 2,
  inputY: 1,
  inputWidth: 7,
  inputHeight: 6,
  coreX: 3,
  coreY: 2,
  coreWidth: 5,
  coreHeight: 3,
});

test("state dtype selects an exact four-channel atlas format", () => {
  assert.deepEqual(createTemporalStateAtlasSpec("float16", 64), {
    dtype: "float16",
    channels: 64,
    layers: 16,
    elementBytes: 2,
    textureFormat: "rgba16float",
    wgslType: "f16",
  });
  assert.deepEqual(createTemporalStateAtlasSpec("float32", 12), {
    dtype: "float32",
    channels: 12,
    layers: 3,
    elementBytes: 4,
    textureFormat: "rgba32float",
    wgslType: "f32",
  });
  assert.equal(
    Object.isFrozen(createTemporalStateAtlasSpec("float16", 4)),
    true,
  );
  assert.throws(
    () => createTemporalStateAtlasSpec("int32", 64),
    /float16.*float32/,
  );
  assert.throws(
    () => createTemporalStateAtlasSpec("float32", 6),
    /divisible by four/,
  );
  assert.throws(
    () => createTemporalStateAtlasSpec("float32", 0),
    /positive u32/,
  );
});

test("float16 and float32 state shaders use only their audited WGSL types", () => {
  const fp16 = createTemporalStateAtlasShaders(
    createTemporalStateAtlasSpec("float16", 64),
  );
  const fp32 = createTemporalStateAtlasShaders(
    createTemporalStateAtlasSpec("float32", 64),
  );

  for (const source of [fp16.pack, fp16.unpack, fp32.pack, fp32.unpack]) {
    assert.match(source, /@workgroup_size\(8, 8, 1\)/);
    assert.match(source, /gid\.z/);
  }
  assert.match(fp16.pack, /^enable f16;/);
  assert.match(fp16.unpack, /^enable f16;/);
  assert.match(fp16.pack, /tile_state: array<f16>/);
  assert.match(fp16.unpack, /tile_state: array<f16>/);
  assert.match(fp16.unpack, /texture_storage_2d_array<rgba16float, write>/);
  assert.doesNotMatch(fp32.pack, /enable f16/);
  assert.doesNotMatch(fp32.unpack, /enable f16/);
  assert.match(fp32.pack, /tile_state: array<f32>/);
  assert.match(fp32.unpack, /tile_state: array<f32>/);
  assert.match(fp32.unpack, /texture_storage_2d_array<rgba32float, write>/);
  assert.match(fp16.pack, /texture_2d_array<f32>/);
  assert.match(fp16.unpack, /textureStore\(candidate_atlas/);
  assert.doesNotMatch(fp16.unpack, /mix\s*\(/);
  assert.equal(Object.isFrozen(fp16), true);
});

test("state and auxiliary uniforms have exact stable byte layouts", () => {
  const spec = createTemporalStateAtlasSpec("float32", 12);
  const pack = buildTemporalStatePackUniform(oddTile, spec);
  const unpack = buildTemporalStateUnpackUniform(oddTile, spec);
  const auxiliary = buildTemporalAuxiliaryPackUniform(oddTile, 2);

  assert.deepEqual([...pack], [7, 6, 2, 1, 3, 0, 0, 0]);
  assert.deepEqual(
    [...unpack],
    [7, 6, 1, 1, 3, 2, 5, 3, 3, 0, 0, 0],
  );
  assert.deepEqual([...auxiliary], [11, 9, 2, 1, 7, 6, 2, 0]);
  assert.equal(pack.byteLength, TEMPORAL_ATLAS_UNIFORM_BYTES.statePack);
  assert.equal(unpack.byteLength, TEMPORAL_ATLAS_UNIFORM_BYTES.stateUnpack);
  assert.equal(
    auxiliary.byteLength,
    TEMPORAL_ATLAS_UNIFORM_BYTES.auxiliaryPack,
  );
  assert.equal(TEMPORAL_ATLAS_WORKGROUP_SIZE, 8);
});

test("state pack maps array layers and RGBA components to planar NCHW", () => {
  const spec = createTemporalStateAtlasSpec("float16", 12);
  const plane = oddTile.inputWidth * oddTile.inputHeight;
  for (const channel of [0, 3, 4, 7, 8, 11]) {
    const address = temporalStatePackAddress(
      oddTile,
      spec,
      channel,
      5,
      4,
    );
    assert.deepEqual(address, {
      atlasX: 7,
      atlasY: 5,
      layer: Math.floor(channel / 4),
      component: channel % 4,
      bufferIndex: channel * plane + 4 * oddTile.inputWidth + 5,
    });
    assert.deepEqual(
      temporalStateChannelAddress(spec, channel),
      { layer: Math.floor(channel / 4), component: channel % 4 },
    );
  }
  assert.equal(temporalPlanarIndex(11, 5, 4, 7, 6, 12), 11 * plane + 33);
});

test("state unpack reads the halo-offset core and writes only global core pixels", () => {
  const spec = createTemporalStateAtlasSpec("float32", 8);
  const destinations = new Set();
  for (let channel = 0; channel < spec.channels; channel++) {
    for (let y = 0; y < oddTile.coreHeight; y++) {
      for (let x = 0; x < oddTile.coreWidth; x++) {
        const address = temporalStateUnpackAddress(
          oddTile,
          spec,
          channel,
          x,
          y,
        );
        assert.equal(address.tileX, 1 + x);
        assert.equal(address.tileY, 1 + y);
        assert.equal(address.atlasX, oddTile.coreX + x);
        assert.equal(address.atlasY, oddTile.coreY + y);
        assert.ok(address.atlasX >= oddTile.coreX);
        assert.ok(address.atlasX < oddTile.coreX + oddTile.coreWidth);
        assert.ok(address.atlasY >= oddTile.coreY);
        assert.ok(address.atlasY < oddTile.coreY + oddTile.coreHeight);
        destinations.add(
          `${channel}:${address.atlasX}:${address.atlasY}`,
        );
      }
    }
  }
  assert.equal(
    destinations.size,
    spec.channels * oddTile.coreWidth * oddTile.coreHeight,
  );
  assert.throws(
    () => temporalStateUnpackAddress(
      oddTile,
      spec,
      0,
      oddTile.coreWidth,
      0,
    ),
    /outside the owned core/,
  );
});

test("FP32 auxiliary packing copies the matching full-frame planar subregion", () => {
  assert.match(
    TEMPORAL_AUXILIARY_PACK_WGSL,
    /source_tensor: array<f32>/,
  );
  assert.match(
    TEMPORAL_AUXILIARY_PACK_WGSL,
    /tile_tensor: array<f32>/,
  );
  assert.match(
    TEMPORAL_AUXILIARY_PACK_WGSL,
    /@workgroup_size\(8, 8, 1\)/,
  );
  assert.match(TEMPORAL_AUXILIARY_PACK_WGSL, /gid\.z/);
  assert.doesNotMatch(TEMPORAL_AUXILIARY_PACK_WGSL, /texture/);

  const address = temporalAuxiliaryPackAddress(
    oddTile,
    2,
    1,
    5,
    4,
  );
  assert.deepEqual(address, {
    sourceX: 7,
    sourceY: 5,
    sourceIndex: 11 * 9 + 5 * 11 + 7,
    tileIndex: 7 * 6 + 4 * 7 + 5,
  });
});

test("real-edge clipping uses only the declared source rectangle", () => {
  const edge = normalizeTemporalAtlasTileGeometry({
    sourceWidth: 13,
    sourceHeight: 7,
    inputX: 0,
    inputY: 0,
    inputWidth: 13,
    inputHeight: 7,
    coreX: 0,
    coreY: 0,
    coreWidth: 9,
    coreHeight: 5,
  });
  assert.equal(edge.stateCropX, 0);
  assert.equal(edge.stateCropY, 0);
  assert.deepEqual(
    temporalStatePackAddress(
      edge,
      createTemporalStateAtlasSpec("float32", 4),
      3,
      12,
      6,
    ),
    {
      atlasX: 12,
      atlasY: 6,
      layer: 0,
      component: 3,
      bufferIndex: 3 * 13 * 7 + 6 * 13 + 12,
    },
  );
});

test("invalid channels and geometry are rejected before uniform creation", () => {
  const spec = createTemporalStateAtlasSpec("float32", 4);
  assert.throws(
    () => normalizeTemporalAtlasTileGeometry({
      ...oddTile,
      inputX: 5,
      inputWidth: 7,
    }),
    /input rectangle exceeds/,
  );
  assert.throws(
    () => normalizeTemporalAtlasTileGeometry({
      ...oddTile,
      coreX: 1,
    }),
    /core must be contained/,
  );
  assert.throws(
    () => normalizeTemporalAtlasTileGeometry({
      ...oddTile,
      coreWidth: 0,
    }),
    /positive u32/,
  );
  assert.throws(
    () => buildTemporalAuxiliaryPackUniform(oddTile, 0),
    /positive u32/,
  );
  assert.throws(
    () => temporalStatePackAddress(oddTile, spec, 4, 0, 0),
    /outside the atlas/,
  );
  assert.throws(
    () => temporalAuxiliaryPackAddress(oddTile, 2, 0, 7, 0),
    /outside the input tile/,
  );
  assert.throws(
    () => normalizeTemporalAtlasTileGeometry({
      sourceWidth: 65_536,
      sourceHeight: 65_536,
      inputX: 0,
      inputY: 0,
      inputWidth: 1,
      inputHeight: 1,
      coreX: 0,
      coreY: 0,
      coreWidth: 1,
      coreHeight: 1,
    }),
    /source plane exceeds the u32 range/,
  );
});
