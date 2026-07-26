// Shader ABI helpers for temporal state-atlas tiling.
//
// State textures pack four NCHW channels per array layer. Pack passes copy a
// clipped real-image tile into one whole ORT GPUBuffer; unpack passes copy only
// the tile's exact-owned core into a global candidate atlas. No pass blends,
// pads, or writes outside that core.

export const TEMPORAL_ATLAS_WORKGROUP_SIZE = 8;
export const TEMPORAL_ATLAS_UNIFORM_BYTES = Object.freeze({
  statePack: 32,
  stateUnpack: 48,
  auxiliaryPack: 32,
});

const MAX_U32 = 0xffff_ffff;
const STATE_DTYPES = Object.freeze({
  float16: Object.freeze({
    elementBytes: 2,
    textureFormat: "rgba16float",
    wgslType: "f16",
  }),
  float32: Object.freeze({
    elementBytes: 4,
    textureFormat: "rgba32float",
    wgslType: "f32",
  }),
});

function objectValue(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value;
}

function u32(value, label, { positive = false } = {}) {
  const minimum = positive ? 1 : 0;
  if (!Number.isSafeInteger(value) || value < minimum || value > MAX_U32) {
    throw new Error(
      `${label} must be ${positive ? "a positive" : "a non-negative"} u32 integer`,
    );
  }
  return value;
}

function checkedAdd(left, right, label) {
  u32(left, label);
  u32(right, label);
  const sum = left + right;
  if (!Number.isSafeInteger(sum) || sum > MAX_U32) {
    throw new Error(`${label} exceeds the u32 range`);
  }
  return sum;
}

function checkedProduct(values, label) {
  let product = 1;
  for (const value of values) {
    u32(value, label, { positive: true });
    if (product > Math.floor(MAX_U32 / value)) {
      throw new Error(`${label} exceeds the u32 range`);
    }
    product *= value;
  }
  return product;
}

export function createTemporalStateAtlasSpec(dtype, channels) {
  const selected = STATE_DTYPES[dtype];
  if (!selected) {
    throw new Error(
      `temporal state dtype must be 'float16' or 'float32' (received '${dtype}')`,
    );
  }
  u32(channels, "temporal state channels", { positive: true });
  if (channels % 4 !== 0) {
    throw new Error("temporal state channels must be divisible by four");
  }
  return Object.freeze({
    dtype,
    channels,
    layers: channels / 4,
    elementBytes: selected.elementBytes,
    textureFormat: selected.textureFormat,
    wgslType: selected.wgslType,
  });
}

function normalizeSpec(value) {
  objectValue(value, "temporal state atlas spec");
  return createTemporalStateAtlasSpec(value.dtype, value.channels);
}

export function normalizeTemporalAtlasTileGeometry(value) {
  objectValue(value, "temporal atlas tile geometry");
  const sourceWidth = u32(
    value.sourceWidth,
    "temporal atlas sourceWidth",
    { positive: true },
  );
  const sourceHeight = u32(
    value.sourceHeight,
    "temporal atlas sourceHeight",
    { positive: true },
  );
  const inputX = u32(value.inputX, "temporal atlas inputX");
  const inputY = u32(value.inputY, "temporal atlas inputY");
  const inputWidth = u32(
    value.inputWidth,
    "temporal atlas inputWidth",
    { positive: true },
  );
  const inputHeight = u32(
    value.inputHeight,
    "temporal atlas inputHeight",
    { positive: true },
  );
  const coreX = u32(value.coreX, "temporal atlas coreX");
  const coreY = u32(value.coreY, "temporal atlas coreY");
  const coreWidth = u32(
    value.coreWidth,
    "temporal atlas coreWidth",
    { positive: true },
  );
  const coreHeight = u32(
    value.coreHeight,
    "temporal atlas coreHeight",
    { positive: true },
  );
  const inputRight = checkedAdd(
    inputX,
    inputWidth,
    "temporal atlas input right",
  );
  const inputBottom = checkedAdd(
    inputY,
    inputHeight,
    "temporal atlas input bottom",
  );
  const coreRight = checkedAdd(
    coreX,
    coreWidth,
    "temporal atlas core right",
  );
  const coreBottom = checkedAdd(
    coreY,
    coreHeight,
    "temporal atlas core bottom",
  );
  if (inputRight > sourceWidth || inputBottom > sourceHeight) {
    throw new Error("temporal atlas input rectangle exceeds the source");
  }
  if (coreRight > sourceWidth || coreBottom > sourceHeight) {
    throw new Error("temporal atlas core rectangle exceeds the source");
  }
  if (coreX < inputX || coreY < inputY ||
      coreRight > inputRight || coreBottom > inputBottom) {
    throw new Error("temporal atlas core must be contained by the input rectangle");
  }
  checkedProduct(
    [sourceWidth, sourceHeight],
    "temporal atlas source plane",
  );
  checkedProduct(
    [inputWidth, inputHeight],
    "temporal atlas tile plane",
  );
  return Object.freeze({
    sourceWidth,
    sourceHeight,
    inputX,
    inputY,
    inputWidth,
    inputHeight,
    coreX,
    coreY,
    coreWidth,
    coreHeight,
    stateCropX: coreX - inputX,
    stateCropY: coreY - inputY,
  });
}

function normalizeChannels(channels, label) {
  return u32(channels, label, { positive: true });
}

export function buildTemporalStatePackUniform(geometry, declaredSpec) {
  const tile = normalizeTemporalAtlasTileGeometry(geometry);
  const spec = normalizeSpec(declaredSpec);
  checkedProduct(
    [tile.inputWidth, tile.inputHeight, spec.channels],
    "temporal state tile elements",
  );
  return new Uint32Array([
    tile.inputWidth,
    tile.inputHeight,
    tile.inputX,
    tile.inputY,
    spec.layers,
    0,
    0,
    0,
  ]);
}

export function buildTemporalStateUnpackUniform(geometry, declaredSpec) {
  const tile = normalizeTemporalAtlasTileGeometry(geometry);
  const spec = normalizeSpec(declaredSpec);
  checkedProduct(
    [tile.inputWidth, tile.inputHeight, spec.channels],
    "temporal state tile elements",
  );
  return new Uint32Array([
    tile.inputWidth,
    tile.inputHeight,
    tile.stateCropX,
    tile.stateCropY,
    tile.coreX,
    tile.coreY,
    tile.coreWidth,
    tile.coreHeight,
    spec.layers,
    0,
    0,
    0,
  ]);
}

export function buildTemporalAuxiliaryPackUniform(geometry, channels) {
  const tile = normalizeTemporalAtlasTileGeometry(geometry);
  const channelCount = normalizeChannels(
    channels,
    "temporal auxiliary channels",
  );
  checkedProduct(
    [tile.sourceWidth, tile.sourceHeight, channelCount],
    "temporal auxiliary source elements",
  );
  checkedProduct(
    [tile.inputWidth, tile.inputHeight, channelCount],
    "temporal auxiliary tile elements",
  );
  return new Uint32Array([
    tile.sourceWidth,
    tile.sourceHeight,
    tile.inputX,
    tile.inputY,
    tile.inputWidth,
    tile.inputHeight,
    channelCount,
    0,
  ]);
}

export function temporalPlanarIndex(
  channel,
  x,
  y,
  width,
  height,
  channels,
) {
  const channelCount = normalizeChannels(channels, "temporal planar channels");
  u32(channel, "temporal planar channel");
  u32(x, "temporal planar x");
  u32(y, "temporal planar y");
  u32(width, "temporal planar width", { positive: true });
  u32(height, "temporal planar height", { positive: true });
  if (channel >= channelCount) {
    throw new Error("temporal planar channel is outside the tensor");
  }
  if (x >= width || y >= height) {
    throw new Error("temporal planar coordinate is outside the tensor");
  }
  const plane = checkedProduct(
    [width, height],
    "temporal planar tensor plane",
  );
  const channelOffset = channel * plane;
  const rowOffset = y * width;
  const index = channelOffset + rowOffset + x;
  if (!Number.isSafeInteger(index) || index > MAX_U32) {
    throw new Error("temporal planar index exceeds the u32 range");
  }
  return index;
}

export function temporalStateChannelAddress(declaredSpec, channel) {
  const spec = normalizeSpec(declaredSpec);
  u32(channel, "temporal state channel");
  if (channel >= spec.channels) {
    throw new Error("temporal state channel is outside the atlas");
  }
  return Object.freeze({
    layer: Math.floor(channel / 4),
    component: channel % 4,
  });
}

export function temporalStatePackAddress(
  geometry,
  declaredSpec,
  channel,
  tileX,
  tileY,
) {
  const tile = normalizeTemporalAtlasTileGeometry(geometry);
  const spec = normalizeSpec(declaredSpec);
  u32(tileX, "temporal state pack x");
  u32(tileY, "temporal state pack y");
  if (tileX >= tile.inputWidth || tileY >= tile.inputHeight) {
    throw new Error("temporal state pack coordinate is outside the input tile");
  }
  const { layer, component } = temporalStateChannelAddress(spec, channel);
  return Object.freeze({
    atlasX: tile.inputX + tileX,
    atlasY: tile.inputY + tileY,
    layer,
    component,
    bufferIndex: temporalPlanarIndex(
      channel,
      tileX,
      tileY,
      tile.inputWidth,
      tile.inputHeight,
      spec.channels,
    ),
  });
}

export function temporalStateUnpackAddress(
  geometry,
  declaredSpec,
  channel,
  coreLocalX,
  coreLocalY,
) {
  const tile = normalizeTemporalAtlasTileGeometry(geometry);
  const spec = normalizeSpec(declaredSpec);
  u32(coreLocalX, "temporal state unpack x");
  u32(coreLocalY, "temporal state unpack y");
  if (coreLocalX >= tile.coreWidth || coreLocalY >= tile.coreHeight) {
    throw new Error("temporal state unpack coordinate is outside the owned core");
  }
  const tileX = tile.stateCropX + coreLocalX;
  const tileY = tile.stateCropY + coreLocalY;
  const { layer, component } = temporalStateChannelAddress(spec, channel);
  return Object.freeze({
    tileX,
    tileY,
    atlasX: tile.coreX + coreLocalX,
    atlasY: tile.coreY + coreLocalY,
    layer,
    component,
    bufferIndex: temporalPlanarIndex(
      channel,
      tileX,
      tileY,
      tile.inputWidth,
      tile.inputHeight,
      spec.channels,
    ),
  });
}

export function temporalAuxiliaryPackAddress(
  geometry,
  channels,
  channel,
  tileX,
  tileY,
) {
  const tile = normalizeTemporalAtlasTileGeometry(geometry);
  const channelCount = normalizeChannels(
    channels,
    "temporal auxiliary channels",
  );
  u32(tileX, "temporal auxiliary pack x");
  u32(tileY, "temporal auxiliary pack y");
  if (tileX >= tile.inputWidth || tileY >= tile.inputHeight) {
    throw new Error("temporal auxiliary coordinate is outside the input tile");
  }
  return Object.freeze({
    sourceX: tile.inputX + tileX,
    sourceY: tile.inputY + tileY,
    sourceIndex: temporalPlanarIndex(
      channel,
      tile.inputX + tileX,
      tile.inputY + tileY,
      tile.sourceWidth,
      tile.sourceHeight,
      channelCount,
    ),
    tileIndex: temporalPlanarIndex(
      channel,
      tileX,
      tileY,
      tile.inputWidth,
      tile.inputHeight,
      channelCount,
    ),
  });
}

function statePackShader(spec) {
  const enable = spec.dtype === "float16" ? "enable f16;\n" : "";
  const cast = spec.dtype === "float16" ? "f16" : "f32";
  return `${enable}struct StatePackParams {
  tile_width: u32,
  tile_height: u32,
  input_x: u32,
  input_y: u32,
  layer_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var source_atlas: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read_write> tile_state: array<${spec.wgslType}>;
@group(0) @binding(2) var<uniform> params: StatePackParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.tile_width ||
      gid.y >= params.tile_height ||
      gid.z >= params.layer_count) {
    return;
  }
  let atlas_position = vec2<i32>(
    i32(params.input_x + gid.x),
    i32(params.input_y + gid.y),
  );
  let value = textureLoad(source_atlas, atlas_position, i32(gid.z), 0);
  let plane = params.tile_width * params.tile_height;
  let pixel = gid.y * params.tile_width + gid.x;
  let base = gid.z * 4u * plane + pixel;
  tile_state[base] = ${cast}(value.x);
  tile_state[base + plane] = ${cast}(value.y);
  tile_state[base + 2u * plane] = ${cast}(value.z);
  tile_state[base + 3u * plane] = ${cast}(value.w);
}`;
}

function stateUnpackShader(spec) {
  const enable = spec.dtype === "float16" ? "enable f16;\n" : "";
  return `${enable}struct StateUnpackParams {
  tile_width: u32,
  tile_height: u32,
  state_crop_x: u32,
  state_crop_y: u32,
  core_x: u32,
  core_y: u32,
  core_width: u32,
  core_height: u32,
  layer_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> tile_state: array<${spec.wgslType}>;
@group(0) @binding(1) var candidate_atlas:
  texture_storage_2d_array<${spec.textureFormat}, write>;
@group(0) @binding(2) var<uniform> params: StateUnpackParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.core_width ||
      gid.y >= params.core_height ||
      gid.z >= params.layer_count) {
    return;
  }
  let tile_x = params.state_crop_x + gid.x;
  let tile_y = params.state_crop_y + gid.y;
  let plane = params.tile_width * params.tile_height;
  let pixel = tile_y * params.tile_width + tile_x;
  let base = gid.z * 4u * plane + pixel;
  let value = vec4<f32>(
    f32(tile_state[base]),
    f32(tile_state[base + plane]),
    f32(tile_state[base + 2u * plane]),
    f32(tile_state[base + 3u * plane]),
  );
  let atlas_position = vec2<i32>(
    i32(params.core_x + gid.x),
    i32(params.core_y + gid.y),
  );
  textureStore(candidate_atlas, atlas_position, i32(gid.z), value);
}`;
}

export function createTemporalStateAtlasShaders(declaredSpec) {
  const spec = normalizeSpec(declaredSpec);
  return Object.freeze({
    pack: statePackShader(spec),
    unpack: stateUnpackShader(spec),
  });
}

export const TEMPORAL_AUXILIARY_PACK_WGSL = `struct AuxiliaryPackParams {
  source_width: u32,
  source_height: u32,
  input_x: u32,
  input_y: u32,
  tile_width: u32,
  tile_height: u32,
  channel_count: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read> source_tensor: array<f32>;
@group(0) @binding(1) var<storage, read_write> tile_tensor: array<f32>;
@group(0) @binding(2) var<uniform> params: AuxiliaryPackParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.tile_width ||
      gid.y >= params.tile_height ||
      gid.z >= params.channel_count) {
    return;
  }
  let source_plane = params.source_width * params.source_height;
  let source_pixel =
    (params.input_y + gid.y) * params.source_width + params.input_x + gid.x;
  let tile_plane = params.tile_width * params.tile_height;
  let tile_pixel = gid.y * params.tile_width + gid.x;
  let source_index = gid.z * source_plane + source_pixel;
  let tile_index = gid.z * tile_plane + tile_pixel;
  tile_tensor[tile_index] = source_tensor[source_index];
}`;
