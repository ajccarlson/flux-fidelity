// fsrcnnx-neural.js — tiled ONNX neural upscaler.
//
// Runs a vetted bundled RGB super-resolution model on an ORT WebGPU device.
// Production hosts this engine in an extension-owned frame, where Neural output
// bypasses luma/chroma recombination and enters the frame's SSimDS/sharpen
// presentation tail as rgba16float.
//
// The inference core stays GPU-resident after its source upload; the isolated
// frame adapter performs CPU staging only at the renderer-process boundaries:
//   snapshot the expiring external/texture source before any await
//   -> split the snapshot into bounded, overlapping core tiles
//   -> pack each clipped halo into dynamic NCHW FP32 and run ORT sequentially
//   -> crop each tile's halo into one full rgba16float output texture
//   -> caller presents. The frame has no policy resolution ceiling; only the
//   adapter's unavoidable source/final-texture dimensions can reject it.
//
// Legacy v1 models retain that tiled single-graph ABI. Contract-v2 models may
// instead expose named full-frame spatial or reset/recurrent graph sessions,
// auxiliary GPU-buffer inputs, and recurrent GPU state. The model catalog lives
// in model/neural/manifest.json.

import {
  createOrtSession,
  ensureOrt,
  getOrtSessionDevice,
  resolvePackagedAssetUrl,
} from "./fsrcnnx-rife.js";

const NEURAL_KEY = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const NEURAL_FILE = /^[A-Za-z0-9][A-Za-z0-9._-]*\.onnx$/;
const TENSOR_NAME = /^[A-Za-z_][A-Za-z0-9_.-]{0,127}$/;
const STATE_KEY = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const GRAPH_KEY = /^[A-Za-z][A-Za-z0-9_-]{0,63}$/;
const V2_INPUT_ROLES = new Set([
  "rgb",
  "motion",
  "residual",
  "state-in",
  "reset",
  "auxiliary",
]);
const V2_OUTPUT_ROLES = new Set(["rgb", "state-out", "auxiliary"]);
const V2_DTYPES = new Set([
  "float16",
  "float32",
  "int32",
  "int64",
  "uint8",
  "bool",
]);
const V2_LAYOUTS = new Set(["nchw", "scalar"]);
const V2_AUXILIARY_PROVIDERS = new Set(["decoded-cda-v1"]);
const validatedContracts = new WeakMap();
export const DEFAULT_NEURAL_TILE_SIZE = 512;
export const DEFAULT_NEURAL_TILE_OVERLAP = 24;
const MIN_NEURAL_TILE_SIZE = 64;
const MAX_NEURAL_TILE_SIZE = 768;
const MIN_NEURAL_TILE_OVERLAP = 18;
const MAX_NEURAL_TILE_OVERLAP = 192;
const MAX_NEURAL_TILE_INPUT_EXTENT = 896;

export function isValidNeuralModelKey(value) {
  return typeof value === "string" && NEURAL_KEY.test(value);
}

function freezeTensorDescriptor(name, value, kind, at) {
  if (!TENSOR_NAME.test(name)) throw new Error(`${at} has an invalid ${kind} tensor name '${name}'`);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${at} ${kind} '${name}' must be an object`);
  }
  const roles = kind === "input" ? V2_INPUT_ROLES : V2_OUTPUT_ROLES;
  if (!roles.has(value.role)) {
    throw new Error(`${at} ${kind} '${name}' has an invalid role`);
  }
  if (!V2_DTYPES.has(value.dtype)) {
    throw new Error(`${at} ${kind} '${name}' has an invalid dtype`);
  }
  const layout = value.layout ?? (value.role === "reset" ? "scalar" : "nchw");
  if (!V2_LAYOUTS.has(layout)) {
    throw new Error(`${at} ${kind} '${name}' has an invalid layout`);
  }
  const channels = value.channels ?? (layout === "scalar" ? 1 : null);
  if (!Number.isInteger(channels) || channels < 1 || channels > 4096) {
    throw new Error(`${at} ${kind} '${name}' has invalid channels`);
  }
  if (layout === "scalar" && channels !== 1) {
    throw new Error(`${at} ${kind} '${name}' scalar layout must have one channel`);
  }
  if (value.role === "rgb" && (value.dtype !== "float32" || channels !== 3 || layout !== "nchw")) {
    throw new Error(`${at} ${kind} '${name}' RGB tensors must be float32 NCHW with three channels`);
  }
  if (value.role === "motion" && (channels !== 2 || layout !== "nchw")) {
    throw new Error(`${at} ${kind} '${name}' motion tensors must be NCHW with two channels`);
  }
  if (value.role === "residual" && (channels !== 1 || layout !== "nchw")) {
    throw new Error(`${at} ${kind} '${name}' residual tensors must be NCHW with one channel`);
  }
  if (value.role === "reset" && kind !== "input") {
    throw new Error(`${at} reset tensors may only be inputs`);
  }

  const isState = value.role === "state-in" || value.role === "state-out";
  if (isState) {
    if (typeof value.state !== "string" || !STATE_KEY.test(value.state)) {
      throw new Error(`${at} ${kind} '${name}' has an invalid state key`);
    }
    if (layout !== "nchw") {
      throw new Error(`${at} ${kind} '${name}' state tensors must use NCHW layout`);
    }
  } else if (value.state != null) {
    throw new Error(`${at} ${kind} '${name}' may not declare a state key`);
  }
  let provider;
  if (value.provider != null) {
    if ((value.role !== "motion" && value.role !== "residual") ||
        !V2_AUXILIARY_PROVIDERS.has(value.provider)) {
      throw new Error(`${at} ${kind} '${name}' has an invalid auxiliary provider`);
    }
    provider = value.provider;
  }

  let reset;
  if (value.role === "state-in") {
    reset = value.reset ?? "zeros";
    if (reset !== "zeros" && reset !== "required") {
      throw new Error(`${at} ${kind} '${name}' has an invalid reset policy`);
    }
  } else if (value.reset != null) {
    throw new Error(`${at} ${kind} '${name}' may not declare a reset policy`);
  }

  return Object.freeze({
    name,
    role: value.role,
    dtype: value.dtype,
    channels,
    layout,
    ...(isState ? { state: value.state } : {}),
    ...(reset ? { reset } : {}),
    ...(provider ? { provider } : {}),
  });
}

function freezeDescriptorMap(value, kind, at) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${at} ${kind}s must be a named descriptor map`);
  }
  const entries = Object.entries(value);
  if (!entries.length) throw new Error(`${at} must declare at least one ${kind}`);
  const descriptors = {};
  for (const [name, descriptor] of entries) {
    descriptors[name] = freezeTensorDescriptor(name, descriptor, kind, at);
  }
  return Object.freeze(descriptors);
}

function requireSingleRole(descriptors, role, kind, at) {
  const matches = Object.values(descriptors).filter((descriptor) => descriptor.role === role);
  if (matches.length !== 1) {
    throw new Error(`${at} must declare exactly one ${kind} with role '${role}'`);
  }
  return matches[0];
}

function stateDescriptors(descriptors, role) {
  return new Map(Object.values(descriptors)
    .filter((descriptor) => descriptor.role === role)
    .map((descriptor) => [descriptor.state, descriptor]));
}

function requireMatchingStateSets(left, right, at) {
  if (left.size !== right.size || [...left.keys()].some((key) => !right.has(key))) {
    throw new Error(`${at} state-in/state-out keys do not match`);
  }
  for (const [key, input] of left) {
    const output = right.get(key);
    if (input.dtype !== output.dtype ||
        input.channels !== output.channels ||
        input.layout !== output.layout) {
      throw new Error(`${at} state '${key}' input/output contracts do not match`);
    }
  }
}

function freezeV2Graph(name, value, at) {
  const graphAt = `${at} graph '${name}'`;
  if (!GRAPH_KEY.test(name)) throw new Error(`${at} has an invalid graph name '${name}'`);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${graphAt} must be an object`);
  }
  if (typeof value.file !== "string" || !NEURAL_FILE.test(value.file)) {
    throw new Error(`${graphAt} has an invalid model filename`);
  }
  const inputs = freezeDescriptorMap(value.inputs, "input", graphAt);
  const outputs = freezeDescriptorMap(value.outputs, "output", graphAt);
  requireSingleRole(inputs, "rgb", "input", graphAt);
  requireSingleRole(outputs, "rgb", "output", graphAt);
  const roleCounts = new Map();
  for (const descriptor of Object.values(inputs)) {
    if (descriptor.role === "auxiliary" || descriptor.role === "state-in") continue;
    roleCounts.set(descriptor.role, (roleCounts.get(descriptor.role) || 0) + 1);
  }
  for (const [role, count] of roleCounts) {
    if (count > 1) throw new Error(`${graphAt} declares ambiguous '${role}' inputs`);
  }
  const stateOut = stateDescriptors(outputs, "state-out");
  if (stateOut.size !== Object.values(outputs).filter(({ role }) => role === "state-out").length) {
    throw new Error(`${graphAt} duplicates a state-out key`);
  }
  const stateIn = stateDescriptors(inputs, "state-in");
  if (stateIn.size !== Object.values(inputs).filter(({ role }) => role === "state-in").length) {
    throw new Error(`${graphAt} duplicates a state-in key`);
  }
  return Object.freeze({ name, file: value.file, inputs, outputs });
}

export function normalizeNeuralModelContract(entry, at = "neural manifest entry") {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
    throw new Error(`${at} must be an object`);
  }
  const declared = entry.contract;
  if (declared == null || declared?.version === 1) {
    if (declared != null &&
        (!declared || typeof declared !== "object" || Array.isArray(declared))) {
      throw new Error(`${at} has an invalid v1 contract`);
    }
    return Object.freeze({
      version: 1,
      mode: "spatial",
      inputs: Object.freeze([Object.freeze({
        name: entry.input ?? null,
        role: "rgb",
        dtype: "float32",
        channels: 3,
        layout: "nchw",
      })]),
      outputs: Object.freeze([Object.freeze({
        name: entry.output ?? null,
        role: "rgb",
        dtype: "float32",
        channels: 3,
        layout: "nchw",
      })]),
      states: Object.freeze([]),
    });
  }
  if (!declared || typeof declared !== "object" || Array.isArray(declared) ||
      declared.version !== 2) {
    throw new Error(`${at} has an unsupported neural contract version`);
  }
  if (entry.file != null || entry.input != null || entry.output != null) {
    throw new Error(`${at} v2 contract may not mix legacy file/input/output fields`);
  }
  const mode = declared.mode;
  if (mode !== "spatial" && mode !== "temporal") {
    throw new Error(`${at} v2 contract has an invalid mode`);
  }
  if (!declared.graphs || typeof declared.graphs !== "object" || Array.isArray(declared.graphs)) {
    throw new Error(`${at} v2 contract graphs must be a named map`);
  }
  const graphEntries = Object.entries(declared.graphs);
  if (!graphEntries.length || graphEntries.length > 8) {
    throw new Error(`${at} v2 contract must declare between one and eight graphs`);
  }
  const graphs = {};
  for (const [name, graph] of graphEntries) graphs[name] = freezeV2Graph(name, graph, at);

  let resetGraph;
  let recurrentGraph;
  if (mode === "temporal") {
    resetGraph = declared.resetGraph ?? "initialize";
    recurrentGraph = declared.recurrentGraph ?? "recurrent";
    if (!Object.hasOwn(graphs, resetGraph) || !Object.hasOwn(graphs, recurrentGraph) ||
        resetGraph === recurrentGraph) {
      throw new Error(`${at} temporal contract must name distinct reset and recurrent graphs`);
    }
    const reset = graphs[resetGraph];
    const recurrent = graphs[recurrentGraph];
    const resetInputs = stateDescriptors(reset.inputs, "state-in");
    if (resetInputs.size) throw new Error(`${at} reset graph may not consume recurrent state`);
    const resetOutputs = stateDescriptors(reset.outputs, "state-out");
    const recurrentInputs = stateDescriptors(recurrent.inputs, "state-in");
    const recurrentOutputs = stateDescriptors(recurrent.outputs, "state-out");
    if (!resetOutputs.size) throw new Error(`${at} temporal contract must produce recurrent state`);
    requireMatchingStateSets(recurrentInputs, recurrentOutputs, `${at} recurrent graph`);
    requireMatchingStateSets(recurrentInputs, resetOutputs, `${at} reset/recurrent graphs`);
  } else {
    resetGraph = declared.resetGraph ?? graphEntries[0][0];
    recurrentGraph = null;
    if (!Object.hasOwn(graphs, resetGraph)) {
      throw new Error(`${at} spatial contract names an unknown graph`);
    }
    for (const graph of Object.values(graphs)) {
      if (stateDescriptors(graph.inputs, "state-in").size ||
          stateDescriptors(graph.outputs, "state-out").size) {
        throw new Error(`${at} spatial contract may not declare recurrent state`);
      }
    }
  }

  const stateKeys = mode === "temporal"
    ? [...stateDescriptors(graphs[recurrentGraph].inputs, "state-in").keys()]
    : [];
  return Object.freeze({
    version: 2,
    mode,
    resetGraph,
    recurrentGraph,
    graphs: Object.freeze(graphs),
    states: Object.freeze(stateKeys),
  });
}

export function neuralModelFiles(entry, at = "neural manifest entry") {
  const contract = normalizeNeuralModelContract(entry, at);
  return Object.freeze(contract.version === 1
    ? [entry.file]
    : Object.values(contract.graphs).map((graph) => graph.file));
}

export function validateNeuralManifest(value) {
  const raw = Array.isArray(value) ? value : value?.models;
  if (!Array.isArray(raw)) throw new Error("neural manifest must be an array or {models: array}");
  const keys = new Set();
  const files = new Set();
  return raw.map((entry, index) => {
    const at = `neural manifest entry ${index}`;
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) throw new Error(`${at} must be an object`);
    if (!isValidNeuralModelKey(entry.key)) throw new Error(`${at} has an invalid key`);
    if (keys.has(entry.key)) throw new Error(`${at} duplicates key '${entry.key}'`);
    if (!Number.isInteger(entry.scale) || entry.scale < 1 || entry.scale > 16) throw new Error(`${at} has an invalid scale`);
    if (entry.padMultiple != null &&
        (!Number.isInteger(entry.padMultiple) || entry.padMultiple < 1 || entry.padMultiple > 256)) {
      throw new Error(`${at} has an invalid padMultiple`);
    }
    if (entry.label != null && (typeof entry.label !== "string" || !entry.label.trim() || entry.label.length > 160)) {
      throw new Error(`${at} has an invalid label`);
    }
    for (const field of ["input", "output"]) {
      if (entry[field] != null && (typeof entry[field] !== "string" || !TENSOR_NAME.test(entry[field]))) {
        throw new Error(`${at} has an invalid ${field} tensor name`);
      }
    }
    const contract = normalizeNeuralModelContract(entry, at);
    if (contract.version === 2) {
      if (entry.padMultiple != null && entry.padMultiple !== 1) {
        throw new Error(`${at} v2 full-frame contract does not support padded input`);
      }
      if (entry.tileSize != null || entry.tileOverlap != null) {
        throw new Error(`${at} v2 full-frame contract may not declare tile controls`);
      }
    }
    const entryFiles = neuralModelFiles(entry, at);
    for (const file of entryFiles) {
      if (typeof file !== "string" || !NEURAL_FILE.test(file)) {
        throw new Error(`${at} has an invalid model filename`);
      }
      if (files.has(file)) throw new Error(`${at} duplicates model file '${file}'`);
      files.add(file);
    }
    if (entry.fp16 != null && typeof entry.fp16 !== "boolean") throw new Error(`${at} has an invalid fp16 flag`);
    const tileSize = entry.tileSize ?? DEFAULT_NEURAL_TILE_SIZE;
    const tileOverlap = entry.tileOverlap ?? DEFAULT_NEURAL_TILE_OVERLAP;
    if (!Number.isInteger(tileSize) ||
        tileSize < MIN_NEURAL_TILE_SIZE || tileSize > MAX_NEURAL_TILE_SIZE) {
      throw new Error(`${at} has an invalid tileSize`);
    }
    if (!Number.isInteger(tileOverlap) ||
        tileOverlap < MIN_NEURAL_TILE_OVERLAP || tileOverlap > MAX_NEURAL_TILE_OVERLAP ||
        tileSize + 2 * tileOverlap > MAX_NEURAL_TILE_INPUT_EXTENT) {
      throw new Error(`${at} has an invalid tileOverlap`);
    }
    keys.add(entry.key);
    const normalized = Object.freeze({
      ...entry,
      ...(contract.version === 2 ? { contract } : {}),
    });
    validatedContracts.set(normalized, contract);
    return normalized;
  });
}

function requirePositiveSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive safe integer`);
  }
  return value;
}

// Pure geometry helper kept public so the tile ABI can be checked without a
// GPU. Boundary halos are clipped to the real image: the model therefore sees
// its own native boundary padding at the same edges as whole-frame inference.
// Only padMultiple expansion replicates the last real pixel in the pack pass.
export function planNeuralTiles(srcW, srcH, {
  tileSize = DEFAULT_NEURAL_TILE_SIZE,
  tileOverlap = DEFAULT_NEURAL_TILE_OVERLAP,
  padMultiple = 1,
  scale = 1,
} = {}) {
  requirePositiveSafeInteger(srcW, "neural source width");
  requirePositiveSafeInteger(srcH, "neural source height");
  requirePositiveSafeInteger(scale, "neural scale");
  requirePositiveSafeInteger(padMultiple, "neural pad multiple");
  if (!Number.isInteger(tileSize) ||
      tileSize < MIN_NEURAL_TILE_SIZE || tileSize > MAX_NEURAL_TILE_SIZE) {
    throw new Error("neural tile size is outside the supported range");
  }
  if (!Number.isInteger(tileOverlap) ||
      tileOverlap < MIN_NEURAL_TILE_OVERLAP || tileOverlap > MAX_NEURAL_TILE_OVERLAP ||
      tileSize + 2 * tileOverlap > MAX_NEURAL_TILE_INPUT_EXTENT) {
    throw new Error("neural tile overlap is outside the supported range");
  }
  if (srcW > Math.floor(Number.MAX_SAFE_INTEGER / scale) ||
      srcH > Math.floor(Number.MAX_SAFE_INTEGER / scale)) {
    throw new Error("neural output dimensions exceed the safe integer range");
  }

  const tiles = [];
  for (let coreY = 0; coreY < srcH; coreY += tileSize) {
    const coreH = Math.min(tileSize, srcH - coreY);
    for (let coreX = 0; coreX < srcW; coreX += tileSize) {
      const coreW = Math.min(tileSize, srcW - coreX);
      const inputX = Math.max(0, coreX - tileOverlap);
      const inputY = Math.max(0, coreY - tileOverlap);
      const inputRight = Math.min(srcW, coreX + coreW + tileOverlap);
      const inputBottom = Math.min(srcH, coreY + coreH + tileOverlap);
      const inputW = inputRight - inputX;
      const inputH = inputBottom - inputY;
      const padW = Math.ceil(inputW / padMultiple) * padMultiple;
      const padH = Math.ceil(inputH / padMultiple) * padMultiple;
      if (!Number.isSafeInteger(padW) || !Number.isSafeInteger(padH)) {
        throw new Error("neural padded tile dimensions exceed the safe integer range");
      }
      tiles.push(Object.freeze({
        coreX,
        coreY,
        coreW,
        coreH,
        inputX,
        inputY,
        inputW,
        inputH,
        padW,
        padH,
        cropX: (coreX - inputX) * scale,
        cropY: (coreY - inputY) * scale,
        dstX: coreX * scale,
        dstY: coreY * scale,
        outW: coreW * scale,
        outH: coreH * scale,
      }));
    }
  }
  return Object.freeze(tiles);
}

function sessionNames(session, kind) {
  const names = session?.[`${kind}Names`];
  if (!Array.isArray(names) || names.length === 0 ||
      names.some((name) => typeof name !== "string" || !name)) {
    throw new Error(`neural session has no valid ${kind} names`);
  }
  if (new Set(names).size !== names.length) {
    throw new Error(`neural session has duplicate ${kind} names`);
  }
  return names;
}

function selectSessionName(names, requested, kind) {
  if (requested != null) {
    if (!names.includes(requested)) {
      throw new Error(`manifest ${kind} '${requested}' is not exposed by the neural session`);
    }
    return requested;
  }
  if (names.length !== 1) {
    throw new Error(
      `neural session exposes ${names.length} ${kind}s; manifest must name the ${kind} tensor`,
    );
  }
  return names[0];
}

function sessionMetadata(session, kind, name, names) {
  const collection = session?.[`${kind}Metadata`];
  if (collection == null) return null;
  if (Array.isArray(collection)) {
    const hasNamedMetadata = collection.some(
      (metadata) => typeof metadata?.name === "string",
    );
    const metadata = hasNamedMetadata
      ? collection.find((candidate) => candidate?.name === name)
      : collection[names.indexOf(name)];
    if (metadata == null) throw new Error(`neural session has no metadata for ${kind} '${name}'`);
    return metadata;
  }
  if (collection instanceof Map) {
    if (!collection.has(name)) throw new Error(`neural session has no metadata for ${kind} '${name}'`);
    return collection.get(name);
  }
  if (typeof collection === "object") {
    if (!Object.hasOwn(collection, name)) {
      throw new Error(`neural session has no metadata for ${kind} '${name}'`);
    }
    return collection[name];
  }
  throw new Error(`neural session exposes invalid ${kind} metadata`);
}

function isDynamicDimension(value) {
  return value == null ||
    (typeof value === "string" && value.length > 0) ||
    (Number.isInteger(value) && value < 0);
}

function validateTensorMetadata(metadata, kind, name) {
  if (metadata == null) return;
  if (metadata.isTensor === false) {
    throw new Error(`neural ${kind} '${name}' is not a tensor`);
  }
  const type = metadata.type ?? metadata.dataType;
  if (type != null && type !== "float32") {
    throw new Error(`neural ${kind} '${name}' has dtype '${type}' (expected float32)`);
  }
  const shape = metadata.shape ?? metadata.dims ?? metadata.dimensions;
  if (shape == null) return;
  if (!Array.isArray(shape) || shape.length !== 4) {
    const shown = Array.isArray(shape) ? `[${shape.join(",")}]` : String(shape);
    throw new Error(`neural ${kind} '${name}' has shape ${shown} (expected NCHW rank 4)`);
  }
  if (shape[0] !== 1 && !isDynamicDimension(shape[0])) {
    throw new Error(`neural ${kind} '${name}' batch dimension must be 1 or dynamic`);
  }
  if (shape[1] !== 3) {
    throw new Error(`neural ${kind} '${name}' channel dimension must be RGB (3)`);
  }
  if (!isDynamicDimension(shape[2]) || !isDynamicDimension(shape[3])) {
    throw new Error(`neural ${kind} '${name}' spatial dimensions must be dynamic`);
  }
}

function validateV2TensorMetadata(metadata, kind, descriptor) {
  if (metadata == null) return;
  const { name, dtype, channels, layout } = descriptor;
  if (metadata.isTensor === false) {
    throw new Error(`neural ${kind} '${name}' is not a tensor`);
  }
  const type = metadata.type ?? metadata.dataType;
  if (type != null && type !== dtype) {
    throw new Error(`neural ${kind} '${name}' has dtype '${type}' (expected ${dtype})`);
  }
  const shape = metadata.shape ?? metadata.dims ?? metadata.dimensions;
  if (shape == null) return;
  if (!Array.isArray(shape)) {
    throw new Error(`neural ${kind} '${name}' has invalid tensor shape metadata`);
  }
  if (layout === "scalar") {
    if (!(shape.length === 0 || (shape.length === 1 && shape[0] === 1))) {
      throw new Error(`neural ${kind} '${name}' must be a scalar or one-element tensor`);
    }
    return;
  }
  if (shape.length !== 4) {
    throw new Error(`neural ${kind} '${name}' has shape [${shape.join(",")}] (expected NCHW rank 4)`);
  }
  if (shape[0] !== 1 && !isDynamicDimension(shape[0])) {
    throw new Error(`neural ${kind} '${name}' batch dimension must be 1 or dynamic`);
  }
  if (shape[1] !== channels) {
    throw new Error(`neural ${kind} '${name}' channel dimension must be ${channels}`);
  }
  for (const dimension of shape.slice(2)) {
    if (!isDynamicDimension(dimension) &&
        (!Number.isSafeInteger(dimension) || dimension < 1)) {
      throw new Error(`neural ${kind} '${name}' has invalid spatial dimensions`);
    }
  }
}

function sameNames(actual, declared) {
  return actual.length === declared.length &&
    declared.every((name) => actual.includes(name));
}

function contractForEntry(entry) {
  return validatedContracts.get(entry) || normalizeNeuralModelContract(entry);
}

export function validateNeuralSessionContract(session, entry, graphName = null) {
  const declaredContract = contractForEntry(entry);
  const inputs = sessionNames(session, "input");
  const outputs = sessionNames(session, "output");
  if (declaredContract.version === 2) {
    const selectedGraphName = graphName ?? declaredContract.resetGraph;
    const graph = declaredContract.graphs[selectedGraphName];
    if (!graph) throw new Error(`neural contract names unknown graph '${selectedGraphName}'`);
    const declaredInputs = Object.keys(graph.inputs);
    const declaredOutputs = Object.keys(graph.outputs);
    if (!sameNames(inputs, declaredInputs)) {
      throw new Error(
        `neural graph '${selectedGraphName}' inputs [${inputs.join(",")}] do not match manifest [${declaredInputs.join(",")}]`,
      );
    }
    if (!sameNames(outputs, declaredOutputs)) {
      throw new Error(
        `neural graph '${selectedGraphName}' outputs [${outputs.join(",")}] do not match manifest [${declaredOutputs.join(",")}]`,
      );
    }
    for (const descriptor of Object.values(graph.inputs)) {
      validateV2TensorMetadata(
        sessionMetadata(session, "input", descriptor.name, inputs),
        "input",
        descriptor,
      );
    }
    for (const descriptor of Object.values(graph.outputs)) {
      validateV2TensorMetadata(
        sessionMetadata(session, "output", descriptor.name, outputs),
        "output",
        descriptor,
      );
    }
    return Object.freeze({
      version: 2,
      mode: declaredContract.mode,
      graphName: selectedGraphName,
      graph,
      inputName: requireSingleRole(graph.inputs, "rgb", "input", `neural graph '${selectedGraphName}'`).name,
      outputName: requireSingleRole(graph.outputs, "rgb", "output", `neural graph '${selectedGraphName}'`).name,
    });
  }
  const selectedInput = selectSessionName(inputs, entry.input, "input");
  const selectedOutput = selectSessionName(outputs, entry.output, "output");
  validateTensorMetadata(
    sessionMetadata(session, "input", selectedInput, inputs),
    "input",
    selectedInput,
  );
  validateTensorMetadata(
    sessionMetadata(session, "output", selectedOutput, outputs),
    "output",
    selectedOutput,
  );
  return Object.freeze({
    version: 1,
    mode: "spatial",
    graphName: "default",
    graph: null,
    inputName: selectedInput,
    outputName: selectedOutput,
  });
}

function v2GraphRole(graph, role, kind = "input") {
  const descriptors = graph?.[`${kind}s`];
  if (!descriptors) return null;
  return Object.values(descriptors).find((descriptor) => descriptor.role === role) || null;
}

export function resolveNeuralAuxiliaryInputs(contract, graphName, auxiliary = {}) {
  if (!contract || contract.version !== 2) {
    throw new Error("neural auxiliary bindings require a v2 contract");
  }
  const graph = contract.graphs?.[graphName];
  if (!graph) throw new Error(`neural contract names unknown graph '${graphName}'`);
  if (!auxiliary || typeof auxiliary !== "object" || Array.isArray(auxiliary)) {
    throw new Error("neural auxiliary inputs must be a named object");
  }

  const recognized = new Set();
  for (const candidateGraph of Object.values(contract.graphs)) {
    const roleCounts = new Map();
    for (const descriptor of Object.values(candidateGraph.inputs)) {
      recognized.add(descriptor.name);
      roleCounts.set(descriptor.role, (roleCounts.get(descriptor.role) || 0) + 1);
    }
    for (const [role, count] of roleCounts) {
      if (count === 1) recognized.add(role);
    }
  }
  for (const key of Object.keys(auxiliary)) {
    if (!recognized.has(key)) {
      throw new Error(`neural auxiliary input '${key}' is not declared by the contract`);
    }
  }

  const resolved = {};
  for (const descriptor of Object.values(graph.inputs)) {
    if (descriptor.role === "rgb" || descriptor.role === "state-in") continue;
    const byName = auxiliary[descriptor.name];
    const byRole = auxiliary[descriptor.role];
    if (byName != null && byRole != null && byName !== byRole) {
      throw new Error(
        `neural auxiliary input '${descriptor.name}' was provided by both name and role`,
      );
    }
    const value = byName ?? byRole;
    if (value == null) {
      throw new Error(
        `neural graph '${graphName}' requires auxiliary input '${descriptor.name}' (${descriptor.role})`,
      );
    }
    resolved[descriptor.name] = value;
  }
  return Object.freeze(resolved);
}

const SNAPSHOT_EXT_WGSL = `
struct P { w:u32, h:u32, _pad0:u32, _pad1:u32 }
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_external;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> u: P;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.w || gid.y >= u.h) { return; }
  let uv = (vec2<f32>(vec2<u32>(gid.xy)) + vec2<f32>(0.5, 0.5)) /
    vec2<f32>(f32(u.w), f32(u.h));
  let c = textureSampleBaseClampToEdge(src, samp, uv).rgb;
  textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(c, 1.0));
}`;

// texture_2d twin: decoder frames and frames synthesized upstream are captured
// with identical sampling before run() first yields.
const SNAPSHOT_TEX_WGSL = SNAPSHOT_EXT_WGSL
  .replace("texture_external", "texture_2d<f32>")
  .replace("textureSampleBaseClampToEdge(src, samp, uv)", "textureSampleLevel(src, samp, uv, 0.0)");

const PACK_TILE_WGSL = `
struct P {
  padW:u32, padH:u32, tileW:u32, tileH:u32,
  srcX:u32, srcY:u32, srcW:u32, srcH:u32
}
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: P;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.padW || gid.y >= u.padH) { return; }
  // Halos are clipped at actual image boundaries. Replication is exclusively
  // the model's padMultiple expansion on the right/bottom.
  let xx = min(gid.x, u.tileW - 1u);
  let yy = min(gid.y, u.tileH - 1u);
  let pos = vec2<i32>(i32(u.srcX + xx), i32(u.srcY + yy));
  let c = textureLoad(src, pos, 0).rgb;
  let plane = u.padW * u.padH;
  let idx = gid.y * u.padW + gid.x;
  dst[idx] = c.r;
  dst[plane + idx] = c.g;
  dst[2u * plane + idx] = c.b;
}`;

const COMPOSITE_WGSL = `
struct P {
  strideW:u32, plane:u32, srcX:u32, srcY:u32,
  dstX:u32, dstY:u32, outW:u32, outH:u32
}
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> u: P;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.outW || gid.y >= u.outH) { return; }
  let idx = (u.srcY + gid.y) * u.strideW + u.srcX + gid.x;
  let r = clamp(src[idx], 0.0, 1.0);
  let g = clamp(src[u.plane + idx], 0.0, 1.0);
  let b = clamp(src[2u * u.plane + idx], 0.0, 1.0);
  let dstPos = vec2<i32>(i32(u.dstX + gid.x), i32(u.dstY + gid.y));
  textureStore(dst, dstPos, vec4<f32>(r, g, b, 1.0));
}`;

export function createNeuralEngine({ log = console.log, warn = console.warn } = {}) {
  let ort = null;
  let session = null;
  let device = null;
  // tileSize/tileOverlap tune bounded work only; omitted entries use 512/24.
  // They never limit the source dimensions or skip a frame by policy.
  let manifest = null;          // validated legacy-v1 or named-graph-v2 model entries
  let active = null;            // manifest entry of the loaded model
  let inputName = "input", outputName = "output";
  let activeContract = null;
  let temporalState = new Map();
  const sessionGroups = new WeakMap();
  const releasedSessionGroups = new WeakSet();
  let fp16Model = false;
  let initGeneration = 0;
  let initTail = Promise.resolve();
  let latestInitPromise = null;
  let sessionGeneration = 0;
  let lifecycleGeneration = 0;
  let activeRuns = 0;
  let runIdleResolvers = [];
  let runBusy = false;
  let deferredSessionReleases = [];
  const retirements = new Set();
  let disposalPromise = null;
  const watchedDevices = new WeakSet();
  const lostDevices = new WeakSet();
  const deviceInvalidations = new WeakMap();
  const activeDeviceInvalidations = new Set();
  const deviceInvalidationFailures = new Set();
  let deviceInvalidationTail = Promise.resolve();
  let deviceInvalidationPromise = null;
  let sessionReleaseFailures = [];

  // GPU resources (allocated on ORT's device)
  let sampler = null;
  let snapshotExtPipe = null, snapshotTexPipe = null;
  let packTilePipe = null, compPipe = null;
  let inBuf = null, inBufSize = 0;
  let snapshotU = null, packU = null, compU = null;
  let snapshotTex = null, snapshotTexW = 0, snapshotTexH = 0;
  let outTex = null, outTexW = 0, outTexH = 0;

  // instrumentation (mirrors RIFE's readout vocabulary)
  const stats = {
    last: 0,
    mu: 0,
    n: 0,
    skip: 0,
    fails: 0,
    lastTiles: 0,
    tileRuns: 0,
    maxTileW: 0,
    maxTileH: 0,
  };

  function whenRunsIdle() {
    if (activeRuns === 0) return Promise.resolve();
    return new Promise((resolve) => runIdleResolvers.push(resolve));
  }

  function endRun() {
    activeRuns = Math.max(0, activeRuns - 1);
    if (activeRuns === 0 && runIdleResolvers.length) {
      for (const resolve of runIdleResolvers.splice(0)) resolve();
    }
  }

  function trackRetirement(promise) {
    const tracked = Promise.resolve(promise).catch(() => {});
    retirements.add(tracked);
    tracked.finally(() => retirements.delete(tracked));
    return tracked;
  }

  function afterSubmittedWork(ownerDevice, callback) {
    let fence;
    try { fence = ownerDevice?.queue?.onSubmittedWorkDone?.() || Promise.resolve(); }
    catch { fence = Promise.resolve(); }
    return trackRetirement(Promise.resolve(fence).catch(() => {}).then(callback));
  }

  function retireGpuObjects(objects, ownerDevice = device) {
    const live = objects.filter(Boolean);
    if (!live.length) return Promise.resolve();
    return afterSubmittedWork(ownerDevice, () => {
      for (const object of live) { try { object.destroy?.(); } catch {} }
    });
  }

  function retireTensors(ownerDevice, ...tensors) {
    const live = [...new Set(tensors.filter(Boolean))];
    if (!live.length) return;
    afterSubmittedWork(ownerDevice, () => {
      for (const tensor of live) {
        try { tensor?.dispose?.(); } catch {}
      }
    });
  }

  async function releaseManagedSession(primarySession) {
    if (!primarySession || releasedSessionGroups.has(primarySession)) return;
    releasedSessionGroups.add(primarySession);
    const group = sessionGroups.get(primarySession);
    sessionGroups.delete(primarySession);
    const sessions = group?.sessions || [primarySession];
    const failures = [];
    for (const candidate of new Set(sessions.filter(Boolean))) {
      try { await candidate?.release?.(); }
      catch (error) { failures.push(error); }
    }
    if (failures.length === 1) throw failures[0];
    if (failures.length) {
      throw new AggregateError(failures, "neural session group release failed");
    }
  }

  function clearTemporalState(ownerDevice = device) {
    const stale = [...temporalState.values()];
    temporalState = new Map();
    retireTensors(ownerDevice, ...stale);
  }

  async function releaseDeferredSessions() {
    const pending = deferredSessionReleases;
    deferredSessionReleases = [];
    const failures = [];
    for (const oldSession of pending) {
      try { await releaseManagedSession(oldSession); }
      catch (error) {
        failures.push(error);
        sessionReleaseFailures.push(error);
        warn("neural: deferred old session release failed:", error.message);
      }
    }
    if (failures.length) {
      throw new AggregateError(failures, "neural deferred session release failed");
    }
  }

  function watchDevice(ownerDevice) {
    if (!ownerDevice || watchedDevices.has(ownerDevice) || !ownerDevice.lost?.then) return;
    watchedDevices.add(ownerDevice);
    ownerDevice.lost.then((info) => {
      lostDevices.add(ownerDevice);
      if (device === ownerDevice) {
        warn(`neural: GPU device lost: ${info?.message || info?.reason || "unknown reason"}`);
      }
      // A replaced session can still be the lifetime guard for main's old
      // adopted device. Route every watched loss through the identity-aware
      // invalidator so a non-current guard is claimed and released as well.
      invalidateDevice(ownerDevice).catch((error) =>
        warn("neural: device-loss cleanup failed:", error.message));
    }).catch(() => {});
  }

  function refreshDeviceInvalidationBarrier() {
    if (!activeDeviceInvalidations.size && !deviceInvalidationFailures.size) {
      deviceInvalidationPromise = null;
      return null;
    }
    const operations = [...activeDeviceInvalidations];
    const latchedFailures = [...deviceInvalidationFailures];
    const barrier = Promise.allSettled(operations).then((results) => {
      const failures = new Set(latchedFailures);
      for (const result of results) {
        if (result.status === "rejected") addDeviceInvalidationFailure(failures, result.reason);
      }
      if (failures.size) {
        throw new AggregateError([...failures], "neural device invalidation barrier failed");
      }
    });
    // The provider watcher observes each per-device promise. This independent
    // aggregate can reject before init()/dispose() consumes it, so attach a
    // passive observer while preserving the rejection for those callers.
    barrier.catch(() => {});
    deviceInvalidationPromise = barrier;
    return barrier;
  }

  function addDeviceInvalidationFailure(failures, error) {
    if (error instanceof AggregateError && error.errors) {
      for (const cause of error.errors) addDeviceInvalidationFailure(failures, cause);
      return;
    }
    failures.add(error);
  }

  function exposeDeviceInvalidation(ownerDevice, cleanup, previousBarrier) {
    const operation = Promise.allSettled([
      previousBarrier || Promise.resolve(),
      cleanup,
    ]).then(([previous, current]) => {
      const failures = new Set();
      if (previous.status === "rejected") {
        addDeviceInvalidationFailure(failures, previous.reason);
      }
      if (current.status === "rejected") {
        addDeviceInvalidationFailure(failures, current.reason);
      }
      if (failures.size) {
        throw new AggregateError([...failures], "neural device invalidation failed");
      }
      return current.value;
    });
    deviceInvalidations.set(ownerDevice, operation);
    activeDeviceInvalidations.add(operation);
    // Register settlement bookkeeping before building allSettled snapshots so
    // a disposal continuation always sees the failure latch populated.
    operation.then(
      () => {
        activeDeviceInvalidations.delete(operation);
        refreshDeviceInvalidationBarrier();
      },
      (error) => {
        activeDeviceInvalidations.delete(operation);
        addDeviceInvalidationFailure(deviceInvalidationFailures, error);
        refreshDeviceInvalidationBarrier();
      },
    );
    refreshDeviceInvalidationBarrier();
    return operation;
  }

  async function loadManifest() {
    if (manifest) return manifest;
    try {
      const r = await fetch(resolvePackagedAssetUrl("model/neural/manifest.json"));
      if (!r.ok) throw new Error("HTTP " + r.status);
      manifest = validateNeuralManifest(await r.json());
    } catch (e) {
      warn("neural manifest rejected:", e.message);
      manifest = [];
    }
    return manifest;
  }

  function ensurePipelines() {
    if (snapshotExtPipe) return;
    const mk = (code) => device.createShaderModule({ code });
    let nextSnapshotU = null, nextPackU = null, nextCompU = null;
    try {
      const nextSnapshotExtPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(SNAPSHOT_EXT_WGSL), entryPoint: "main" } });
      const nextSnapshotTexPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(SNAPSHOT_TEX_WGSL), entryPoint: "main" } });
      const nextPackTilePipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(PACK_TILE_WGSL), entryPoint: "main" } });
      const nextCompPipe = device.createComputePipeline({ layout: "auto", compute: { module: mk(COMPOSITE_WGSL), entryPoint: "main" } });
      const nextSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
      nextSnapshotU = device.createBuffer({ label: "neural-snapshotU", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      nextPackU = device.createBuffer({ label: "neural-packU", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      nextCompU = device.createBuffer({ label: "neural-compU", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      snapshotExtPipe = nextSnapshotExtPipe;
      snapshotTexPipe = nextSnapshotTexPipe;
      packTilePipe = nextPackTilePipe;
      compPipe = nextCompPipe;
      sampler = nextSampler;
      snapshotU = nextSnapshotU;
      packU = nextPackU;
      compU = nextCompU;
    } catch (error) {
      try { nextSnapshotU?.destroy?.(); } catch {}
      try { nextPackU?.destroy?.(); } catch {}
      try { nextCompU?.destroy?.(); } catch {}
      throw error;
    }
  }

  function ensureInBuf(padW, padH) {
    const need = checkedProduct("neural input buffer", padW, padH, 3, 4);
    if (inBuf && inBufSize >= need) return;
    const old = inBuf;
    const candidate = device.createBuffer({
      label: `neural-in-${padW}x${padH}`,
      size: need,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    inBuf = candidate;
    inBufSize = need;
    if (old) retireGpuObjects([old]);
  }

  function ensureSnapshotTex(w, h) {
    if (snapshotTex && snapshotTexW === w && snapshotTexH === h) return;
    const old = snapshotTex;
    const candidate = device.createTexture({
      label: `neural-snapshot-${w}x${h}`,
      size: { width: w, height: h },
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    snapshotTex = candidate;
    snapshotTexW = w;
    snapshotTexH = h;
    if (old) retireGpuObjects([old]);
  }

  function ensureOutTex(w, h) {
    if (outTex && outTexW === w && outTexH === h) return;
    const old = outTex;
    const candidate = device.createTexture({
      label: `neural-out-${w}x${h}`,
      size: { width: w, height: h },
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    outTex = candidate;
    outTexW = w; outTexH = h;
    if (old) retireGpuObjects([old]);
  }

  function checkedProduct(label, ...values) {
    let product = 1;
    for (const value of values) {
      if (!Number.isSafeInteger(value) || value <= 0 || product > Number.MAX_SAFE_INTEGER / value) {
        const error = new Error(`${label} exceeds the safe integer range`);
        error.code = "NEURAL_LIMIT";
        throw error;
      }
      product *= value;
    }
    return product;
  }

  function allocationLimits() {
    const maxDimension = Math.max(1, Number(device?.limits?.maxTextureDimension2D) || 8192);
    const maxBuffer = Math.max(1, Math.min(
      Number(device?.limits?.maxBufferSize) || 256 * 1024 * 1024,
      Number(device?.limits?.maxStorageBufferBindingSize) || 128 * 1024 * 1024,
    ));
    const maxGroups = Math.max(1, Number(device?.limits?.maxComputeWorkgroupsPerDimension) || 65535);
    return { maxDimension, maxBuffer, maxGroups };
  }

  function validateFrameAllocationLimits(srcW, srcH, outW, outH) {
    const { maxDimension, maxGroups } = allocationLimits();
    const dimensions = [srcW, srcH, outW, outH];
    if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 1 || value > maxDimension)) {
      const error = new Error(`neural dimensions exceed the device texture limit ${maxDimension}`);
      error.code = "NEURAL_LIMIT";
      throw error;
    }
    if (Math.ceil(srcW / 8) > maxGroups || Math.ceil(srcH / 8) > maxGroups) {
      const error = new Error(`neural snapshot dispatch exceeds the device workgroup limit ${maxGroups}`);
      error.code = "NEURAL_LIMIT";
      throw error;
    }
  }

  function validateTileAllocationLimits(tile, scale) {
    const { maxBuffer, maxGroups } = allocationLimits();
    const { padW, padH, outW, outH } = tile;
    const paddedOutW = checkedProduct("neural padded output width", padW, scale);
    const paddedOutH = checkedProduct("neural padded output height", padH, scale);
    if (Math.ceil(padW / 8) > maxGroups || Math.ceil(padH / 8) > maxGroups ||
        Math.ceil(outW / 8) > maxGroups || Math.ceil(outH / 8) > maxGroups) {
      const error = new Error(`neural tile dispatch exceeds the device workgroup limit ${maxGroups}`);
      error.code = "NEURAL_LIMIT";
      throw error;
    }
    const inputBytes = checkedProduct("neural input buffer", padW, padH, 3, 4);
    const outputBytes = checkedProduct("neural output buffer", paddedOutW, paddedOutH, 3, 4);
    if (inputBytes > maxBuffer || outputBytes > maxBuffer) {
      const error = new Error(`neural tensor buffer exceeds the device binding limit ${maxBuffer}`);
      error.code = "NEURAL_LIMIT";
      throw error;
    }
  }

  async function initOne(key, generation) {
    const list = await loadManifest();
    if (!list.length) throw new Error("no neural models: model/neural/manifest.json missing or empty");
    const entry = list.find((m) => m.key === key) || list[0];
    const modelContract = contractForEntry(entry);
    if (generation !== initGeneration) return null;
    if (session && device && active?.key === entry.key) return active;

    ort = await ensureOrt();
    if (generation !== initGeneration) return null;
    // Create the new session before releasing the old one. The ORT device's
    // lifetime follows its sessions, so its refcount must not reach zero during
    // a live model swap.
    const opts = {
      executionProviders: [{ name: "webgpu" }],
      graphOptimizationLevel: "all",
      enableGraphCapture: false,
      preferredOutputLocation: "gpu-buffer",
    };
    const requestedExecutionFp16 = entry.fp16 === true;
    let executionFp16 = requestedExecutionFp16;
    const graphSpecs = modelContract.version === 1
      ? [{ name: "default", file: entry.file }]
      : Object.values(modelContract.graphs);
    const graphSessions = new Map();
    const createdSessions = [];
    const releaseCreatedSessions = async () => {
      for (const stale of createdSessions.splice(0)) {
        try { await stale?.release?.(); } catch {}
      }
    };
    for (const graphSpec of graphSpecs) {
      const url = resolvePackagedAssetUrl("model/neural/" + graphSpec.file);
      let created;
      if (requestedExecutionFp16) {
        try {
          created = await createOrtSession(url, opts, { enableFp16: true });
        } catch (fp16Error) {
          if (generation !== initGeneration) {
            await releaseCreatedSessions();
            return null;
          }
          executionFp16 = false;
          warn(`neural: FP16 execution session failed for ${graphSpec.file}; retrying FP32: ${fp16Error.message}`);
          try {
            created = await createOrtSession(url, opts, { enableFp16: false });
          } catch (fp32Error) {
            if (generation !== initGeneration) {
              await releaseCreatedSessions();
              return null;
            }
            await releaseCreatedSessions();
            throw new Error(`neural session create failed (${graphSpec.file}): ${fp32Error.message}`);
          }
        }
      } else {
        try {
          created = await createOrtSession(url, opts, { enableFp16: false });
        } catch (fp32Error) {
          if (generation !== initGeneration) {
            await releaseCreatedSessions();
            return null;
          }
          await releaseCreatedSessions();
          throw new Error(`neural session create failed (${graphSpec.file}): ${fp32Error.message}`);
        }
      }
      graphSessions.set(graphSpec.name, created);
      createdSessions.push(created);
      if (generation !== initGeneration) {
        await releaseCreatedSessions();
        return null;
      }
    }
    const primaryGraphName = modelContract.version === 1
      ? "default"
      : modelContract.resetGraph;
    const next = graphSessions.get(primaryGraphName);
    sessionGroups.set(next, {
      sessions: createdSessions,
      graphs: graphSessions,
      contract: modelContract,
      resolvedGraphs: null,
    });
    if (generation !== initGeneration) {
      try { await releaseManagedSession(next); } catch {}
      return null;
    }
    const resolvedGraphs = new Map();
    try {
      for (const [graphName, graphSession] of graphSessions) {
        resolvedGraphs.set(
          graphName,
          validateNeuralSessionContract(
            graphSession,
            entry,
            modelContract.version === 1 ? null : graphName,
          ),
        );
      }
    } catch (contractError) {
      try {
        await releaseManagedSession(next);
      } catch (releaseError) {
        throw new AggregateError(
          [contractError, releaseError],
          `neural model contract rejection cleanup failed (${graphSpecs.map(({ file }) => file).join(", ")})`,
        );
      }
      throw new Error(`neural model contract rejected (${graphSpecs.map(({ file }) => file).join(", ")}): ${contractError.message}`, {
        cause: contractError,
      });
    }
    sessionGroups.get(next).resolvedGraphs = resolvedGraphs;
    const nextOrt = await ensureOrt();
    const nextDevice = getOrtSessionDevice(next);
    if (!nextDevice) {
      try { await releaseManagedSession(next); } catch {}
      throw new Error("ORT device unavailable after neural session create");
    }
    if (createdSessions.some((candidate) => getOrtSessionDevice(candidate) !== nextDevice)) {
      try { await releaseManagedSession(next); } catch {}
      throw new Error("neural graph sessions did not share one ORT device");
    }
    // Subscribe before publication and yield once. A device can be returned with
    // an already-settled `lost` promise; accepting that session would let init()
    // report success just before the watcher clears it on the next turn.
    watchDevice(nextDevice);
    await Promise.resolve();
    if (generation !== initGeneration) {
      try { await releaseManagedSession(next); } catch {}
      return null;
    }
    if (lostDevices.has(nextDevice)) {
      try { await releaseManagedSession(next); } catch {}
      throw new Error("ORT device was lost during neural session initialization");
    }

    // Do not swap or release the old session while run() is using its names,
    // buffers, output tensor, or device. New runs cannot begin during this task's
    // synchronous commit once the idle promise resolves.
    await whenRunsIdle();
    if (generation !== initGeneration) {
      try { await releaseManagedSession(next); } catch {}
      return null;
    }

    const oldSession = session;
    const oldDevice = device;
    if (oldDevice && oldDevice !== nextDevice) destroyGpuResources(oldDevice);
    device = nextDevice;
    try {
      ensurePipelines();
    } catch (error) {
      destroyGpuResources(nextDevice);
      device = oldDevice;
      try { await releaseManagedSession(next); } catch {}
      throw error;
    }

    ort = nextOrt;
    session = next;
    active = entry;
    activeContract = modelContract;
    const primaryContract = resolvedGraphs.get(primaryGraphName);
    inputName = primaryContract.inputName;
    outputName = primaryContract.outputName;
    clearTemporalState(nextDevice);
    fp16Model = graphSpecs.some(({ file }) => /fp16/i.test(file)) || entry.fp16 === true;
    sessionGeneration++;
    // Transfer the prior session out of this initializer's local ownership before
    // yielding. A concurrent dispose/device-loss can clear the just-published
    // session at the microtask below; without this handoff, the identity check
    // would throw before the prior session reached either release path.
    if (oldSession && !deferredSessionReleases.includes(oldSession)) {
      deferredSessionReleases.push(oldSession);
    }
    // Re-check after synchronous publication as well. This closes the narrow
    // window between the pre-commit yield and assigning the public session.
    await Promise.resolve();
    const initializationCancelled = generation !== initGeneration;
    if (lostDevices.has(nextDevice) || device !== nextDevice || session !== next) {
      await invalidateDevice(nextDevice);
      throw new Error("ORT device was lost during neural session initialization");
    }
    if (oldSession) {
      if (oldDevice === nextDevice) {
        // The new session retains the same shared device, so the old reference can
        // be released immediately without dropping its device refcount to zero.
        deferredSessionReleases = deferredSessionReleases.filter(
          (candidate) => candidate !== oldSession,
        );
        try { await releaseManagedSession(oldSession); }
        catch (error) {
          sessionReleaseFailures.push(error);
          warn("neural: old session release failed:", error.message);
          throw new AggregateError([error], "neural replaced session release failed");
        }
      }
      // For a different device, keep the pre-yield deferred ownership until the
      // first new-device run. The presentation owner adopts the new device before
      // calling run(), so this cannot orphan its current pipelines.
    }

    if (initializationCancelled) return null;

    log(`neural: session ready — ${entry.label || entry.key} (${entry.scale}x, ${fp16Model ? "fp16" : "fp32"} weights, ${executionFp16 ? "FP16" : "FP32"} execution, dynamic dims)`);
    return entry;
  }

  function init(key) {
    if (disposalPromise) {
      return disposalPromise.then(() => init(key));
    }
    // A device invalidation owns the old ORT session until its GPU fence and
    // release() have both completed. Allocate the init generation only after
    // that physical cleanup finishes; otherwise invalidateDevice() would both
    // overlap session creation and cancel the recovery request it is gating.
    if (deviceInvalidationPromise) {
      return deviceInvalidationPromise.then(() => init(key));
    }
    const generation = ++initGeneration;
    const raw = initTail.catch(() => {}).then(() => initOne(key, generation));
    initTail = raw.then(() => undefined, () => undefined);
    let exposed;
    exposed = raw.then(
      (entry) => {
        if (entry) return entry;
        const latest = latestInitPromise;
        if (latest && latest !== exposed) return latest;
        throw new Error("neural initialization cancelled");
      },
      (error) => {
        const latest = latestInitPromise;
        if (generation !== initGeneration && latest && latest !== exposed) return latest;
        throw error;
      },
    );
    latestInitPromise = exposed;
    return exposed;
  }

  function destroyGpuResources(ownerDevice = device) {
    clearTemporalState(ownerDevice);
    const resources = [inBuf, snapshotU, packU, compU, snapshotTex, outTex];
    inBuf = null; inBufSize = 0;
    snapshotU = null; packU = null; compU = null;
    snapshotTex = null; snapshotTexW = 0; snapshotTexH = 0;
    outTex = null; outTexW = 0; outTexH = 0;
    snapshotExtPipe = null; snapshotTexPipe = null;
    packTilePipe = null; compPipe = null; sampler = null;
    const cleanup = whenRunsIdle().then(async () => {
      try { await ownerDevice?.queue?.onSubmittedWorkDone?.(); } catch {}
      for (const resource of resources) { try { resource?.destroy?.(); } catch {} }
    });
    return trackRetirement(cleanup);
  }

  function invalidateDevice(lostDevice) {
    if (!lostDevice) return Promise.resolve(false);
    const existing = deviceInvalidations.get(lostDevice);
    if (existing) return existing;
    const previousBarrier = deviceInvalidationPromise;
    const affectsCurrent = device === lostDevice;
    const oldSession = affectsCurrent ? session : null;
    const lostDeferred = deferredSessionReleases.filter(
      (candidate) => getOrtSessionDevice(candidate) === lostDevice,
    );
    if (lostDeferred.length) {
      deferredSessionReleases = deferredSessionReleases.filter(
        (candidate) => getOrtSessionDevice(candidate) !== lostDevice,
      );
    }

    // Claim and unpublish logical ownership in this stack. Recovery and run()
    // must not observe a lost current session during a prior invalidation's
    // queued physical release. A loss of an old cross-device guard leaves the
    // healthy current session published while still claiming that guard once.
    let cleanup = Promise.resolve();
    if (affectsCurrent) {
      ++initGeneration;
      ++lifecycleGeneration;
      ++sessionGeneration;
      session = null;
      device = null;
      runBusy = false;
      cleanup = destroyGpuResources(lostDevice);
    }
    const claimedSessions = new Set([...lostDeferred, oldSession].filter(Boolean));
    const claimed = affectsCurrent || claimedSessions.size > 0;
    const operation = deviceInvalidationTail.catch(() => {}).then(async () => {
      if (!claimed) return false;
      const failures = [];
      try { await cleanup; } catch (error) { failures.push(error); }
      for (const candidate of claimedSessions) {
        try { await releaseManagedSession(candidate); }
        catch (error) {
          failures.push(error);
          warn("neural: device-loss session release failed:", error.message);
        }
      }
      if (failures.length) throw new AggregateError(failures, "neural device-loss cleanup failed");
      return true;
    });
    deviceInvalidationTail = operation.then(() => undefined, () => undefined);
    // A GPUDevice cannot recover after `lost` settles. Retain its exact cleanup
    // promise in a WeakMap so every observer (provider watcher, renderer, and
    // disposal path) sees the claiming operation's result, including failures.
    return exposeDeviceInvalidation(lostDevice, operation, previousBarrier);
  }

  // src: external texture (default) or { tex } for the texture_2d twin.
  // Returns { tex, outW, outH } or throws.
  function validateOutputTensor(tensor, padW, padH, scale, expectedOutputName) {
    if (!tensor?.gpuBuffer) throw new Error(`output not on GPU (${expectedOutputName})`);
    const type = tensor.type || tensor.dataType;
    if (type !== "float32") throw new Error(`unsupported neural output dtype '${type || "unknown"}' (expected float32)`);

    const expectedDims = [1, 3, padH * scale, padW * scale];
    const dims = Array.from(tensor.dims || [], Number);
    if (dims.length !== expectedDims.length || dims.some((value, index) => value !== expectedDims[index])) {
      throw new Error(`neural output shape [${dims.join(",")}] does not match manifest scale/padding [${expectedDims.join(",")}]`);
    }
    const elements = expectedDims.reduce((product, value) => product * value, 1);
    if (!Number.isSafeInteger(elements)) throw new Error("neural output tensor size exceeds safe integer range");
    if (tensor.size != null && Number(tensor.size) !== elements) {
      throw new Error(`neural output element count ${tensor.size} does not match expected ${elements}`);
    }
    const expectedBytes = elements * 4;
    const bufferBytes = Number(tensor.gpuBuffer.size);
    if (!Number.isFinite(bufferBytes) || bufferBytes < expectedBytes) {
      throw new Error(`neural output GPU buffer is ${bufferBytes || 0} bytes; expected at least ${expectedBytes}`);
    }
    return { strideW: expectedDims[3], plane: expectedDims[2] * expectedDims[3] };
  }

  function tensorElementBytes(dtype) {
    switch (dtype) {
      case "float16": return 2;
      case "float32":
      case "int32": return 4;
      case "int64": return 8;
      case "uint8":
      case "bool": return 1;
      default: throw new Error(`unsupported neural tensor dtype '${dtype}'`);
    }
  }

  function validateRuntimeTensor(value, descriptor, label, expectedSpatial = null) {
    if (!value || typeof value !== "object" || !value.gpuBuffer) {
      throw new Error(`${label} must provide a GPU buffer`);
    }
    const dtype = value.dataType ?? value.type;
    if (dtype !== descriptor.dtype) {
      throw new Error(`${label} has dtype '${dtype || "unknown"}' (expected ${descriptor.dtype})`);
    }
    if (descriptor.provider != null && value.provider !== descriptor.provider) {
      throw new Error(
        `${label} came from provider '${value.provider || "unknown"}' (expected ${descriptor.provider})`,
      );
    }
    const dims = Array.from(value.dims || [], Number);
    if (descriptor.layout === "scalar") {
      if (!(dims.length === 0 || (dims.length === 1 && dims[0] === 1))) {
        throw new Error(`${label} must be a scalar or one-element tensor`);
      }
    } else {
      if (dims.length !== 4 || dims[0] !== 1 || dims[1] !== descriptor.channels ||
          dims.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1)) {
        throw new Error(
          `${label} shape [${dims.join(",")}] does not match NCHW channels ${descriptor.channels}`,
        );
      }
      if (expectedSpatial &&
          (dims[2] !== expectedSpatial.height || dims[3] !== expectedSpatial.width)) {
        throw new Error(
          `${label} spatial shape ${dims[3]}x${dims[2]} does not match ${expectedSpatial.width}x${expectedSpatial.height}`,
        );
      }
    }
    const elements = dims.length
      ? dims.reduce((product, dimension) => checkedProduct(label, product, dimension), 1)
      : 1;
    const expectedBytes = checkedProduct(label, elements, tensorElementBytes(dtype));
    const bufferBytes = Number(value.gpuBuffer.size);
    if (!Number.isFinite(bufferBytes) || bufferBytes < expectedBytes) {
      throw new Error(`${label} GPU buffer is ${bufferBytes || 0} bytes; expected at least ${expectedBytes}`);
    }
    return { dataType: dtype, dims };
  }

  function fullFrameTile(srcW, srcH, scale) {
    return Object.freeze({
      coreX: 0,
      coreY: 0,
      coreW: srcW,
      coreH: srcH,
      inputX: 0,
      inputY: 0,
      inputW: srcW,
      inputH: srcH,
      padW: srcW,
      padH: srcH,
      cropX: 0,
      cropY: 0,
      dstX: 0,
      dstY: 0,
      outW: checkedProduct("neural output width", srcW, scale),
      outH: checkedProduct("neural output height", srcH, scale),
    });
  }

  // Contract-v2 options:
  //   { temporal?: { reset?: boolean }, reset?: boolean,
  //     auxiliary?: { [tensorNameOrUniqueRole]: { provider?, gpuBuffer, dataType, dims } } }
  // The provider owns GPUBuffer lifetime; this engine owns the ORT wrappers and
  // recurrent state tensors it receives from the graph.
  async function run(src, srcW, srcH, options = undefined) {
    if (!session || !device) throw new Error("neural engine not initialized");
    if (runBusy) throw new Error("neural inference already in progress");
    if (!Number.isInteger(srcW) || !Number.isInteger(srcH) || srcW <= 0 || srcH <= 0) {
      throw new Error(`invalid neural input dimensions ${srcW}x${srcH}`);
    }

    const runDevice = device;
    const runOrt = ort;
    const runEntry = active;
    const runPrimarySession = session;
    const runModelContract = activeContract || contractForEntry(runEntry);
    const isV2 = runModelContract.version === 2;
    let runSession = session;
    let runInputName = inputName;
    let runOutputName = outputName;
    let runGraph = null;
    let runGraphName = "default";
    let runAuxiliary = Object.freeze({});
    let resettingTemporalState = false;
    if (isV2) {
      if (options != null && (!options || typeof options !== "object" || Array.isArray(options))) {
        throw new Error("neural v2 run options must be an object");
      }
      const temporal = options?.temporal;
      if (temporal != null &&
          (!temporal || typeof temporal !== "object" || Array.isArray(temporal))) {
        throw new Error("neural v2 temporal metadata must be an object");
      }
      if (options?.reset != null && temporal?.reset != null &&
          options.reset !== temporal.reset) {
        throw new Error("neural v2 reset conflicts with temporal metadata");
      }
      const reset = options?.reset ?? temporal?.reset ?? false;
      if (typeof reset !== "boolean") throw new Error("neural v2 reset must be boolean");
      const stateReady = runModelContract.states.every((key) => temporalState.has(key));
      resettingTemporalState = runModelContract.mode === "temporal" && (reset || !stateReady);
      runGraphName = runModelContract.mode === "temporal" && !resettingTemporalState
        ? runModelContract.recurrentGraph
        : runModelContract.resetGraph;
      runGraph = runModelContract.graphs[runGraphName];
      const group = sessionGroups.get(session);
      runSession = group?.graphs?.get(runGraphName);
      if (!runSession) throw new Error(`neural graph session '${runGraphName}' is unavailable`);
      runInputName = v2GraphRole(runGraph, "rgb", "input").name;
      runOutputName = v2GraphRole(runGraph, "rgb", "output").name;
      runAuxiliary = resolveNeuralAuxiliaryInputs(
        runModelContract,
        runGraphName,
        options?.auxiliary ?? {},
      );
      if (resettingTemporalState) clearTemporalState(runDevice);
      for (const descriptor of Object.values(runGraph.inputs)) {
        if (descriptor.role === "state-in" && !temporalState.has(descriptor.state)) {
          throw new Error(
            `neural graph '${runGraphName}' is missing recurrent state '${descriptor.state}'`,
          );
        }
      }
    }
    const runSessionGeneration = sessionGeneration;
    const runLifecycleGeneration = lifecycleGeneration;
    const mult = isV2 ? 1 : Math.max(1, runEntry.padMultiple | 0 || 1);
    const scale = runEntry.scale;
    const outW = checkedProduct("neural output width", srcW, scale);
    const outH = checkedProduct("neural output height", srcH, scale);
    validateFrameAllocationLimits(srcW, srcH, outW, outH);
    const tiles = isV2
      ? Object.freeze([fullFrameTile(srcW, srcH, scale)])
      : planNeuralTiles(srcW, srcH, {
        tileSize: runEntry.tileSize ?? DEFAULT_NEURAL_TILE_SIZE,
        tileOverlap: runEntry.tileOverlap ?? DEFAULT_NEURAL_TILE_OVERLAP,
        padMultiple: mult,
        scale,
      });
    for (const tile of tiles) validateTileAllocationLimits(tile, scale);
    const largestTile = tiles.reduce((largest, tile) =>
      tile.padW * tile.padH > largest.padW * largest.padH ? tile : largest);
    const t0 = performance.now();
    runBusy = true;
    activeRuns++;
    stats.lastTiles = tiles.length;
    stats.maxTileW = Math.max(stats.maxTileW, ...tiles.map((tile) => tile.padW));
    stats.maxTileH = Math.max(stats.maxTileH, ...tiles.map((tile) => tile.padH));

    try {
      ensurePipelines();
      ensureInBuf(largestTile.padW, largestTile.padH);
      ensureSnapshotTex(srcW, srcH);
      ensureOutTex(outW, outH);
      const runInBuf = inBuf;
      const runSnapshotU = snapshotU;
      const runPackU = packU;
      const runCompU = compU;
      const runSnapshotExtPipe = snapshotExtPipe;
      const runSnapshotTexPipe = snapshotTexPipe;
      const runPackTilePipe = packTilePipe;
      const runCompPipe = compPipe;
      const runSampler = sampler;
      const runSnapshotTex = snapshotTex;
      const runOutTex = outTex;

      const writePackUniform = (tile) => {
        runDevice.queue.writeBuffer(runPackU, 0, new Uint32Array([
          tile.padW,
          tile.padH,
          tile.inputW,
          tile.inputH,
          tile.inputX,
          tile.inputY,
          srcW,
          srcH,
        ]));
      };
      const encodePack = (encoder, tile) => {
        const bg = runDevice.createBindGroup({
          layout: runPackTilePipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: runSnapshotTex.createView() },
            { binding: 1, resource: { buffer: runInBuf } },
            { binding: 2, resource: { buffer: runPackU } },
          ],
        });
        const cp = encoder.beginComputePass();
        cp.setPipeline(runPackTilePipe);
        cp.setBindGroup(0, bg);
        cp.dispatchWorkgroups(Math.ceil(tile.padW / 8), Math.ceil(tile.padH / 8));
        cp.end();
      };

      // Capture the whole external/texture source and pack the first tile in one
      // submission before the first await. External textures expire at task end;
      // every later tile reads only the persistent snapshot.
      runDevice.queue.writeBuffer(runSnapshotU, 0, new Uint32Array([srcW, srcH, 0, 0]));
      writePackUniform(tiles[0]);
      {
        const isTex = !!(src && src.tex);
        const pipe = isTex ? runSnapshotTexPipe : runSnapshotExtPipe;
        const enc = runDevice.createCommandEncoder();
        const snapshotBg = runDevice.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: runSampler },
            { binding: 1, resource: isTex ? src.tex.createView() : src },
            { binding: 2, resource: runSnapshotTex.createView() },
            { binding: 3, resource: { buffer: runSnapshotU } },
          ],
        });
        const cp = enc.beginComputePass();
        cp.setPipeline(pipe);
        cp.setBindGroup(0, snapshotBg);
        cp.dispatchWorkgroups(Math.ceil(srcW / 8), Math.ceil(srcH / 8));
        cp.end();
        encodePack(enc, tiles[0]);
        runDevice.queue.submit([enc.finish()]);
      }

      // The external source is now consumed on the new device. It is safe to let
      // sessions retaining a prior device go because the presentation owner
      // adopts this device before it can call run().
      await releaseDeferredSessions();
      if (runLifecycleGeneration !== lifecycleGeneration) throw new Error("neural inference cancelled by stop");

      for (let tileIndex = 0; tileIndex < tiles.length; tileIndex++) {
        const tile = tiles[tileIndex];
        let inTensor = null;
        let auxiliaryTensors = [];
        let resultTensors = [];
        let retainedResultTensors = new Set();
        try {
          if (tileIndex > 0) {
            if (runLifecycleGeneration !== lifecycleGeneration) {
              throw new Error("neural inference cancelled by stop");
            }
            writePackUniform(tile);
            const packEncoder = runDevice.createCommandEncoder();
            encodePack(packEncoder, tile);
            runDevice.queue.submit([packEncoder.finish()]);
          }

          inTensor = runOrt.Tensor.fromGpuBuffer(runInBuf, {
            dataType: "float32",
            dims: [1, 3, tile.padH, tile.padW],
          });
          const feeds = { [runInputName]: inTensor };
          let fetches = [runOutputName];
          if (isV2) {
            for (const descriptor of Object.values(runGraph.inputs)) {
              if (descriptor.role === "rgb") continue;
              if (descriptor.role === "state-in") {
                feeds[descriptor.name] = temporalState.get(descriptor.state);
                continue;
              }
              const provided = runAuxiliary[descriptor.name];
              const expectedSpatial =
                descriptor.role === "motion" || descriptor.role === "residual"
                  ? { width: tile.padW, height: tile.padH }
                  : null;
              const tensorOptions = validateRuntimeTensor(
                provided,
                descriptor,
                `neural auxiliary input '${descriptor.name}'`,
                expectedSpatial,
              );
              const wrapper = runOrt.Tensor.fromGpuBuffer(provided.gpuBuffer, tensorOptions);
              auxiliaryTensors.push(wrapper);
              feeds[descriptor.name] = wrapper;
            }
            fetches = Object.keys(runGraph.outputs);
          }
          const result = await runSession.run(feeds, fetches);
          if (!result || typeof result !== "object") {
            throw new Error("neural session returned an invalid result");
          }
          resultTensors = [...new Set(Object.values(result).filter(Boolean))];
          const outT = result[runOutputName];
          if (runLifecycleGeneration !== lifecycleGeneration) {
            throw new Error("neural inference cancelled by stop");
          }
          if (runSessionGeneration !== sessionGeneration || runPrimarySession !== session) {
            throw new Error("neural session changed during inference");
          }
          let nextTemporalState = null;
          if (isV2 && runModelContract.mode === "temporal") {
            nextTemporalState = new Map();
            for (const descriptor of Object.values(runGraph.outputs)) {
              if (descriptor.role !== "state-out") continue;
              const stateTensor = result[descriptor.name];
              validateRuntimeTensor(
                stateTensor,
                descriptor,
                `neural state output '${descriptor.name}'`,
              );
              nextTemporalState.set(descriptor.state, stateTensor);
            }
            if (nextTemporalState.size !== runModelContract.states.length ||
                runModelContract.states.some((key) => !nextTemporalState.has(key))) {
              throw new Error(`neural graph '${runGraphName}' returned incomplete recurrent state`);
            }
          }
          const shape = validateOutputTensor(
            outT,
            tile.padW,
            tile.padH,
            scale,
            runOutputName,
          );

          runDevice.queue.writeBuffer(runCompU, 0, new Uint32Array([
            shape.strideW,
            shape.plane,
            tile.cropX,
            tile.cropY,
            tile.dstX,
            tile.dstY,
            tile.outW,
            tile.outH,
          ]));
          const enc = runDevice.createCommandEncoder();
          const bg = runDevice.createBindGroup({
            layout: runCompPipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: outT.gpuBuffer } },
              { binding: 1, resource: runOutTex.createView() },
              { binding: 2, resource: { buffer: runCompU } },
            ],
          });
          const cp = enc.beginComputePass();
          cp.setPipeline(runCompPipe);
          cp.setBindGroup(0, bg);
          cp.dispatchWorkgroups(Math.ceil(tile.outW / 8), Math.ceil(tile.outH / 8));
          cp.end();
          runDevice.queue.submit([enc.finish()]);
          if (nextTemporalState) {
            const previousState = temporalState;
            temporalState = nextTemporalState;
            retainedResultTensors = new Set(nextTemporalState.values());
            retireTensors(
              runDevice,
              ...[...previousState.values()].filter(
                (tensor) => !retainedResultTensors.has(tensor),
              ),
            );
          }
          stats.tileRuns++;
        } finally {
          // Each tile can release its wrappers after its composite is submitted;
          // this bounds wrapper/output lifetime instead of retaining a frame's
          // worth of ONNX tensors until every tile finishes.
          retireTensors(
            runDevice,
            inTensor,
            ...auxiliaryTensors,
            ...resultTensors.filter((tensor) => !retainedResultTensors.has(tensor)),
          );
        }
      }

      const dt = performance.now() - t0;
      stats.last = dt;
      stats.mu = stats.n === 0 ? dt : stats.mu * 0.9 + dt * 0.1;
      stats.n++;
      return { tex: runOutTex, outW, outH };
    } catch (error) {
      stats.fails++;
      if (isV2 && runModelContract.mode === "temporal") clearTemporalState(runDevice);
      throw error;
    } finally {
      runBusy = false;
      endRun();
    }
  }

  function cancel() {
    // Publish logical cancellation synchronously. Recurrent wrappers retire
    // behind the queue fence; stop/dispose owns the remaining GPU/session cleanup
    // after the active session.run() observes this generation change and unwinds.
    ++lifecycleGeneration;
    ++initGeneration;
    if (activeRuns === 0) clearTemporalState();
  }

  function stop() {
    // Release transient GPU resources but keep the session alive. Releasing the
    // only session would tear down the device still used by the presentation
    // owner; the session persists until engine disposal.
    cancel();
    return destroyGpuResources();
  }

  async function quiesce() {
    // Cancel unpublished initialization and prevent active inference from
    // reaching a newer continuation, while retaining the committed session as
    // a device-lifetime guard until the presentation owner releases its resources.
    cancel();
    const pendingInit = initTail;
    const cleanup = destroyGpuResources();
    try { await pendingInit; } catch {}
    await whenRunsIdle();
    try { await cleanup; } catch {}
    while (retirements.size) {
      try { await Promise.allSettled([...retirements]); } catch {}
    }
  }

  function dispose() {
    if (disposalPromise) return disposalPromise;

    // Publish the empty state synchronously. This makes ready()/device() truthful
    // while an active run, queue fence, or stale session creation is draining.
    ++lifecycleGeneration;
    ++initGeneration;
    ++sessionGeneration;
    const oldSession = session;
    const oldDevice = device;
    const oldDeferredSessions = deferredSessionReleases;
    const priorSessionReleaseFailures = sessionReleaseFailures;
    const pendingInit = initTail;
    const pendingDeviceInvalidation = deviceInvalidationPromise;
    session = null;
    device = null;
    active = null;
    activeContract = null;
    inputName = "input";
    outputName = "output";
    fp16Model = false;
    runBusy = false;
    latestInitPromise = null;
    deferredSessionReleases = [];
    sessionReleaseFailures = [];
    const cleanup = destroyGpuResources(oldDevice);

    const operation = (async () => {
      const failures = new Set(priorSessionReleaseFailures);
      // invalidateDevice() may have unpublished the same session and retained
      // sole ownership of releasing it. Disposal is not complete until that
      // release is complete, even when there is nothing left to capture above.
      try { await pendingDeviceInvalidation; }
      catch (error) { failures.add(error); }
      // A watcher can publish another identity-specific invalidation while the
      // captured aggregate drains. Wait until the live set is empty, then consume
      // the settled-failure latch exactly once through this disposal report.
      while (activeDeviceInvalidations.size) {
        try { await refreshDeviceInvalidationBarrier(); }
        catch (error) { failures.add(error); }
      }
      deviceInvalidationFailures.clear();
      refreshDeviceInvalidationBarrier();
      try { await pendingInit; } catch {}
      await whenRunsIdle();
      try { await cleanup; } catch {}
      while (retirements.size) {
        try { await Promise.allSettled([...retirements]); } catch {}
      }
      // A losing initializer may transfer its prior committed session into the
      // replacement array while pendingInit drains. Close disposal over both
      // snapshots and the committed session, releasing each physical guard once.
      const sessionsToRelease = new Set([
        ...oldDeferredSessions,
        ...deferredSessionReleases,
        oldSession,
      ].filter(Boolean));
      deferredSessionReleases = [];
      for (const error of sessionReleaseFailures) failures.add(error);
      sessionReleaseFailures = [];
      for (const old of sessionsToRelease) {
        try { await releaseManagedSession(old); }
        catch (error) {
          failures.add(error);
          warn("neural: session release failed:", error.message);
        }
      }
      if (failures.size) {
        throw new AggregateError([...failures], "neural session disposal failed");
      }
    })().finally(() => {
      if (disposalPromise === operation) disposalPromise = null;
    });
    disposalPromise = operation;
    // New init() calls are serialized behind physical session release.
    initTail = operation.then(() => undefined, () => undefined);
    return operation;
  }

  return {
    models: loadManifest,
    init,
    run,
    cancel,
    stop,
    quiesce,
    dispose,
    invalidateDevice,
    ready: () => !!(session && device),
    activeEntry: () => active,
    activeContract: () => activeContract,
    device: () => device,
    stats: () => ({ ...stats }),
    bumpSkip: () => { stats.skip++; },
  };
}
