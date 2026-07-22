// Shared validation and dimension preflight for the generated WebGPU models.
// This module deliberately has no DOM, WebGPU-global, or Node dependencies so it
// can be consumed by extension runtime code, the browser validator, and CI.

// WebGPU exposes texture dimensions and binding limits, but not a trustworthy
// available-VRAM budget. Keep the calculated working set as diagnostics and
// allow callers to opt into a known deployment-specific cap; the portable
// default only rejects unsafe arithmetic and real device-limit violations.
export const DEFAULT_MODEL_WORKING_SET_BYTES = Number.MAX_SAFE_INTEGER;

const BYTES_PER_RGBA16FLOAT_TEXEL = 8;
const MODEL_NAME = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;
const RESOURCE_NAME = /^[A-Za-z_][A-Za-z0-9_]{0,127}$/;
const DEFAULT_LIMITS = Object.freeze({
  maxTextureDimension2D: 8192,
  maxBindingsPerBindGroup: 1000,
  maxSampledTexturesPerShaderStage: 16,
  maxStorageTexturesPerShaderStage: 4,
});

export class ModelBundleError extends Error {
  constructor(message, code = "MODEL_BUNDLE_INVALID") {
    super(message);
    this.name = "ModelBundleError";
    this.code = code;
  }
}

function invalid(message, code) {
  throw new ModelBundleError(message, code);
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) invalid(`${label} must be a positive safe integer`);
  return value;
}

function finitePositive(value, label) {
  if (!Number.isFinite(value) || value <= 0) invalid(`${label} must be finite and positive`);
  return value;
}

function limitValue(limits, name) {
  const value = Number(limits?.[name]);
  return Number.isSafeInteger(value) && value > 0 ? value : DEFAULT_LIMITS[name];
}

function checkedProduct(values, label) {
  let product = 1;
  for (const value of values) {
    positiveInteger(value, label);
    product *= value;
    if (!Number.isSafeInteger(product)) invalid(`${label} exceeds the safe integer range`, "MODEL_DIMENSIONS_UNSAFE");
  }
  return product;
}

function checkedSum(left, right, label) {
  const value = left + right;
  if (!Number.isSafeInteger(value)) invalid(`${label} exceeds the safe integer range`, "MODEL_DIMENSIONS_UNSAFE");
  return value;
}

function requireObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) invalid(`${label} must be an object`);
}

function requireResourceName(value, label) {
  if (typeof value !== "string" || !RESOURCE_NAME.test(value)) {
    invalid(`${label} must be a valid resource name`);
  }
}

function validateBindingLimits(pass, index, limits) {
  const sampled = limitValue(limits, "maxSampledTexturesPerShaderStage");
  const storage = limitValue(limits, "maxStorageTexturesPerShaderStage");
  const total = limitValue(limits, "maxBindingsPerBindGroup");
  if (pass.binds.length > sampled) {
    invalid(`pass ${index} has ${pass.binds.length} sampled textures; device limit is ${sampled}`,
      "MODEL_BINDING_LIMIT");
  }
  if (storage < 1 || pass.binds.length + 1 > total) {
    invalid(`pass ${index} exceeds the device bind-group limits`, "MODEL_BINDING_LIMIT");
  }
}

// Return a strict pass-index -> complete shader source map. The prelude before
// the first marker is prepended to every entry, matching the generated format.
export function splitModelEntries(wgslSource) {
  if (typeof wgslSource !== "string" || !wgslSource.length) invalid("WGSL source must be a non-empty string");

  const markerMatches = [...wgslSource.matchAll(/^\/\/==== ENTRY[^\r\n]*$/gm)];
  if (!markerMatches.length) invalid("WGSL source is missing //==== ENTRY markers");

  const parsed = markerMatches.map((match) => {
    const marker = match[0].match(/^\/\/==== ENTRY pass(0|[1-9]\d*)(?:\s*:\s*.*)?\s*$/);
    if (!marker) invalid(`malformed WGSL entry marker: ${match[0]}`);
    return { index: Number(marker[1]), offset: match.index };
  });
  const prelude = wgslSource.slice(0, parsed[0].offset);
  const entries = new Map();

  parsed.forEach((entry, position) => {
    if (entries.has(entry.index)) invalid(`duplicate WGSL entry marker for pass ${entry.index}`);
    if (entry.index !== position) {
      invalid(`WGSL entry markers must be ordered and contiguous; position ${position} is pass ${entry.index}`);
    }
    const end = position + 1 < parsed.length ? parsed[position + 1].offset : wgslSource.length;
    const code = prelude + wgslSource.slice(entry.offset, end);
    const computeCount = (code.match(/@compute\b/g) || []).length;
    const mainCount = (code.match(/\bfn\s+main\s*\(/g) || []).length;
    if (computeCount !== 1 || mainCount !== 1) {
      invalid(`WGSL entry pass ${entry.index} must contain exactly one @compute fn main`);
    }
    entries.set(entry.index, code);
  });
  return entries;
}

function stripWgslComments(source) {
  // WGSL has no string literals, so comments can be removed before inspecting
  // resource declarations without accidentally changing declaration text.
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/[^\r\n]*/g, "");
}

function validateBindingDeclarations(entries, passes) {
  for (const pass of passes) {
    const code = stripWgslComments(entries.get(pass.index));
    const expectedCount = pass.binds.length + 1;
    const groups = [...code.matchAll(/@group\s*\(\s*(\d+)\s*\)/g)]
      .map((match) => Number(match[1]));
    if (groups.some((group) => group !== 0)) {
      invalid(`WGSL entry pass ${pass.index} may only declare bind group 0`);
    }

    const bindingAttributes = [...code.matchAll(/@binding\s*\(\s*(\d+)\s*\)/g)];
    const declarations = [...code.matchAll(
      /@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var\s+([A-Za-z_]\w*)\s*:\s*([^;]+);/g,
    )];
    if (groups.length !== expectedCount || bindingAttributes.length !== expectedCount ||
        declarations.length !== expectedCount) {
      invalid(`WGSL entry pass ${pass.index} bindings do not match its manifest inputs/output`);
    }

    const byBinding = new Map();
    for (const declaration of declarations) {
      const group = Number(declaration[1]);
      const binding = Number(declaration[2]);
      if (group !== 0 || !Number.isSafeInteger(binding) || binding < 0 || byBinding.has(binding)) {
        invalid(`WGSL entry pass ${pass.index} bindings do not match its manifest inputs/output`);
      }
      byBinding.set(binding, declaration[4].replace(/\s+/g, ""));
    }
    if (byBinding.size !== expectedCount ||
        Array.from({ length: expectedCount }, (_, binding) => binding)
          .some((binding) => !byBinding.has(binding))) {
      invalid(`WGSL entry pass ${pass.index} bindings do not match its manifest inputs/output`);
    }

    for (let binding = 0; binding < pass.binds.length; binding++) {
      if (byBinding.get(binding) !== "texture_2d<f32>") {
        invalid(`WGSL entry pass ${pass.index} input binding ${binding} must be a sampled texture_2d<f32>`);
      }
    }
    if (byBinding.get(pass.binds.length) !== "texture_storage_2d<rgba16float,write>") {
      invalid(`WGSL entry pass ${pass.index} output binding ${pass.binds.length} must be a writable rgba16float storage texture`);
    }
  }
}

function immutableManifestCopy(manifest) {
  const passes = manifest.passes.map((pass) => Object.freeze({
    ...pass,
    binds: Object.freeze([...pass.binds]),
  }));
  return Object.freeze({ ...manifest, passes: Object.freeze(passes) });
}

function validateCommonManifest(manifest, { expectedName, deviceLimits } = {}) {
  requireObject(manifest, "model manifest");
  if (typeof manifest.name !== "string" || !MODEL_NAME.test(manifest.name)) {
    invalid("manifest name is missing or invalid");
  }
  if (expectedName != null && manifest.name !== expectedName) {
    invalid(`manifest name ${manifest.name} does not match expected model ${expectedName}`);
  }
  finitePositive(manifest.whenThreshold, "manifest whenThreshold");
  if (!Array.isArray(manifest.passes) || !manifest.passes.length) invalid("manifest passes must be a non-empty array");
  if (manifest.passes.length > 512) invalid("manifest contains too many passes");

  const available = new Set(["LUMA"]);
  const dimensionsBySave = new Map();
  for (let index = 0; index < manifest.passes.length; index++) {
    const pass = manifest.passes[index];
    requireObject(pass, `pass ${index}`);
    if (pass.index !== index) invalid(`pass ${index} has non-contiguous index ${pass.index}`);
    if (typeof pass.desc !== "string" || !pass.desc.trim()) invalid(`pass ${index} has no description`);
    if (!Array.isArray(pass.binds) || !pass.binds.length) invalid(`pass ${index} has no inputs`);
    const seenBinds = new Set();
    for (const bind of pass.binds) {
      requireResourceName(bind, `pass ${index} input`);
      if (seenBinds.has(bind)) invalid(`pass ${index} binds ${bind} more than once`);
      seenBinds.add(bind);
      if (!available.has(bind)) invalid(`pass ${index} binds unavailable resource ${bind}`);
    }
    positiveInteger(pass.widthMul, `pass ${index} widthMul`);
    positiveInteger(pass.heightMul, `pass ${index} heightMul`);
    validateBindingLimits(pass, index, deviceLimits);

    if (pass.save != null) {
      requireResourceName(pass.save, `pass ${index} save`);
      if (pass.save === "LUMA" || pass.save === "OUTPUT") {
        invalid(`pass ${index} cannot overwrite reserved resource ${pass.save}`);
      }
      if (seenBinds.has(pass.save)) invalid(`pass ${index} reads and writes ${pass.save}`);
      const oldDims = dimensionsBySave.get(pass.save);
      if (oldDims && (oldDims[0] !== pass.widthMul || oldDims[1] !== pass.heightMul)) {
        invalid(`resource ${pass.save} is reused with inconsistent dimensions`);
      }
      dimensionsBySave.set(pass.save, [pass.widthMul, pass.heightMul]);
      available.add(pass.save);
    }
  }
  return { dimensionsBySave };
}

function validateFsrcnnxManifest(manifest, options) {
  validateCommonManifest(manifest, options);
  let terminalCount = 0;
  let scale = null;
  for (let index = 0; index < manifest.passes.length; index++) {
    const pass = manifest.passes[index];
    if (pass.kind === "conv") {
      if (index === manifest.passes.length - 1) invalid("FSRCNNX final pass must be shuffle");
      if (pass.save == null) invalid(`FSRCNNX conv pass ${index} must save an output`);
      if (pass.widthMul !== 1 || pass.heightMul !== 1) {
        invalid(`FSRCNNX conv pass ${index} must use native dimensions`);
      }
      if (!Number.isInteger(pass.components) || pass.components < 1 || pass.components > 4) {
        invalid(`FSRCNNX conv pass ${index} has invalid components`);
      }
    } else if (pass.kind === "shuffle") {
      terminalCount++;
      if (index !== manifest.passes.length - 1) invalid("FSRCNNX shuffle pass must be final");
      if (pass.save != null) invalid("FSRCNNX shuffle pass cannot save a logical intermediate");
      if (pass.widthMul !== pass.heightMul) invalid("FSRCNNX shuffle scale must be square");
      scale = pass.widthMul;
    } else {
      invalid(`FSRCNNX pass ${index} has unsupported kind ${pass.kind}`);
    }
  }
  if (terminalCount !== 1) invalid("FSRCNNX manifest must contain exactly one terminal shuffle pass");
  return scale;
}

function validateArtCnnManifest(manifest, options) {
  const { dimensionsBySave } = validateCommonManifest(manifest, options);
  const scale = positiveInteger(manifest.scale, "ArtCNN scale");
  if (scale !== 2) invalid("ArtCNN runtime currently supports exactly 2x models");
  let terminalCount = 0;

  for (let index = 0; index < manifest.passes.length; index++) {
    const pass = manifest.passes[index];
    if (pass.kind === "conv") {
      if (index === manifest.passes.length - 1) invalid("ArtCNN final pass must be d2s");
      if (pass.save == null) invalid(`ArtCNN conv pass ${index} must save an output`);
      if (!Number.isInteger(pass.numResults) || ![1, 8].includes(pass.numResults)) {
        invalid(`ArtCNN conv pass ${index} has unsupported result count`);
      }
      const packed = pass.numResults === 8;
      if ((packed && (pass.widthMul !== 4 || pass.heightMul !== 2)) ||
          (!packed && (pass.widthMul !== 1 || pass.heightMul !== 1))) {
        invalid(`ArtCNN conv pass ${index} dimensions do not match its result layout`);
      }
      if (typeof pass.relu !== "boolean" || typeof pass.skipSum !== "boolean") {
        invalid(`ArtCNN conv pass ${index} has invalid activation metadata`);
      }
      if (pass.binds.length !== (pass.skipSum ? 2 : 1)) {
        invalid(`ArtCNN conv pass ${index} inputs do not match skipSum`);
      }
    } else if (pass.kind === "d2s") {
      terminalCount++;
      if (index !== manifest.passes.length - 1) invalid("ArtCNN d2s pass must be final");
      if (pass.save != null) invalid("ArtCNN d2s pass cannot save a logical intermediate");
      if (pass.binds.length !== 1) invalid("ArtCNN d2s pass must have one input");
      if (pass.widthMul !== scale || pass.heightMul !== scale || pass.numResults !== 0) {
        invalid("ArtCNN d2s metadata does not match manifest scale");
      }
      const inputDims = dimensionsBySave.get(pass.binds[0]);
      if (!inputDims || inputDims[0] !== 1 || inputDims[1] !== 1) {
        invalid("ArtCNN d2s input must be a native-resolution saved texture");
      }
    } else {
      invalid(`ArtCNN pass ${index} has unsupported kind ${pass.kind}`);
    }
  }
  if (terminalCount !== 1) invalid("ArtCNN manifest must contain exactly one terminal d2s pass");
  return scale;
}

export function validateModelBundle(kind, manifest, wgslSource, options = {}) {
  if (kind !== "fsrcnnx" && kind !== "artcnn") invalid(`unsupported model bundle kind ${kind}`);
  const scale = kind === "fsrcnnx"
    ? validateFsrcnnxManifest(manifest, options)
    : validateArtCnnManifest(manifest, options);
  const entries = splitModelEntries(wgslSource);
  if (entries.size !== manifest.passes.length) {
    invalid(`manifest has ${manifest.passes.length} passes but WGSL has ${entries.size} entries`);
  }
  for (let index = 0; index < manifest.passes.length; index++) {
    if (!entries.has(index)) invalid(`WGSL is missing entry for pass ${index}`);
  }
  validateBindingDeclarations(entries, manifest.passes);
  // Runtime behavior must not be changeable by mutating the parsed JSON object
  // after it has passed validation.
  return { entries, scale, manifest: immutableManifestCopy(manifest) };
}

export function preflightModelDimensions(kind, manifest, lumaW, lumaH, options = {}) {
  if (kind !== "fsrcnnx" && kind !== "artcnn") invalid(`unsupported model kind ${kind}`);
  positiveInteger(lumaW, "model input width");
  positiveInteger(lumaH, "model input height");
  const maxDimension = limitValue(options.deviceLimits, "maxTextureDimension2D");
  const maxWorkingSetBytes = options.maxWorkingSetBytes ?? DEFAULT_MODEL_WORKING_SET_BYTES;
  positiveInteger(maxWorkingSetBytes, "model working-set budget");
  if (lumaW > maxDimension || lumaH > maxDimension) {
    invalid(`model input ${lumaW}x${lumaH} exceeds maxTextureDimension2D ${maxDimension}`,
      "MODEL_DIMENSION_LIMIT");
  }

  const scale = kind === "artcnn"
    ? positiveInteger(manifest.scale, "ArtCNN scale")
    : positiveInteger(manifest.passes.find((pass) => pass.kind === "shuffle")?.widthMul,
      "FSRCNNX scale");
  const resources = [];
  const saves = new Map();
  for (const pass of manifest.passes) {
    if (pass.save == null || saves.has(pass.save)) continue;
    const width = checkedProduct([lumaW, pass.widthMul], `${pass.save} width`);
    const height = checkedProduct([lumaH, pass.heightMul], `${pass.save} height`);
    saves.set(pass.save, { name: pass.save, width, height });
  }
  resources.push(...saves.values());
  const outputWidth = checkedProduct([lumaW, scale], "model output width");
  const outputHeight = checkedProduct([lumaH, scale], "model output height");
  resources.push({ name: "OUTPUT", width: outputWidth, height: outputHeight });

  let texels = 0;
  for (const resource of resources) {
    if (resource.width > maxDimension || resource.height > maxDimension) {
      invalid(`${resource.name} texture ${resource.width}x${resource.height} exceeds maxTextureDimension2D ${maxDimension}`,
        "MODEL_DIMENSION_LIMIT");
    }
    texels = checkedSum(texels, checkedProduct([resource.width, resource.height], `${resource.name} area`),
      "model texture area");
  }
  const workingSetBytes = checkedProduct([texels, BYTES_PER_RGBA16FLOAT_TEXEL], "model working set");
  if (workingSetBytes > maxWorkingSetBytes) {
    invalid(`model working set ${workingSetBytes} bytes exceeds budget ${maxWorkingSetBytes}`,
      "MODEL_WORKING_SET_LIMIT");
  }

  return {
    inputWidth: lumaW,
    inputHeight: lumaH,
    outputWidth,
    outputHeight,
    workingSetBytes,
    maxWorkingSetBytes,
    maxTextureDimension2D: maxDimension,
    resources,
  };
}

export function preflightModelChain(stages, lumaW, lumaH, label = "model") {
  if (!Array.isArray(stages) || !stages.length) invalid(`${label} chain must contain at least one stage`);
  positiveInteger(lumaW, `${label} chain input width`);
  positiveInteger(lumaH, `${label} chain input height`);
  let width = lumaW, height = lumaH, workingSetBytes = 0;
  const plans = [];
  const budget = Math.min(...stages.map((stage, index) => {
    if (!stage || typeof stage.preflight !== "function") invalid(`${label} stage ${index} cannot be preflighted`);
    const value = stage.maxWorkingSetBytes ?? DEFAULT_MODEL_WORKING_SET_BYTES;
    return positiveInteger(value, `${label} stage ${index} working-set budget`);
  }));
  for (let index = 0; index < stages.length; index++) {
    const plan = stages[index].preflight(width, height);
    requireObject(plan, `${label} stage ${index} plan`);
    positiveInteger(plan.workingSetBytes, `${label} stage ${index} working set`);
    positiveInteger(plan.outputWidth, `${label} stage ${index} output width`);
    positiveInteger(plan.outputHeight, `${label} stage ${index} output height`);
    plans.push(plan);
    workingSetBytes = checkedSum(workingSetBytes, plan.workingSetBytes, `${label} chain working set`);
    width = plan.outputWidth;
    height = plan.outputHeight;
  }
  if (workingSetBytes > budget) {
    invalid(`${label} chain working set ${workingSetBytes} bytes exceeds budget ${budget}`,
      "MODEL_WORKING_SET_LIMIT");
  }
  return { plans, workingSetBytes, maxWorkingSetBytes: budget, outputWidth: width, outputHeight: height };
}

// Stage allocations cannot all publish atomically through WebGPU's synchronous
// creation API. If any later stage fails, clear every stage's retryable allocation
// cache so callers never observe or reuse a mixed old/new chain generation.
export function allocateModelChain(stages, lumaW, lumaH, lumaTexture, label = "model") {
  const chainPlan = preflightModelChain(stages, lumaW, lumaH, label);
  let width = lumaW, height = lumaH, input = lumaTexture;
  try {
    for (let index = 0; index < stages.length; index++) {
      const stage = stages[index];
      if (typeof stage.allocate !== "function" || typeof stage.resetAllocation !== "function") {
        invalid(`${label} stage ${index} does not implement retryable allocation`);
      }
      stage.allocate(width, height, input);
      if (!stage.outputTexture) invalid(`${label} stage ${index} did not publish an output texture`);
      input = stage.outputTexture;
      width = chainPlan.plans[index].outputWidth;
      height = chainPlan.plans[index].outputHeight;
    }
    return chainPlan;
  } catch (error) {
    for (const stage of stages) {
      try { stage?.resetAllocation?.(); } catch {}
    }
    throw error;
  }
}
