// Pure geometry and device-limit planning for contract-v2 temporal models.
//
// The planner owns no GPU resources and applies no quality policy. A selected
// model's audited execution profile determines its halo and largest logical
// tensor footprint; source dimensions only determine how many exact-owned core
// tiles are required.

const MIB = 1024 * 1024;
const DEFAULT_LIMITS = Object.freeze({
  maxBufferSize: 256 * MIB,
  maxStorageBufferBindingSize: 128 * MIB,
  maxTextureDimension2D: 8192,
  maxTextureArrayLayers: 256,
  maxComputeWorkgroupsPerDimension: 65_535,
});

export const CDA_TEMPORAL_LOGICAL_BYTES = Object.freeze({
  fp32: 776,
  // The 194-channel recurrent concat is 388 B/pixel in FP16, but the
  // 128-channel motion-warp value crosses the audited FP32 GridSample island.
  // Its 512 B/pixel buffer is therefore the mixed graph's binding maximum.
  mixed: 512,
});

function invalid(message) {
  throw new Error(`invalid temporal tiling contract: ${message}`);
}

function limitError(message) {
  const error = new Error(message);
  error.code = "NEURAL_LIMIT";
  return error;
}

function positiveSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive safe integer`);
  }
  return value;
}

function nonNegativeSafeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a non-negative safe integer`);
  }
  return value;
}

function checkedProduct(label, ...values) {
  let product = 1;
  for (const value of values) {
    positiveSafeInteger(value, label);
    if (product > Math.floor(Number.MAX_SAFE_INTEGER / value)) {
      throw limitError(`${label} exceeds the safe integer range`);
    }
    product *= value;
  }
  return product;
}

function checkedSum(label, left, right) {
  nonNegativeSafeInteger(left, label);
  nonNegativeSafeInteger(right, label);
  if (left > Number.MAX_SAFE_INTEGER - right) {
    throw limitError(`${label} exceeds the safe integer range`);
  }
  return left + right;
}

function checkedScale(label, value, scale) {
  nonNegativeSafeInteger(value, label);
  positiveSafeInteger(scale, label);
  if (value > Math.floor(Number.MAX_SAFE_INTEGER / scale)) {
    throw limitError(`${label} exceeds the safe integer range`);
  }
  return value * scale;
}

function alignDown(value, alignment) {
  return Math.floor(value / alignment) * alignment;
}

function alignUpPhysical(value, alignment, label) {
  const remainder = value % alignment;
  return remainder === 0
    ? value
    : checkedSum(label, value, alignment - remainder);
}

function normalizedLimit(limits, name) {
  const selected = limits[name] ?? DEFAULT_LIMITS[name];
  if (!Number.isSafeInteger(selected) || selected <= 0) {
    throw new Error(`device limit ${name} must be a positive safe integer`);
  }
  return selected;
}

function normalizeLimits(value) {
  if (value == null) value = {};
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("temporal tiling device limits must be an object");
  }
  return Object.freeze({
    maxBufferSize: normalizedLimit(value, "maxBufferSize"),
    maxStorageBufferBindingSize: normalizedLimit(
      value,
      "maxStorageBufferBindingSize",
    ),
    maxTextureDimension2D: normalizedLimit(value, "maxTextureDimension2D"),
    maxTextureArrayLayers: normalizedLimit(value, "maxTextureArrayLayers"),
    maxComputeWorkgroupsPerDimension: normalizedLimit(
      value,
      "maxComputeWorkgroupsPerDimension",
    ),
  });
}

function normalizeProfile(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    invalid("profile must be an object");
  }
  const kind = value.kind ?? "temporal-state-atlas-v1";
  if (kind !== "temporal-state-atlas-v1") invalid(`unsupported kind '${kind}'`);

  const scale = value.scale;
  const halo = value.halo;
  const largestLogicalBytesPerSourcePixel =
    value.largestLogicalBytesPerSourcePixel;
  const preferredInputExtent = value.preferredInputExtent ?? 512;
  const inputAlignment = value.inputAlignment ?? 8;
  const stateArrayLayers = value.stateArrayLayers ?? 16;
  const workgroupSize = value.workgroupSize ?? 8;

  for (const [label, candidate] of [
    ["scale", scale],
    ["largestLogicalBytesPerSourcePixel", largestLogicalBytesPerSourcePixel],
    ["preferredInputExtent", preferredInputExtent],
    ["inputAlignment", inputAlignment],
    ["stateArrayLayers", stateArrayLayers],
    ["workgroupSize", workgroupSize],
  ]) {
    if (!Number.isSafeInteger(candidate) || candidate <= 0) {
      invalid(`${label} must be a positive safe integer`);
    }
  }
  if (!Number.isSafeInteger(halo) || halo < 0) {
    invalid("halo must be a non-negative safe integer");
  }
  if (halo > Math.floor((Number.MAX_SAFE_INTEGER - 1) / 2)) {
    invalid("halo is too large");
  }
  if (preferredInputExtent <= halo * 2) {
    invalid("preferredInputExtent must leave a positive core after both halos");
  }

  return Object.freeze({
    kind,
    scale,
    halo,
    largestLogicalBytesPerSourcePixel,
    preferredInputExtent,
    inputAlignment,
    stateArrayLayers,
    workgroupSize,
  });
}

export function deriveCdaTemporalHalo(searchRadius = 8) {
  nonNegativeSafeInteger(searchRadius, "CDA temporal search radius");
  const recurrentRadius = checkedSum(
    "CDA temporal recurrent radius",
    searchRadius,
    35,
  );
  const aligned = alignUpPhysical(
    recurrentRadius,
    8,
    "CDA temporal recurrent radius",
  );
  return Math.max(64, aligned);
}

export function createCdaTemporalTilingProfile({
  searchRadius = 8,
  largestLogicalBytesPerSourcePixel = CDA_TEMPORAL_LOGICAL_BYTES.fp32,
  preferredInputExtent = 512,
} = {}) {
  return normalizeProfile({
    kind: "temporal-state-atlas-v1",
    scale: 4,
    halo: deriveCdaTemporalHalo(searchRadius),
    largestLogicalBytesPerSourcePixel,
    preferredInputExtent,
    inputAlignment: 8,
    stateArrayLayers: 16,
    workgroupSize: 8,
  });
}

function hardInputExtent(bindingLimit, bytesPerPixel) {
  const quotient = Math.floor(bindingLimit / bytesPerPixel);
  let extent = Math.floor(Math.sqrt(quotient));
  while (extent > 0 && extent * extent > quotient) extent--;
  while ((extent + 1) * (extent + 1) <= quotient) extent++;
  return extent;
}

function validatePhysicalFrame(width, height, profile, limits) {
  if (width > limits.maxTextureDimension2D ||
      height > limits.maxTextureDimension2D) {
    throw limitError(
      `temporal source dimensions exceed the device texture limit ` +
      `${limits.maxTextureDimension2D}`,
    );
  }
  if (profile.stateArrayLayers > limits.maxTextureArrayLayers) {
    throw limitError(
      `temporal state atlas requires ${profile.stateArrayLayers} layers; ` +
      `device limit is ${limits.maxTextureArrayLayers}`,
    );
  }

  const outputWidth = checkedProduct(
    "temporal neural output width",
    width,
    profile.scale,
  );
  const outputHeight = checkedProduct(
    "temporal neural output height",
    height,
    profile.scale,
  );
  if (outputWidth > limits.maxTextureDimension2D ||
      outputHeight > limits.maxTextureDimension2D) {
    throw limitError(
      `temporal neural output dimensions exceed the device texture limit ` +
      `${limits.maxTextureDimension2D}`,
    );
  }
  if (Math.ceil(width / profile.workgroupSize) >
        limits.maxComputeWorkgroupsPerDimension ||
      Math.ceil(height / profile.workgroupSize) >
        limits.maxComputeWorkgroupsPerDimension) {
    throw limitError(
      `temporal source dispatch exceeds the device workgroup limit ` +
      `${limits.maxComputeWorkgroupsPerDimension}`,
    );
  }
  return Object.freeze({ outputWidth, outputHeight });
}

export function planTemporalNeuralTiling(
  sourceWidth,
  sourceHeight,
  declaredProfile,
  deviceLimits = {},
) {
  positiveSafeInteger(sourceWidth, "temporal source width");
  positiveSafeInteger(sourceHeight, "temporal source height");
  const profile = normalizeProfile(declaredProfile);
  const limits = normalizeLimits(deviceLimits);
  const { outputWidth, outputHeight } = validatePhysicalFrame(
    sourceWidth,
    sourceHeight,
    profile,
    limits,
  );

  const bindingLimit = Math.min(
    limits.maxBufferSize,
    limits.maxStorageBufferBindingSize,
  );
  let hardExtent = hardInputExtent(
    bindingLimit,
    profile.largestLogicalBytesPerSourcePixel,
  );
  while (hardExtent > 0) {
    const bytes = alignUpPhysical(
      checkedProduct(
        "temporal tile logical tensor",
        hardExtent,
        hardExtent,
        profile.largestLogicalBytesPerSourcePixel,
      ),
      16,
      "temporal tile logical tensor",
    );
    if (bytes <= bindingLimit) break;
    hardExtent--;
  }
  if (hardExtent < 1) {
    throw limitError("device cannot bind one temporal source pixel");
  }

  const unalignedExtent = Math.min(
    hardExtent,
    profile.preferredInputExtent,
  );
  const alignedExtent = alignDown(
    unalignedExtent,
    profile.inputAlignment,
  );
  const maxInputExtent = alignedExtent || unalignedExtent;
  const fullCoreExtent = maxInputExtent - profile.halo * 2;
  const requiresTiling =
    sourceWidth > maxInputExtent || sourceHeight > maxInputExtent;
  if (requiresTiling && fullCoreExtent < 1) {
    throw limitError(
      `device tensor limits allow input extent ${maxInputExtent}, ` +
      `which cannot contain both ${profile.halo}-pixel halos`,
    );
  }
  const coreExtent = requiresTiling
    ? fullCoreExtent
    : Math.max(sourceWidth, sourceHeight);

  const tiles = [];
  let maxTileLogicalBytes = 0;
  let row = 0;
  for (let coreY = 0; coreY < sourceHeight; coreY += coreExtent, row++) {
    const coreHeight = Math.min(coreExtent, sourceHeight - coreY);
    let column = 0;
    for (let coreX = 0; coreX < sourceWidth; coreX += coreExtent, column++) {
      const coreWidth = Math.min(coreExtent, sourceWidth - coreX);
      const inputX = Math.max(0, coreX - profile.halo);
      const inputY = Math.max(0, coreY - profile.halo);
      const inputRight = Math.min(
        sourceWidth,
        checkedSum(
          "temporal tile input right",
          coreX,
          checkedSum(
            "temporal tile input right",
            coreWidth,
            profile.halo,
          ),
        ),
      );
      const inputBottom = Math.min(
        sourceHeight,
        checkedSum(
          "temporal tile input bottom",
          coreY,
          checkedSum(
            "temporal tile input bottom",
            coreHeight,
            profile.halo,
          ),
        ),
      );
      const inputWidth = inputRight - inputX;
      const inputHeight = inputBottom - inputY;
      if (inputWidth > maxInputExtent || inputHeight > maxInputExtent) {
        throw limitError("temporal tile geometry exceeds its planned input extent");
      }

      const logicalBytes = alignUpPhysical(
        checkedProduct(
          "temporal tile logical tensor",
          inputWidth,
          inputHeight,
          profile.largestLogicalBytesPerSourcePixel,
        ),
        16,
        "temporal tile logical tensor",
      );
      if (logicalBytes > bindingLimit) {
        throw limitError(
          `temporal tile logical tensor exceeds the device binding limit ` +
          `${bindingLimit}`,
        );
      }

      const stateCropX = coreX - inputX;
      const stateCropY = coreY - inputY;
      const cropX = checkedScale(
        "temporal RGB crop x",
        stateCropX,
        profile.scale,
      );
      const cropY = checkedScale(
        "temporal RGB crop y",
        stateCropY,
        profile.scale,
      );
      const dstX = checkedScale(
        "temporal RGB destination x",
        coreX,
        profile.scale,
      );
      const dstY = checkedScale(
        "temporal RGB destination y",
        coreY,
        profile.scale,
      );
      const outWidth = checkedProduct(
        "temporal tile output width",
        coreWidth,
        profile.scale,
      );
      const outHeight = checkedProduct(
        "temporal tile output height",
        coreHeight,
        profile.scale,
      );
      if (Math.ceil(inputWidth / profile.workgroupSize) >
            limits.maxComputeWorkgroupsPerDimension ||
          Math.ceil(inputHeight / profile.workgroupSize) >
            limits.maxComputeWorkgroupsPerDimension ||
          Math.ceil(outWidth / profile.workgroupSize) >
            limits.maxComputeWorkgroupsPerDimension ||
          Math.ceil(outHeight / profile.workgroupSize) >
            limits.maxComputeWorkgroupsPerDimension) {
        throw limitError(
          `temporal tile dispatch exceeds the device workgroup limit ` +
          `${limits.maxComputeWorkgroupsPerDimension}`,
        );
      }

      maxTileLogicalBytes = Math.max(maxTileLogicalBytes, logicalBytes);
      tiles.push(Object.freeze({
        index: tiles.length,
        row,
        column,
        coreX,
        coreY,
        coreWidth,
        coreHeight,
        inputX,
        inputY,
        inputWidth,
        inputHeight,
        stateCropX,
        stateCropY,
        cropX,
        cropY,
        dstX,
        dstY,
        outWidth,
        outHeight,
        logicalBytes,
      }));
    }
  }

  return Object.freeze({
    sourceWidth,
    sourceHeight,
    outputWidth,
    outputHeight,
    scale: profile.scale,
    halo: profile.halo,
    hardInputExtent: hardExtent,
    maxInputExtent,
    coreExtent,
    bindingLimit,
    maxTileLogicalBytes,
    profile,
    limits,
    tiles: Object.freeze(tiles),
  });
}
