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

const TEMPORAL_PROFILE_FIELDS = Object.freeze([
  "kind",
  "scale",
  "halo",
  "haloDerivation",
  "largestLogicalBytesPerSourcePixel",
  "preferredInputExtent",
  "inputAlignment",
  "workgroupSize",
  "stateAtlas",
]);
const HALO_DERIVATION_FIELDS = Object.freeze([
  "motionSearchRadius",
  "fixedRecurrentRadius",
  "minimum",
  "alignment",
]);
const STATE_ATLAS_FIELDS = Object.freeze([
  "stateCount",
  "channelsPerState",
  "arrayLayersPerState",
  "textureFormat",
]);
const LEGACY_PROFILE_FIELDS = Object.freeze([
  "scale",
  "halo",
  "largestLogicalBytesPerSourcePixel",
  "preferredInputExtent",
  "inputAlignment",
  "stateArrayLayers",
  "workgroupSize",
]);

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

function profileObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    invalid(`${label} must be an object`);
  }
  return value;
}

function requireExactFields(value, expected, label) {
  const expectedSet = new Set(expected);
  const actual = Object.keys(value);
  const missing = expected.filter((field) => !Object.hasOwn(value, field));
  const extra = actual.filter((field) => !expectedSet.has(field));
  if (missing.length) {
    invalid(`${label} is missing ${missing.join(", ")}`);
  }
  if (extra.length) {
    invalid(`${label} has unknown ${extra.join(", ")}`);
  }
}

function requireLegacyFields(value) {
  const allowed = new Set(["kind", ...LEGACY_PROFILE_FIELDS]);
  const missing = LEGACY_PROFILE_FIELDS.filter(
    (field) => !Object.hasOwn(value, field),
  );
  const extra = Object.keys(value).filter((field) => !allowed.has(field));
  if (missing.length) {
    invalid(`legacy profile is missing ${missing.join(", ")}`);
  }
  if (extra.length) {
    invalid(`legacy profile has unknown ${extra.join(", ")}`);
  }
}

function profilePositiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    invalid(`${label} must be a positive safe integer`);
  }
  return value;
}

function profileNonNegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    invalid(`${label} must be a non-negative safe integer`);
  }
  return value;
}

function deriveTemporalHalo({
  motionSearchRadius,
  fixedRecurrentRadius,
  minimum,
  alignment,
}) {
  const radius = motionSearchRadius + fixedRecurrentRadius;
  if (!Number.isSafeInteger(radius)) {
    invalid("haloDerivation radius exceeds the safe integer range");
  }
  const remainder = radius % alignment;
  const aligned = remainder === 0 ? radius : radius + alignment - remainder;
  if (!Number.isSafeInteger(aligned)) {
    invalid("haloDerivation result exceeds the safe integer range");
  }
  return Math.max(minimum, aligned);
}

export function normalizeTemporalNeuralTilingProfile(value) {
  profileObject(value, "profile");
  requireExactFields(value, TEMPORAL_PROFILE_FIELDS, "profile");
  if (value.kind !== "temporal-state-atlas-v1") {
    invalid(`unsupported kind '${value.kind}'`);
  }

  const scale = profilePositiveInteger(value.scale, "scale");
  const halo = profileNonNegativeInteger(value.halo, "halo");
  const largestLogicalBytesPerSourcePixel =
    profilePositiveInteger(
      value.largestLogicalBytesPerSourcePixel,
      "largestLogicalBytesPerSourcePixel",
    );
  const preferredInputExtent = profilePositiveInteger(
    value.preferredInputExtent,
    "preferredInputExtent",
  );
  const inputAlignment = profilePositiveInteger(
    value.inputAlignment,
    "inputAlignment",
  );
  const workgroupSize = profilePositiveInteger(
    value.workgroupSize,
    "workgroupSize",
  );
  if (workgroupSize !== 8) {
    invalid("workgroupSize must be 8");
  }
  if (halo > Math.floor((Number.MAX_SAFE_INTEGER - 1) / 2)) {
    invalid("halo is too large");
  }
  if (preferredInputExtent <= halo * 2) {
    invalid("preferredInputExtent must leave a positive core after both halos");
  }

  const rawDerivation = profileObject(
    value.haloDerivation,
    "haloDerivation",
  );
  requireExactFields(
    rawDerivation,
    HALO_DERIVATION_FIELDS,
    "haloDerivation",
  );
  const haloDerivation = Object.freeze({
    motionSearchRadius: profileNonNegativeInteger(
      rawDerivation.motionSearchRadius,
      "haloDerivation.motionSearchRadius",
    ),
    fixedRecurrentRadius: profileNonNegativeInteger(
      rawDerivation.fixedRecurrentRadius,
      "haloDerivation.fixedRecurrentRadius",
    ),
    minimum: profileNonNegativeInteger(
      rawDerivation.minimum,
      "haloDerivation.minimum",
    ),
    alignment: profilePositiveInteger(
      rawDerivation.alignment,
      "haloDerivation.alignment",
    ),
  });
  if (haloDerivation.motionSearchRadius !== 8) {
    invalid("haloDerivation.motionSearchRadius must be 8");
  }
  if (haloDerivation.fixedRecurrentRadius !== 35) {
    invalid("haloDerivation.fixedRecurrentRadius must be 35");
  }
  if (haloDerivation.minimum !== 64) {
    invalid("haloDerivation.minimum must be 64");
  }
  if (haloDerivation.alignment !== 8) {
    invalid("haloDerivation.alignment must be 8");
  }
  const derivedHalo = deriveTemporalHalo(haloDerivation);
  if (halo !== derivedHalo) {
    invalid(`halo ${halo} does not match derived halo ${derivedHalo}`);
  }

  const rawStateAtlas = profileObject(value.stateAtlas, "stateAtlas");
  requireExactFields(rawStateAtlas, STATE_ATLAS_FIELDS, "stateAtlas");
  const stateAtlas = Object.freeze({
    stateCount: profilePositiveInteger(
      rawStateAtlas.stateCount,
      "stateAtlas.stateCount",
    ),
    channelsPerState: profilePositiveInteger(
      rawStateAtlas.channelsPerState,
      "stateAtlas.channelsPerState",
    ),
    arrayLayersPerState: profilePositiveInteger(
      rawStateAtlas.arrayLayersPerState,
      "stateAtlas.arrayLayersPerState",
    ),
    textureFormat: rawStateAtlas.textureFormat,
  });
  if (stateAtlas.stateCount !== 2) {
    invalid("stateAtlas.stateCount must be 2");
  }
  if (stateAtlas.channelsPerState !== 64) {
    invalid("stateAtlas.channelsPerState must be 64");
  }
  if (stateAtlas.arrayLayersPerState !== 16 ||
      stateAtlas.arrayLayersPerState * 4 !== stateAtlas.channelsPerState) {
    invalid("stateAtlas.arrayLayersPerState must be 16");
  }
  if (stateAtlas.textureFormat !== "rgba16float" &&
      stateAtlas.textureFormat !== "rgba32float") {
    invalid("stateAtlas.textureFormat must be rgba16float or rgba32float");
  }

  return Object.freeze({
    kind: value.kind,
    scale,
    halo,
    haloDerivation,
    largestLogicalBytesPerSourcePixel,
    preferredInputExtent,
    inputAlignment,
    workgroupSize,
    stateAtlas,
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
  textureFormat = largestLogicalBytesPerSourcePixel ===
    CDA_TEMPORAL_LOGICAL_BYTES.fp32
    ? "rgba32float"
    : "rgba16float",
} = {}) {
  const haloDerivation = Object.freeze({
    motionSearchRadius: searchRadius,
    fixedRecurrentRadius: 35,
    minimum: 64,
    alignment: 8,
  });
  return normalizeTemporalNeuralTilingProfile({
    kind: "temporal-state-atlas-v1",
    scale: 4,
    halo: deriveCdaTemporalHalo(searchRadius),
    haloDerivation,
    largestLogicalBytesPerSourcePixel,
    preferredInputExtent,
    inputAlignment: 8,
    workgroupSize: 8,
    stateAtlas: {
      stateCount: 2,
      channelsPerState: 64,
      arrayLayersPerState: 16,
      textureFormat,
    },
  });
}

function normalizeLegacyPlannerProfile(value) {
  // Contract manifests use the strict exported normalizer. This adapter only
  // preserves the planner's historical flat profile used by direct unit calls.
  profileObject(value, "profile");
  requireLegacyFields(value);
  const kind = value.kind ?? "temporal-state-atlas-v1";
  if (kind !== "temporal-state-atlas-v1") invalid(`unsupported kind '${kind}'`);
  if (value.halo !== deriveCdaTemporalHalo(8)) {
    invalid("legacy profile halo must be the audited CDA default");
  }
  if (value.stateArrayLayers !== 16) {
    invalid("legacy profile stateArrayLayers must be 16");
  }
  return normalizeTemporalNeuralTilingProfile({
    kind,
    scale: value.scale,
    halo: value.halo,
    haloDerivation: {
      motionSearchRadius: 8,
      fixedRecurrentRadius: 35,
      minimum: 64,
      alignment: 8,
    },
    largestLogicalBytesPerSourcePixel:
      value.largestLogicalBytesPerSourcePixel,
    preferredInputExtent: value.preferredInputExtent,
    inputAlignment: value.inputAlignment,
    workgroupSize: value.workgroupSize,
    stateAtlas: {
      stateCount: 2,
      channelsPerState: 64,
      arrayLayersPerState: 16,
      textureFormat:
        value.largestLogicalBytesPerSourcePixel ===
          CDA_TEMPORAL_LOGICAL_BYTES.fp32
          ? "rgba32float"
          : "rgba16float",
    },
  });
}

function normalizePlannerProfile(value) {
  profileObject(value, "profile");
  if (Object.hasOwn(value, "haloDerivation") ||
      Object.hasOwn(value, "stateAtlas")) {
    return normalizeTemporalNeuralTilingProfile(value);
  }
  return normalizeLegacyPlannerProfile(value);
}

function hardInputExtent(bindingLimit, bytesPerPixel) {
  const quotient = Math.floor(bindingLimit / bytesPerPixel);
  let extent = Math.floor(Math.sqrt(quotient));
  while (extent > 0 && extent * extent > quotient) extent--;
  while ((extent + 1) * (extent + 1) <= quotient) extent++;
  return extent;
}

function logicalTileBytes(inputWidth, inputHeight, bytesPerPixel) {
  return alignUpPhysical(
    checkedProduct(
      "temporal tile logical tensor",
      inputWidth,
      inputHeight,
      bytesPerPixel,
    ),
    16,
    "temporal tile logical tensor",
  );
}

function preferredInputPixels(profile, bindingLimit) {
  // The audited preferred extent defines a square-equivalent work/memory
  // envelope. A rectangular tensor may trade width for height within the same
  // logical-pixel budget; the binding itself is linear and area-constrained.
  const preferredPixels = checkedProduct(
    "temporal preferred input area",
    profile.preferredInputExtent,
    profile.preferredInputExtent,
  );
  let pixels = Math.min(
    preferredPixels,
    Math.floor(bindingLimit / profile.largestLogicalBytesPerSourcePixel),
  );
  while (pixels > 0) {
    const logicalBytes = alignUpPhysical(
      checkedProduct(
        "temporal tile logical tensor",
        pixels,
        profile.largestLogicalBytesPerSourcePixel,
      ),
      16,
      "temporal tile logical tensor",
    );
    if (logicalBytes <= bindingLimit) return pixels;
    pixels--;
  }
  throw limitError("device cannot bind one temporal source pixel");
}

function axisCapacity(maximumInputLength, index, parts, halo) {
  if (maximumInputLength <= 0) return 0;
  if (parts === 1) return maximumInputLength;
  const edge = index === 0 || index === parts - 1;
  return Math.max(0, maximumInputLength - halo * (edge ? 1 : 2));
}

function axisTargetFeasible(
  length,
  parts,
  halo,
  alignment,
  maximumInputLength,
) {
  if (parts === 1) return length <= maximumInputLength;
  if (maximumInputLength >= length) return true;
  const effectiveAlignment = length >= parts * alignment ? alignment : 1;
  const lastRemainder = length % effectiveAlignment;
  const lastMinimum = lastRemainder || effectiveAlignment;
  let capacity = 0;
  for (let index = 0; index < parts; index++) {
    let partCapacity = axisCapacity(
      maximumInputLength,
      index,
      parts,
      halo,
    );
    if (index < parts - 1) {
      partCapacity = alignDown(partCapacity, effectiveAlignment);
      if (partCapacity < effectiveAlignment) return false;
    } else {
      if (partCapacity < lastMinimum) return false;
      partCapacity = lastMinimum +
        alignDown(partCapacity - lastMinimum, effectiveAlignment);
    }
    capacity = checkedSum(
      "temporal tile axis capacity",
      capacity,
      partCapacity,
    );
  }
  return capacity >= length;
}

function planTileAxis(length, parts, halo, alignment) {
  positiveSafeInteger(length, "temporal tile axis length");
  positiveSafeInteger(parts, "temporal tile axis parts");
  if (parts > length) {
    throw limitError("temporal tile grid cannot create empty cores");
  }
  if (parts === 1) {
    return Object.freeze({
      parts,
      maxInputLength: length,
      totalInputLength: length,
      segments: Object.freeze([Object.freeze({
        coreStart: 0,
        coreLength: length,
        inputStart: 0,
        inputLength: length,
      })]),
    });
  }

  let low = 1;
  let high = length;
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2);
    if (axisTargetFeasible(length, parts, halo, alignment, middle)) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  const maximumInputLength = low;
  const effectiveAlignment = length >= parts * alignment ? alignment : 1;
  const lastRemainder = length % effectiveAlignment;
  const lastMinimum = lastRemainder || effectiveAlignment;
  const widths = Array(parts).fill(effectiveAlignment);
  widths[parts - 1] = lastMinimum;
  const capacities = widths.map((_, index) => {
    const raw = maximumInputLength >= length
      ? length
      : axisCapacity(maximumInputLength, index, parts, halo);
    if (index < parts - 1) return alignDown(raw, effectiveAlignment);
    return lastMinimum + alignDown(raw - lastMinimum, effectiveAlignment);
  });
  const assigned = widths.reduce((sum, width) =>
    checkedSum("temporal tile axis assignment", sum, width), 0);
  let remaining = length - assigned;
  while (remaining > 0) {
    let selected = -1;
    let selectedSpan = Number.POSITIVE_INFINITY;
    for (let index = 0; index < parts; index++) {
      if (widths[index] + effectiveAlignment > capacities[index]) continue;
      const edge = index === 0 || index === parts - 1;
      const span = widths[index] + halo * (edge ? 1 : 2);
      if (span < selectedSpan) {
        selected = index;
        selectedSpan = span;
      }
    }
    if (selected < 0 || remaining < effectiveAlignment) {
      throw limitError("temporal tile axis could not satisfy its aligned budget");
    }
    widths[selected] += effectiveAlignment;
    remaining -= effectiveAlignment;
  }

  const segments = [];
  let coreStart = 0;
  let maxInputLength = 0;
  let totalInputLength = 0;
  for (const coreLength of widths) {
    const coreEnd = checkedSum(
      "temporal tile core end",
      coreStart,
      coreLength,
    );
    const inputStart = Math.max(0, coreStart - halo);
    const inputEnd = Math.min(
      length,
      checkedSum("temporal tile input end", coreEnd, halo),
    );
    const inputLength = inputEnd - inputStart;
    if (inputLength > maximumInputLength) {
      throw limitError("temporal tile axis exceeds its planned input span");
    }
    maxInputLength = Math.max(maxInputLength, inputLength);
    totalInputLength = checkedSum(
      "temporal tile total input span",
      totalInputLength,
      inputLength,
    );
    segments.push(Object.freeze({
      coreStart,
      coreLength,
      inputStart,
      inputLength,
    }));
    coreStart = coreEnd;
  }
  if (coreStart !== length) {
    throw limitError("temporal tile axis does not cover its source");
  }
  return Object.freeze({
    parts,
    maxInputLength,
    totalInputLength,
    segments: Object.freeze(segments),
  });
}

function compareGridCandidates(left, right) {
  if (!right) return -1;
  if (left.tileCount !== right.tileCount) {
    return left.tileCount - right.tileCount;
  }
  if (left.totalInputPixels !== right.totalInputPixels) {
    return left.totalInputPixels - right.totalInputPixels;
  }
  if (left.maxInputPixels !== right.maxInputPixels) {
    return left.maxInputPixels - right.maxInputPixels;
  }
  if (left.rows !== right.rows) return left.rows - right.rows;
  return left.columns - right.columns;
}

function selectRectangularGrid(
  sourceWidth,
  sourceHeight,
  profile,
  maximumInputPixels,
  conservativeCoreExtent,
) {
  const sourcePixels = checkedProduct(
    "temporal source area",
    sourceWidth,
    sourceHeight,
  );
  if (sourcePixels <= maximumInputPixels) {
    return Object.freeze({
      columns: 1,
      rows: 1,
      tileCount: 1,
      maxInputPixels: sourcePixels,
      totalInputPixels: sourcePixels,
      x: planTileAxis(sourceWidth, 1, profile.halo, profile.inputAlignment),
      y: planTileAxis(sourceHeight, 1, profile.halo, profile.inputAlignment),
    });
  }
  if (conservativeCoreExtent < 1) {
    throw limitError(
      `device tensor limits cannot contain both ${profile.halo}-pixel ` +
      `halos around a temporal core`,
    );
  }

  const fallbackColumns = Math.ceil(sourceWidth / conservativeCoreExtent);
  const fallbackRows = Math.ceil(sourceHeight / conservativeCoreExtent);
  let maximumTileCount = checkedProduct(
    "temporal fallback tile count",
    fallbackColumns,
    fallbackRows,
  );
  const xPlans = new Map();
  const yPlans = new Map();
  const xPlan = (parts) => {
    let plan = xPlans.get(parts);
    if (!plan) {
      plan = planTileAxis(
        sourceWidth,
        parts,
        profile.halo,
        profile.inputAlignment,
      );
      xPlans.set(parts, plan);
    }
    return plan;
  };
  const yPlan = (parts) => {
    let plan = yPlans.get(parts);
    if (!plan) {
      plan = planTileAxis(
        sourceHeight,
        parts,
        profile.halo,
        profile.inputAlignment,
      );
      yPlans.set(parts, plan);
    }
    return plan;
  };
  let best = null;
  const maximumColumns = Math.min(sourceWidth, maximumTileCount);
  for (let columns = 1; columns <= maximumColumns; columns++) {
    const maximumRows = Math.min(
      sourceHeight,
      Math.floor(maximumTileCount / columns),
    );
    for (let rows = 1; rows <= maximumRows; rows++) {
      const tileCount = checkedProduct(
        "temporal tile count",
        columns,
        rows,
      );
      const x = xPlan(columns);
      const y = yPlan(rows);
      const maxInputPixels = checkedProduct(
        "temporal maximum tile area",
        x.maxInputLength,
        y.maxInputLength,
      );
      if (maxInputPixels > maximumInputPixels) continue;
      const totalInputPixels = checkedProduct(
        "temporal total tile input area",
        x.totalInputLength,
        y.totalInputLength,
      );
      const candidate = {
        columns,
        rows,
        tileCount,
        maxInputPixels,
        totalInputPixels,
        x,
        y,
      };
      if (compareGridCandidates(candidate, best) < 0) {
        best = candidate;
        maximumTileCount = tileCount;
      }
    }
  }
  if (!best) {
    throw limitError("device tensor limits cannot form a temporal tile grid");
  }
  return Object.freeze(best);
}

function validatePhysicalFrame(width, height, profile, limits) {
  if (width > limits.maxTextureDimension2D ||
      height > limits.maxTextureDimension2D) {
    throw limitError(
      `temporal source dimensions exceed the device texture limit ` +
      `${limits.maxTextureDimension2D}`,
    );
  }
  if (profile.stateAtlas.arrayLayersPerState >
      limits.maxTextureArrayLayers) {
    throw limitError(
      `temporal state atlas requires ` +
      `${profile.stateAtlas.arrayLayersPerState} layers; ` +
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
  const profile = normalizePlannerProfile(declaredProfile);
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
  const maxInputPixels = preferredInputPixels(profile, bindingLimit);
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
  const grid = selectRectangularGrid(
    sourceWidth,
    sourceHeight,
    profile,
    maxInputPixels,
    fullCoreExtent,
  );
  // Keep the historical square-equivalent diagnostics for callers that
  // display device capacity. Rectangular tiles are bounded by maxInputPixels
  // and may have one side longer than maxInputExtent.
  const coreExtent = grid.tileCount > 1
    ? fullCoreExtent
    : Math.max(sourceWidth, sourceHeight);

  const tiles = [];
  let maxTileLogicalBytes = 0;
  for (let row = 0; row < grid.y.segments.length; row++) {
    const y = grid.y.segments[row];
    for (let column = 0; column < grid.x.segments.length; column++) {
      const x = grid.x.segments[column];
      const coreX = x.coreStart;
      const coreY = y.coreStart;
      const coreWidth = x.coreLength;
      const coreHeight = y.coreLength;
      const inputX = x.inputStart;
      const inputY = y.inputStart;
      const inputWidth = x.inputLength;
      const inputHeight = y.inputLength;
      const tileInputPixels = checkedProduct(
        "temporal tile input area",
        inputWidth,
        inputHeight,
      );
      if (tileInputPixels > maxInputPixels) {
        throw limitError("temporal tile geometry exceeds its planned input area");
      }

      const logicalBytes = logicalTileBytes(
        inputWidth,
        inputHeight,
        profile.largestLogicalBytesPerSourcePixel,
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
    maxInputPixels,
    coreExtent,
    bindingLimit,
    maxTileLogicalBytes,
    maxTileInputPixels: grid.maxInputPixels,
    maxTileInputWidth: grid.x.maxInputLength,
    maxTileInputHeight: grid.y.maxInputLength,
    totalTileInputPixels: grid.totalInputPixels,
    tileColumns: grid.columns,
    tileRows: grid.rows,
    profile,
    limits,
    tiles: Object.freeze(tiles),
  });
}
