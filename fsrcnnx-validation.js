// Browser-validation primitives shared by the extension page and unit tests.

import { LUMA_EXTRACT_WGSL, RECOMBINE_WGSL } from "./fsrcnnx-color.js";
import { buildSharpenShader } from "./fsrcnnx-sharpen.js";
import {
  buildL2Shader,
  buildMeanShader,
  SSIMDS_FINAL_WGSL,
  SSIMDS_MR_WGSL,
} from "./fsrcnnx-ssimds.js";

export const VALIDATION_TIMEOUT_MS = 30_000;

export const ONNX_VALIDATION_CHECKS = Object.freeze([
  Object.freeze({ id: "onnx:rife-v4.26-fp16", label: "ORT RIFE 4.26 FP16 WASM asset/inference smoke" }),
  Object.freeze({ id: "onnx:rife-v4.26", label: "ORT default RIFE 4.26 WebGPU inference smoke" }),
]);

export function createValidationPlan(catalog) {
  const plan = [
    Object.freeze({ id: "webgpu", label: "WebGPU device" }),
    Object.freeze({ id: "core:pipelines", label: "Supporting color/filter pipelines" }),
    Object.freeze({ id: "webgpu:errors", label: "GPU error channel" }),
    ...ONNX_VALIDATION_CHECKS,
  ];
  for (const spec of catalog) {
    plan.push(
      Object.freeze({ id: `${spec.name}:topology`, label: `${spec.label} topology` }),
      Object.freeze({ id: `${spec.name}:pipelines`, label: `${spec.label} pipelines` }),
      Object.freeze({ id: `${spec.name}:inference`, label: `${spec.label} inference smoke` }),
    );
  }
  if (new Set(plan.map(({ id }) => id)).size !== plan.length) {
    throw new Error("validation plan contains duplicate check identifiers");
  }
  return Object.freeze(plan);
}

export async function inspectOrtFloatTensor(tensor, expectedDims, label = "ORT output") {
  if (!tensor || typeof tensor !== "object") throw new TypeError(`${label} is missing`);
  if (!Array.isArray(expectedDims) || !expectedDims.length ||
      expectedDims.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
    throw new TypeError(`${label} expected dimensions are invalid`);
  }

  const type = tensor.type || tensor.dataType;
  if (type !== "float32") throw new Error(`${label} has dtype '${type || "unknown"}'; expected float32`);
  const dims = Array.from(tensor.dims || [], Number);
  if (dims.length !== expectedDims.length || dims.some((value, index) => value !== expectedDims[index])) {
    throw new Error(`${label} has shape [${dims.join(",")}]; expected [${expectedDims.join(",")}]`);
  }

  let expectedElements = 1;
  for (const value of expectedDims) {
    if (expectedElements > Number.MAX_SAFE_INTEGER / value) {
      throw new RangeError(`${label} expected element count exceeds the safe integer range`);
    }
    expectedElements *= value;
  }
  let data;
  if (tensor.location && tensor.location !== "cpu") {
    if (typeof tensor.getData !== "function") {
      throw new Error(`${label} is at '${tensor.location}' but has no getData() readback`);
    }
    data = await tensor.getData();
  } else {
    data = tensor.data;
  }
  if (!(data instanceof Float32Array)) {
    throw new Error(`${label} data is not Float32Array`);
  }
  if (data.length !== expectedElements) {
    throw new Error(`${label} has ${data.length} values; expected ${expectedElements}`);
  }

  let min = Infinity;
  let max = -Infinity;
  let nonFinite = 0;
  for (const value of data) {
    if (!Number.isFinite(value)) nonFinite++;
    else {
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  if (nonFinite) throw new Error(`${label} has ${nonFinite}/${data.length} non-finite values`);
  if (!(max - min > 1e-6)) throw new Error(`${label} is constant (${min})`);
  return Object.freeze({
    dims: Object.freeze([...dims]),
    elements: data.length,
    min,
    max,
  });
}

export function summarizeValidation(plan, results) {
  const known = new Set(plan.map(({ id }) => id));
  const counts = { pass: 0, fail: 0, skip: 0, pending: 0 };
  for (const check of plan) {
    const status = results.get(check.id)?.status;
    if (status === "pass" || status === "fail" || status === "skip") counts[status]++;
    else counts.pending++;
  }
  for (const id of results.keys()) {
    if (!known.has(id)) throw new Error(`result does not belong to validation plan: ${id}`);
  }
  return Object.freeze({
    ...counts,
    total: plan.length,
    complete: counts.pending === 0,
    ok: counts.pending === 0 && counts.fail === 0 && counts.skip === 0,
  });
}

export function withTimeout(operation, timeoutMs = VALIDATION_TIMEOUT_MS, label = "operation") {
  const promise = typeof operation === "function" ? Promise.resolve().then(operation) : Promise.resolve(operation);
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return promise;
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs} ms`)), timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

export async function acquireValidationDevice(gpu, timeoutMs = VALIDATION_TIMEOUT_MS) {
  if (!gpu) throw new Error("navigator.gpu is unavailable");
  const adapter = await withTimeout(
    gpu.requestAdapter({ powerPreference: "high-performance" }),
    timeoutMs,
    "WebGPU adapter request",
  );
  if (!adapter) throw new Error("no compatible adapter");
  const requiredFeatures = [];
  if (adapter.features.has("float32-filterable")) requiredFeatures.push("float32-filterable");
  const deviceRequest = Promise.resolve().then(() => adapter.requestDevice({ requiredFeatures }));
  let device;
  try {
    device = await withTimeout(deviceRequest, timeoutMs, "WebGPU device request");
  } catch (error) {
    // requestDevice has no cancellation API. If it settles after our deadline,
    // immediately retire the device whose ownership would otherwise be lost.
    void deviceRequest.then(
      (lateDevice) => { try { lateDevice?.destroy?.(); } catch {} },
      () => {},
    );
    throw error;
  }
  const info = adapter.info || {};
  const adapterName = [info.vendor, info.architecture, info.device, info.description]
    .filter(Boolean).join(" ") || "adapter";
  return Object.freeze({
    device,
    detail: `${adapterName}; ${requiredFeatures.length ? requiredFeatures.join(", ") : "no optional features"}`,
  });
}

export async function withGpuErrorScopes(device, label, operation, timeoutMs = VALIDATION_TIMEOUT_MS) {
  const filters = ["internal", "out-of-memory", "validation"];
  for (const filter of filters) device.pushErrorScope(filter);
  let value;
  let operationError = null;
  try {
    // The operation owns its timeout and cleanup. Racing it here would pop the
    // scopes and allow the next model to start while delayed GPU work was still
    // touching resources from this one.
    value = await operation();
  } catch (error) {
    operationError = error;
  }

  const scopeErrors = [];
  for (let index = filters.length - 1; index >= 0; index--) {
    try {
      const error = await withTimeout(device.popErrorScope(), timeoutMs, `${label} error scope`);
      if (error) scopeErrors.push(`${filters[index]}: ${error.message || String(error)}`);
    } catch (error) {
      scopeErrors.push(`${filters[index]} scope: ${error.message || String(error)}`);
    }
  }
  if (operationError) {
    if (!scopeErrors.length) throw operationError;
    const combined = new Error(
      `${operationError.message || String(operationError)}; ${label}: ${scopeErrors.join("; ")}`,
    );
    combined.cause = operationError;
    if (operationError.code) combined.code = operationError.code;
    throw combined;
  }
  if (scopeErrors.length) throw new Error(`${label}: ${scopeErrors.join("; ")}`);
  return value;
}

export function float16ToNumber(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

export function numberToFloat16(value) {
  const scratch = new Float32Array(1);
  const words = new Uint32Array(scratch.buffer);
  scratch[0] = value;
  const word = words[0];
  const sign = (word >>> 16) & 0x8000;
  const absolute = word & 0x7fffffff;
  const sourceExponent = absolute >>> 23;
  let fraction = absolute & 0x7fffff;
  if (sourceExponent === 0xff) return sign | (fraction ? 0x7e00 : 0x7c00);

  let exponent = sourceExponent - 127 + 15;
  if (exponent >= 0x1f) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    fraction |= 0x800000;
    return sign | roundShiftToEven(fraction, 14 - exponent);
  }

  let halfFraction = roundShiftToEven(fraction, 13);
  if (halfFraction === 0x400) {
    halfFraction = 0;
    exponent++;
    if (exponent >= 0x1f) return sign | 0x7c00;
  }
  return sign | (exponent << 10) | halfFraction;
}

function roundShiftToEven(value, shift) {
  const divisor = 2 ** shift;
  const quotient = Math.floor(value / divisor);
  const remainder = value - quotient * divisor;
  const halfway = divisor / 2;
  return quotient + (remainder > halfway || (remainder === halfway && quotient % 2 === 1) ? 1 : 0);
}

export function inspectRgba16Float(words, width, height, wordsPerRow = width * 4) {
  if (!(words instanceof Uint16Array)) throw new TypeError("rgba16float data must be a Uint16Array");
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError("rgba16float dimensions must be positive safe integers");
  }
  if (!Number.isSafeInteger(wordsPerRow) || wordsPerRow < width * 4 || words.length < wordsPerRow * height) {
    throw new RangeError("rgba16float data is shorter than its declared row layout");
  }
  let min = Infinity;
  let max = -Infinity;
  let nonFinite = 0;
  const channelMin = [Infinity, Infinity, Infinity, Infinity];
  const channelMax = [-Infinity, -Infinity, -Infinity, -Infinity];
  for (let y = 0; y < height; y++) {
    const row = y * wordsPerRow;
    for (let index = 0; index < width * 4; index++) {
      const value = float16ToNumber(words[row + index]);
      if (!Number.isFinite(value)) nonFinite++;
      else {
        min = Math.min(min, value);
        max = Math.max(max, value);
        const channel = index % 4;
        channelMin[channel] = Math.min(channelMin[channel], value);
        channelMax[channel] = Math.max(channelMax[channel], value);
      }
    }
  }
  return Object.freeze({
    texels: width * height,
    components: width * height * 4,
    nonFinite,
    min,
    max,
    channelMin: Object.freeze(channelMin),
    channelMax: Object.freeze(channelMax),
  });
}

export function buildCorePipelines(device, canvasFormat) {
  const pipelines = [];
  const render = (code, format = canvasFormat) => {
    const module = device.createShaderModule({ code });
    pipelines.push(device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    }));
  };
  const textureVariant = (source) => source
    .replace(/texture_external/g, "texture_2d<f32>")
    .replace(/textureSampleBaseClampToEdge\(([^)]*)\)/g, "textureSampleLevel($1, 0.0)");

  pipelines.push(device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: LUMA_EXTRACT_WGSL }), entryPoint: "main" },
  }));
  pipelines.push(device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: textureVariant(LUMA_EXTRACT_WGSL) }), entryPoint: "main" },
  }));
  render(RECOMBINE_WGSL);
  render(RECOMBINE_WGSL, "rgba16float");
  render(textureVariant(RECOMBINE_WGSL));
  render(textureVariant(RECOMBINE_WGSL), "rgba16float");
  render(buildSharpenShader(1, false));
  render(buildMeanShader(2, 2), "rgba16float");
  render(buildL2Shader(1, 2), "rgba16float");
  render(buildL2Shader(0, 2), "rgba16float");
  render(SSIMDS_MR_WGSL, "rgba16float");
  render(SSIMDS_FINAL_WGSL, "rgba16float");
  return pipelines;
}

export async function runModelInference(device, model, options = {}) {
  const width = options.width ?? 32;
  const height = options.height ?? 8;
  const timeoutMs = options.timeoutMs ?? VALIDATION_TIMEOUT_MS;
  const inputBytesPerRow = width * 8;
  if (inputBytesPerRow % 256 !== 0) throw new Error("validation input row pitch must be 256-byte aligned");

  let inputTexture = null;
  let readBuffer = null;
  let mapped = false;
  try {
    inputTexture = device.createTexture({
      label: "validation-luma-input",
      size: { width, height },
      format: "rgba16float",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    });
    const input = new Uint16Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const offset = (y * width + x) * 4;
        const value = 0.1 + 0.8 * ((x + y) / (width + height - 2));
        input[offset] = numberToFloat16(value);
        input[offset + 1] = 0;
        input[offset + 2] = 0;
        input[offset + 3] = numberToFloat16(1);
      }
    }
    device.queue.writeTexture(
      { texture: inputTexture },
      input,
      { bytesPerRow: inputBytesPerRow, rowsPerImage: height },
      { width, height },
    );

    const plan = model.allocate(width, height, inputTexture);
    const outputWidth = plan.outputWidth;
    const outputHeight = plan.outputHeight;
    const outputBytesPerRow = Math.ceil(outputWidth * 8 / 256) * 256;
    readBuffer = device.createBuffer({
      label: "validation-model-readback",
      size: outputBytesPerRow * outputHeight,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder({ label: "validation-model-inference" });
    const output = model.run(encoder, inputTexture);
    if (output.width !== outputWidth || output.height !== outputHeight) {
      throw new Error(`runtime output is ${output.width}x${output.height}; expected ${outputWidth}x${outputHeight}`);
    }
    encoder.copyTextureToBuffer(
      { texture: output },
      { buffer: readBuffer, bytesPerRow: outputBytesPerRow, rowsPerImage: outputHeight },
      { width: outputWidth, height: outputHeight },
    );
    device.queue.submit([encoder.finish()]);
    await withTimeout(device.queue.onSubmittedWorkDone(), timeoutMs, "GPU inference submission");
    await withTimeout(readBuffer.mapAsync(GPUMapMode.READ), timeoutMs, "GPU inference readback");
    mapped = true;
    const words = new Uint16Array(readBuffer.getMappedRange()).slice();
    const stats = inspectRgba16Float(words, outputWidth, outputHeight, outputBytesPerRow / 2);
    if (stats.nonFinite) throw new Error(`${stats.nonFinite}/${stats.components} output components are non-finite`);
    const lumaRange = stats.channelMax[0] - stats.channelMin[0];
    if (!(lumaRange > 1e-4)) throw new Error(`output luma is constant (${stats.channelMin[0]})`);
    if (Math.abs(stats.channelMin[3] - 1) > 1e-3 || Math.abs(stats.channelMax[3] - 1) > 1e-3) {
      throw new Error(`output alpha is outside the expected opaque value ` +
        `[${stats.channelMin[3]}, ${stats.channelMax[3]}]`);
    }
    return Object.freeze({ width: outputWidth, height: outputHeight, ...stats });
  } finally {
    if (mapped) {
      try { readBuffer?.unmap(); } catch {}
    }
    try { readBuffer?.destroy(); } catch {}
    try { inputTexture?.destroy(); } catch {}
  }
}
