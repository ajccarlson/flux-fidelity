// Numerical-reference validation shared by the extension validation page and
// deterministic unit tests. The ordinary extension runtime does not import it.

import { LUMA_EXTRACT_WGSL, RECOMBINE_WGSL } from "./fsrcnnx-color.js";
import {
  REFERENCE_CASES,
  REFERENCE_FIXTURE_SCHEMA_VERSION,
  REFERENCE_INPUT_VERSION,
  createReferenceInput,
  getReferenceCase,
} from "./reference-fixtures.js";
import { buildSharpenShader } from "./fsrcnnx-sharpen.js";
import { SsimDownscaler } from "./fsrcnnx-ssimds-runtime.js";
import {
  alignedBytesPerRow,
  float16ToNumber,
  numberToFloat16,
  VALIDATION_TIMEOUT_MS,
  withTimeout,
} from "./fsrcnnx-validation.js";

export const REFERENCE_FIXTURE_MANIFEST_PATH = "validation/reference-fixtures.json";

function requireDimension(value, label) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
  return value;
}

function requireTolerance(value, label) {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${label} must be finite and non-negative`);
  return value;
}

export function validateReferenceManifest(manifest) {
  if (!manifest || typeof manifest !== "object" || !Array.isArray(manifest.cases)) {
    throw new TypeError("reference fixture manifest must contain a cases array");
  }
  if (manifest.schemaVersion !== REFERENCE_FIXTURE_SCHEMA_VERSION) {
    throw new Error(
      `reference fixture schema is ${manifest.schemaVersion}; expected ${REFERENCE_FIXTURE_SCHEMA_VERSION}`,
    );
  }
  if (manifest.inputVersion !== REFERENCE_INPUT_VERSION) {
    throw new Error(`reference input version is '${manifest.inputVersion}'; expected '${REFERENCE_INPUT_VERSION}'`);
  }
  const ids = new Set();
  for (const fixtureCase of manifest.cases) {
    if (!fixtureCase || typeof fixtureCase !== "object" ||
        typeof fixtureCase.id !== "string" || !fixtureCase.id) {
      throw new Error("reference fixture case has an invalid id");
    }
    if (ids.has(fixtureCase.id)) throw new Error(`duplicate reference fixture case ${fixtureCase.id}`);
    ids.add(fixtureCase.id);
    if (!Array.isArray(fixtureCase.inputs) || !fixtureCase.inputs.length) {
      throw new Error(`${fixtureCase.id}: reference inputs are missing`);
    }
    const roles = new Set();
    for (const input of fixtureCase.inputs) {
      if (!input || typeof input.role !== "string" || !input.role || typeof input.id !== "string" || !input.id) {
        throw new Error(`${fixtureCase.id}: invalid reference input`);
      }
      if (roles.has(input.role)) throw new Error(`${fixtureCase.id}: duplicate input role ${input.role}`);
      roles.add(input.role);
    }
    if (!fixtureCase.output || typeof fixtureCase.output !== "object") {
      throw new Error(`${fixtureCase.id}: reference output is missing`);
    }
    requireDimension(fixtureCase.output.width, `${fixtureCase.id} output width`);
    requireDimension(fixtureCase.output.height, `${fixtureCase.id} output height`);
    if (!["red", "rgb"].includes(fixtureCase.output.comparison)) {
      throw new Error(`${fixtureCase.id}: output comparison must be 'red' or 'rgb'`);
    }
    if (fixtureCase.output.kind !== "computed") {
      if (fixtureCase.output.format !== "rgb16le" || typeof fixtureCase.output.path !== "string" ||
          !fixtureCase.output.path || !/^[a-f0-9]{64}$/.test(fixtureCase.output.sha256 || "")) {
        throw new Error(`${fixtureCase.id}: pinned reference output metadata is invalid`);
      }
      const expectedLength = fixtureCase.output.width * fixtureCase.output.height * 6;
      if (fixtureCase.output.byteLength !== expectedLength) {
        throw new Error(`${fixtureCase.id}: reference output byte length is invalid`);
      }
    }
    if (!fixtureCase.tolerances || typeof fixtureCase.tolerances !== "object") {
      throw new Error(`${fixtureCase.id}: tolerances are missing`);
    }
    let toleranceCount = 0;
    for (const name of ["rmse", "p99", "max"]) {
      if (fixtureCase.tolerances[name] == null) continue;
      requireTolerance(fixtureCase.tolerances[name], `${fixtureCase.id} ${name} tolerance`);
      toleranceCount++;
    }
    if (!toleranceCount) throw new Error(`${fixtureCase.id}: no numerical tolerances are defined`);
  }
  if (ids.size !== REFERENCE_CASES.length) {
    throw new Error(`reference fixture manifest has ${ids.size} cases; expected ${REFERENCE_CASES.length}`);
  }
  for (const canonical of REFERENCE_CASES) {
    const fixtureCase = manifest.cases.find(({ id }) => id === canonical.id);
    if (!fixtureCase) throw new Error(`reference fixture case ${canonical.id} is missing`);
    const inputIdentity = (input) => `${input.role}:${input.id}`;
    if (fixtureCase.inputs.map(inputIdentity).join("|") !== canonical.inputs.map(inputIdentity).join("|")) {
      throw new Error(`${canonical.id}: reference input roles differ from the canonical case`);
    }
    for (const name of ["kind", "path", "width", "height", "format", "comparison", "sha256"]) {
      if ((fixtureCase.output[name] ?? null) !== (canonical.output[name] ?? null)) {
        throw new Error(`${canonical.id}: output ${name} differs from the canonical case`);
      }
    }
    if (fixtureCase.oracle?.kind !== canonical.oracle?.kind) {
      throw new Error(`${canonical.id}: oracle kind differs from the canonical case`);
    }
    for (const [name, value] of Object.entries(canonical.oracle || {})) {
      if (fixtureCase.oracle?.[name] !== value) {
        throw new Error(`${canonical.id}: oracle ${name} differs from the canonical case`);
      }
    }
    for (const name of ["rmse", "p99", "max"]) {
      if ((fixtureCase.tolerances[name] ?? null) !== (canonical.tolerances[name] ?? null)) {
        throw new Error(`${canonical.id}: ${name} tolerance differs from the canonical case`);
      }
    }
  }
  return Object.freeze({ manifest, cases: new Map(manifest.cases.map((fixtureCase) => [fixtureCase.id, fixtureCase])) });
}

export async function loadReferenceManifest(path = REFERENCE_FIXTURE_MANIFEST_PATH) {
  const response = await withTimeout(fetch(path), VALIDATION_TIMEOUT_MS, "reference fixture manifest fetch");
  if (!response.ok) throw new Error(`reference fixture manifest fetch failed (${response.status})`);
  return validateReferenceManifest(await response.json());
}

function bytesToHex(bytes) {
  return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
}

export async function loadPinnedReference(fixtureCase) {
  const { output } = fixtureCase;
  if (!output || output.kind === "computed") {
    throw new Error(`${fixtureCase.id}: case has no pinned binary output`);
  }
  const response = await withTimeout(fetch(output.path), VALIDATION_TIMEOUT_MS, `${fixtureCase.id} reference fetch`);
  if (!response.ok) throw new Error(`${fixtureCase.id}: reference fetch failed (${response.status})`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== output.byteLength) {
    throw new Error(`${fixtureCase.id}: reference is ${buffer.byteLength} bytes; expected ${output.byteLength}`);
  }
  const digest = bytesToHex(new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)));
  if (digest !== output.sha256) {
    throw new Error(`${fixtureCase.id}: reference SHA-256 mismatch (${digest})`);
  }
  return decodeRgb16le(buffer, output.width, output.height);
}

export function decodeRgb16le(buffer, width, height) {
  requireDimension(width, "reference width");
  requireDimension(height, "reference height");
  const bytes = buffer instanceof ArrayBuffer
    ? new Uint8Array(buffer)
    : ArrayBuffer.isView(buffer)
      ? new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
      : null;
  if (!bytes) throw new TypeError("rgb16le reference must be an ArrayBuffer or typed-array view");
  const expectedLength = width * height * 6;
  if (bytes.byteLength !== expectedLength) {
    throw new RangeError(`rgb16le reference is ${bytes.byteLength} bytes; expected ${expectedLength}`);
  }
  const samples = new Uint16Array(width * height * 3);
  for (let index = 0; index < samples.length; index++) {
    samples[index] = bytes[index * 2] | (bytes[index * 2 + 1] << 8);
  }
  return samples;
}

export function computeErrorMetrics(actual, expected) {
  if (!actual || !expected || actual.length !== expected.length || !actual.length) {
    throw new RangeError("metric inputs must be non-empty arrays of equal length");
  }
  const errors = new Float64Array(actual.length);
  let squared = 0;
  let maximum = 0;
  for (let index = 0; index < actual.length; index++) {
    const actualValue = Number(actual[index]);
    const expectedValue = Number(expected[index]);
    if (!Number.isFinite(actualValue) || !Number.isFinite(expectedValue)) {
      throw new Error(`metric input ${index} is non-finite`);
    }
    const error = Math.abs(actualValue - expectedValue);
    errors[index] = error;
    squared += error * error;
    maximum = Math.max(maximum, error);
  }
  errors.sort();
  return Object.freeze({
    samples: errors.length,
    rmse: Math.sqrt(squared / errors.length),
    p99: errors[Math.max(0, Math.ceil(errors.length * 0.99) - 1)],
    max: maximum,
  });
}

export function assertReferenceMetrics(metrics, tolerances, label = "numerical reference") {
  const failures = [];
  for (const name of ["rmse", "p99", "max"]) {
    if (tolerances[name] == null) continue;
    const limit = requireTolerance(tolerances[name], `${label} ${name} tolerance`);
    if (!Number.isFinite(metrics[name]) || metrics[name] > limit) {
      failures.push(`${name} ${formatMetric(metrics[name])} > ${formatMetric(limit)}`);
    }
  }
  if (failures.length) throw new Error(`${label} mismatch: ${failures.join(", ")}`);
  return metrics;
}

export function formatReferenceMetrics(metrics) {
  return `RMSE ${formatMetric(metrics.rmse)}, p99 ${formatMetric(metrics.p99)}, max ${formatMetric(metrics.max)} ` +
    `(${metrics.samples} samples)`;
}

function formatMetric(value) {
  return Number(value).toExponential(3);
}

function quantizedSample(sample) {
  return Math.fround(float16ToNumber(numberToFloat16(sample / 65535)));
}

const f32 = Math.fround;
const f32Add = (left, right) => f32(f32(left) + f32(right));
const f32Sub = (left, right) => f32(f32(left) - f32(right));
const f32Mul = (left, right) => f32(f32(left) * f32(right));
const f32Div = (left, right) => f32(f32(left) / f32(right));

function bt709Luma(red, green, blue) {
  return f32Add(
    f32Add(f32Mul(red, 0.2126), f32Mul(green, 0.7152)),
    f32Mul(blue, 0.0722),
  );
}

function inputRole(caseSpec, role) {
  const match = caseSpec.inputs.find((input) => input.role === role);
  if (!match) throw new Error(`${caseSpec.id}: input role '${role}' is missing`);
  return createReferenceInput(match.id);
}

export function buildBt709ExtractOracle(source) {
  if (source.channels !== 3 || source.data.length !== source.width * source.height * 3) {
    throw new Error("BT.709 extraction oracle requires a packed RGB input");
  }
  const expected = new Float64Array(source.width * source.height);
  for (let pixel = 0; pixel < expected.length; pixel++) {
    const base = pixel * 3;
    const red = quantizedSample(source.data[base]);
    const green = quantizedSample(source.data[base + 1]);
    const blue = quantizedSample(source.data[base + 2]);
    expected[pixel] = float16ToNumber(numberToFloat16(bt709Luma(red, green, blue)));
  }
  return expected;
}

function sampleRgbBilinear(source, u, v, channel) {
  const px = f32Sub(f32Mul(u, source.width), 0.5);
  const py = f32Sub(f32Mul(v, source.height), 0.5);
  const x0 = Math.floor(px);
  const y0 = Math.floor(py);
  const tx = f32Sub(px, x0);
  const ty = f32Sub(py, y0);
  const sample = (x, y) => {
    const clampedX = Math.max(0, Math.min(source.width - 1, x));
    const clampedY = Math.max(0, Math.min(source.height - 1, y));
    return quantizedSample(source.data[(clampedY * source.width + clampedX) * 3 + channel]);
  };
  const top = f32Add(
    f32Mul(sample(x0, y0), f32Sub(1, tx)),
    f32Mul(sample(x0 + 1, y0), tx),
  );
  const bottom = f32Add(
    f32Mul(sample(x0, y0 + 1), f32Sub(1, tx)),
    f32Mul(sample(x0 + 1, y0 + 1), tx),
  );
  return f32Add(f32Mul(top, f32Sub(1, ty)), f32Mul(bottom, ty));
}

export function buildBt709RecombineOracle(source, luma) {
  if (source.channels !== 3 || source.data.length !== source.width * source.height * 3) {
    throw new Error("BT.709 recombination oracle requires a packed RGB source");
  }
  if (luma.channels !== 1 || luma.data.length !== luma.width * luma.height) {
    throw new Error("BT.709 recombination oracle requires a packed grayscale luma input");
  }
  const expected = new Float64Array(luma.width * luma.height * 3);
  for (let y = 0; y < luma.height; y++) {
    for (let x = 0; x < luma.width; x++) {
      const u = f32Div(f32Add(x, 0.5), luma.width);
      const v = f32Div(f32Add(y, 0.5), luma.height);
      const red = sampleRgbBilinear(source, u, v, 0);
      const green = sampleRgbBilinear(source, u, v, 1);
      const blue = sampleRgbBilinear(source, u, v, 2);
      const sourceLuma = bt709Luma(red, green, blue);
      const cb = f32Div(f32Sub(blue, sourceLuma), 1.8556);
      const cr = f32Div(f32Sub(red, sourceLuma), 1.5748);
      const highLuma = quantizedSample(luma.data[y * luma.width + x]);
      const outRed = f32Add(highLuma, f32Mul(1.5748, cr));
      const outBlue = f32Add(highLuma, f32Mul(1.8556, cb));
      const outGreen = f32Div(
        f32Sub(f32Sub(highLuma, f32Mul(0.2126, outRed)), f32Mul(0.0722, outBlue)),
        0.7152,
      );
      const base = (y * luma.width + x) * 3;
      for (const [channel, value] of [[0, outRed], [1, outGreen], [2, outBlue]]) {
        expected[base + channel] = float16ToNumber(numberToFloat16(Math.max(0, Math.min(1, value))));
      }
    }
  }
  return expected;
}

export function referenceCaseSpec(id) {
  const spec = getReferenceCase(id);
  if (!spec) throw new Error(`unknown reference fixture case ${id}`);
  return spec;
}

export function rgba16ReadbackValues(words, width, height, wordsPerRow, channels, clampValues = false) {
  if (!(words instanceof Uint16Array)) throw new TypeError("rgba16 readback must be Uint16Array");
  requireDimension(width, "readback width");
  requireDimension(height, "readback height");
  if (!Number.isSafeInteger(wordsPerRow) || wordsPerRow < width * 4 || words.length < wordsPerRow * height) {
    throw new RangeError("rgba16 readback is shorter than its padded row layout");
  }
  if (!Array.isArray(channels) || !channels.length || channels.some((channel) => ![0, 1, 2, 3].includes(channel))) {
    throw new RangeError("readback channels must select RGBA channel indices");
  }
  const values = new Float64Array(width * height * channels.length);
  let destination = 0;
  for (let y = 0; y < height; y++) {
    const row = y * wordsPerRow;
    for (let x = 0; x < width; x++) {
      for (const channel of channels) {
        let value = float16ToNumber(words[row + x * 4 + channel]);
        if (clampValues) value = Math.max(0, Math.min(1, value));
        values[destination++] = value;
      }
    }
  }
  return values;
}

export function rgb16ReferenceValues(samples, width, height, channels) {
  if (!(samples instanceof Uint16Array) || samples.length !== width * height * 3) {
    throw new RangeError("RGB16 reference samples do not match the declared dimensions");
  }
  if (!Array.isArray(channels) || !channels.length || channels.some((channel) => ![0, 1, 2].includes(channel))) {
    throw new RangeError("reference channels must select RGB channel indices");
  }
  const values = new Float64Array(width * height * channels.length);
  let destination = 0;
  for (let pixel = 0; pixel < width * height; pixel++) {
    for (const channel of channels) values[destination++] = samples[pixel * 3 + channel] / 65535;
  }
  return values;
}

function compareReadback(readback, expected, channels, tolerances, label, clampValues = false) {
  const actual = rgba16ReadbackValues(
    readback.words,
    readback.width,
    readback.height,
    readback.wordsPerRow,
    channels,
    clampValues,
  );
  const metrics = computeErrorMetrics(actual, expected);
  assertReferenceMetrics(metrics, tolerances, label);
  return metrics;
}

function modelMismatchDiagnostics(actual, expected, width, height) {
  let minimum = Infinity;
  let maximum = -Infinity;
  let maxError = -1;
  let maxIndex = 0;
  for (let index = 0; index < actual.length; index++) {
    minimum = Math.min(minimum, actual[index]);
    maximum = Math.max(maximum, actual[index]);
    const error = Math.abs(actual[index] - expected[index]);
    if (error > maxError) {
      maxError = error;
      maxIndex = index;
    }
  }
  const clampedMetrics = computeErrorMetrics(
    Array.from(actual, (value) => Math.max(0, Math.min(1, value))),
    expected,
  );
  const details = [
    `range [${formatMetric(minimum)},${formatMetric(maximum)}]`,
    `clamped RMSE ${formatMetric(clampedMetrics.rmse)}/p99 ${formatMetric(clampedMetrics.p99)}/max ${formatMetric(clampedMetrics.max)}`,
    `max@${maxIndex % width},${Math.floor(maxIndex / width)}`,
  ];
  for (const border of [1, 2, 4, 8]) {
    if (width <= border * 2 || height <= border * 2) continue;
    const innerActual = [];
    const innerExpected = [];
    for (let y = border; y < height - border; y++) {
      for (let x = border; x < width - border; x++) {
        const index = y * width + x;
        innerActual.push(actual[index]);
        innerExpected.push(expected[index]);
      }
    }
    const metrics = computeErrorMetrics(innerActual, innerExpected);
    details.push(`inner${border} RMSE ${formatMetric(metrics.rmse)}/max ${formatMetric(metrics.max)}`);
  }
  const permutations = [];
  const buildPermutation = (prefix, remaining) => {
    if (!remaining.length) {
      permutations.push(prefix);
      return;
    }
    for (let index = 0; index < remaining.length; index++) {
      buildPermutation(
        [...prefix, remaining[index]],
        [...remaining.slice(0, index), ...remaining.slice(index + 1)],
      );
    }
  };
  buildPermutation([], [0, 1, 2, 3]);
  let best = null;
  for (const permutation of permutations) {
    let squared = 0;
    let samples = 0;
    for (let y = 0; y + 1 < height; y += 2) {
      for (let x = 0; x + 1 < width; x += 2) {
        for (let parity = 0; parity < 4; parity++) {
          const actualX = x + Math.floor(parity / 2);
          const actualY = y + (parity % 2);
          const expectedParity = permutation[parity];
          const expectedX = x + Math.floor(expectedParity / 2);
          const expectedY = y + (expectedParity % 2);
          const delta = actual[actualY * width + actualX] - expected[expectedY * width + expectedX];
          squared += delta * delta;
          samples++;
        }
      }
    }
    const rmse = Math.sqrt(squared / samples);
    if (!best || rmse < best.rmse) best = { permutation, rmse };
  }
  details.push(`best 2x2 map ${best.permutation.join("")} RMSE ${formatMetric(best.rmse)}`);
  return details.join(", ");
}

function assertOutputDimensions(fixtureCase, width, height) {
  if (fixtureCase.output.width !== width || fixtureCase.output.height !== height) {
    throw new Error(
      `${fixtureCase.id}: runtime output is ${width}x${height}; ` +
      `reference is ${fixtureCase.output.width}x${fixtureCase.output.height}`,
    );
  }
}

function assertOutputComparison(fixtureCase, expected) {
  if (fixtureCase.output.comparison !== expected) {
    throw new Error(
      `${fixtureCase.id}: output comparison is '${fixtureCase.output.comparison || "missing"}'; expected '${expected}'`,
    );
  }
}

function createReferenceTexture(device, input, label) {
  requireDimension(input.width, `${label} width`);
  requireDimension(input.height, `${label} height`);
  if (![1, 3].includes(input.channels) || !(input.data instanceof Uint16Array) ||
      input.data.length !== input.width * input.height * input.channels) {
    throw new Error(`${label}: fixture input layout is invalid`);
  }
  const texture = device.createTexture({
    label,
    size: { width: input.width, height: input.height },
    format: "rgba16float",
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  try {
    const bytesPerRow = alignedBytesPerRow(input.width);
    const wordsPerRow = bytesPerRow / 2;
    const words = new Uint16Array(wordsPerRow * input.height);
    for (let y = 0; y < input.height; y++) {
      for (let x = 0; x < input.width; x++) {
        const destination = y * wordsPerRow + x * 4;
        const source = (y * input.width + x) * input.channels;
        words[destination] = numberToFloat16(input.data[source] / 65535);
        if (input.channels === 3) {
          words[destination + 1] = numberToFloat16(input.data[source + 1] / 65535);
          words[destination + 2] = numberToFloat16(input.data[source + 2] / 65535);
        }
        words[destination + 3] = numberToFloat16(1);
      }
    }
    device.queue.writeTexture(
      { texture },
      words,
      { bytesPerRow, rowsPerImage: input.height },
      { width: input.width, height: input.height },
    );
    return texture;
  } catch (error) {
    try { texture.destroy(); } catch {}
    throw error;
  }
}

function createOutputTexture(device, width, height, label, render = false) {
  return device.createTexture({
    label,
    size: { width, height },
    format: "rgba16float",
    usage: GPUTextureUsage.COPY_SRC |
      (render ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.STORAGE_BINDING),
  });
}

async function submitTextureReadback(device, encoder, texture, width, height, label) {
  const bytesPerRow = alignedBytesPerRow(width);
  const readBuffer = device.createBuffer({
    label: `${label}-buffer`,
    size: bytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    encoder.copyTextureToBuffer(
      { texture },
      { buffer: readBuffer, bytesPerRow, rowsPerImage: height },
      { width, height },
    );
    device.queue.submit([encoder.finish()]);
    await withTimeout(device.queue.onSubmittedWorkDone(), VALIDATION_TIMEOUT_MS, `${label} GPU submission`);
    await withTimeout(readBuffer.mapAsync(GPUMapMode.READ), VALIDATION_TIMEOUT_MS, `${label} GPU readback`);
    mapped = true;
    return Object.freeze({
      width,
      height,
      wordsPerRow: bytesPerRow / 2,
      words: new Uint16Array(readBuffer.getMappedRange()).slice(),
    });
  } finally {
    if (mapped) {
      try { readBuffer.unmap(); } catch {}
    }
    try { readBuffer.destroy(); } catch {}
  }
}

function texture2dVariant(source) {
  const translated = source
    .replace(/texture_external/g, "texture_2d<f32>")
    .replace(/textureSampleBaseClampToEdge\(([^()]*)\)/g, "textureSampleLevel($1, 0.0)");
  if (translated.includes("texture_external") || translated.includes("textureSampleBaseClampToEdge")) {
    throw new Error("external-texture validation shader translation was incomplete");
  }
  return translated;
}

function createLinearSampler(device) {
  return device.createSampler({
    addressModeU: "clamp-to-edge",
    addressModeV: "clamp-to-edge",
    magFilter: "linear",
    minFilter: "linear",
  });
}

function createRenderPipeline(device, code, label) {
  const module = device.createShaderModule({ label: `${label}-shader`, code });
  return device.createRenderPipeline({
    label: `${label}-pipeline`,
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    primitive: { topology: "triangle-list" },
  });
}

function recordFullscreenRender(encoder, pipeline, bindGroup, output, label) {
  const pass = encoder.beginRenderPass({
    label,
    colorAttachments: [{
      view: output.createView(),
      loadOp: "clear",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: "store",
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
}

export async function runModelReference(device, model, fixtureCase) {
  const spec = referenceCaseSpec(fixtureCase.id);
  const source = inputRole(spec, "source");
  assertOutputComparison(fixtureCase, "red");
  const reference = await loadPinnedReference(fixtureCase);
  let inputTexture = null;
  try {
    inputTexture = createReferenceTexture(device, source, `${fixtureCase.id}-reference-input`);
    const plan = model.allocate(source.width, source.height, inputTexture);
    assertOutputDimensions(fixtureCase, plan.outputWidth, plan.outputHeight);
    const encoder = device.createCommandEncoder({ label: `${fixtureCase.id}-reference-inference` });
    const output = model.run(encoder, inputTexture);
    const readback = await submitTextureReadback(
      device,
      encoder,
      output,
      plan.outputWidth,
      plan.outputHeight,
      `${fixtureCase.id} reference`,
    );
    // mpv screenshots of the grayscale hooks can differ by up to two 16-bit
    // code values between RGB channels. The generator/checker bound that spread;
    // red is the pinned comparison channel for the WebGPU luma texture.
    const expected = rgb16ReferenceValues(reference, plan.outputWidth, plan.outputHeight, [0]);
    const actual = rgba16ReadbackValues(
      readback.words,
      readback.width,
      readback.height,
      readback.wordsPerRow,
      [0],
      fixtureCase.oracle?.presentationClamp === true,
    );
    const metrics = computeErrorMetrics(actual, expected);
    try {
      assertReferenceMetrics(metrics, fixtureCase.tolerances, `${fixtureCase.id} upstream reference`);
    } catch (error) {
      throw new Error(
        `${error.message}; ${modelMismatchDiagnostics(actual, expected, plan.outputWidth, plan.outputHeight)}`,
        { cause: error },
      );
    }
    return Object.freeze({ width: plan.outputWidth, height: plan.outputHeight, ...metrics });
  } finally {
    try { inputTexture?.destroy(); } catch {}
  }
}

async function runExtractReference(device, fixtureCase, spec) {
  const source = inputRole(spec, "source");
  assertOutputComparison(fixtureCase, "red");
  assertOutputDimensions(fixtureCase, source.width, source.height);
  let inputTexture = null;
  let outputTexture = null;
  try {
    inputTexture = createReferenceTexture(device, source, "bt709-extract-source");
    outputTexture = createOutputTexture(device, source.width, source.height, "bt709-extract-output");
    const pipeline = device.createComputePipeline({
      label: "bt709-extract-reference-pipeline",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: texture2dVariant(LUMA_EXTRACT_WGSL) }),
        entryPoint: "main",
      },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: createLinearSampler(device) },
        { binding: 1, resource: inputTexture.createView() },
        { binding: 2, resource: outputTexture.createView() },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "bt709-extract-reference" });
    const pass = encoder.beginComputePass({ label: "bt709-extract-reference-pass" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(source.width / 8), Math.ceil(source.height / 8));
    pass.end();
    const readback = await submitTextureReadback(
      device, encoder, outputTexture, source.width, source.height, "BT.709 extraction reference",
    );
    const metrics = compareReadback(
      readback,
      buildBt709ExtractOracle(source),
      [0],
      fixtureCase.tolerances,
      "BT.709 extraction reference",
    );
    return Object.freeze({ width: source.width, height: source.height, ...metrics });
  } finally {
    try { outputTexture?.destroy(); } catch {}
    try { inputTexture?.destroy(); } catch {}
  }
}

async function runRecombineReference(device, fixtureCase, spec) {
  const source = inputRole(spec, "source");
  const luma = inputRole(spec, "luma");
  assertOutputComparison(fixtureCase, "rgb");
  assertOutputDimensions(fixtureCase, luma.width, luma.height);
  let sourceTexture = null;
  let lumaTexture = null;
  let outputTexture = null;
  try {
    sourceTexture = createReferenceTexture(device, source, "bt709-recombine-source");
    lumaTexture = createReferenceTexture(device, luma, "bt709-recombine-luma");
    outputTexture = createOutputTexture(device, luma.width, luma.height, "bt709-recombine-output", true);
    const pipeline = createRenderPipeline(device, texture2dVariant(RECOMBINE_WGSL), "bt709-recombine-reference");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: createLinearSampler(device) },
        { binding: 1, resource: sourceTexture.createView() },
        { binding: 2, resource: lumaTexture.createView() },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "bt709-recombine-reference" });
    recordFullscreenRender(encoder, pipeline, bindGroup, outputTexture, "bt709-recombine-reference-pass");
    const readback = await submitTextureReadback(
      device, encoder, outputTexture, luma.width, luma.height, "BT.709 recombination reference",
    );
    const metrics = compareReadback(
      readback,
      buildBt709RecombineOracle(source, luma),
      [0, 1, 2],
      fixtureCase.tolerances,
      "BT.709 recombination reference",
    );
    return Object.freeze({ width: luma.width, height: luma.height, ...metrics });
  } finally {
    try { outputTexture?.destroy(); } catch {}
    try { lumaTexture?.destroy(); } catch {}
    try { sourceTexture?.destroy(); } catch {}
  }
}

async function runSsimReference(device, fixtureCase, spec) {
  const source = inputRole(spec, "source");
  assertOutputComparison(fixtureCase, "rgb");
  const reference = await loadPinnedReference(fixtureCase);
  let inputTexture = null;
  let downscaler = null;
  try {
    inputTexture = createReferenceTexture(device, source, "ssimds-reference-source");
    downscaler = new SsimDownscaler(device);
    const prepared = downscaler.prepare(
      source.width,
      source.height,
      fixtureCase.output.width,
      fixtureCase.output.height,
      inputTexture,
    );
    if (!prepared) throw new Error(`SSimDownscaler unexpectedly bypassed: ${downscaler.lastPlan?.reason || "unknown"}`);
    const encoder = device.createCommandEncoder({ label: "ssimds-reference" });
    const output = downscaler.run(encoder, inputTexture);
    const readback = await submitTextureReadback(
      device,
      encoder,
      output,
      fixtureCase.output.width,
      fixtureCase.output.height,
      "SSimDownscaler reference",
    );
    const expected = rgb16ReferenceValues(reference, readback.width, readback.height, [0, 1, 2]);
    const metrics = compareReadback(
      readback,
      expected,
      [0, 1, 2],
      fixtureCase.tolerances,
      "SSimDownscaler upstream reference",
    );
    return Object.freeze({ width: readback.width, height: readback.height, ...metrics });
  } finally {
    try { downscaler?.destroy(); } catch {}
    try { inputTexture?.destroy(); } catch {}
  }
}

async function runSharpenReference(device, fixtureCase, spec) {
  const source = inputRole(spec, "source");
  assertOutputDimensions(fixtureCase, source.width, source.height);
  assertOutputComparison(fixtureCase, "rgb");
  const reference = await loadPinnedReference(fixtureCase);
  let inputTexture = null;
  let outputTexture = null;
  try {
    inputTexture = createReferenceTexture(device, source, "sharpen-reference-source");
    outputTexture = createOutputTexture(device, source.width, source.height, "sharpen-reference-output", true);
    const pipeline = createRenderPipeline(device, buildSharpenShader(1, false), "sharpen-reference");
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: createLinearSampler(device) },
        { binding: 1, resource: inputTexture.createView() },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "sharpen-reference" });
    recordFullscreenRender(encoder, pipeline, bindGroup, outputTexture, "sharpen-reference-pass");
    const readback = await submitTextureReadback(
      device, encoder, outputTexture, source.width, source.height, "adaptive sharpen reference",
    );
    const expected = rgb16ReferenceValues(reference, source.width, source.height, [0, 1, 2]);
    const metrics = compareReadback(
      readback,
      expected,
      [0, 1, 2],
      fixtureCase.tolerances,
      "adaptive sharpen upstream reference",
      fixtureCase.oracle?.presentationClamp === true,
    );
    return Object.freeze({ width: source.width, height: source.height, ...metrics });
  } finally {
    try { outputTexture?.destroy(); } catch {}
    try { inputTexture?.destroy(); } catch {}
  }
}

export async function runSupportingReference(device, fixtureCase) {
  const spec = referenceCaseSpec(fixtureCase.id);
  switch (fixtureCase.id) {
    case "color:extract-reference": return runExtractReference(device, fixtureCase, spec);
    case "color:recombine-reference": return runRecombineReference(device, fixtureCase, spec);
    case "filter:ssimds-reference": return runSsimReference(device, fixtureCase, spec);
    case "filter:sharpen-reference": return runSharpenReference(device, fixtureCase, spec);
    default: throw new Error(`unsupported supporting reference ${fixtureCase.id}`);
  }
}
