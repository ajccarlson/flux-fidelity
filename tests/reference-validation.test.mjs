import assert from "node:assert/strict";
import test from "node:test";

import {
  REFERENCE_CASES,
  REFERENCE_FIXTURE_SCHEMA_VERSION,
  REFERENCE_INPUTS,
  REFERENCE_INPUT_VERSION,
  createReferenceInput,
  getReferenceCase,
} from "../reference-fixtures.js";
import {
  assertReferenceMetrics,
  buildBt709ExtractOracle,
  buildBt709RecombineOracle,
  computeErrorMetrics,
  decodeRgb16le,
  rgb16ReferenceValues,
  rgba16ReadbackValues,
  validateReferenceManifest,
} from "../fsrcnnx-reference-validation.js";
import { float16ToNumber, numberToFloat16 } from "../fsrcnnx-validation.js";

test("reference definitions cover every numerical path with deterministic structured inputs", () => {
  assert.equal(Object.isFrozen(REFERENCE_CASES), true);
  assert.equal(Object.isFrozen(REFERENCE_INPUTS), true);
  assert.deepEqual(REFERENCE_CASES.map(({ id }) => id), [
    "FSRCNNX_x2_16-0-4-1",
    "FSRCNNX_x2_56-16-4-1",
    "ArtCNN_C4F32",
    "ArtCNN_C4F32_DN",
    "ArtCNN_C4F32_DS",
    "filter:ssimds-reference",
    "filter:sharpen-reference",
    "color:extract-reference",
    "color:recombine-reference",
  ]);
  for (const spec of Object.values(REFERENCE_INPUTS)) {
    const first = createReferenceInput(spec.id);
    const second = createReferenceInput(spec.id);
    assert.equal(first.width, spec.width);
    assert.equal(first.height, spec.height);
    assert.equal(first.channels, spec.channels);
    assert.deepEqual(first.data, second.data, `${spec.id} input changed between calls`);
    assert.ok(new Set(first.data).size > 8, `${spec.id} must exercise structured variation`);
  }
  assert.equal(getReferenceCase("filter:ssimds-reference").output.width, 31);
  assert.throws(() => getReferenceCase("unknown"), /Unknown reference case/);
});

test("RGB16LE decoding and channel selection are explicit and endian-stable", () => {
  const bytes = Uint8Array.from([
    0x34, 0x12, 0xcd, 0xab, 0xff, 0x00,
    0x00, 0x80, 0x01, 0x00, 0xff, 0xff,
  ]);
  const samples = decodeRgb16le(bytes, 2, 1);
  assert.deepEqual([...samples], [0x1234, 0xabcd, 0x00ff, 0x8000, 0x0001, 0xffff]);
  assert.deepEqual(
    [...rgb16ReferenceValues(samples, 2, 1, [2, 0])],
    [0x00ff / 65535, 0x1234 / 65535, 1, 0x8000 / 65535],
  );
  assert.throws(() => decodeRgb16le(bytes.subarray(1), 2, 1), /expected 12/);
});

test("reference metrics use full-frame RMSE, nearest-rank p99, and maximum error", () => {
  const actual = new Float64Array(100);
  actual[98] = 0.5;
  actual[99] = 1;
  const expected = new Float64Array(100);
  const metrics = computeErrorMetrics(actual, expected);
  assert.equal(metrics.samples, 100);
  assert.equal(metrics.rmse, Math.sqrt(1.25 / 100));
  assert.equal(metrics.p99, 0.5);
  assert.equal(metrics.max, 1);
  assert.equal(assertReferenceMetrics(metrics, { rmse: 0.12, p99: 0.5, max: 1 }), metrics);
  assert.throws(
    () => assertReferenceMetrics(metrics, { rmse: 0.1, max: 1 }, "fixture"),
    /fixture mismatch: rmse/,
  );
  assert.throws(() => computeErrorMetrics([0, NaN], [0, 0]), /non-finite/);
});

test("padded rgba16 readback ignores row padding and selects only requested channels", () => {
  const width = 37;
  const wordsPerRow = 256;
  const words = new Uint16Array(wordsPerRow * 2).fill(0x7e00);
  const values = [0.1, 0.2, 0.3, 1, -0.1, 1.1, 0.7, 1];
  for (let y = 0; y < 2; y++) {
    const row = y * wordsPerRow;
    words.fill(0, row, row + width * 4);
    for (let channel = 0; channel < 4; channel++) {
      words[row + channel] = numberToFloat16(values[y * 4 + channel]);
    }
  }
  const extracted = rgba16ReadbackValues(words, width, 2, wordsPerRow, [0, 2], true);
  assert.equal(extracted.length, width * 2 * 2);
  assert.deepEqual([...extracted.slice(0, 2)], [
    float16ToNumber(numberToFloat16(0.1)),
    float16ToNumber(numberToFloat16(0.3)),
  ]);
  assert.deepEqual([...extracted.slice(width * 2, width * 2 + 2)], [
    0,
    float16ToNumber(numberToFloat16(0.7)),
  ]);
});

test("BT.709 CPU oracles include source and output float16 quantization", () => {
  const red = Object.freeze({
    width: 1,
    height: 1,
    channels: 3,
    data: new Uint16Array([65535, 0, 0]),
  });
  assert.deepEqual(
    [...buildBt709ExtractOracle(red)],
    [float16ToNumber(numberToFloat16(0.2126))],
  );

  const graySample = 16384;
  const grayRgb = Object.freeze({
    width: 1,
    height: 1,
    channels: 3,
    data: new Uint16Array([graySample, graySample, graySample]),
  });
  const grayLuma = Object.freeze({
    width: 1,
    height: 1,
    channels: 1,
    data: new Uint16Array([graySample]),
  });
  const quantized = float16ToNumber(numberToFloat16(graySample / 65535));
  assert.deepEqual([...buildBt709RecombineOracle(grayRgb, grayLuma)], [quantized, quantized, quantized]);
});

test("browser manifest validation rejects drift-prone output metadata", () => {
  const cases = REFERENCE_CASES.map((spec) => ({
    ...spec,
    inputs: spec.inputs.map((input) => ({ ...input })),
    output: spec.output.kind === "computed"
      ? { ...spec.output }
      : {
          ...spec.output,
          byteLength: spec.output.width * spec.output.height * 6,
        },
    tolerances: { ...spec.tolerances },
  }));
  const valid = {
    schemaVersion: REFERENCE_FIXTURE_SCHEMA_VERSION,
    inputVersion: REFERENCE_INPUT_VERSION,
    cases,
  };
  const validated = validateReferenceManifest(valid);
  assert.equal(validated.cases.get(REFERENCE_CASES[0].id), valid.cases[0]);
  assert.throws(
    () => validateReferenceManifest({ ...valid, cases: [valid.cases[0], valid.cases[0], ...valid.cases.slice(2)] }),
    /duplicate reference fixture case/,
  );
  assert.throws(
    () => validateReferenceManifest({
      ...valid,
      cases: [{ ...valid.cases[0], output: { ...valid.cases[0].output, byteLength: 10 } }, ...valid.cases.slice(1)],
    }),
    /byte length is invalid/,
  );
  assert.throws(
    () => validateReferenceManifest({ ...valid, schemaVersion: REFERENCE_FIXTURE_SCHEMA_VERSION + 1 }),
    /expected/,
  );
  assert.throws(
    () => validateReferenceManifest({
      ...valid,
      cases: [
        { ...valid.cases[0], tolerances: { ...valid.cases[0].tolerances, rmse: 1 } },
        ...valid.cases.slice(1),
      ],
    }),
    /tolerance differs from the canonical case/,
  );
  assert.throws(
    () => validateReferenceManifest({
      ...valid,
      cases: [
        { ...valid.cases[0], oracle: { ...valid.cases[0].oracle, presentationClamp: false } },
        ...valid.cases.slice(1),
      ],
    }),
    /oracle presentationClamp differs from the canonical case/,
  );
  assert.throws(
    () => validateReferenceManifest({
      ...valid,
      cases: [
        { ...valid.cases[0], output: { ...valid.cases[0].output, sha256: "a".repeat(64) } },
        ...valid.cases.slice(1),
      ],
    }),
    /output sha256 differs from the canonical case/,
  );
});
