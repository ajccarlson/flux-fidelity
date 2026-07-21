import test from "node:test";
import assert from "node:assert/strict";
import { selectModel } from "../fsrcnnx-runtime.js";
import { LUMA_EXTRACT_WGSL, RECOMBINE_WGSL } from "../fsrcnnx-color.js";

test("model selection chooses the highest eligible threshold", () => {
  const models = [
    { scale: 4, manifest: { whenThreshold: 3.4 } },
    { scale: 2, manifest: { whenThreshold: 1.3 } },
    { scale: 3, manifest: { whenThreshold: 2.4 } },
  ];
  assert.equal(selectModel(models, 100, 100), null);
  assert.equal(selectModel(models, 150, 100)?.scale, 2);
  assert.equal(selectModel(models, 280, 100)?.scale, 3);
  assert.equal(selectModel(models, 400, 100)?.scale, 4);
});

test("color shaders expose extraction and recombination entry points", () => {
  assert.match(LUMA_EXTRACT_WGSL, /@compute\s+@workgroup_size\(8, 8\)/);
  assert.match(RECOMBINE_WGSL, /fn rgb2ycbcr/);
  assert.match(RECOMBINE_WGSL, /fn ycbcr2rgb/);
  assert.match(RECOMBINE_WGSL, /@fragment/);
});
