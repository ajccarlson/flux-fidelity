import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import { splitModelEntries } from "../src/core/fsrcnnx-model-bundle.js";

const root = resolve(import.meta.dirname, "..");
const modelNames = ["ArtCNN_C4F32", "ArtCNN_C4F32_DN", "ArtCNN_C4F32_DS"];

function read(path) {
  return readFileSync(resolve(root, path), "utf8");
}

function zeroPaddedBlocks(entry) {
  return [...entry.matchAll(
    /  var (inp_(\d+)_(\d+)_(\d+)) : (f32|vec4f) = ([^;]+);\n  if \(all\(tap_(\d+)_(\d+) >= vec2i\(0\)\) && all\(tap_(\d+)_(\d+) < (sdim|logicalDim)\)\) \{\n([\s\S]*?)\n  \}/g,
  )];
}

test("every ArtCNN convolution tap is explicitly zero padded", () => {
  for (const name of modelNames) {
    const manifest = JSON.parse(read(`model/${name}.artcnn.json`));
    const source = read(`model/${name}.artcnn.wgsl`);
    const entries = splitModelEntries(source);

    assert.doesNotMatch(
      source,
      /textureLoad\([^\n;]*\bclamp\s*\(/,
      `${name} must not turn convolution padding into replicated edge samples`,
    );

    for (const pass of manifest.passes.filter(({ kind }) => kind === "conv")) {
      const entry = entries.get(pass.index);
      const blocks = zeroPaddedBlocks(entry);
      const expectedTaps = pass.index === 0 ? 9 : 72;
      assert.equal(blocks.length, expectedTaps, `${name} pass ${pass.index} guarded tap count`);

      let guardedLoads = 0;
      for (const match of blocks) {
        const [
          , inputName, , inputDx, inputDy, type, initializer,
          lowerDx, lowerDy, upperDx, upperDy, bound, body,
        ] = match;
        assert.equal(lowerDx, inputDx);
        assert.equal(lowerDy, inputDy);
        assert.equal(upperDx, inputDx);
        assert.equal(upperDy, inputDy);
        assert.equal(type, pass.index === 0 ? "f32" : "vec4f");
        assert.equal(initializer, pass.index === 0 ? "0.0" : "vec4f(0.0)");
        assert.equal(bound, pass.index === 0 ? "sdim" : "logicalDim");
        assert.match(body, new RegExp(`\\b${inputName} = textureLoad\\(`));

        const loads = (body.match(/textureLoad\(/g) || []).length;
        assert.equal(loads, pass.skipSum ? 2 : 1, `${name} pass ${pass.index} ${inputName} load count`);
        guardedLoads += loads;
      }

      assert.equal(
        (entry.match(/textureLoad\(/g) || []).length,
        guardedLoads,
        `${name} pass ${pass.index} has no convolution load outside a zero-padding guard`,
      );
      if (pass.index > 0) {
        assert.match(entry, /let logicalDim = sdim \/ vec2i\(4, 2\);/);
      }
    }

    const residual = entries.get(6);
    assert.match(
      residual,
      /var inp_0_0_0 : vec4f = vec4f\(0\.0\);[\s\S]*?if \(all\(tap_0_0 >= vec2i\(0\)\) && all\(tap_0_0 < logicalDim\)\) \{[\s\S]*?inp_0_0_0 = textureLoad\(t_conv2d, packed_0_0_0, 0\) \+ textureLoad\(t_conv2d_5, packed_0_0_0, 0\);[\s\S]*?\n  \}/,
      `${name} residual inputs must leave both source reads behind the same logical-boundary guard`,
    );
  }
});

test("ArtCNN corner taps contribute zero while in-range taps retain their source values", () => {
  const source = read("model/ArtCNN_C4F32.artcnn.wgsl");
  const firstPass = splitModelEntries(source).get(0);
  assert.match(firstPass, /let tap_0_0 = sp \+ vec2i\(-1, -1\);/);
  assert.match(
    firstPass,
    /var inp_0_0_0 : f32 = 0\.0;\n  if \(all\(tap_0_0 >= vec2i\(0\)\) && all\(tap_0_0 < sdim\)\) \{\n    inp_0_0_0 = textureLoad\(t_LUMA, tap_0_0, 0\)\.x;\n  \}/,
  );

  const sample = (pixels, width, height, x, y, dx, dy) => {
    const tapX = x + dx - 1;
    const tapY = y + dy - 1;
    return tapX >= 0 && tapY >= 0 && tapX < width && tapY < height
      ? pixels[tapY * width + tapX]
      : 0;
  };
  const pixels = [1, 2, 3, 4];
  assert.equal(sample(pixels, 2, 2, 0, 0, 0, 0), 0, "top-left outside tap is zero");
  assert.equal(sample(pixels, 2, 2, 0, 0, 1, 1), 1, "center tap remains the corner pixel");
  assert.equal(sample(pixels, 2, 2, 1, 1, 2, 2), 0, "bottom-right outside tap is zero");
  assert.notEqual(sample(pixels, 2, 2, 0, 0, 0, 0), pixels[0], "padding does not replicate the edge");
});

test("FSRCNNX retains its sampled clamp-to-edge boundary convention", () => {
  const source = read("model/FSRCNNX_x2_16-0-4-1.wgsl");
  assert.match(source, /textureLoad\(t_LUMA, clampCoord\(p \+ vec2i\(-2, -2\), textureDimensions\(t_LUMA\)\), 0\)\.x/);
});
