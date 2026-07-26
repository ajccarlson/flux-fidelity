import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { basename, isAbsolute, relative, resolve } from "node:path";
import { validateModelBundle } from "../src/core/fsrcnnx-model-bundle.js";
import {
  neuralModelFiles,
  validateNeuralManifest,
} from "../src/core/fsrcnnx-neural.js";
import { PACKAGE_FILES } from "./package-files.mjs";

const root = resolve(import.meta.dirname, "..");
const modelDir = resolve(root, "model");
const errors = [];

// Pin reviewed model, runtime, source, license, and rebuilding bytes so a
// package never silently substitutes different material while retaining the
// same provenance and compliance record.
const pinnedArtifacts = {
  "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
  "LICENSES/GPL-3.0.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
  "LICENSES/LGPL-3.0.txt": "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118",
  "LICENSES/Real-ESRGAN-BSD-3-Clause.txt": "4a699ec4863d96a91fc265948a0c90033f7e8735d515524dcf3444736406e0c2",
  "NOTICE": "42dfd832a3c91892045eb6a2c5f76df4aa185c794145f0c3726fa20054c5daaa",
  "docs/compliance/LGPL_REBUILDING.md": "1e7843deaf5da74a884a78b98205f99d034cf12cdfece0b8bec1b72cf36843bd",
  "model/rife_v4.26.onnx": "af25762dfec02a4bbb949decea63988b01fa56c46c0ff9dc66ac8e2f12cbb661",
  "model/rife_v4.26_fp16.onnx": "d5672f39b493609220c95c709542d6b99204145a67d9ca496d4500cd8895301f",
  "model/neural/cda-vsr-initializer.onnx": "7773490658a7cad663e9b4f7e9cc8269b3d0c7a9a8e5840ec3151e895c1161f1",
  "model/neural/cda-vsr-recurrent.onnx": "442be6f8d356889070ed70acdb49f9d2d77f24b6947e51e823404ca5a6d66a05",
  "model/neural/realesrganv2_animevideo_xsx2.fp16.onnx": "f674a410b528aec55bb9f9f594cb1aaea580237adb29abd9dc32296d34b690a0",
  "vendor/ort/LICENSE": "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c",
  "vendor/ort/ThirdPartyNotices.txt": "0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2",
  "vendor/ort/ort.webgpu.min.mjs": "46988a5a025f49449850f39f95eb0d21e40e67b3beb13a0b54efd3ab5d83f60e",
  "vendor/ort/ort-wasm-simd.asyncify.mjs": "457bb6e6fc849b7c18fd39b75812e4ccf41f3ec482a4eb486c76ad6f2d43c811",
  "vendor/ort/ort-wasm-simd.asyncify.wasm": "c425ba45af30512459007c34c536aa43ea9e3367c5e765baee3d1a19c28e46d1",
  "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl": "d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965",
  "shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl": "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
  "shaders/upstream/SSimDownscaler.glsl": "f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804",
  "shaders/upstream/adaptive-sharpen.glsl": "827fb3d662ac9a91b4075e9117fe6e1dbc1c06d85959ba719cdb954dfb7fb8e4",
  "shaders/upstream/ArtCNN_C4F32.glsl": "f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3",
  "shaders/upstream/ArtCNN_C4F32_DN.glsl": "6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c",
  "shaders/upstream/ArtCNN_C4F32_DS.glsl": "a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e",
  "model/FSRCNNX_x2_16-0-4-1.wgsl": "2b005b9c4e60c59445708b2f503c9afb01fd70ee9efb1615782274e7b7707f26",
  "model/FSRCNNX_x2_16-0-4-1.passes.json": "1378fc336deb2588f75ddf8b9ed6ec70109256f2c3fa0477ca742adf830fb3e0",
  "model/FSRCNNX_x2_56-16-4-1.wgsl": "19a5327c8f96b7cb0593512f846f75ef266a3d857a84532c4dc5a374296e3d11",
  "model/FSRCNNX_x2_56-16-4-1.passes.json": "4b7512ca17fd9788f4876f2681207fa8fb3b10c46d314ea2b3ce684864fb4d70",
  "model/ArtCNN_C4F32.artcnn.wgsl": "ab6fe4c88e88eb0cc3b5482e68ca9279c802c0b7844699c40f9f15eb3aac8138",
  "model/ArtCNN_C4F32.artcnn.json": "4ab29b29a6121e0fa3d3880b890bedabb3ea1f49356ef46704ad1770b143077a",
  "model/ArtCNN_C4F32_DN.artcnn.wgsl": "c319ff51ff358558cd4daa1fc897da4bfc0064c175cca3f9fd29052ac29af280",
  "model/ArtCNN_C4F32_DN.artcnn.json": "b5911c707c83462c79dcf954bcaf422efd2d6b42efd4d08228361ab8ea52fe79",
  "model/ArtCNN_C4F32_DS.artcnn.wgsl": "41a1e37c67bfb76a74ce07b52324d961fb4e9351eee44581fba783f8d69341af",
  "model/ArtCNN_C4F32_DS.artcnn.json": "f98bbd5e834cbfb2ed66ba07865889f76466279e356bfbd62c33df73e95b30cb",
  "src/core/fsrcnnx-ssimds.js": "5bec6c839d512e504e44789765c0fb1edba3b9888755a8625becde09e6386101",
  "src/core/fsrcnnx-sharpen.js": "9312f5445791792634679bac74f01d3292e8e776c6fc7e3be348435f2913ef8a",
  "tools/package.json": "609158e6c5fbc237939fa3ddf7faab80ab690bdc0c8d584414a885130103c4e8",
  "tools/transpile.js": "6abd739bc5356ea9fc151c754f6c4d9e017c39283d5e5ba477a70aafe814003a",
};

for (const [path, expected] of Object.entries(pinnedArtifacts)) {
  try {
    const actual = createHash("sha256").update(readFileSync(resolve(root, path))).digest("hex");
    if (actual !== expected) errors.push(`${path}: SHA-256 ${actual}, expected ${expected}`);
  } catch (error) {
    errors.push(`${path}: cannot verify pinned artifact (${error.message})`);
  }
}

try {
  const ortGlue = readFileSync(
    resolve(root, "vendor/ort/ort-wasm-simd.asyncify.mjs"),
    "utf8",
  );
  if (!ortGlue.includes("ort-wasm-simd.asyncify.wasm")) {
    errors.push("single-thread ORT glue does not name its paired WebAssembly file");
  }
  if (/shared\s*:\s*(?:!0|true)/.test(ortGlue)) {
    errors.push("single-thread ORT glue still allocates shared WebAssembly memory");
  }
} catch (error) {
  errors.push(`single-thread ORT glue: cannot verify runtime contract (${error.message})`);
}

const modificationNotice =
  "Transpiled in 2026 from the mpv/libplacebo GLSL hook format to WGSL compute passes and a JSON pass manifest for FSRCNNX-EXT; model weights and pass order are preserved.";
const fsrcnnxSources = Object.freeze([
  Object.freeze({
    label: "standard",
    name: "FSRCNNX_x2_16-0-4-1",
    upstream: "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_16-0-4-1.glsl",
    license: "LGPL-3.0-or-later",
    sourcePath: "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
    sourceSha256: "d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965",
  }),
  Object.freeze({
    label: "High",
    name: "FSRCNNX_x2_56-16-4-1",
    upstream: "https://github.com/igv/FSRCNN-TensorFlow/releases/download/1.1/FSRCNNX_x2_56-16-4-1.glsl",
    license: "LGPL-3.0-or-later",
    sourcePath: "shaders/upstream/FSRCNNX_x2_56-16-4-1.glsl",
    sourceSha256: "34cd5d0087ebb6ae5f9bff2578382205457da53baa364d52de8021d6925b7fd6",
  }),
]);
for (const source of fsrcnnxSources) {
  try {
    const manifest = JSON.parse(readFileSync(
      resolve(modelDir, `${source.name}.passes.json`),
      "utf8",
    ));
    for (const [field, expected] of Object.entries({
      license: source.license,
      sourcePath: source.sourcePath,
      sourceSha256: source.sourceSha256,
      modificationNotice,
    })) {
      if (manifest[field] !== expected) {
        errors.push(`FSRCNNX ${source.label} manifest: ${field} does not match the pinned source record`);
      }
    }
    const wgsl = readFileSync(resolve(modelDir, `${source.name}.wgsl`), "utf8");
    for (const line of [
      `// License: ${source.license}`,
      `// Upstream: ${source.upstream}`,
      `// Source path: ${source.sourcePath}`,
      `// Source SHA-256: ${source.sourceSha256}`,
      `// Modification notice: ${modificationNotice}`,
    ]) {
      if (!wgsl.startsWith("// License:") || !wgsl.includes(`${line}\n`)) {
        errors.push(`FSRCNNX ${source.label} WGSL: missing pinned metadata line ${line}`);
      }
    }
  } catch (error) {
    errors.push(`FSRCNNX ${source.label} metadata: cannot verify generated record (${error.message})`);
  }
}

for (const name of readdirSync(modelDir).filter((file) => /\.(?:passes|artcnn)\.json$/.test(file)).sort()) {
  try {
    const manifest = JSON.parse(readFileSync(resolve(modelDir, name), "utf8"));
    const artcnn = name.endsWith(".artcnn.json");
    const wgslName = name.replace(/\.passes\.json$/, ".wgsl").replace(/\.artcnn\.json$/, ".artcnn.wgsl");
    const wgsl = readFileSync(resolve(modelDir, wgslName), "utf8");
    const expectedName = name.replace(/\.passes\.json$/, "").replace(/\.artcnn\.json$/, "");
    validateModelBundle(artcnn ? "artcnn" : "fsrcnnx", manifest, wgsl, { expectedName });
  } catch (error) {
    errors.push(`${name}: ${error.message}`);
  }
}

const neuralDir = resolve(modelDir, "neural");
try {
  const neuralManifest = validateNeuralManifest(
    JSON.parse(readFileSync(resolve(neuralDir, "manifest.json"), "utf8")),
  );
  for (const entry of neuralManifest) {
    for (const modelFile of neuralModelFiles(entry)) {
      const file = resolve(neuralDir, modelFile);
      const localPath = relative(neuralDir, file);
      if (!localPath || isAbsolute(localPath) || localPath === ".." || localPath.startsWith("../")) {
        errors.push(`neural manifest: model path escapes neural directory (${modelFile})`);
        continue;
      }
      const packagePath = `model/neural/${modelFile}`;
      if (!PACKAGE_FILES.includes(packagePath)) {
        errors.push(`neural manifest: ${packagePath} is outside the package boundary`);
      }
      if (!Object.hasOwn(pinnedArtifacts, packagePath)) {
        errors.push(`neural manifest: ${packagePath} has no pinned SHA-256`);
      }
      try {
        if (!statSync(file).isFile()) throw new Error();
      } catch {
        errors.push(`neural manifest: missing model ${modelFile}`);
      }
    }
  }
} catch (error) {
  errors.push(`neural manifest: ${error.message}`);
}

if (errors.length) {
  for (const error of errors) console.error(error);
  process.exit(1);
}
console.log(`Model manifests: ok (${basename(modelDir)})`);
