import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { basename, isAbsolute, relative, resolve } from "node:path";
import { validateModelBundle } from "../fsrcnnx-model-bundle.js";
import { validateNeuralManifest } from "../fsrcnnx-neural.js";
import { PACKAGE_FILES } from "./package-files.mjs";

const root = resolve(import.meta.dirname, "..");
const modelDir = resolve(root, "model");
const errors = [];

// Pin reviewed model, runtime, source, license, and rebuilding bytes so a
// package never silently substitutes different material while retaining the
// same provenance and compliance record.
const pinnedArtifacts = {
  "GPL-3.0.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
  "LGPL-3.0.txt": "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118",
  "LGPL_REBUILDING.md": "b23f3a64a3db81248f1fe99dc1138d4b55aace39b162cfaefaa667836240effe",
  "LICENSE": "29018b491588ba5338f7cceb3eec504a7958985ed69bf1a91c03daac4c2e3fbf",
  "model/rife_v4.26.onnx": "af25762dfec02a4bbb949decea63988b01fa56c46c0ff9dc66ac8e2f12cbb661",
  "model/rife_v4.26_fp16.onnx": "d5672f39b493609220c95c709542d6b99204145a67d9ca496d4500cd8895301f",
  "vendor/ort/LICENSE": "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c",
  "vendor/ort/ThirdPartyNotices.txt": "0e07b95f3a8d6230037707c5c4a2b554d12c4cb67369669ac255635528ffcee2",
  "vendor/ort/ort.webgpu.min.mjs": "46988a5a025f49449850f39f95eb0d21e40e67b3beb13a0b54efd3ab5d83f60e",
  "vendor/ort/ort-wasm-simd-threaded.asyncify.mjs": "7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655",
  "vendor/ort/ort-wasm-simd-threaded.asyncify.wasm": "7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a",
  "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl": "d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965",
  "shaders/upstream/SSimDownscaler.glsl": "f46f4710a162d17058b9d82ed8610588b0c04d7be07cef6bf2a8c4077828f804",
  "shaders/upstream/ArtCNN_C4F32.glsl": "f773bce6cf5fe7e5e5d599a695edd40df5cd7a20c3d08c4d164d07591d5bead3",
  "shaders/upstream/ArtCNN_C4F32_DN.glsl": "6b51a6f7d75826c9492c3f78b5e60acffa24a71928e2d47c4a329423922a143c",
  "shaders/upstream/ArtCNN_C4F32_DS.glsl": "a04c9cba6fbb8e6db9239d61848390208aedf8e348ef116e12174c803d22077e",
  "model/FSRCNNX_x2_16-0-4-1.wgsl": "2b005b9c4e60c59445708b2f503c9afb01fd70ee9efb1615782274e7b7707f26",
  "model/FSRCNNX_x2_16-0-4-1.passes.json": "1378fc336deb2588f75ddf8b9ed6ec70109256f2c3fa0477ca742adf830fb3e0",
  "model/ArtCNN_C4F32.artcnn.wgsl": "af2b1911fe4ec1f77354b71d5aa6796c93b4d53eb73cd693b59af7e8cfb9d654",
  "model/ArtCNN_C4F32.artcnn.json": "4ab29b29a6121e0fa3d3880b890bedabb3ea1f49356ef46704ad1770b143077a",
  "model/ArtCNN_C4F32_DN.artcnn.wgsl": "f204b33d52614e87bc9d8d31ba43822ad3bad1ff75da8d425ca6fbd90a2032a9",
  "model/ArtCNN_C4F32_DN.artcnn.json": "b5911c707c83462c79dcf954bcaf422efd2d6b42efd4d08228361ab8ea52fe79",
  "model/ArtCNN_C4F32_DS.artcnn.wgsl": "f6de86466a0ae261c178f53d72d2cb79032ade94b8ec452f51fa1315b93be3c5",
  "model/ArtCNN_C4F32_DS.artcnn.json": "f98bbd5e834cbfb2ed66ba07865889f76466279e356bfbd62c33df73e95b30cb",
  "fsrcnnx-ssimds.js": "0f55f8f2b49bea3cb8ee2e4c801a663f21d4dfabb88efaf2de23b709c6ade3c6",
  "fsrcnnx-sharpen.js": "9312f5445791792634679bac74f01d3292e8e776c6fc7e3be348435f2913ef8a",
  "transpile.js": "2ad45126cd36d52ce1064e8da1e189e10b5d256d8edc28a9dec3737957f4f631",
};

for (const [path, expected] of Object.entries(pinnedArtifacts)) {
  try {
    const actual = createHash("sha256").update(readFileSync(resolve(root, path))).digest("hex");
    if (actual !== expected) errors.push(`${path}: SHA-256 ${actual}, expected ${expected}`);
  } catch (error) {
    errors.push(`${path}: cannot verify pinned artifact (${error.message})`);
  }
}

const standardSourceMetadata = Object.freeze({
  license: "LGPL-3.0-or-later",
  sourcePath: "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
  sourceSha256: "d5a24a271e5d9a3f7f7a053b150c460a44c25b3cf7f770857d57cc3a2e1c9965",
  modificationNotice:
    "Transpiled in 2026 from the mpv/libplacebo GLSL hook format to WGSL compute passes and a JSON pass manifest for FSRCNNX-EXT; model weights and pass order are preserved.",
});
try {
  const manifest = JSON.parse(readFileSync(
    resolve(modelDir, "FSRCNNX_x2_16-0-4-1.passes.json"),
    "utf8",
  ));
  for (const [field, expected] of Object.entries(standardSourceMetadata)) {
    if (manifest[field] !== expected) {
      errors.push(`FSRCNNX standard manifest: ${field} does not match the pinned source record`);
    }
  }
  const wgsl = readFileSync(resolve(modelDir, "FSRCNNX_x2_16-0-4-1.wgsl"), "utf8");
  for (const line of [
    `// License: ${standardSourceMetadata.license}`,
    `// Source path: ${standardSourceMetadata.sourcePath}`,
    `// Source SHA-256: ${standardSourceMetadata.sourceSha256}`,
    `// Modification notice: ${standardSourceMetadata.modificationNotice}`,
  ]) {
    if (!wgsl.startsWith("// License:") || !wgsl.includes(`${line}\n`)) {
      errors.push(`FSRCNNX standard WGSL: missing pinned metadata line ${line}`);
    }
  }
} catch (error) {
  errors.push(`FSRCNNX standard metadata: cannot verify generated record (${error.message})`);
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
    const file = resolve(neuralDir, entry.file);
    const localPath = relative(neuralDir, file);
    if (!localPath || isAbsolute(localPath) || localPath === ".." || localPath.startsWith("../")) {
      errors.push(`neural manifest: model path escapes neural directory (${entry.file})`);
      continue;
    }
    const packagePath = `model/neural/${entry.file}`;
    if (!PACKAGE_FILES.includes(packagePath)) {
      errors.push(`neural manifest: ${packagePath} is outside the package boundary`);
    }
    if (!Object.hasOwn(pinnedArtifacts, packagePath)) {
      errors.push(`neural manifest: ${packagePath} has no pinned SHA-256`);
    }
    try {
      if (!statSync(file).isFile()) throw new Error();
    } catch {
      errors.push(`neural manifest: missing model ${entry.file}`);
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
