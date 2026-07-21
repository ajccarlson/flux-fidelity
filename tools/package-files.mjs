export const EXPECTED_PACKAGE_FILE_COUNT = 54;

// This list is the package boundary. Additions and removals must be deliberate:
// package creation and package reference validation both consume this exact set.
export const PACKAGE_FILES = Object.freeze([
  "LICENSE",
  "MODEL_PROVENANCE.md",
  "THIRD_PARTY_NOTICES.md",
  "background.js",
  "content.js",
  "fsrcnnx-artcnn-runtime.js",
  "fsrcnnx-color.js",
  "fsrcnnx-deband.js",
  "fsrcnnx-grab.js",
  "fsrcnnx-images.js",
  "fsrcnnx-interpolate.js",
  "fsrcnnx-main.js",
  "fsrcnnx-neural.js",
  "fsrcnnx-rife-gpu.js",
  "fsrcnnx-rife.js",
  "fsrcnnx-runtime.js",
  "fsrcnnx-sharpen.js",
  "fsrcnnx-ssimds-runtime.js",
  "fsrcnnx-ssimds.js",
  "icons/icon-128.png",
  "icons/icon-16.png",
  "icons/icon-32.png",
  "icons/icon-48.png",
  "icons/icon-off-128.png",
  "icons/icon-off-16.png",
  "icons/icon-off-32.png",
  "icons/icon-off-48.png",
  "manifest.json",
  "model/ArtCNN_C4F32.artcnn.json",
  "model/ArtCNN_C4F32.artcnn.wgsl",
  "model/ArtCNN_C4F32_DN.artcnn.json",
  "model/ArtCNN_C4F32_DN.artcnn.wgsl",
  "model/ArtCNN_C4F32_DS.artcnn.json",
  "model/ArtCNN_C4F32_DS.artcnn.wgsl",
  "model/FSRCNNX_x2_16-0-4-1.passes.json",
  "model/FSRCNNX_x2_16-0-4-1.wgsl",
  "model/FSRCNNX_x2_56-16-4-1.passes.json",
  "model/FSRCNNX_x2_56-16-4-1.wgsl",
  "model/FSRCNNX_x3_16-0-4-1.passes.json",
  "model/FSRCNNX_x3_16-0-4-1.wgsl",
  "model/FSRCNNX_x4_16-0-4-1.passes.json",
  "model/FSRCNNX_x4_16-0-4-1.wgsl",
  "model/neural/manifest.json",
  "model/neural/span2x_smoke.fp16.onnx",
  "model/rife.onnx",
  "model/rife_v4.26.onnx",
  "model/rife_v4.26_fp16.onnx",
  "popup.html",
  "popup.js",
  "validate.html",
  "validate.js",
  "vendor/ort/ort-wasm-simd-threaded.asyncify.mjs",
  "vendor/ort/ort-wasm-simd-threaded.asyncify.wasm",
  "vendor/ort/ort.webgpu.min.mjs",
]);

function assertPackageBoundary() {
  if (PACKAGE_FILES.length !== EXPECTED_PACKAGE_FILE_COUNT) {
    throw new Error(`Package allowlist must contain exactly ${EXPECTED_PACKAGE_FILE_COUNT} entries`);
  }
  if (new Set(PACKAGE_FILES).size !== PACKAGE_FILES.length) {
    throw new Error("Package allowlist contains duplicate entries");
  }
  if (PACKAGE_FILES.some((file) =>
    !file || file.startsWith("/") || file.includes("\\") || file.split("/").includes("..")
  )) {
    throw new Error("Package allowlist entries must be normalized relative paths");
  }
  const sorted = [...PACKAGE_FILES].sort();
  if (PACKAGE_FILES.some((file, index) => file !== sorted[index])) {
    throw new Error("Package allowlist must remain sorted");
  }
}

assertPackageBoundary();
