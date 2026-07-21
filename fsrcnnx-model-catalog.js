// Canonical inventory for generated WebGPU models shipped with the extension.
// Runtime selection, package validation, automated tests, and the browser
// validator all consume this module so a model cannot be added in only one
// place.

function model(name, label, kind, role) {
  const artcnn = kind === "artcnn";
  return Object.freeze({
    name,
    label,
    kind,
    role,
    manifestPath: `model/${name}.${artcnn ? "artcnn" : "passes"}.json`,
    shaderPath: `model/${name}.${artcnn ? "artcnn." : ""}wgsl`,
  });
}

export const GENERATED_MODEL_CATALOG = Object.freeze([
  model("FSRCNNX_x2_16-0-4-1", "FSRCNNX x2", "fsrcnnx", "standard"),
  model("ArtCNN_C4F32", "ArtCNN", "artcnn", "artcnn"),
  model("ArtCNN_C4F32_DN", "ArtCNN denoise", "artcnn", "artcnn"),
  model("ArtCNN_C4F32_DS", "ArtCNN denoise/sharpen", "artcnn", "artcnn"),
]);

export const FSRCNNX_STANDARD_MODEL_NAMES = Object.freeze(
  GENERATED_MODEL_CATALOG
    .filter(({ role }) => role === "standard")
    .map(({ name }) => name),
);

export const ARTCNN_MODEL_NAMES = Object.freeze(
  GENERATED_MODEL_CATALOG
    .filter(({ role }) => role === "artcnn")
    .map(({ name }) => name),
);

export const GENERATED_MODEL_ASSET_PATHS = Object.freeze(
  GENERATED_MODEL_CATALOG
    .flatMap(({ manifestPath, shaderPath }) => [manifestPath, shaderPath])
    .sort(),
);
