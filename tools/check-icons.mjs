import { createHash } from "node:crypto";
import { existsSync, lstatSync, readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(import.meta.dirname, "..");
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const LEGACY_ICON_SOURCES = Object.freeze(["icon.svg", "icon-generator.py"]);

// These committed PNGs are the canonical icon artwork. Exact hashes preserve the
// current rendering without relying on an external SVG renderer, native graphics
// libraries, or unpinned development packages.
export const ICON_ASSET_CONTRACT = Object.freeze([
  Object.freeze({ path: "icons/icon-16.png", size: 16, sha256: "b49d5afbe24df36213aca04d29c4a69b5dfe2d6d98ddf24842481d0152b4525c" }),
  Object.freeze({ path: "icons/icon-32.png", size: 32, sha256: "ea468bd1927deaf064cf9030ea8d9297aaf4c14e55f5eeb29811b83f7d8ad556" }),
  Object.freeze({ path: "icons/icon-48.png", size: 48, sha256: "5d08ffcf98394ad200d97152f042b919687425a8336f59af9cdfd57a5dc5d78f" }),
  Object.freeze({ path: "icons/icon-128.png", size: 128, sha256: "b3d7766c1d94e839b67bd39707ab4a01b74129512f9fc83a2bce7d5ee8c6f240" }),
  Object.freeze({ path: "icons/icon-off-16.png", size: 16, sha256: "1ae0feb993f80eee8c66a1a310ac5b417ac408a291e9e350f6d9c8463965040f" }),
  Object.freeze({ path: "icons/icon-off-32.png", size: 32, sha256: "511c29a3f053b199c315ccaeed08d76ad525d3573a33e1c90d32a4aee8736904" }),
  Object.freeze({ path: "icons/icon-off-48.png", size: 48, sha256: "726d18ab2c1b9eee5938cf513366dddfca4ac84b38aeb6e1f6dd0b720b3b4808" }),
  Object.freeze({ path: "icons/icon-off-128.png", size: 128, sha256: "13b94b3cbf4db6078bbf8c17b749a70ff5fe16abbb40f0e0bebaf40d6328a2be" }),
]);

function inspectPngHeader(data, spec, errors) {
  if (data.length < 33 || !data.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    errors.push(`${spec.path}: not a PNG file`);
    return;
  }
  if (data.readUInt32BE(8) !== 13 || data.toString("ascii", 12, 16) !== "IHDR") {
    errors.push(`${spec.path}: malformed PNG IHDR`);
    return;
  }

  const width = data.readUInt32BE(16);
  const height = data.readUInt32BE(20);
  if (width !== spec.size || height !== spec.size) {
    errors.push(`${spec.path}: expected ${spec.size}x${spec.size}, found ${width}x${height}`);
  }
  const [bitDepth, colorType, compression, filter, interlace] = data.subarray(24, 29);
  if (bitDepth !== 8 || colorType !== 6 || compression !== 0 || filter !== 0 || interlace !== 0) {
    errors.push(`${spec.path}: expected non-interlaced 8-bit RGBA PNG encoding`);
  }
}

export function inspectIconAssets({ rootDir = root } = {}) {
  const errors = [];
  for (const path of LEGACY_ICON_SOURCES) {
    if (existsSync(resolve(rootDir, path))) {
      errors.push(`${path}: competing legacy icon source must remain removed`);
    }
  }

  let actualNames = [];
  try {
    actualNames = readdirSync(resolve(rootDir, "icons"))
      .filter((name) => /^icon(?:-off)?-\d+\.png$/.test(name))
      .sort();
  } catch (error) {
    errors.push(`icons: ${error.code || error.message}`);
  }
  const expectedNames = ICON_ASSET_CONTRACT.map(({ path }) => path.slice("icons/".length)).sort();
  if (actualNames.join("\n") !== expectedNames.join("\n")) {
    errors.push(`icons: expected exactly ${expectedNames.join(", ")}; found ${actualNames.join(", ") || "none"}`);
  }

  for (const spec of ICON_ASSET_CONTRACT) {
    const absolute = resolve(rootDir, spec.path);
    let metadata;
    try {
      metadata = lstatSync(absolute);
    } catch (error) {
      errors.push(`${spec.path}: ${error.code || error.message}`);
      continue;
    }
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      errors.push(`${spec.path}: must be a regular, non-symlink file`);
      continue;
    }

    const data = readFileSync(absolute);
    inspectPngHeader(data, spec, errors);
    const actualHash = createHash("sha256").update(data).digest("hex");
    if (actualHash !== spec.sha256) {
      errors.push(`${spec.path}: hash is ${actualHash}, expected ${spec.sha256}`);
    }
  }
  return errors;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const errors = inspectIconAssets();
  if (errors.length) {
    for (const error of errors) console.error(error);
    process.exitCode = 1;
  } else {
    console.log(`Icon assets: ok (${ICON_ASSET_CONTRACT.length} canonical PNGs)`);
  }
}
