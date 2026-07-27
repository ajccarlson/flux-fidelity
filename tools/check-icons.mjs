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
  Object.freeze({ path: "icons/icon-16.png", size: 16, sha256: "7457945c00cb94a8ed6e6314ac1e9f7c3882c1f889cf9e6b75ffc7476fcecca6" }),
  Object.freeze({ path: "icons/icon-32.png", size: 32, sha256: "08d4b8f186924c75e946a1919ecadca32c972b1518d3bbc302fcb2c18b9a78ca" }),
  Object.freeze({ path: "icons/icon-48.png", size: 48, sha256: "e6628ae87e20f9f41fe03d6160119cfea09cc3287bc30c762a63536cea569f16" }),
  Object.freeze({ path: "icons/icon-128.png", size: 128, sha256: "7bc8dabdaba70db76c285e8b41b6c1aaa399d37ab67ad8853609cf498ac8990f" }),
  Object.freeze({ path: "icons/icon-off-16.png", size: 16, sha256: "74514bdbc00532f286bac2f91c3ac02dd6c75c2b070ec6904fe6b457b7a814ef" }),
  Object.freeze({ path: "icons/icon-off-32.png", size: 32, sha256: "33400d725c83cf48a66ad22862fb5ffb62481b47e062204a300a033f3c79afe7" }),
  Object.freeze({ path: "icons/icon-off-48.png", size: 48, sha256: "ef5b423ebb336a6cc67d1a82897db3bf3c613d95aaa864e84f798bc90f39a551" }),
  Object.freeze({ path: "icons/icon-off-128.png", size: 128, sha256: "4bd6feef42c9df5e2bb64f5b531ab8befe911e1f0f5c06ad4ee9350d8d170f71" }),
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
