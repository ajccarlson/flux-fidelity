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
  Object.freeze({ path: "icons/icon-16.png", size: 16, sha256: "6bad5618b39be4f1f2e9bd4ac885883dec6daa1867ee711c3918cbb4bdc8f229" }),
  Object.freeze({ path: "icons/icon-32.png", size: 32, sha256: "5382c1ef63657e13c39f65e624f8cd43a4dd05e8816b0b8ddb40f957b8cbae6d" }),
  Object.freeze({ path: "icons/icon-48.png", size: 48, sha256: "c13b88fade0561479b6ec1ceb25a00df4098eb984004b31e0d219448bd2a6b97" }),
  Object.freeze({ path: "icons/icon-128.png", size: 128, sha256: "c0557108bc16182e53b5cf0849af20bec30485d2c204442812f239c05737d237" }),
  Object.freeze({ path: "icons/icon-off-16.png", size: 16, sha256: "9fc9e822a9ef0abced19f3a7207b24af69a17d93eff053f122dc8122af57a2f3" }),
  Object.freeze({ path: "icons/icon-off-32.png", size: 32, sha256: "176297a941ee8864bf6772e88b8d2ea1371349b3bf80adfa027e91c54fabb134" }),
  Object.freeze({ path: "icons/icon-off-48.png", size: 48, sha256: "9a22b404eac37c6ad4a456f2825198f3045ed9c30bb04256248c089ea4cbbafd" }),
  Object.freeze({ path: "icons/icon-off-128.png", size: 128, sha256: "4e6f23fa3ee73f80d13390243a582dd899a5f0f58ada114fcf45c9f984180c37" }),
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
