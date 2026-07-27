import { createHash } from "node:crypto";
import { deflateRawSync } from "node:zlib";
import { lstatSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PACKAGE_FILES } from "./package-files.mjs";

const root = resolve(import.meta.dirname, "..");
const fixedMode = 0o100644;

const crcTable = new Uint32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
  crcTable[n] = c >>> 0;
}
function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}
function u16(value) { const b = Buffer.alloc(2); b.writeUInt16LE(value); return b; }
function u32(value) { const b = Buffer.alloc(4); b.writeUInt32LE(value >>> 0); return b; }

function isChromiumVersion(value) {
  if (typeof value !== "string") return false;
  const parts = value.split(".");
  return parts.length >= 1 && parts.length <= 4 &&
    parts.some((part) => part !== "0") &&
    parts.every((part) => /^(?:0|[1-9]\d*)$/.test(part) && Number(part) <= 65535);
}

function makeZip(entries) {
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const entry of entries) {
    const name = Buffer.from(entry.name, "utf8");
    const compressed = deflateRawSync(entry.data, { level: 9 });
    const crc = crc32(entry.data);
    const local = Buffer.concat([
      u32(0x04034b50), u16(20), u16(0x0800), u16(8), u16(0), u16(33),
      u32(crc), u32(compressed.length), u32(entry.data.length), u16(name.length), u16(0), name, compressed,
    ]);
    const central = Buffer.concat([
      u32(0x02014b50), u16(0x031e), u16(20), u16(0x0800), u16(8), u16(0), u16(33),
      u32(crc), u32(compressed.length), u32(entry.data.length), u16(name.length), u16(0), u16(0),
      u16(0), u16(0), u32(fixedMode << 16), u32(offset), name,
    ]);
    locals.push(local); centrals.push(central); offset += local.length;
  }
  const central = Buffer.concat(centrals);
  const end = Buffer.concat([
    u32(0x06054b50), u16(0), u16(0), u16(entries.length), u16(entries.length),
    u32(central.length), u32(offset), u16(0),
  ]);
  return Buffer.concat([...locals, central, end]);
}

export function buildPackage({ rootDir = root, distDir = resolve(rootDir, "dist") } = {}) {
  const pkg = JSON.parse(readFileSync(resolve(rootDir, "package.json"), "utf8"));
  const manifest = JSON.parse(readFileSync(resolve(rootDir, "manifest.json"), "utf8"));
  const version = pkg.version;
  if (!isChromiumVersion(version)) {
    throw new Error(`Package version is not a valid Chromium extension version: ${version || "missing"}`);
  }
  if (manifest.version !== version) {
    throw new Error(`Manifest version ${manifest.version || "missing"} differs from package ${version}`);
  }
  const stage = resolve(distDir, "flux-fidelity");
  const archive = resolve(distDir, `flux-fidelity-${version}.zip`);
  const checksums = resolve(distDir, "SHA256SUMS");

  rmSync(stage, { recursive: true, force: true });
  mkdirSync(stage, { recursive: true });
  const entries = [];
  for (const file of PACKAGE_FILES) {
    const source = resolve(rootDir, file);
    const metadata = lstatSync(source);
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      throw new Error(`Package input must be a regular file: ${file}`);
    }
    const data = readFileSync(source);
    const target = resolve(stage, file);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, data, { mode: 0o644 });
    entries.push({ name: file, data });
  }

  const zip = makeZip(entries);
  mkdirSync(distDir, { recursive: true });
  writeFileSync(archive, zip);
  const digest = createHash("sha256").update(zip).digest("hex");
  writeFileSync(checksums, `${digest}  ${relative(distDir, archive)}\n`);
  return { archive, checksums, digest, fileCount: entries.length, stage, version };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = buildPackage();
  console.log(`Packaged ${result.fileCount} files: ${relative(root, result.archive)}`);
  console.log(`SHA-256 ${result.digest}`);
}
