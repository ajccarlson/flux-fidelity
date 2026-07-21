import { createHash } from "node:crypto";
import { deflateRawSync } from "node:zlib";
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const dist = resolve(root, "dist");
const stage = resolve(dist, "fsrcnnx-ext");
const archive = resolve(dist, "fsrcnnx-ext.zip");
const fixedMode = 0o100644;

function walk(dir, prefix = "") {
  const out = [];
  for (const entry of readdirSync(dir).sort()) {
    const path = resolve(dir, entry);
    const rel = prefix ? `${prefix}/${entry}` : entry;
    if (statSync(path).isDirectory()) out.push(...walk(path, rel));
    else out.push(rel);
  }
  return out;
}

function runtimeFiles() {
  const files = [
    "manifest.json", "popup.html", "popup.js", "background.js", "content.js",
    "validate.html", "validate.js", "LICENSE", "THIRD_PARTY_NOTICES.md", "MODEL_PROVENANCE.md",
  ];
  files.push(...readdirSync(root).filter((name) => /^fsrcnnx-.*\.js$/.test(name)));
  for (const dir of ["icons", "model", "vendor/ort"]) files.push(...walk(resolve(root, dir), dir));
  return [...new Set(files)].sort();
}

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

rmSync(stage, { recursive: true, force: true });
mkdirSync(stage, { recursive: true });
const entries = [];
for (const file of runtimeFiles()) {
  const source = resolve(root, file);
  const data = readFileSync(source);
  const target = resolve(stage, file);
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, data, { mode: 0o644 });
  entries.push({ name: file, data });
}

const zip = makeZip(entries);
writeFileSync(archive, zip);
const digest = createHash("sha256").update(zip).digest("hex");
writeFileSync(resolve(dist, "SHA256SUMS"), `${digest}  ${relative(dist, archive)}\n`);
console.log(`Packaged ${entries.length} files: ${relative(root, archive)}`);
console.log(`SHA-256 ${digest}`);
