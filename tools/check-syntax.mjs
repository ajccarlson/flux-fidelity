import { createHash, timingSafeEqual } from "node:crypto";
import { lstatSync, readFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const root = resolve(import.meta.dirname, "..");
const REGULAR_FILE_MODE = 0o100000;
const MODE_TYPE_MASK = 0o170000;

function gitDirectory(rootDir) {
  const dotGit = join(rootDir, ".git");
  const metadata = lstatSync(dotGit);
  if (metadata.isDirectory()) return dotGit;
  if (metadata.isSymbolicLink() || !metadata.isFile()) {
    throw new Error(".git must be a directory or worktree pointer file");
  }
  const pointer = readFileSync(dotGit, "utf8").match(/^gitdir:\s*(.+?)\s*$/i);
  if (!pointer) throw new Error(".git worktree pointer is invalid");
  return resolve(rootDir, pointer[1]);
}

function checkedRange(buffer, offset, length, label, contentEnd) {
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) ||
      offset < 0 || length < 0 || offset + length > contentEnd) {
    throw new Error(`Git index is truncated while reading ${label}`);
  }
}

function decodeV4StripCount(buffer, offset, contentEnd) {
  checkedRange(buffer, offset, 1, "v4 path compression", contentEnd);
  let byte = buffer[offset++];
  let value = byte & 0x7f;
  while (byte & 0x80) {
    if (value > (Number.MAX_SAFE_INTEGER - 1) / 128) {
      throw new Error("Git index v4 path compression exceeds the safe integer range");
    }
    value++;
    checkedRange(buffer, offset, 1, "v4 path compression", contentEnd);
    byte = buffer[offset++];
    value = value * 128 + (byte & 0x7f);
  }
  return { value, offset };
}

function parseIndexWithObjectId(buffer, objectIdLength) {
  const hashName = objectIdLength === 32 ? "sha256" : "sha1";
  if (buffer.length < 12 + objectIdLength) throw new Error("Git index is too short");
  const contentEnd = buffer.length - objectIdLength;
  const expectedChecksum = buffer.subarray(contentEnd);
  const actualChecksum = createHash(hashName).update(buffer.subarray(0, contentEnd)).digest();
  if (!timingSafeEqual(actualChecksum, expectedChecksum)) return null;
  if (buffer.toString("ascii", 0, 4) !== "DIRC") throw new Error("Git index signature is invalid");
  const version = buffer.readUInt32BE(4);
  if (version < 2 || version > 4) throw new Error(`Unsupported Git index version ${version}`);
  const entryCount = buffer.readUInt32BE(8);
  const entries = [];
  let previousPath = Buffer.alloc(0);
  let offset = 12;

  for (let index = 0; index < entryCount; index++) {
    const entryStart = offset;
    const flagsOffset = entryStart + 40 + objectIdLength;
    checkedRange(buffer, entryStart, 40 + objectIdLength + 2, "cache entry", contentEnd);
    const mode = buffer.readUInt32BE(entryStart + 24);
    const flags = buffer.readUInt16BE(flagsOffset);
    offset = flagsOffset + 2;
    if (flags & 0x4000) {
      checkedRange(buffer, offset, 2, "extended cache-entry flags", contentEnd);
      offset += 2;
    }

    let pathBytes;
    if (version === 4) {
      const decoded = decodeV4StripCount(buffer, offset, contentEnd);
      offset = decoded.offset;
      if (decoded.value > previousPath.length) {
        throw new Error("Git index v4 path compression removes more bytes than the previous path");
      }
      const suffixEnd = buffer.indexOf(0, offset);
      if (suffixEnd < 0 || suffixEnd >= contentEnd) {
        throw new Error("Git index v4 path is not NUL-terminated");
      }
      pathBytes = Buffer.concat([
        previousPath.subarray(0, previousPath.length - decoded.value),
        buffer.subarray(offset, suffixEnd),
      ]);
      offset = suffixEnd + 1;
    } else {
      const declaredLength = flags & 0x0fff;
      let pathEnd;
      if (declaredLength < 0x0fff) {
        pathEnd = offset + declaredLength;
        checkedRange(buffer, offset, declaredLength + 1, "cache-entry path", contentEnd);
        if (buffer[pathEnd] !== 0) throw new Error("Git index path length does not end at NUL");
      } else {
        pathEnd = buffer.indexOf(0, offset);
        if (pathEnd < 0 || pathEnd >= contentEnd) {
          throw new Error("Git index path is not NUL-terminated");
        }
      }
      pathBytes = buffer.subarray(offset, pathEnd);
      offset = entryStart + Math.ceil((pathEnd + 1 - entryStart) / 8) * 8;
      checkedRange(buffer, entryStart, offset - entryStart, "cache-entry padding", contentEnd);
    }

    const path = pathBytes.toString("utf8");
    if (!path || !Buffer.from(path, "utf8").equals(pathBytes)) {
      throw new Error("Git index contains an empty or non-UTF-8 path");
    }
    entries.push(Object.freeze({ path, mode }));
    previousPath = Buffer.from(pathBytes);
  }

  while (offset < contentEnd) {
    checkedRange(buffer, offset, 8, "index extension", contentEnd);
    const signature = buffer.toString("ascii", offset, offset + 4);
    const size = buffer.readUInt32BE(offset + 4);
    checkedRange(buffer, offset + 8, size, `index extension ${signature}`, contentEnd);
    if (signature === "link") {
      throw new Error("Split Git indexes are unsupported; disable splitIndex before validation");
    }
    offset += 8 + size;
  }
  if (offset !== contentEnd) throw new Error("Git index has trailing unparsed data");
  return Object.freeze(entries);
}

export function parseGitIndex(buffer) {
  if (!Buffer.isBuffer(buffer)) throw new TypeError("Git index input must be a Buffer");
  for (const objectIdLength of [20, 32]) {
    const entries = parseIndexWithObjectId(buffer, objectIdLength);
    if (entries) return entries;
  }
  throw new Error("Git index checksum is invalid");
}

export function readTrackedIndex(rootDir = root) {
  return parseGitIndex(readFileSync(join(gitDirectory(rootDir), "index")));
}

export function selectTrackedJavaScriptFiles(rootDir, entries) {
  if (!Array.isArray(entries) || entries.length === 0) {
    throw new Error("Tracked file inventory is empty");
  }
  const files = [];
  for (const entry of entries) {
    if (!entry || typeof entry.path !== "string" || !/\.(?:js|mjs)$/.test(entry.path)) continue;
    if ((entry.mode & MODE_TYPE_MASK) !== REGULAR_FILE_MODE) continue;
    const file = resolve(rootDir, entry.path);
    const local = relative(rootDir, file);
    if (isAbsolute(local) || local === ".." || local.startsWith(`..${sep}`)) {
      throw new Error(`Tracked JavaScript path escapes the repository: ${entry.path}`);
    }
    let metadata;
    try {
      metadata = lstatSync(file);
    } catch (error) {
      throw new Error(`Tracked JavaScript source is missing: ${entry.path}`, { cause: error });
    }
    // Never follow a working-tree symlink, even if it replaced a regular index
    // entry locally. A tracked symlink is already excluded by its index mode.
    if (metadata.isSymbolicLink()) continue;
    if (!metadata.isFile()) {
      throw new Error(`Tracked JavaScript source is not a regular file: ${entry.path}`);
    }
    files.push(file);
  }
  return [...new Set(files)].sort();
}

export function checkTrackedJavaScriptSyntax(rootDir = root) {
  const { SourceTextModule } = vm;
  if (typeof SourceTextModule !== "function") {
    throw new Error("Syntax validation requires Node's --experimental-vm-modules flag");
  }
  const files = selectTrackedJavaScriptFiles(rootDir, readTrackedIndex(rootDir));
  if (!files.some((file) => relative(rootDir, file).replaceAll("\\", "/") === "tools/check-syntax.mjs")) {
    throw new Error("Tracked JavaScript inventory does not contain tools/check-syntax.mjs");
  }
  let failed = false;
  for (const file of files) {
    try {
      new SourceTextModule(readFileSync(file, "utf8"), { identifier: file });
    } catch (error) {
      failed = true;
      process.stderr.write(`${relative(rootDir, file)}: ${error.message}\n`);
    }
  }
  return { ok: !failed, count: files.length };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = checkTrackedJavaScriptSyntax();
  if (!result.ok) process.exitCode = 1;
  else console.log(`JavaScript syntax: ok (${result.count} tracked files)`);
}
