import { readdirSync, statSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const ignored = new Set([".git", "node_modules", "dist", "build"]);

function walk(dir) {
  const files = [];
  for (const entry of readdirSync(dir)) {
    if (ignored.has(entry)) continue;
    const path = resolve(dir, entry);
    if (statSync(path).isDirectory()) files.push(...walk(path));
    else if (/\.(?:js|mjs)$/.test(entry)) files.push(path);
  }
  return files;
}

let failed = false;
for (const file of walk(root)) {
  const result = spawnSync(process.execPath, ["--check", file], { encoding: "utf8" });
  if (result.status !== 0) {
    failed = true;
    process.stderr.write(result.stderr || result.stdout);
  }
}

if (failed) process.exit(1);
console.log("JavaScript syntax: ok");
