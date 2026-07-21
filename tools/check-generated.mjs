import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = resolve(import.meta.dirname, "..");
const temp = mkdtempSync(join(tmpdir(), "fsrcnnx-generated-"));
const originalArgv = process.argv;

function same(actual, expected) {
  return readFileSync(actual).equals(readFileSync(expected));
}

async function runScript(script, args, token) {
  process.argv = [process.execPath, resolve(root, script), ...args];
  await import(`${pathToFileURL(resolve(root, script)).href}?validation=${token}`);
}

try {
  const fsrcnnxOut = resolve(temp, "fsrcnnx");
  await runScript("transpile.js", [
    "shaders/upstream/FSRCNNX_x2_16-0-4-1.glsl",
    "--out", fsrcnnxOut,
  ], "fsrcnnx");

  for (const name of ["FSRCNNX_x2_16-0-4-1"]) {
    for (const suffix of ["passes.json", "wgsl"]) {
      const actual = resolve(fsrcnnxOut, `${name}.${suffix}`);
      const expected = resolve(root, "model", `${name}.${suffix}`);
      if (!same(actual, expected)) throw new Error(`${name}.${suffix} differs from regenerated output`);
    }
  }

  const artcnnOut = resolve(temp, "artcnn");
  await runScript("transpile-artcnn.js", [
    "shaders/upstream/ArtCNN_C4F32.glsl",
    "shaders/upstream/ArtCNN_C4F32_DN.glsl",
    "shaders/upstream/ArtCNN_C4F32_DS.glsl",
    "--out", artcnnOut,
  ], "artcnn");

  for (const name of ["ArtCNN_C4F32", "ArtCNN_C4F32_DN", "ArtCNN_C4F32_DS"]) {
    for (const suffix of ["artcnn.json", "artcnn.wgsl"]) {
      const actual = resolve(artcnnOut, `${name}.${suffix}`);
      const expected = resolve(root, "model", `${name}.${suffix}`);
      if (!same(actual, expected)) throw new Error(`${name}.${suffix} differs from regenerated output`);
    }
  }
  console.log("Generated model assets: reproducible");
} finally {
  process.argv = originalArgv;
  rmSync(temp, { recursive: true, force: true });
}
