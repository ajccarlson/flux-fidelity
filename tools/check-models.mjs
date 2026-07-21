import { readFileSync, readdirSync, statSync } from "node:fs";
import { basename, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const modelDir = resolve(root, "model");
const errors = [];

for (const name of readdirSync(modelDir).filter((file) => /\.(?:passes|artcnn)\.json$/.test(file)).sort()) {
  const manifest = JSON.parse(readFileSync(resolve(modelDir, name), "utf8"));
  if (!Array.isArray(manifest.passes) || manifest.passes.length === 0) {
    errors.push(`${name}: missing passes`);
    continue;
  }

  const available = new Set(["LUMA"]);
  manifest.passes.forEach((pass, index) => {
    if (pass.index !== index) errors.push(`${name}: pass ${index} has index ${pass.index}`);
    if (!Array.isArray(pass.binds) || pass.binds.length === 0) errors.push(`${name}: pass ${index} has no inputs`);
    for (const bind of pass.binds || []) {
      if (!available.has(bind)) errors.push(`${name}: pass ${index} binds unavailable ${bind}`);
    }
    if (pass.save) {
      // FSRCNNX intentionally reuses logical names across mapping bands. A later
      // pass replaces the prior texture in the runtime's logical resource map.
      available.add(pass.save);
    }
  });

  const wgslName = name.replace(/\.passes\.json$/, ".wgsl").replace(/\.artcnn\.json$/, ".artcnn.wgsl");
  const wgsl = readFileSync(resolve(modelDir, wgslName), "utf8");
  const entries = [...wgsl.matchAll(/@compute\s+@workgroup_size[\s\S]{0,180}?fn\s+(\w+)/g)];
  if (entries.length !== manifest.passes.length) {
    errors.push(`${name}: ${manifest.passes.length} passes but ${entries.length} WGSL compute entries`);
  }
}

const neuralManifest = JSON.parse(readFileSync(resolve(modelDir, "neural", "manifest.json"), "utf8"));
const neuralKeys = new Set();
for (const entry of neuralManifest) {
  if (!entry.key || neuralKeys.has(entry.key)) errors.push(`neural manifest: invalid/duplicate key ${entry.key}`);
  neuralKeys.add(entry.key);
  if (!Number.isInteger(entry.scale) || entry.scale < 1) errors.push(`neural manifest: invalid scale for ${entry.key}`);
  const file = resolve(modelDir, "neural", entry.file || "");
  try {
    if (!statSync(file).isFile()) throw new Error();
  } catch {
    errors.push(`neural manifest: missing model ${entry.file}`);
  }
}

if (errors.length) {
  for (const error of errors) console.error(error);
  process.exit(1);
}
console.log(`Model manifests: ok (${basename(modelDir)})`);
