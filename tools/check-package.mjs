import { readFileSync, readdirSync, statSync } from "node:fs";
import { resolve, relative, dirname, normalize } from "node:path";

const root = resolve(import.meta.dirname, "..");
const skipped = new Set([".git", "node_modules", "dist", "build"]);
const errors = [];

function walk(dir = root) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    if (skipped.has(entry)) continue;
    const path = resolve(dir, entry);
    if (statSync(path).isDirectory()) out.push(...walk(path));
    else out.push(relative(root, path).replaceAll("\\", "/"));
  }
  return out;
}

// Authoring documents are intentionally retained locally and excluded by
// .gitignore. They are not extension package inputs.
const files = walk().filter((file) => !/\.docx$/i.test(file));
const fileSet = new Set(files);
const manifest = JSON.parse(readFileSync(resolve(root, "manifest.json"), "utf8"));
const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));

if (manifest.manifest_version !== 3) errors.push("manifest_version must be 3");
if (manifest.version !== pkg.version) errors.push(`manifest version ${manifest.version} differs from package ${pkg.version}`);

function requireFile(path, source) {
  if (!path || !fileSet.has(path)) errors.push(`${source}: missing ${path || "<empty>"}`);
}

requireFile(manifest.action?.default_popup, "action.default_popup");
for (const [size, path] of Object.entries(manifest.icons || {})) requireFile(path, `icons.${size}`);
for (const [size, path] of Object.entries(manifest.action?.default_icon || {})) requireFile(path, `action.default_icon.${size}`);
requireFile(manifest.background?.service_worker, "background.service_worker");
for (const script of manifest.content_scripts || []) {
  for (const path of [...(script.js || []), ...(script.css || [])]) requireFile(path, "content_scripts");
}

function globRegex(pattern) {
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replaceAll("*", "[^/]*");
  return new RegExp(`^${escaped}$`);
}

for (const group of manifest.web_accessible_resources || []) {
  for (const resource of group.resources || []) {
    if (resource.includes("*")) {
      if (!files.some((file) => globRegex(resource).test(file))) errors.push(`web_accessible_resources: ${resource} matches nothing`);
    } else requireFile(resource, "web_accessible_resources");
  }
}

for (const html of files.filter((file) => file.endsWith(".html"))) {
  const source = readFileSync(resolve(root, html), "utf8");
  for (const match of source.matchAll(/<(?:script|link)\b[^>]+(?:src|href)=["']([^"']+)["']/gi)) {
    const ref = match[1];
    if (/^(?:https?:|data:|#)/.test(ref)) continue;
    const local = normalize(resolve(dirname(resolve(root, html)), ref));
    requireFile(relative(root, local).replaceAll("\\", "/"), html);
  }
}

const popupSource = readFileSync(resolve(root, "popup.js"), "utf8");
const popupHtml = readFileSync(resolve(root, "popup.html"), "utf8");
const requiredPopupIds = new Set([...popupSource.matchAll(/\$\(["']([^"']+)["']\)/g)].map((match) => match[1]));
const popupIds = [...popupHtml.matchAll(/\bid=["']([^"']+)["']/g)].map((match) => match[1]);
const popupIdSet = new Set(popupIds);
for (const id of requiredPopupIds) if (!popupIdSet.has(id)) errors.push(`popup.html: missing #${id}`);
for (const id of popupIdSet) if (popupIds.filter((candidate) => candidate === id).length > 1) errors.push(`popup.html: duplicate #${id}`);
if (/<script\b(?![^>]*\bsrc=)[^>]*>/i.test(popupHtml)) errors.push("popup.html: inline scripts violate MV3 CSP");

for (const js of files.filter((file) => /\.(?:js|mjs)$/.test(file))) {
  const source = readFileSync(resolve(root, js), "utf8");
  for (const match of source.matchAll(/(?:from\s*|import\s*\()\s*["'](\.[^"']+)["']/g)) {
    const local = relative(root, resolve(dirname(resolve(root, js)), match[1])).replaceAll("\\", "/");
    requireFile(local, js);
  }
  for (const match of source.matchAll(/chrome\.runtime\.getURL\(["']([^"']+)["']\)/g)) {
    if (!match[1].endsWith("/")) requireFile(match[1], js);
  }
}

for (const file of files) {
  if (statSync(resolve(root, file)).size >= 100 * 1024 * 1024) errors.push(`${file}: exceeds GitHub's 100 MiB object limit`);
}

if (errors.length) {
  for (const error of errors) console.error(error);
  process.exit(1);
}
console.log(`Extension package references: ok (${files.length} files inspected)`);
