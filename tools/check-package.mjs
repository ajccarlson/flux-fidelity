import { lstatSync, readFileSync } from "node:fs";
import { dirname, normalize, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PACKAGE_FILES, REQUIRED_RUNTIME_MODEL_FILES } from "./package-files.mjs";

const root = resolve(import.meta.dirname, "..");

function normalizedRelativePath(rootDir, sourceFile, reference) {
  const withoutQuery = reference.split(/[?#]/, 1)[0];
  const local = normalize(resolve(dirname(resolve(rootDir, sourceFile)), withoutQuery));
  return relative(rootDir, local).replaceAll("\\", "/");
}

function globRegex(pattern) {
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replaceAll("*", "[^/]*");
  return new RegExp(`^${escaped}$`);
}

function localImportReferences(source) {
  const references = [];
  const patterns = [
    /\b(?:import|export)\s+(?:[^"'`]*?\s+from\s*)?["'](\.[^"'`]+)["']/g,
    /\bimport\s*\(\s*["'](\.[^"'`]+)["']\s*\)/g,
  ];
  for (const pattern of patterns) {
    for (const match of source.matchAll(pattern)) references.push(match[1]);
  }
  return references;
}

function runtimeUrlReferences(source) {
  const references = [];
  const getUrlPattern = /chrome\.runtime\.getURL\(\s*(["'`])([^"'`]*)\1/g;
  for (const match of source.matchAll(getUrlPattern)) {
    const quote = match[1];
    const expressionTail = source.slice(match.index + match[0].length);
    const templateMarker = quote === "`" ? match[2].indexOf("${") : -1;
    const expressionContinues = !/^\s*\)/.test(expressionTail);
    if (templateMarker >= 0) {
      references.push({ type: "prefix", value: match[2].slice(0, templateMarker) });
    } else if (expressionContinues || match[2].endsWith("/")) {
      references.push({ type: "prefix", value: match[2] });
    } else {
      references.push({ type: "file", value: match[2] });
    }
  }
  return references;
}

function parseJson(rootDir, file, errors) {
  try {
    return JSON.parse(readFileSync(resolve(rootDir, file), "utf8"));
  } catch (error) {
    errors.push(`${file}: ${error.message}`);
    return null;
  }
}

function readText(rootDir, file, errors) {
  try {
    return readFileSync(resolve(rootDir, file), "utf8");
  } catch (error) {
    errors.push(`${file}: ${error.message}`);
    return null;
  }
}

export function validatePackage({ rootDir = root, packageFiles = PACKAGE_FILES } = {}) {
  const errors = [];
  const fileSet = new Set(packageFiles);

  if (fileSet.size !== packageFiles.length) errors.push("package allowlist contains duplicate entries");

  for (const file of packageFiles) {
    if (!file || file.startsWith("/") || file.includes("\\") || file.split("/").includes("..")) {
      errors.push(`package allowlist: invalid path ${file || "<empty>"}`);
      continue;
    }
    try {
      const metadata = lstatSync(resolve(rootDir, file));
      if (!metadata.isFile() || metadata.isSymbolicLink()) {
        errors.push(`${file}: package input must be a regular file`);
      } else if (metadata.size >= 100 * 1024 * 1024) {
        errors.push(`${file}: exceeds GitHub's 100 MiB object limit`);
      }
    } catch (error) {
      errors.push(`${file}: package input is missing (${error.code || error.message})`);
    }
  }

  function requireFile(path, source) {
    if (!path || !fileSet.has(path)) errors.push(`${source}: missing ${path || "<empty>"} from package`);
  }

  function requirePrefix(prefix, source) {
    if (!prefix || !packageFiles.some((file) => file.startsWith(prefix))) {
      errors.push(`${source}: package contains no file matching ${prefix || "<dynamic path>"}`);
    }
  }

  for (const file of REQUIRED_RUNTIME_MODEL_FILES) {
    requireFile(file, "runtime model assets");
  }

  const manifest = fileSet.has("manifest.json") ? parseJson(rootDir, "manifest.json", errors) : null;
  const pkg = parseJson(rootDir, "package.json", errors);

  if (!fileSet.has("manifest.json")) errors.push("package allowlist: missing manifest.json");
  if (manifest && manifest.manifest_version !== 3) errors.push("manifest_version must be 3");
  if (manifest && pkg && manifest.version !== pkg.version) {
    errors.push(`manifest version ${manifest.version} differs from package ${pkg.version}`);
  }

  if (manifest) {
    requireFile(manifest.action?.default_popup, "action.default_popup");
    for (const [size, path] of Object.entries(manifest.icons || {})) requireFile(path, `icons.${size}`);
    for (const [size, path] of Object.entries(manifest.action?.default_icon || {})) {
      requireFile(path, `action.default_icon.${size}`);
    }
    requireFile(manifest.background?.service_worker, "background.service_worker");
    for (const script of manifest.content_scripts || []) {
      for (const path of [...(script.js || []), ...(script.css || [])]) requireFile(path, "content_scripts");
    }

    for (const group of manifest.web_accessible_resources || []) {
      for (const resource of group.resources || []) {
        if (resource.includes("*")) {
          if (!packageFiles.some((file) => globRegex(resource).test(file))) {
            errors.push(`web_accessible_resources: ${resource} matches nothing in package`);
          }
        } else requireFile(resource, "web_accessible_resources");
      }
    }

    const webAccessiblePatterns = (manifest.web_accessible_resources || [])
      .flatMap((group) => group.resources || []);
    const isWebAccessible = (file) => webAccessiblePatterns.some((pattern) => (
      pattern.includes("*") ? globRegex(pattern).test(file) : pattern === file
    ));
    const requireWebAccessible = (file, source) => {
      if (!isWebAccessible(file)) {
        errors.push(
          `${source}: content-script dependency ${file} is not declared in web_accessible_resources`,
        );
      }
    };

    // Content scripts execute against arbitrary pages. Walk every module and
    // runtime URL reachable from those entry points so a packaged file cannot
    // work in extension-owned pages while failing only when the content script
    // attempts to load it. Extension pages and the service worker deliberately
    // remain outside this graph and do not need to be exposed to page origins.
    const pendingContentModules = (manifest.content_scripts || [])
      .flatMap((script) => script.js || []);
    const visitedContentModules = new Set();
    while (pendingContentModules.length) {
      const sourceFile = pendingContentModules.shift();
      if (visitedContentModules.has(sourceFile) || !fileSet.has(sourceFile)) continue;
      visitedContentModules.add(sourceFile);
      const source = readText(rootDir, sourceFile, errors);
      if (source === null) continue;

      const referencedFiles = [];
      for (const reference of localImportReferences(source)) {
        referencedFiles.push(normalizedRelativePath(rootDir, sourceFile, reference));
      }
      for (const reference of runtimeUrlReferences(source)) {
        if (reference.type === "file") {
          referencedFiles.push(reference.value);
        } else {
          referencedFiles.push(...packageFiles.filter((file) => file.startsWith(reference.value)));
        }
      }

      for (const referencedFile of new Set(referencedFiles)) {
        // The general reference pass below reports missing package inputs. Only
        // packaged dependencies can be checked for accessibility or traversed.
        if (!fileSet.has(referencedFile)) continue;
        requireWebAccessible(referencedFile, sourceFile);
        if (/\.(?:js|mjs)$/.test(referencedFile)) pendingContentModules.push(referencedFile);
      }
    }
  }

  for (const html of packageFiles.filter((file) => file.endsWith(".html"))) {
    const source = readText(rootDir, html, errors);
    if (source === null) continue;
    for (const match of source.matchAll(/<(?:script|link)\b[^>]+(?:src|href)=["']([^"']+)["']/gi)) {
      const ref = match[1];
      if (/^(?:https?:|data:|#)/.test(ref)) continue;
      requireFile(normalizedRelativePath(rootDir, html, ref), html);
    }
  }

  if (fileSet.has("popup.js") && fileSet.has("popup.html")) {
    const popupSource = readText(rootDir, "popup.js", errors);
    const popupHtml = readText(rootDir, "popup.html", errors);
    if (popupSource !== null && popupHtml !== null) {
      const requiredPopupIds = new Set(
        [...popupSource.matchAll(/\$\(["']([^"']+)["']\)/g)].map((match) => match[1]),
      );
      const popupIds = [...popupHtml.matchAll(/\bid=["']([^"']+)["']/g)].map((match) => match[1]);
      const popupIdSet = new Set(popupIds);
      for (const id of requiredPopupIds) if (!popupIdSet.has(id)) errors.push(`popup.html: missing #${id}`);
      for (const id of popupIdSet) {
        if (popupIds.filter((candidate) => candidate === id).length > 1) {
          errors.push(`popup.html: duplicate #${id}`);
        }
      }
      if (/<script\b(?![^>]*\bsrc=)[^>]*>/i.test(popupHtml)) {
        errors.push("popup.html: inline scripts violate MV3 CSP");
      }
    }
  }

  for (const js of packageFiles.filter((file) => /\.(?:js|mjs)$/.test(file))) {
    const source = readText(rootDir, js, errors);
    if (source === null) continue;
    for (const reference of localImportReferences(source)) {
      requireFile(normalizedRelativePath(rootDir, js, reference), js);
    }

    for (const reference of runtimeUrlReferences(source)) {
      if (reference.type === "prefix") {
        requirePrefix(reference.value, js);
      } else {
        requireFile(reference.value, js);
      }
    }
  }

  return errors;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const errors = validatePackage();
  if (errors.length) {
    for (const error of errors) console.error(error);
    process.exitCode = 1;
  } else {
    console.log(`Extension package references: ok (${PACKAGE_FILES.length} packaged files inspected)`);
  }
}
