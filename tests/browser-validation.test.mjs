import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import {
  browserVersion,
  fixtureDisplayDimensions,
} from "../tools/browser-validation.mjs";

const root = resolve(import.meta.dirname, "..");
const validator = readFileSync(resolve(root, "tools/browser-validation.mjs"), "utf8");
const workflow = readFileSync(resolve(root, ".github/workflows/ci.yml"), "utf8");
const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));

test("browser validator covers the packaged extension and real-video runtime", () => {
  assert.match(validator, /--extension-root/);
  assert.match(validator, /runRealVideoIntegration/);
  assert.match(validator, /Page\.setWebLifecycleState/);
  assert.match(validator, /DevTools target .* retirement/);
  assert.match(validator, /required: true/);
  assert.match(validator, /color-hdr-unsupported/);
  assert.match(validator, /color-wide-gamut-unsupported/);
  assert.match(validator, /data-fsrcnnx-overlay/);
  assert.match(validator, /Page\.captureScreenshot/);
  assert.match(validator, /getImageData/);
  assert.match(validator, /requirePixelProgression/);
  assert.match(validator, /changePopupControl\(popupPage\.client, "engine", "neural"\)/);
  assert.match(validator, /Neural ONNX presentation without fallback/);
  assert.match(validator, /status\.renderer\?\.fallback == null/);
  assert.match(validator, /status\.neuralRuntime\?\.phase === "active"/);
  assert.match(validator, /status\.neural\?\.n >= neuralRuns \+ 2/);
  assert.match(validator, /status\.presentation\.output\.width === status\.presentation\.source\.width \* scale/);
  assert.match(validator, /hostIsolation\?\.crossOriginIsolated === false/);
  assert.match(validator, /hostIsolation\?\.sharedArrayBuffer === "undefined"/);
  assert.match(validator, /Neural to FSRCNNX transition/);
  assert.match(validator, /__FSRCNNX_PIXEL_ISOLATION_RESTORE__/);
  assert.match(validator, /brightness\(0\)/);
  assert.match(validator, /element !== target/);
  assert.match(validator, /framesPresented >= interpolationGeneration/);
  assert.match(validator, /\/json\/new\?\$\{encodeURIComponent\(url\)\}/);
  assert.doesNotMatch(validator, /encodeURIComponent\("about:blank"\)/);
  assert.match(validator, /async function waitForDevToolsHttp/);
  assert.match(validator, /new URL\("json\/version", httpBase\)/);
  assert.match(validator, /await waitForDevToolsHttp\(httpBase, signal\)/);
  assert.match(validator, /deadline = Date\.now\(\) \+ STARTUP_TIMEOUT_MS/);
  assert.match(validator, /chrome\.tabs\.create/);
  assert.match(validator, /manifest\.background\.service_worker/);
  assert.match(validator, /service-worker bootstrap/);
  assert.doesNotMatch(validator, /extensionIdFromPath/);
  assert.match(validator, /127\.0\.0\.1/);
  assert.doesNotMatch(validator, /--no-sandbox/);
});

test("browser validator accepts configured paths on Windows and POSIX", () => {
  assert.match(validator, /isAbsolute\(candidate\)/);
  assert.ok(validator.includes('candidate.includes("\\\\")'));
  assert.match(validator, /process\.platform !== "win32" && !process\.env\.DISPLAY/);
  assert.match(validator, /softwareGpuArguments = process\.platform === "linux"/);
  assert.match(validator, /--disable-backgrounding-occluded-windows/);
  assert.doesNotMatch(validator, /msEdgeDisableEnhancedSecurityMode/);
  assert.doesNotMatch(validator, /state\.readyState === "complete" && state\.operationText\) break/);
});

test("fixture display scaling maps requested physical pixels through the live DPR", () => {
  assert.deepEqual(fixtureDisplayDimensions(160, 90, 1.35, 1), {
    width: 216,
    height: 122,
    physicalWidth: 216,
    physicalHeight: 122,
    devicePixelRatio: 1,
  });
  assert.deepEqual(fixtureDisplayDimensions(160, 90, 1.35, 2), {
    width: 108,
    height: 61,
    physicalWidth: 216,
    physicalHeight: 122,
    devicePixelRatio: 2,
  });
  assert.deepEqual(fixtureDisplayDimensions(160, 90, 1.35, 1.25), {
    width: 172.8,
    height: 97.6,
    physicalWidth: 216,
    physicalHeight: 122,
    devicePixelRatio: 1.25,
  });
  assert.equal(fixtureDisplayDimensions(160, 90, 1.35, 0).devicePixelRatio, 1);
  assert.match(validator, /window\.devicePixelRatio/);
});

test("Windows browser version detection reads executable metadata without launching Edge", async () => {
  const calls = [];
  const spawnProcess = (command, args, options) => {
    calls.push({ command, args, options });
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.kill = () => {};
    queueMicrotask(() => {
      child.stdout.emit("data", Buffer.from("150.0.4078.83\r\n"));
      child.emit("exit", 0, null);
    });
    return child;
  };
  const browser = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
  const version = await browserVersion(browser, undefined, {
    platform: "win32",
    spawnProcess,
    environment: { SystemRoot: "C:\\Windows" },
  });

  assert.equal(version, "150.0.4078.83");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, "powershell.exe");
  assert.ok(!calls[0].args.includes("--version"));
  assert.equal(calls[0].options.env.FSRCNNX_BROWSER_VERSION_TARGET, browser);
});

test("browser fixture server pins a fail-closed content security policy", () => {
  const expected =
    "default-src 'none'; script-src 'self'; style-src 'self'; media-src 'self'; " +
    "base-uri 'none'; frame-ancestors 'none'";
  const declaration = validator.match(
    /const FIXTURE_CONTENT_SECURITY_POLICY =\s*("[^"\n]*") \+\s*("[^"\n]*");/,
  );
  assert.ok(declaration, "fixture CSP constant is missing or no longer an exact literal");
  assert.equal(JSON.parse(declaration[1]) + JSON.parse(declaration[2]), expected);
  assert.ok(validator.includes(`"Content-Security-Policy": FIXTURE_CONTENT_SECURITY_POLICY`));
  for (const directive of [
    "default-src 'none'", "script-src 'self'", "style-src 'self'", "media-src 'self'",
    "base-uri 'none'", "frame-ancestors 'none'",
  ]) assert.ok(expected.includes(directive));
});

test("browser validation is release-blocking without slowing internal packaging", () => {
  const releaseCheck = packageJson.scripts["release:check"];
  assert.match(releaseCheck, /npm run package:stage/);
  assert.match(releaseCheck, /npm run validate:browser -- --extension-root dist\/fsrcnnx-ext/);
  assert.ok(releaseCheck.indexOf("npm run package:stage") < releaseCheck.indexOf("npm run validate:browser"));
  assert.equal(packageJson.scripts.package, "npm run release:check");
  assert.doesNotMatch(packageJson.scripts["package:internal"], /validate:browser/);
  assert.match(packageJson.scripts.check, /check:browser-fixtures/);
});

test("CI validates the staged package under Xvfb without disabling the sandbox", () => {
  assert.match(workflow, /browser-integration:/);
  assert.match(workflow, /name: Validate packaged extension in Chromium/);
  assert.match(workflow, /FSRCNNX_BROWSER: chromium/);
  assert.match(workflow, /"\$FSRCNNX_BROWSER" --version/);
  assert.match(workflow, /readlink -f "\$\(command -v "\$FSRCNNX_BROWSER"\)"/);
  assert.match(workflow, /profile fsrcnnx-chromium-ci \$chromium_path flags=\(unconfined\)/);
  assert.match(workflow, /'  userns,'/);
  assert.match(workflow, /apparmor_parser -r \/etc\/apparmor\.d\/fsrcnnx-chromium-ci/);
  assert.match(workflow, /apparmor_status \| grep -F 'fsrcnnx-chromium-ci'/);
  assert.doesNotMatch(workflow, /CHROME_DEVEL_SANDBOX/);
  assert.doesNotMatch(workflow, /apparmor_restrict_unprivileged_userns=0/);
  assert.doesNotMatch(workflow, /FSRCNNX_BROWSER: google-chrome/);
  assert.match(workflow, /npm run package:internal/);
  assert.match(workflow, /--extension-root dist\/fsrcnnx-ext/);
  assert.match(workflow, /xvfb-run/);
  assert.doesNotMatch(workflow, /--no-sandbox/);
});
