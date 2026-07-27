import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

import {
  browserVersion,
  fixtureDisplayDimensions,
  isRecoveredInterpolationWatchdogWarning,
  isUnsupportedNeuralF16Fallback,
  isUnsupportedNeuralF16Warning,
  parseArguments,
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
  assert.match(
    validator,
    /neuralModelKey: options\.neuralModelKey/,
  );
  assert.match(validator, /FSRCNNX_SETNEURALMODEL/);
  assert.match(
    validator,
    /status\.neuralRuntime\.activeModel === neuralModelKey/,
  );
  assert.match(validator, /status\.neural\?\.temporalResetRuns >= 1/);
  assert.match(validator, /status\.neural\?\.temporalRecurrentRuns >= 2/);
  assert.match(validator, /status\.renderer\?\.fallback == null/);
  assert.match(validator, /status\.neuralRuntime\?\.phase === "active"/);
  assert.match(validator, /status\.neural\?\.n >= neuralRuns \+ 2/);
  assert.match(validator, /status\.presentation\.output\.width === status\.presentation\.source\.width \* outputScale/);
  assert.match(validator, /status\.policy === "force2" \? 2 : scale/);
  assert.match(validator, /Neural exact x2 sampled and SSimDownscaler presentation/);
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

test("browser validator can pin one neural model for external-artifact probes", () => {
  const options = parseArguments([
    "--extension-root",
    "dist/flux-fidelity",
    "--neural-model-key",
    "cda-vsr-4x-local-probe",
    "--require-temporal-neural-runs",
  ]);
  assert.equal(options.extensionRoot, resolve(root, "dist/flux-fidelity"));
  assert.equal(options.neuralModelKey, "cda-vsr-4x-local-probe");
  assert.equal(options.requireTemporalNeuralRuns, true);
  assert.equal(options.allowNeuralF16Unavailable, false);
  assert.throws(
    () => parseArguments(["--neural-model-key", "../outside"]),
    /not a valid neural model key/,
  );
  assert.throws(
    () => parseArguments(["--require-temporal-neural-runs"]),
    /requires --neural-model-key/,
  );
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

test("CI may accept only the explicit shader-f16 hardware fallback", () => {
  const status = {
    renderer: {
      fallback: {
        from: "neural",
        to: "fsrcnnx",
        code: "neural-init-failed",
        detail: "Program Transpose requires f16 but the device does not support it.",
      },
    },
    neuralRuntime: {
      requested: true,
      phase: "fallback",
      lastFailure: {
        code: "neural-init-failed",
        detail: "Program Transpose requires f16 but the device does not support it.",
      },
    },
  };
  assert.equal(isUnsupportedNeuralF16Fallback(status), true);

  for (const mutate of [
    (value) => { value.renderer.fallback.to = "off"; },
    (value) => { value.renderer.fallback.code = "neural-run-failed"; },
    (value) => { value.neuralRuntime.phase = "failed"; },
    (value) => { value.neuralRuntime.lastFailure.detail = "model output is malformed"; },
  ]) {
    const changed = structuredClone(status);
    mutate(changed);
    assert.equal(isUnsupportedNeuralF16Fallback(changed), false);
  }

  const warning = {
    kind: "console",
    type: "warning",
    text: "[FSRCNNX] neural init failed: Program Transpose requires f16 but the device does not support it.",
  };
  assert.equal(isUnsupportedNeuralF16Warning(warning), true);
  assert.equal(isUnsupportedNeuralF16Warning({ ...warning, type: "error" }), false);
  assert.equal(isUnsupportedNeuralF16Warning({
    ...warning,
    text: "[FSRCNNX] another warning: Program Transpose requires f16 but the device does not support it.",
  }), false);
});

test("software-GPU validation may accept only a recovered interpolation watchdog", () => {
  const warning = {
    kind: "console",
    type: "warning",
    text: "[FSRCNNX] interp WATCHDOG: present stalled 1020ms (q=9, headDue=0ms, headTs=0.38s) — re-anchoring to recover",
  };
  assert.equal(isRecoveredInterpolationWatchdogWarning(warning), true);
  assert.equal(isRecoveredInterpolationWatchdogWarning({ ...warning, type: "error" }), false);
  assert.equal(isRecoveredInterpolationWatchdogWarning({
    ...warning,
    text: warning.text.replace("re-anchoring to recover", "giving up"),
  }), false);
  assert.equal(isRecoveredInterpolationWatchdogWarning({
    ...warning,
    text: "[FSRCNNX] interp WATCHDOG: arbitrary warning",
  }), false);
  assert.match(validator, /usingSoftwareGpu: process\.platform === "linux"/);
  assert.match(
    validator,
    /usingSoftwareGpu && isRecoveredInterpolationWatchdogWarning\(event\)/,
  );
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
  assert.match(releaseCheck, /npm run validate:browser -- --extension-root dist\/flux-fidelity/);
  assert.match(releaseCheck, /--neural-model-key cda-vsr-4x/);
  assert.match(releaseCheck, /--require-temporal-neural-runs/);
  assert.ok(releaseCheck.indexOf("npm run package:stage") < releaseCheck.indexOf("npm run validate:browser"));
  assert.equal(packageJson.scripts.package, "npm run release:check");
  assert.doesNotMatch(packageJson.scripts["package:internal"], /validate:browser/);
  assert.match(packageJson.scripts.check, /check:browser-fixtures/);
});

test("CI validates the staged package under Xvfb without disabling the sandbox", () => {
  assert.equal(
    (workflow.match(
      /actions\/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405/g,
    ) || []).length,
    2,
  );
  assert.equal((workflow.match(/python-version: "3\.11"/g) || []).length, 2);
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
  assert.match(workflow, /--extension-root dist\/flux-fidelity/);
  assert.match(workflow, /--neural-model-key cda-vsr-4x/);
  assert.match(workflow, /--require-temporal-neural-runs/);
  assert.match(workflow, /--allow-neural-f16-unavailable/);
  assert.match(workflow, /xvfb-run/);
  assert.doesNotMatch(workflow, /--no-sandbox/);
});
