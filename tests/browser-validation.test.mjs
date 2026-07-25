import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

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
