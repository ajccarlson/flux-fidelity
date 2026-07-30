// Shared prelude for test harnesses that slice fsrcnnx-main.js and evaluate the
// slice as a data: URL module.
//
// fsrcnnx-main.js imports its setting enums from fsrcnnx-setting-contract.js, so
// a sliced region referencing CONTRACT_* would throw ReferenceError without this.
// The prelude imports the real module by absolute URL rather than redeclaring the
// values, because a second copy in the tests would recreate exactly the drift the
// contract module exists to prevent — and would let a production change pass
// against a stale test-local constant.
//
// tests/neural-lifecycle.test.mjs already rewrites relative specifiers to
// absolute URLs for the same reason; this is that pattern, shared.
export const CONTRACT_IMPORT = [
  "import {",
  "  ENGINES as CONTRACT_ENGINES,",
  "  INTERPOLATION_MODELS as CONTRACT_INTERPOLATION_MODELS,",
  "  INTERPOLATION_RES_MODES as CONTRACT_INTERPOLATION_RES_MODES,",
  "  MODES as CONTRACT_MODES,",
  "  UPSCALE_POLICIES as CONTRACT_UPSCALE_POLICIES,",
  "  upscalePoliciesForEngine,",
  `} from ${JSON.stringify(
    new URL("../../src/core/fsrcnnx-setting-contract.js", import.meta.url).href,
  )};`,
  "",
].join("\n");
