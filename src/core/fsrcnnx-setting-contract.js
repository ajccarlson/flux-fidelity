// fsrcnnx-setting-contract.js — the single source of truth for every setting
// value that crosses the popup → content-script → page-runtime boundary.
//
// These enums previously existed in three independent copies: the runtime's own
// validation in fsrcnnx-main.js, the content script's message gate, and the
// popup's option lists. Nothing asserted they agreed, and they drifted: the
// neural "native" policy was offered by the popup and accepted by the runtime
// while the content-script gate rejected it, so a shipped feature was
// unreachable through the UI. fsrcnnx-model-catalog.js already establishes this
// pattern for model names; this module extends it to the setting contracts.
//
// The content script cannot be an ES module, so it reads these values from the
// pipeline module it loads rather than importing them directly. Labels stay in
// the popup — only values live here.

export const MODES = Object.freeze(["off", "passthrough", "upscale"]);

export const ENGINES = Object.freeze(["fsrcnnx", "fsrcnnx-hi", "artcnn", "neural"]);

// Policies are per-engine because they describe what that engine can actually
// produce. The message gate validates against the union and the runtime then
// applies the engine-specific set, so an engine-invalid value is rejected by the
// layer that knows the engine.
// fsrcnnx-hi and artcnn are fixed-x2 engines: they encode forced scale through
// cascade depth, so they offer force8 and not force3. The standard engine is the
// other way round.
export const UPSCALE_POLICIES_BY_ENGINE = Object.freeze({
  "fsrcnnx": Object.freeze(["display", "auto", "force2", "force3", "force4"]),
  "fsrcnnx-hi": Object.freeze(["display", "auto", "force2", "force4", "force8"]),
  "artcnn": Object.freeze(["display", "auto", "force2", "force4", "force8"]),
  "neural": Object.freeze(["display", "force2", "native"]),
});

// Derived, never hand-maintained: this is what made the three copies drift.
export const UPSCALE_POLICIES = Object.freeze([
  ...new Set(Object.values(UPSCALE_POLICIES_BY_ENGINE).flat()),
]);

export const INTERPOLATION_MODELS = Object.freeze([
  "rife_v4.26_fp16",
  "rife_v4.26",
  "blend",
]);

export const INTERPOLATION_RES_MODES = Object.freeze(["auto", "full", "half", "quarter"]);

export function upscalePoliciesForEngine(engine) {
  return UPSCALE_POLICIES_BY_ENGINE[engine] || UPSCALE_POLICIES_BY_ENGINE.fsrcnnx;
}
