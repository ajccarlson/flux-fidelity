import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const mainUrl = new URL("../fsrcnnx-main.js", import.meta.url);
let revision = 0;

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);
  return source.slice(start, end);
}

async function loadLifecycle(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(source, "export async function suspendDocument()", "// A video can be unreadable");
  const harness = `
    const deps = globalThis.__documentLifecycleDeps;
    let pageSuspended = false;
    let mode = "upscale", optInterpolate = true, optImages = true;
    let adoptionGeneration = 2, modeSelectionGeneration = 3, videoSelectionGeneration = 4;
    let interpolationSelectionGeneration = 5, imagesSelectionGeneration = 6;
    let imageUpscalerInitGeneration = 7;
    let videoMonitor = { stop: () => deps.events.push("monitor-stop") };
    let canvas = { style: { display: "block", opacity: "0" } };
    let primaryController = { active: true, video: deps.video };
    let neuralEng = { stop: () => deps.events.push("neural-stop") };
    let interpolator = { stop: () => deps.events.push("interp-stop") };
    let imageUpscaler = { stop: () => deps.events.push("image-stop") };
    const initialColorCache = new WeakMap();
    let videoColorSupportCache = initialColorCache;
    const uncheckedColorSupport = (detail) => ({ supported: false, code: "color-not-checked", detail });
    let selectedColorSupport = { supported: true, code: "color-supported" };
    const cancelDeviceRecovery = () => deps.events.push("recovery-cancel");
    const cancelMainLoop = () => deps.events.push("loop-cancel");
    const clearMultiTargets = () => deps.events.push("multi-clear");
    const chainTap = () => deps.events.push("chain-clear");
    let chainInverted = true, _texSource = {};
    const detach = () => { deps.events.push("detach"); primaryController = null; };
    const notifyState = () => deps.events.push("notify");
    const updateVideoMonitor = () => deps.events.push("monitor-update");
    const waitForGpuRetirement = async () => deps.events.push("retirement-wait");
    const retireGpuResources = async (reason) => deps.events.push(["gpu-retire", reason]);
    const ensureImageUpscaler = async () => ({ start: () => deps.events.push("image-start") });
    const startImageUpscalerIfCurrent = (upscaler) => {
      upscaler?.start?.();
      return !!upscaler;
    };
    const findVideo = () => deps.video;
    const queueVideoSelection = async (candidate, options) => {
      deps.events.push(["reconcile", candidate, options.force]);
      primaryController = { active: true, video: candidate };
      return true;
    };
    const warn = () => {};
    ${production}
    export function state() {
      return { pageSuspended, mode, optInterpolate, optImages, adoptionGeneration,
        modeSelectionGeneration, videoSelectionGeneration, interpolationSelectionGeneration,
        imagesSelectionGeneration, imageUpscalerInitGeneration, primaryController,
        display: canvas.style.display, opacity: canvas.style.opacity,
        colorCacheReplaced: videoColorSupportCache !== initialColorCache,
        colorSupport: selectedColorSupport };
    }
  `;
  globalThis.__documentLifecycleDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(harness).toString("base64")}#${++revision}`);
}

test("document suspension is idempotent and invalidates every asynchronous producer", async (t) => {
  const previous = globalThis.__documentLifecycleDeps;
  t.after(() => { globalThis.__documentLifecycleDeps = previous; });
  const deps = { video: { id: "A" }, events: [] };
  const lifecycle = await loadLifecycle(deps);

  assert.deepEqual(await lifecycle.suspendDocument(), { ok: true, suspended: true, changed: true });
  const firstEvents = [...deps.events];
  assert.deepEqual(await lifecycle.suspendDocument(), { ok: true, suspended: true, changed: false });
  assert.deepEqual(
    deps.events.filter((event) => event !== "retirement-wait"),
    firstEvents,
    "duplicate lifecycle signals must not tear down twice",
  );

  const state = lifecycle.state();
  assert.equal(state.mode, "upscale", "durable requested mode is preserved");
  assert.equal(state.optInterpolate, true);
  assert.equal(state.optImages, true);
  assert.equal(state.adoptionGeneration, 3);
  assert.equal(state.modeSelectionGeneration, 4);
  assert.equal(state.videoSelectionGeneration, 5);
  assert.equal(state.interpolationSelectionGeneration, 6);
  assert.equal(state.imagesSelectionGeneration, 7);
  assert.equal(state.imageUpscalerInitGeneration, 8);
  assert.equal(state.primaryController, null);
  assert.equal(state.display, "none");
  assert.equal(state.opacity, "1");
  assert.equal(state.colorCacheReplaced, true, "suspension drops every cached frame classification");
  assert.equal(state.colorSupport.code, "color-not-checked");
  for (const event of ["loop-cancel", "neural-stop", "interp-stop", "image-stop", "multi-clear", "monitor-stop", "detach"]) {
    assert.ok(deps.events.includes(event), `missing teardown event ${event}`);
  }
});

test("document resume restores current intent once and reconciles the current video", async (t) => {
  const previous = globalThis.__documentLifecycleDeps;
  t.after(() => { globalThis.__documentLifecycleDeps = previous; });
  const deps = { video: { id: "B" }, events: [] };
  const lifecycle = await loadLifecycle(deps);
  await lifecycle.suspendDocument();
  deps.events.length = 0;

  assert.deepEqual(
    await lifecycle.resumeDocument(),
    { ok: true, suspended: false, active: true },
  );
  assert.deepEqual(
    await lifecycle.resumeDocument(),
    { ok: true, suspended: false, changed: false },
  );
  assert.equal(deps.events.filter((event) => event === "image-start").length, 1);
  assert.equal(deps.events.filter((event) => Array.isArray(event) && event[0] === "reconcile").length, 1);
  assert.ok(deps.events.includes("monitor-update"));
  assert.equal(lifecycle.state().primaryController.video, deps.video);
});
