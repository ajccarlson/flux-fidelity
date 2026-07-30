import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { CONTRACT_IMPORT } from "./helpers/setting-contract-import.mjs";

const mainUrl = new URL("../src/core/fsrcnnx-main.js", import.meta.url);
let revision = 0;

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function section(source, startMarker, endMarker) {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `missing start marker: ${startMarker}`);
  assert.notEqual(end, -1, `missing end marker: ${endMarker}`);
  return source.slice(start, end);
}

function count(events, expected) {
  return events.filter((event) => event === expected).length;
}

function candidate(id = "secondary") {
  return {
    id,
    videoWidth: 640,
    videoHeight: 360,
    getBoundingClientRect: () => ({ width: 640, height: 360 }),
  };
}

async function loadMultiTargetLifecycle(deps) {
  const source = await readFile(mainUrl, "utf8");
  const targetLifecycle = section(
    source,
    "const multiTargetRetirements = new Set();",
    "// Build per-target model instances",
  );
  const reconciliation = section(
    source,
    "const MAX_SECONDARY_TARGETS",
    "// Render one secondary video",
  );
  const harness = `
    const deps = globalThis.__multiTargetLifecycleDeps;
    const device = deps.device || {
      id: "device",
      queue: { onSubmittedWorkDone: async () => deps.events.push("queue:fence") },
    };
    const format = "bgra8unorm";
    const SRGB_COLOR_SPACE = "srgb";
    let mode = "upscale", optAllVideos = true, pageSuspended = false, adopting = false;
    let gpuResourcePhase = "active";
    let video = deps.primaryVideo || { id: "primary" };
    let videoMonitor = { request() {} };
    const warn = (...args) => deps.warnings.push(args);
    const probeVideo = () => "ok";
    const renderMultiOne = () => {};
    const videoPageVisible = () => true;
    const positionVideoCanvas = () => true;
    const videoPresentationState = () => ({
      pictureInPicture: false,
      directFullscreen: false,
      fullscreenElsewhere: false,
      nativeRequired: false,
    });
    const applyOverlayReveal = () => {};
    const hoverRegionFor = () => null;
    const invalidateVideoColorSupport = () => {};
    const findAllVideos = () => deps.candidates || [];
    const boundedRuntimeDetail = (error, fallback = "cleanup failed") =>
      String(error?.message || error || fallback).replace(/\\s+/g, " ").trim().slice(0, 240);

    class SsimDownscaler {
      constructor(owner) {
        deps.events.push("ssim:create");
        if (deps.ssimConstructorError) throw deps.ssimConstructorError;
      }
      destroy() {
        deps.events.push("ssim:destroy");
        if (deps.ssimDestroyError) throw deps.ssimDestroyError;
      }
    }

    class VideoController {
      constructor(targetVideo, callbacks) {
        this.video = targetVideo;
        this.callbacks = callbacks;
        deps.events.push("controller:construct:" + targetVideo.id);
        if (deps.controllerConstructorError) throw deps.controllerConstructorError;
      }
      start() {
        deps.events.push("controller:start:" + this.video.id);
        if (deps.startError) throw deps.startError;
      }
      scheduleFrame() {
        deps.events.push("controller:schedule:" + this.video.id);
        if (deps.scheduleError) throw deps.scheduleError;
      }
      destroy() {
        deps.events.push("controller:destroy:" + this.video.id);
        if (deps.controllerDestroyError) throw deps.controllerDestroyError;
      }
    }

    const document = {
      createElement() {
        deps.events.push("canvas:create");
        if (deps.createElementError) throw deps.createElementError;
        const context = {
          configure() {
            deps.events.push("context:configure");
            if (deps.configureError) throw deps.configureError;
          },
          unconfigure() {
            deps.events.push("context:unconfigure");
            if (deps.unconfigureError) throw deps.unconfigureError;
          },
        };
        return {
          className: "",
          style: {},
          setAttribute() {
            if (deps.setAttributeError) throw deps.setAttributeError;
          },
          getContext(kind) {
            deps.events.push("context:get:" + kind);
            return deps.contextUnavailable ? null : context;
          },
          remove() {
            deps.events.push("canvas:remove");
            if (deps.canvasRemoveError) throw deps.canvasRemoveError;
          },
        };
      },
    };

    ${targetLifecycle}
    ${reconciliation}

    export function construct(targetVideo) { return new MultiTarget(targetVideo); }
    export function reconcile() { return syncMultiTargets(); }
    export function clear() { return clearMultiTargets(); }
    export function drain() { return drainMultiTargetRetirements(); }
    export function destroy(target) { return target.destroy(); }
    export function register(targetVideo, target) { multiTargets.set(targetVideo, target); }
    export function prime(target, resources) {
      target.models = resources.models || [];
      target.highStages = resources.highStages || [];
      target.artStages = resources.artStages || {};
      target.lumaTexture = resources.lumaTexture || null;
      target.hiRGB = resources.hiRGB || null;
      target.dispRGB = resources.dispRGB || null;
      target.sharpenPipeline = resources.sharpenPipeline || null;
      target.activeModel = resources.activeModel || null;
      target.chainedFsrcnnx = resources.chainedFsrcnnx || null;
      target.chainedHigh = resources.chainedHigh || null;
      target.chainedArt = resources.chainedArt || null;
    }
    export function size() { return multiTargets.size; }
  `;
  globalThis.__multiTargetLifecycleDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(CONTRACT_IMPORT + harness).toString("base64")}#${++revision}`);
}

async function loadIntegratedRetirement(deps) {
  const source = await readFile(mainUrl, "utf8");
  const hardRetirement = section(
    source,
    "function runtimeGpuResourcesRequested()",
    "let webGpuInitPromise = null;",
  );
  const invalidation = section(
    source,
    "function invalidateMainDeviceResources()",
    "function retirePrimaryGpuAllocations(",
  );
  const targetRetirement = section(
    source,
    "const multiTargetRetirements = new Set();",
    "class MultiTarget {",
  );
  const clearTargets = section(
    source,
    "function clearMultiTargets()",
    "// Render one secondary video",
  );
  const harness = `
    const deps = globalThis.__multiTargetLifecycleDeps;
    let pageSuspended = false;
    let device = deps.device, deviceOwnedByMain = true;
    let gpuResourceGeneration = 0, gpuResourcePhase = "active", gpuResourceReason = null;
    let gpuRetirementTail = Promise.resolve(), gpuRetirementPromise = null;
    let gpuAdapterPhase = "ready", gpuDevicePhase = "ready";
    let gpuRecoveryPhase = "idle", gpuRecoveryAttempt = 0;
    let adoptionGeneration = 0, imageUpscalerInitGeneration = 0, adopting = false;
    let webGpuInitPromise = null, adoptionPromise = null, deviceRecoveryPromise = null;
    let deviceLossInvalidationPromise = null;
    let primaryAllocationRetirementPromise = null;
    let modelLoadPromise = null, fsrcnnxStageBuildPromise = null;
    let highStageBuildPromise = null, artStageBuildPromise = null;
    let imageUpscalerInitPromise = null, interpolatorInitPromise = null;
    const deviceRecoveryRequested = () => false;
    const boundedRuntimeDetail = (error, fallback = "cleanup failed") =>
      String(error?.message || error || fallback).replace(/\\s+/g, " ").trim().slice(0, 240);
    const warn = (...args) => deps.events.push(["warn", ...args]);
    const notifyState = () => {};
    const cancelDeviceRecovery = () => {};
    const pauseDeviceProducers = () => {};
    const detachDeviceErrorHandler = () => {};
    const invalidateImageUpscaler = () => null;
    const neuralEng = null;
    const interpolator = null;

    let multiTargets = new Map();
    let models = [], highStages = [], artStages = {};
    let chainTapTex = null, chainTapFrame = 0;
    let lumaTexture = null, lumaW = 0, lumaH = 0;
    let hiRGB = null, hiRGBW = 0, hiRGBH = 0;
    let dispRGB = null, dispRGBW = 0, dispRGBH = 0;
    let ssimds = null, context = null, format = null;
    let extractPipeline = null, recombinePipeline = null, recombine16Pipeline = null, blitPipeline = null;
    let extractPipelineTex = null, recombinePipelineTex = null, recombine16PipelineTex = null;
    let passthroughPipeline = null, sharpenPipeline = null, sharpenStrengthBuilt = null;
    let sampler = null, modelsDevice = null, activeModel = null;
    let chainedFsrcnnx = null, chainedHigh = null, chainedArt = null;
    let fsrcnnxLoadPending = false, highLoadPending = false, artLoadPending = false, _texSource = null;
    const resetScaleSelection = () => {};
    const resetPresentedRuntime = () => {};

    ${targetRetirement}
    ${clearTargets}
    ${invalidation}
    ${hardRetirement}

    export function seedPriorRetirement() {
      return trackMultiTargetRetirement(deps.targetDrain.promise.then(() => {
        deps.events.push("target-physical-destroy");
        return { ok: true, errors: [] };
      }));
    }
    export function retire() { return retireGpuResources("all-features-off"); }
  `;
  globalThis.__multiTargetLifecycleDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(CONTRACT_IMPORT + harness).toString("base64")}#${++revision}`);
}

async function loadTargetModelPreparation(deps) {
  const source = await readFile(mainUrl, "utf8");
  const production = section(
    source,
    "function ensureTargetModels(t)",
    "// Swap a MultiTarget's state into the globals",
  );
  const harness = `
    const deps = globalThis.__multiTargetLifecycleDeps;
    let device = deps.device;
    const lostDevices = new WeakSet();
    let engine = "fsrcnnx-hi", chainDepth = 3, artVariant = "ArtCNN_C4F32";
    const STANDARD_MODEL = "model-standard";
    const HIGH_MODEL = "model-high";
    const srcCache = {
      fsrcnnx: {
        [HIGH_MODEL]: { manifest: { name: HIGH_MODEL }, wgsl: "high-wgsl" },
      },
      artcnn: {},
    };
    const FsrcnnxModel = deps.FsrcnnxModel;
    const ArtCnnModel = deps.ArtCnnModel;
    ${production}
    export { ensureTargetModels };
  `;
  globalThis.__multiTargetLifecycleDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(CONTRACT_IMPORT + harness).toString("base64")}#${++revision}`);
}

test("MultiTarget constructor cleans partial resources at every post-allocation failure point", async (t) => {
  const cases = [
    {
      name: "canvas creation failure",
      options: { createElementError: new Error("canvas creation failed") },
      error: /canvas creation failed/,
      configured: false,
      controllerConstructed: false,
      canvasCreated: false,
      downscalerCreated: false,
    },
    {
      name: "canvas setup failure",
      options: { setAttributeError: new Error("canvas setup failed") },
      error: /canvas setup failed/,
      configured: false,
      controllerConstructed: false,
      canvasCreated: true,
      downscalerCreated: false,
    },
    {
      name: "downscaler construction failure",
      options: { ssimConstructorError: new Error("downscaler failed") },
      error: /downscaler failed/,
      configured: false,
      controllerConstructed: false,
      canvasCreated: true,
      downscalerCreated: true,
      downscalerPublished: false,
    },
    {
      name: "missing canvas context",
      options: { contextUnavailable: true },
      error: /secondary WebGPU canvas context is unavailable/,
      configured: false,
      controllerConstructed: false,
      canvasCreated: true,
      downscalerCreated: true,
      downscalerPublished: true,
    },
    {
      name: "context configure failure",
      options: { configureError: new Error("configure failed") },
      error: /configure failed/,
      configured: true,
      controllerConstructed: false,
      canvasCreated: true,
      downscalerCreated: true,
      downscalerPublished: true,
    },
    {
      name: "controller construction failure",
      options: { controllerConstructorError: new Error("controller failed") },
      error: /controller failed/,
      configured: true,
      controllerConstructed: true,
      canvasCreated: true,
      downscalerCreated: true,
      downscalerPublished: true,
    },
  ];

  for (const scenario of cases) {
    await t.test(scenario.name, async () => {
      const deps = { events: [], warnings: [], candidates: [], ...scenario.options };
      const lifecycle = await loadMultiTargetLifecycle(deps);

      assert.throws(() => lifecycle.construct(candidate()), scenario.error);
      await lifecycle.drain();
      assert.equal(count(deps.events, "ssim:create"), scenario.downscalerCreated ? 1 : 0);
      assert.equal(count(deps.events, "ssim:destroy"), scenario.downscalerPublished ? 1 : 0,
        "an already-published downscaler is destroyed exactly once");
      assert.equal(count(deps.events, "canvas:remove"), scenario.canvasCreated ? 1 : 0,
        "an allocated partial overlay is removed exactly once");
      assert.equal(count(deps.events, "context:unconfigure"), scenario.configured ? 1 : 0,
        "an acquired canvas context is unconfigured during rollback");
      assert.equal(
        deps.events.includes("controller:construct:secondary"),
        scenario.controllerConstructed,
      );
      assert.equal(lifecycle.size(), 0, "a failed constructor never publishes a target");
    });
  }
});

test("MultiTarget startup rollback removes registration and destroys resources", async (t) => {
  const cases = [
    { name: "controller start failure", options: { startError: new Error("start failed") } },
    { name: "initial schedule failure", options: { scheduleError: new Error("schedule failed") } },
  ];

  for (const scenario of cases) {
    await t.test(scenario.name, async () => {
      const secondary = candidate();
      const deps = {
        events: [],
        warnings: [],
        candidates: [secondary],
        ...scenario.options,
      };
      const lifecycle = await loadMultiTargetLifecycle(deps);

      lifecycle.reconcile();
      await lifecycle.drain();

      assert.equal(lifecycle.size(), 0, "startup failure removes the published map entry");
      assert.equal(count(deps.events, "controller:destroy:secondary"), 1);
      assert.equal(count(deps.events, "ssim:destroy"), 1);
      assert.equal(count(deps.events, "context:unconfigure"), 1);
      assert.equal(count(deps.events, "canvas:remove"), 1);
      assert.equal(deps.warnings.length, 1);
      assert.match(deps.warnings[0].join(" "), /secondary target initialization failed/);
    });
  }
});

test("secondary admission keeps its target-count bound without a source-pixel ceiling", async () => {
  const large = candidate("large");
  large.videoWidth = 3840;
  large.videoHeight = 2160;
  large.getBoundingClientRect = () => ({ width: 1920, height: 1080 });
  const small = candidate("small");
  const deps = { events: [], warnings: [], candidates: [small, large] };
  const lifecycle = await loadMultiTargetLifecycle(deps);

  lifecycle.reconcile();

  assert.equal(lifecycle.size(), 2);
  assert.equal(count(deps.events, "controller:construct:large"), 1,
    "a large secondary source must not be excluded by an aggregate pixel quota");
  assert.equal(count(deps.events, "controller:construct:small"), 1);
  await lifecycle.clear();
});

test("secondary High targets receive an independent depth-matched model pool", async (t) => {
  const previous = globalThis.__multiTargetLifecycleDeps;
  t.after(() => { globalThis.__multiTargetLifecycleDeps = previous; });
  const device = { id: "shared-device" };
  const created = [];
  class FakeModel {
    constructor(owner, manifest, wgsl, options) {
      Object.assign(this, { owner, manifest, wgsl, options });
      created.push(this);
    }
  }
  const prep = await loadTargetModelPreparation({
    device,
    FsrcnnxModel: FakeModel,
    ArtCnnModel: FakeModel,
  });
  const target = { device, models: [], highStages: [], artStages: {} };

  assert.equal(prep.ensureTargetModels(target), true);
  assert.equal(target.highStages.length, 3);
  assert.ok(target.highStages.every((stage) => stage.owner === device));
  assert.ok(target.highStages.every((stage) => stage.options.expectedName === "model-high"));
  assert.ok(target.highStages.every((stage) => stage.options.maxWorkingSetBytes === undefined));
  assert.equal(target.models.length, 0, "High targets do not reuse the standard pool");

  assert.equal(prep.ensureTargetModels(target), true);
  assert.equal(created.length, 3, "a prepared target reuses its own High stages");
});

test("clearMultiTargets isolates throwing destructors and removes every map entry", async () => {
  const deps = { events: [], warnings: [], candidates: [] };
  const lifecycle = await loadMultiTargetLifecycle(deps);
  const firstVideo = candidate("first");
  const secondVideo = candidate("second");

  lifecycle.register(firstVideo, {
    destroy() {
      deps.events.push("target:destroy:first");
      throw new Error("first cleanup failed");
    },
  });
  lifecycle.register(secondVideo, {
    destroy() { deps.events.push("target:destroy:second"); },
  });

  const drain = lifecycle.clear();

  assert.equal(lifecycle.size(), 0, "every target is removed synchronously");
  const result = await drain;

  assert.deepEqual(
    deps.events.filter((event) => event.startsWith("target:destroy:")),
    ["target:destroy:first", "target:destroy:second"],
    "one cleanup exception cannot prevent later targets from being destroyed",
  );
  assert.equal(result.ok, false);
  assert.equal(deps.warnings.length, 1);
  assert.match(deps.warnings[0].join(" "), /secondary target cleanup failed: first cleanup failed/);
});

test("MultiTarget destruction unpublishes immediately and retires physical resources after its queue fence", async () => {
  const fence = deferred();
  const events = [];
  const device = {
    queue: {
      onSubmittedWorkDone() {
        events.push("queue:fence");
        return fence.promise;
      },
    },
  };
  const deps = {
    device,
    events,
    warnings: [],
    candidates: [],
    controllerDestroyError: new Error("controller cleanup failed"),
  };
  const lifecycle = await loadMultiTargetLifecycle(deps);
  const video = candidate();
  const target = lifecycle.construct(video);
  const resource = (name, error = null) => ({
    destroy() {
      events.push(`${name}:destroy`);
      if (error) throw error;
    },
  });
  lifecycle.prime(target, {
    models: [resource("model-a", new Error("model cleanup failed")), resource("model-b")],
    highStages: [resource("high-model")],
    artStages: { current: [resource("art-model")] },
    lumaTexture: resource("luma", new Error("texture cleanup failed")),
    hiRGB: resource("hi"),
    dispRGB: resource("display"),
    sharpenPipeline: { id: "pipeline" },
  });
  lifecycle.register(video, target);

  const first = lifecycle.destroy(target);
  const second = lifecycle.destroy(target);

  assert.equal(first, second, "destroy is idempotent and exposes one drain barrier");
  assert.equal(lifecycle.size(), 0, "destroy closes map publication synchronously");
  assert.equal(count(events, "controller:destroy:secondary"), 1);
  assert.equal(count(events, "canvas:remove"), 1, "the overlay detaches synchronously");
  assert.equal(events.some((event) => event.endsWith?.(":destroy") &&
    !event.startsWith("controller:")), false,
  "device-backed objects remain alive until submitted work is complete");

  fence.resolve();
  const result = await first;

  assert.equal(result.ok, false, "isolated cleanup failures remain observable to an awaiting owner");
  for (const expected of [
    "model-a:destroy",
    "model-b:destroy",
    "high-model:destroy",
    "art-model:destroy",
    "luma:destroy",
    "hi:destroy",
    "display:destroy",
    "ssim:destroy",
    "context:unconfigure",
  ]) assert.equal(count(events, expected), 1, `${expected} should run exactly once`);
  assert.equal(result.errors.length, 3);
  assert.equal(deps.warnings.length, 3);
});

test("clearMultiTargets includes retirements that were unpublished immediately before the clear", async () => {
  const fence = deferred();
  const events = [];
  const device = {
    queue: { onSubmittedWorkDone: () => fence.promise },
  };
  const deps = { device, events, warnings: [], candidates: [] };
  const lifecycle = await loadMultiTargetLifecycle(deps);
  const firstVideo = candidate("first");
  const first = lifecycle.construct(firstVideo);
  lifecycle.register(firstVideo, first);

  // This target is no longer in the map, but its device resources are still
  // retained behind the fence and must be part of the next global drain.
  lifecycle.destroy(first);
  assert.equal(lifecycle.size(), 0);

  const secondVideo = candidate("second");
  lifecycle.register(secondVideo, {
    destroy() {
      events.push("target:destroy:second");
      return Promise.reject(new Error("async target cleanup failed"));
    },
  });
  const allDrained = lifecycle.clear();
  assert.equal(lifecycle.size(), 0, "clear removes all publication before awaiting cleanup");

  let settled = false;
  allDrained.then(() => { settled = true; });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(settled, false, "the earlier unpublished target remains in the global drain set");

  fence.resolve();
  const result = await allDrained;
  assert.equal(result.ok, false);
  assert.equal(events.includes("ssim:destroy"), true);
  assert.equal(events.includes("context:unconfigure"), true);
  assert.equal(deps.warnings.some((warning) => warning.join(" ").includes("async target cleanup failed")), true);
});

test("hard retirement awaits a previously unpublished target drain before destroying its owned device", async () => {
  const targetDrain = deferred();
  const events = [];
  const device = {
    queue: { onSubmittedWorkDone: async () => events.push("main-queue-fence") },
    destroy: () => events.push("device-destroy"),
  };
  const runtime = await loadIntegratedRetirement({ device, targetDrain, events });

  // Model a target that was synchronously removed just before the global hard
  // retirement began but is still retaining physical WebGPU resources.
  runtime.seedPriorRetirement();
  const retirement = runtime.retire();
  await new Promise((resolve) => setImmediate(resolve));

  assert.equal(events.includes("main-queue-fence"), true);
  assert.equal(events.includes("device-destroy"), false,
    "the owned device must remain alive while a target retirement is pending");

  targetDrain.resolve();
  assert.equal((await retirement).ok, true);
  assert.ok(events.indexOf("target-physical-destroy") < events.indexOf("device-destroy"));
});
