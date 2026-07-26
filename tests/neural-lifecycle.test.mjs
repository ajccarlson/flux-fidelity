import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let moduleRevision = 0;

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

async function loadNeuralEngine(deps) {
  const url = new URL("../src/core/fsrcnnx-neural.js", import.meta.url);
  const original = await readFile(url, "utf8");
  const dependencyImport = [
    "import {",
    "  createOrtSession,",
    "  ensureOrt,",
    "  getOrtSessionDevice,",
    "  resolvePackagedAssetUrl,",
    `} from ${JSON.stringify("./fsrcnnx-rife.js")};`,
  ].join("\n");
  const source = original.replace(
    dependencyImport,
    "const { createOrtSession, ensureOrt, getOrtSessionDevice } = globalThis.__neuralTestDeps;\n" +
      "const resolvePackagedAssetUrl = (path) => chrome.runtime.getURL(path);",
  );
  assert.notEqual(source, original, "neural test dependency injection must match the source import");
  globalThis.__neuralTestDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}#${++moduleRevision}`);
}

function fakeDevice() {
  const resource = () => ({ destroy() {} });
  return {
    queue: { onSubmittedWorkDone: async () => {}, writeBuffer() {}, submit() {} },
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createSampler: resource,
    createBuffer: resource,
    createTexture: resource,
  };
}

function fakeInferenceDevice() {
  const device = {
    limits: {
      maxTextureDimension2D: 8192,
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    lost: new Promise(() => {}),
    queue: { onSubmittedWorkDone: async () => {}, writeBuffer() {}, submit() {} },
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createSampler: () => ({}),
    createBuffer: ({ size }) => ({ size, destroy() {} }),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
      }),
      finish: () => ({}),
    }),
  };
  return device;
}

test("neural manifests reject ambiguous keys and unsafe model paths", async (t) => {
  const previous = globalThis.__neuralTestDeps;
  t.after(() => { globalThis.__neuralTestDeps = previous; });
  const deps = {
    createOrtSession: async () => null,
    ensureOrt: async () => ({}),
    getOrtSessionDevice: () => null,
  };
  const { validateNeuralManifest } = await loadNeuralEngine(deps);
  assert.deepEqual(validateNeuralManifest([]), []);
  assert.deepEqual(validateNeuralManifest({ models: [] }), []);
  const valid = validateNeuralManifest({ models: [
    {
      key: "span-2x", file: "span.fp16.onnx", label: "SPAN", scale: 2,
      padMultiple: 8, input: "input", output: "output",
    },
  ] });
  assert.equal(valid.length, 1);
  assert.equal(Object.isFrozen(valid[0]), true);
  assert.equal(validateNeuralManifest([
    { key: "a".repeat(64), file: "max-key.onnx", scale: 2 },
  ])[0].key.length, 64);

  const invalid = [
    [{ key: "", file: "model.onnx", scale: 2 }],
    [{ key: "a".repeat(65), file: "model.onnx", scale: 2 }],
    [{ key: "model", file: "../model.onnx", scale: 2 }],
    [{ key: "model", file: "folder/model.onnx", scale: 2 }],
    [{ key: "model", file: "%2e%2e-model.onnx", scale: 2 }],
    [{ key: "model", file: "model.onnx", scale: Infinity }],
    [{ key: "model", file: "model.onnx", scale: 2, input: "" }],
    [
      { key: "model", file: "a.onnx", scale: 2 },
      { key: "model", file: "b.onnx", scale: 2 },
    ],
    [
      { key: "a", file: "same.onnx", scale: 2 },
      { key: "b", file: "same.onnx", scale: 2 },
    ],
  ];
  for (const manifest of invalid) {
    assert.throws(() => validateNeuralManifest(manifest), /neural manifest entry|manifest must/);
  }
});

test("stop-cancelled neural init rejects instead of returning the persistent active session", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [
      { key: "first", file: "first.onnx", label: "First", scale: 2 },
      { key: "second", file: "second.onnx", label: "Second", scale: 2 },
    ] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const device = fakeDevice();
  const secondStarted = deferred();
  const secondSession = deferred();
  let calls = 0;
  let cancelledReleaseCount = 0;
  const makeSession = (release) => ({
    device,
    inputNames: ["input"],
    outputNames: ["output"],
    release,
  });
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async () => {
      calls++;
      if (calls === 1) return makeSession(async () => {});
      secondStarted.resolve();
      return secondSession.promise;
    },
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });

  assert.equal((await engine.init("first")).key, "first");
  const pending = engine.init("second");
  await secondStarted.promise;
  engine.stop();
  secondSession.resolve(makeSession(async () => { cancelledReleaseCount++; }));

  await assert.rejects(pending, /neural initialization cancelled/);
  assert.equal(engine.activeEntry().key, "first");
  assert.equal(cancelledReleaseCount, 1);
});

test("neural device loss invalidates the persistent session before reinitialization", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [{ key: "model", file: "model.onnx", scale: 2 }] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const lost = deferred();
  const deviceA = { ...fakeDevice(), lost: lost.promise };
  const deviceB = { ...fakeDevice(), lost: new Promise(() => {}) };
  let creates = 0;
  let releasesA = 0;
  const sessions = [
    { device: deviceA, inputNames: ["input"], outputNames: ["output"], async release() { releasesA++; } },
    { device: deviceB, inputNames: ["input"], outputNames: ["output"], async release() {} },
  ];
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async () => sessions[creates++],
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });

  await engine.init("model");
  assert.equal(engine.ready(), true);
  assert.equal(engine.device(), deviceA);
  lost.resolve({ message: "adapter reset" });
  for (let i = 0; i < 20 && engine.ready(); i++) await new Promise((resolve) => setImmediate(resolve));
  assert.equal(engine.ready(), false, "lost session must stop being reusable immediately");
  for (let i = 0; i < 20 && releasesA === 0; i++) await new Promise((resolve) => setImmediate(resolve));
  assert.equal(releasesA, 1);

  await engine.init("model");
  assert.equal(creates, 2);
  assert.equal(engine.ready(), true);
  assert.equal(engine.device(), deviceB);
});

test("neural reinitialization waits for invalidation-owned session release", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [{ key: "model", file: "model.onnx", scale: 2 }] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const releaseStarted = deferred();
  const releaseGate = deferred();
  const deviceA = { ...fakeDevice(), lost: new Promise(() => {}) };
  const deviceB = { ...fakeDevice(), lost: new Promise(() => {}) };
  let creates = 0;
  let releaseInProgress = false;
  let createdDuringRelease = false;
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async () => {
      creates++;
      if (creates === 2) createdDuringRelease = releaseInProgress;
      if (creates === 1) {
        return {
          device: deviceA,
          inputNames: ["input"],
          outputNames: ["output"],
          async release() {
            releaseInProgress = true;
            releaseStarted.resolve();
            await releaseGate.promise;
            releaseInProgress = false;
          },
        };
      }
      return {
        device: deviceB,
        inputNames: ["input"],
        outputNames: ["output"],
        async release() {},
      };
    },
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("model");

  const invalidation = engine.invalidateDevice(deviceA);
  assert.equal(engine.ready(), false, "current loss must unpublish readiness synchronously");
  assert.equal(engine.device(), null, "current loss must unpublish its device synchronously");
  await releaseStarted.promise;
  const recovery = engine.init("model");
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(creates, 1, "recovery must not create a session while old release() is pending");

  releaseGate.resolve();
  assert.equal(await invalidation, true);
  assert.equal((await recovery).key, "model");
  assert.equal(creates, 2);
  assert.equal(createdDuringRelease, false);
  assert.equal(engine.ready(), true);
  assert.equal(engine.device(), deviceB);
});

test("natural neural loss releases a non-current cross-device guard without invalidating the replacement", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [
      { key: "first", file: "first.onnx", scale: 2 },
      { key: "second", file: "second.onnx", scale: 2 },
    ] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const guardReleaseStarted = deferred();
  const guardReleaseGate = deferred();
  const guardReleaseFinished = deferred();
  const oldDeviceLoss = deferred();
  const deviceA = { ...fakeDevice(), lost: oldDeviceLoss.promise };
  const deviceB = { ...fakeDevice(), lost: new Promise(() => {}) };
  let creates = 0;
  let releasesA = 0;
  let releasesB = 0;
  const sessions = [
    {
      device: deviceA,
      inputNames: ["input"],
      outputNames: ["output"],
      async release() {
        releasesA++;
        guardReleaseStarted.resolve();
        await guardReleaseGate.promise;
        guardReleaseFinished.resolve();
      },
    },
    {
      device: deviceB,
      inputNames: ["input"],
      outputNames: ["output"],
      async release() { releasesB++; },
    },
  ];
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async () => sessions[creates++],
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("first");
  await engine.init("second");
  assert.equal(engine.device(), deviceB);
  assert.equal(releasesA, 0, "the prior device remains guarded until adoption or loss");

  oldDeviceLoss.resolve({ message: "old adopted device lost" });
  await guardReleaseStarted.promise;
  assert.equal(engine.ready(), true, "old-guard loss must preserve the healthy replacement");
  assert.equal(engine.device(), deviceB);

  guardReleaseGate.resolve();
  await guardReleaseFinished.promise;
  assert.equal(releasesA, 1);
  assert.equal(engine.device(), deviceB);
  await engine.dispose();
  assert.equal(releasesB, 1);
});

test("neural disposal waits for invalidation-owned session release", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [{ key: "model", file: "model.onnx", scale: 2 }] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const releaseStarted = deferred();
  const releaseGate = deferred();
  const device = { ...fakeDevice(), lost: new Promise(() => {}) };
  let releases = 0;
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async () => ({
      device,
      inputNames: ["input"],
      outputNames: ["output"],
      async release() {
        releases++;
        releaseStarted.resolve();
        await releaseGate.promise;
      },
    }),
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("model");

  const invalidation = engine.invalidateDevice(device);
  await releaseStarted.promise;
  const disposal = engine.dispose();
  assert.equal(engine.dispose(), disposal, "dispose remains single-flight behind invalidation");
  assert.equal(engine.ready(), false);
  assert.equal(engine.device(), null);
  let disposed = false;
  disposal.then(() => { disposed = true; });
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(disposed, false, "dispose must not resolve before invalidation release() finishes");

  releaseGate.resolve();
  assert.equal(await invalidation, true);
  await disposal;
  assert.equal(disposed, true);
  assert.equal(releases, 1, "the invalidated session is released exactly once");
});

test("neural disposal during a published replacement handoff releases both sessions", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [
      { key: "first", file: "first.onnx", scale: 2 },
      { key: "second", file: "second.onnx", scale: 2 },
    ] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const deviceA = { ...fakeDevice(), lost: new Promise(() => {}) };
  const deviceB = { ...fakeDevice(), lost: new Promise(() => {}) };
  const replacementSession = deferred();
  let creates = 0;
  let releasesA = 0;
  let releasesB = 0;
  const sessionA = {
    device: deviceA,
    inputNames: ["input"],
    outputNames: ["output"],
    async release() { releasesA++; },
  };
  const sessionB = {
    device: deviceB,
    inputNames: ["input"],
    outputNames: ["output"],
    async release() { releasesB++; },
  };
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async () => {
      creates++;
      return creates === 1 ? sessionA : replacementSession.promise;
    },
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("first");

  const replacement = engine.init("second");
  let disposal = null;
  const disposeAtPublication = new Promise((resolve, reject) => {
    let attempts = 0;
    const poll = () => {
      if (engine.device() === deviceB) {
        disposal = engine.dispose();
        resolve();
        return;
      }
      if (++attempts > 100) {
        reject(new Error("replacement session was not observed at its post-publication yield"));
        return;
      }
      queueMicrotask(poll);
    };
    queueMicrotask(poll);
  });
  replacementSession.resolve(sessionB);
  await disposeAtPublication;

  assert.equal(engine.ready(), false, "dispose must revoke the published replacement synchronously");
  await assert.rejects(replacement, /device was lost during neural session initialization/);
  await disposal;
  assert.equal(releasesA, 1, "the prior-device guard must not remain trapped in the losing initializer");
  assert.equal(releasesB, 1, "the published replacement is released exactly once");
});

test("neural init rejects a session whose device is already lost", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [{ key: "model", file: "model.onnx", scale: 2 }] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const lostDevice = {
    ...fakeDevice(),
    lost: Promise.resolve({ reason: "unknown", message: "already gone" }),
  };
  let releases = 0;
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async () => ({
      device: lostDevice,
      inputNames: ["input"],
      outputNames: ["output"],
      async release() { releases++; },
    }),
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });

  await assert.rejects(engine.init("model"), /device was lost during neural session initialization/);
  assert.equal(engine.ready(), false);
  assert.equal(engine.device(), null);
  assert.equal(releases, 1);
});

test("neural disposal publishes idle state, fences resources, and serializes reinitialization", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [{ key: "model", file: "model.onnx", scale: 2 }] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const fence = deferred();
  const device = fakeDevice();
  device.lost = new Promise(() => {});
  device.queue.onSubmittedWorkDone = () => fence.promise;
  let creates = 0;
  let releases = 0;
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async () => {
      creates++;
      return {
        device,
        inputNames: ["input"],
        outputNames: ["output"],
        async release() { releases++; },
      };
    },
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("model");

  const retirement = engine.dispose();
  assert.equal(engine.ready(), false, "logical ownership is released synchronously");
  assert.equal(engine.device(), null);
  const nextInit = engine.init("model");
  await Promise.resolve();
  assert.equal(creates, 1, "a new session cannot overtake physical retirement");
  assert.equal(releases, 0);

  fence.resolve();
  await retirement;
  assert.equal(releases, 1);
  assert.equal((await nextInit).key, "model");
  assert.equal(creates, 2);
  assert.equal(engine.ready(), true);
});

test("neural loss failures survive duplicate, unrelated, and settled recovery barriers", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [{ key: "model", file: "model.onnx", scale: 2 }] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const loss = deferred();
  const releaseStarted = deferred();
  const releaseGate = deferred();
  const device = { ...fakeDevice(), lost: loss.promise };
  const unrelatedDevice = { ...fakeDevice(), lost: new Promise(() => {}) };
  const physicalFailure = new Error("ORT release failed");
  let creates = 0;
  let releases = 0;
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (candidate) => candidate.device,
    createOrtSession: async () => {
      creates++;
      return {
        device,
        inputNames: ["input"],
        outputNames: ["output"],
        async release() {
          releases++;
          releaseStarted.resolve();
          await releaseGate.promise;
          throw physicalFailure;
        },
      };
    },
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("model");

  // The provider watcher claims the device first, matching production ordering.
  loss.resolve({ message: "natural loss" });
  await releaseStarted.promise;
  const duplicate = engine.invalidateDevice(device);
  assert.strictEqual(engine.invalidateDevice(device), duplicate,
    "all observers of one terminal device loss must share the claiming promise");
  const unrelated = engine.invalidateDevice(unrelatedDevice);
  const recovery = engine.init("model");
  const settled = Promise.allSettled([duplicate, unrelated, recovery]);

  releaseGate.resolve();
  const results = await settled;
  assert.equal(results[0].status, "rejected");
  assert.ok(results[0].reason instanceof AggregateError);
  assert.equal(results[1].status, "rejected",
    "a later device barrier must carry earlier physical-cleanup failures");
  assert.ok(results[1].reason instanceof AggregateError);
  assert.equal(results[2].status, "rejected",
    "the all-active recovery barrier must retain the earlier failure");
  assert.ok(results[2].reason instanceof AggregateError);

  await assert.rejects(engine.init("model"),
    (error) => error instanceof AggregateError && /invalidation barrier/.test(error.message),
    "a settled cleanup failure must remain latched until disposal reports it");
  const disposal = engine.dispose();
  assert.strictEqual(engine.dispose(), disposal);
  await assert.rejects(disposal,
    (error) => error instanceof AggregateError && /session disposal/.test(error.message));
  assert.equal(creates, 1, "failed physical cleanup must block automatic recovery");
  assert.equal(releases, 1, "the claiming invalidation owns release exactly once");
  assert.equal(engine.ready(), false);
});

test("a failed deferred neural device guard is reported by run and later disposal", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    GPUTextureUsage: globalThis.GPUTextureUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.GPUTextureUsage = previous.GPUTextureUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [
      { key: "first", file: "first.onnx", scale: 2 },
      { key: "second", file: "second.onnx", scale: 2 },
    ] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };
  globalThis.GPUTextureUsage = { STORAGE_BINDING: 1, TEXTURE_BINDING: 2 };

  const makeRunDevice = () => {
    const owner = { ...fakeDevice(), lost: new Promise(() => {}) };
    owner.createTexture = () => ({ createView: () => ({}), destroy() {} });
    owner.createBindGroup = () => ({});
    owner.createCommandEncoder = () => ({
      beginComputePass: () => ({
        setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
      }),
      finish: () => ({}),
    });
    return owner;
  };
  const firstDevice = makeRunDevice();
  const secondDevice = makeRunDevice();
  let releasesFirst = 0;
  let releasesSecond = 0;
  const sessions = [
    {
      device: firstDevice,
      inputNames: ["input"], outputNames: ["output"],
      async release() { releasesFirst++; throw new Error("old guard release failed"); },
    },
    {
      device: secondDevice,
      inputNames: ["input"], outputNames: ["output"],
      async release() { releasesSecond++; },
    },
  ];
  let creates = 0;
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (candidate) => candidate.device,
    createOrtSession: async () => sessions[creates++],
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("first");
  await engine.init("second");

  await assert.rejects(
    engine.run({ tex: { createView: () => ({}) } }, 1, 1),
    (error) => error instanceof AggregateError && /deferred session release/.test(error.message),
  );
  await assert.rejects(engine.dispose(),
    (error) => error instanceof AggregateError && /session disposal/.test(error.message));
  assert.equal(releasesFirst, 1);
  assert.equal(releasesSecond, 1);
});

test("neural init validates graph names and FP32 dynamic RGB NCHW metadata", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };
  const dynamic = [1, 3, "height", "width"];
  const cases = [
    {
      label: "ambiguous output",
      entry: { key: "model", file: "model.onnx", scale: 2 },
      session: { inputNames: ["input"], outputNames: ["image", "debug"] },
      expected: /manifest must name the output tensor/,
    },
    {
      label: "unknown explicit input",
      entry: { key: "model", file: "model.onnx", scale: 2, input: "pixels" },
      session: { inputNames: ["input"], outputNames: ["output"] },
      expected: /manifest input 'pixels' is not exposed/,
    },
    {
      label: "non-tensor input",
      entry: { key: "model", file: "model.onnx", scale: 2 },
      session: {
        inputNames: ["input"], outputNames: ["output"],
        inputMetadata: [{ name: "input", isTensor: false }],
      },
      expected: /input 'input' is not a tensor/,
    },
    {
      label: "FP16 output",
      entry: { key: "model", file: "model.onnx", scale: 2 },
      session: {
        inputNames: ["input"], outputNames: ["output"],
        inputMetadata: [{ name: "input", isTensor: true, type: "float32", shape: dynamic }],
        outputMetadata: [{ name: "output", isTensor: true, type: "float16", shape: dynamic }],
      },
      expected: /output 'output' has dtype 'float16'/,
    },
    {
      label: "fixed spatial input",
      entry: { key: "model", file: "model.onnx", scale: 2 },
      session: {
        inputNames: ["input"], outputNames: ["output"],
        inputMetadata: [{ name: "input", isTensor: true, type: "float32", shape: [1, 3, 64, 64] }],
      },
      expected: /input 'input' spatial dimensions must be dynamic/,
    },
    {
      label: "non-RGB output",
      entry: { key: "model", file: "model.onnx", scale: 2 },
      session: {
        inputNames: ["input"], outputNames: ["output"],
        inputMetadata: [{ name: "input", isTensor: true, type: "float32", shape: dynamic }],
        outputMetadata: [{ name: "output", isTensor: true, type: "float32", shape: [1, 1, "height", "width"] }],
      },
      expected: /output 'output' channel dimension must be RGB/,
    },
  ];

  for (const scenario of cases) {
    let releases = 0;
    const session = {
      device: fakeDevice(),
      async release() { releases++; },
      ...scenario.session,
    };
    globalThis.fetch = async () => ({
      ok: true,
      json: async () => ({ models: [scenario.entry] }),
    });
    const deps = {
      ensureOrt: async () => ({}),
      getOrtSessionDevice: (candidate) => candidate.device,
      createOrtSession: async () => session,
    };
    const { createNeuralEngine } = await loadNeuralEngine(deps);
    const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
    await assert.rejects(engine.init("model"), scenario.expected, scenario.label);
    assert.equal(releases, 1, `${scenario.label}: rejected session must be released`);
    assert.equal(engine.ready(), false, scenario.label);
  }
});

test("neural execution defaults to FP32 and only attempts FP16 for a manifest opt-in", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [
      { key: "default", file: "default.fp16.onnx", scale: 2 },
      { key: "opt-in", file: "opt-in.onnx", scale: 2, fp16: true },
    ] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };

  const device = { ...fakeDevice(), lost: new Promise(() => {}) };
  const executionModes = [];
  let releases = 0;
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async (_url, _options, { enableFp16 }) => {
      executionModes.push(enableFp16);
      if (enableFp16) throw new Error("shader-f16 unavailable");
      return {
        device,
        inputNames: ["input"],
        outputNames: ["output"],
        async release() { releases++; },
      };
    },
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const warnings = [];
  const engine = createNeuralEngine({ log: () => {}, warn: (message) => warnings.push(message) });

  await engine.init("default");
  await engine.init("opt-in");
  assert.deepEqual(executionModes, [false, true, false]);
  assert.match(warnings.join("\n"), /retrying FP32/);
  await engine.dispose();
  assert.equal(releases, 2);
});

test("neural run selects the declared output and retires every returned tensor", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    GPUTextureUsage: globalThis.GPUTextureUsage,
    deps: globalThis.__neuralTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.GPUTextureUsage = previous.GPUTextureUsage;
    globalThis.__neuralTestDeps = previous.deps;
  });

  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [{
      key: "model",
      file: "model.onnx",
      scale: 2,
      output: "image",
    }] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };
  globalThis.GPUTextureUsage = { STORAGE_BINDING: 1, TEXTURE_BINDING: 2 };

  const device = fakeInferenceDevice();
  const disposals = { input: 0, selected: 0, unselected: 0, errorOutput: 0 };
  let runFetches = null;
  let runFeeds = null;
  let runCalls = 0;
  const selected = {
    type: "float32",
    dims: [1, 3, 4, 6],
    size: 72,
    gpuBuffer: { size: 72 * 4 },
    dispose() { disposals.selected++; },
  };
  const unselected = {
    type: "float32",
    dims: [1],
    gpuBuffer: { size: 4 },
    dispose() { disposals.unselected++; },
  };
  const errorOutput = {
    type: "float32",
    dims: [1],
    gpuBuffer: { size: 4 },
    dispose() { disposals.errorOutput++; },
  };
  let returned = { image: selected, debug: unselected };
  const session = {
    device,
    inputNames: ["input"],
    outputNames: ["image", "debug"],
    inputMetadata: [
      { name: "input", isTensor: true, type: "float32", shape: [1, 3, "height", "width"] },
    ],
    outputMetadata: [
      { name: "image", isTensor: true, type: "float32", shape: [1, 3, "out_height", "out_width"] },
      { name: "debug", isTensor: true, type: "float32", shape: [1] },
    ],
    async run(feeds, fetches) {
      runCalls++;
      runFeeds = feeds;
      runFetches = fetches;
      return returned;
    },
    async release() {},
  };
  const deps = {
    ensureOrt: async () => ({
      Tensor: {
        fromGpuBuffer: (_buffer, options) => ({
          options,
          dispose() { disposals.input++; },
        }),
      },
    }),
    getOrtSessionDevice: (candidate) => candidate.device,
    createOrtSession: async () => session,
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("model");

  const rendered = await engine.run({ tex: { createView: () => ({}) } }, 3, 2);
  assert.deepEqual({ outW: rendered.outW, outH: rendered.outH }, { outW: 6, outH: 4 });
  assert.deepEqual(runFetches, ["image"]);
  assert.deepEqual(runFeeds.input.options, {
    dataType: "float32",
    dims: [1, 3, 2, 3],
  });
  returned = { debug: errorOutput };
  await assert.rejects(
    engine.run({ tex: { createView: () => ({}) } }, 3, 2),
    /output not on GPU \(image\)/,
  );
  assert.equal(runCalls, 2);
  await engine.quiesce();
  assert.deepEqual(disposals, { input: 2, selected: 1, unselected: 1, errorOutput: 1 });
  await engine.dispose();
});
