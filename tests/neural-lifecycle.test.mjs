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
  const url = new URL("../fsrcnnx-neural.js", import.meta.url);
  const original = await readFile(url, "utf8");
  const dependencyImport = "import { createOrtSession, ensureOrt, getOrtSessionDevice } " +
    `from ${JSON.stringify("./fsrcnnx-rife.js")};`;
  const source = original.replace(
    dependencyImport,
    "const { createOrtSession, ensureOrt, getOrtSessionDevice } = globalThis.__neuralTestDeps;",
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

test("neural manifests reject ambiguous keys and unsafe model paths", async (t) => {
  const previous = globalThis.__neuralTestDeps;
  t.after(() => { globalThis.__neuralTestDeps = previous; });
  const deps = {
    createOrtSession: async () => null,
    ensureOrt: async () => ({}),
    getOrtSessionDevice: () => null,
  };
  const { validateNeuralManifest } = await loadNeuralEngine(deps);
  const valid = validateNeuralManifest({ models: [
    { key: "span-2x", file: "span.fp16.onnx", label: "SPAN", scale: 2, padMultiple: 8, input: "input", output: "output" },
  ] });
  assert.equal(valid.length, 1);
  assert.equal(Object.isFrozen(valid[0]), true);

  const invalid = [
    [{ key: "", file: "model.onnx", scale: 2 }],
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
