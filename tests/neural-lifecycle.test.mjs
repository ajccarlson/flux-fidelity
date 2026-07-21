import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

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
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
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
