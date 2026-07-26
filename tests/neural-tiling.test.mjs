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
    "const { createOrtSession, ensureOrt, getOrtSessionDevice } = globalThis.__neuralTileTestDeps;\n" +
      "const resolvePackagedAssetUrl = (path) => chrome.runtime.getURL(path);",
  )
    .replace(
      JSON.stringify("./fsrcnnx-neural-temporal-tiling.js"),
      JSON.stringify(
        new URL(
          "../src/core/fsrcnnx-neural-temporal-tiling.js",
          import.meta.url,
        ).href,
      ),
    )
    .replace(
      JSON.stringify("./fsrcnnx-neural-temporal-atlas.js"),
      JSON.stringify(
        new URL(
          "../src/core/fsrcnnx-neural-temporal-atlas.js",
          import.meta.url,
        ).href,
      ),
    );
  assert.notEqual(source, original, "neural tile test dependency injection must match");
  globalThis.__neuralTileTestDeps = deps;
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}#tile-${++moduleRevision}`);
}

function installGlobals(t, entry) {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    GPUTextureUsage: globalThis.GPUTextureUsage,
    deps: globalThis.__neuralTileTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.GPUTextureUsage = previous.GPUTextureUsage;
    globalThis.__neuralTileTestDeps = previous.deps;
  });
  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [entry] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };
  globalThis.GPUTextureUsage = { STORAGE_BINDING: 1, TEXTURE_BINDING: 2 };
}

async function createHarness(t, {
  width,
  height,
  scale = 2,
  entry = {},
  runHook,
} = {}) {
  const manifestEntry = {
    key: "tiled",
    file: "tiled.onnx",
    scale,
    input: "input",
    output: "output",
    ...entry,
  };
  installGlobals(t, manifestEntry);

  const writes = [];
  const submissions = [];
  const buffers = [];
  const textures = [];
  const inputTensors = [];
  const outputTensors = [];
  const runCalls = [];
  let releases = 0;

  const device = {
    limits: {
      maxTextureDimension2D: 8192,
      maxBufferSize: 256 * 1024 * 1024,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    lost: new Promise(() => {}),
    queue: {
      onSubmittedWorkDone: async () => {},
      writeBuffer(buffer, _offset, data) {
        writes.push({ label: buffer.label, data: Array.from(data) });
      },
      submit(commands) {
        submissions.push(commands);
      },
    },
    createShaderModule: ({ code }) => ({ code }),
    createComputePipeline: ({ compute }) => ({
      code: compute.module.code,
      getBindGroupLayout: () => ({}),
    }),
    createSampler: () => ({}),
    createBuffer(options) {
      const buffer = {
        ...options,
        destroyed: false,
        destroy() { this.destroyed = true; },
      };
      buffers.push(buffer);
      return buffer;
    },
    createTexture(options) {
      const texture = {
        ...options,
        destroyed: false,
        createView: () => ({ texture: options.label }),
        destroy() { this.destroyed = true; },
      };
      textures.push(texture);
      return texture;
    },
    createBindGroup: ({ entries }) => ({ entries }),
    createCommandEncoder() {
      const passes = [];
      return {
        beginComputePass() {
          const pass = { pipeline: null, bindGroup: null, dispatch: null };
          passes.push(pass);
          return {
            setPipeline(pipeline) { pass.pipeline = pipeline; },
            setBindGroup(_index, bindGroup) { pass.bindGroup = bindGroup; },
            dispatchWorkgroups(x, y) { pass.dispatch = [x, y]; },
            end() {},
          };
        },
        finish: () => ({ passes }),
      };
    },
  };

  const makeOutput = (dims) => {
    const outDims = [1, 3, dims[2] * scale, dims[3] * scale];
    const size = outDims.reduce((product, value) => product * value, 1);
    const tensor = {
      type: "float32",
      dims: outDims,
      size,
      gpuBuffer: { size: size * 4 },
      disposed: false,
      dispose() { this.disposed = true; },
    };
    outputTensors.push(tensor);
    return tensor;
  };
  const session = {
    device,
    inputNames: ["input"],
    outputNames: ["output"],
    inputMetadata: [
      { name: "input", isTensor: true, type: "float32", shape: [1, 3, "height", "width"] },
    ],
    outputMetadata: [
      { name: "output", isTensor: true, type: "float32", shape: [1, 3, "out_height", "out_width"] },
    ],
    async run(feeds, fetches) {
      const call = {
        index: runCalls.length,
        dims: [...feeds.input.options.dims],
        buffer: feeds.input.buffer,
        fetches,
      };
      runCalls.push(call);
      if (runHook) {
        return runHook({ ...call, makeOutput: () => makeOutput(call.dims) });
      }
      return { output: makeOutput(call.dims) };
    },
    async release() { releases++; },
  };
  const ort = {
    Tensor: {
      fromGpuBuffer(buffer, options) {
        const tensor = {
          buffer,
          options,
          disposed: false,
          dispose() { this.disposed = true; },
        };
        inputTensors.push(tensor);
        return tensor;
      },
    },
  };
  const deps = {
    ensureOrt: async () => ort,
    getOrtSessionDevice: (candidate) => candidate.device,
    createOrtSession: async () => session,
  };
  const module = await loadNeuralEngine(deps);
  const engine = module.createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("tiled");

  let sourceViewCalls = 0;
  const source = {
    tex: {
      createView() {
        sourceViewCalls++;
        return { source: `${width}x${height}` };
      },
    },
  };
  return {
    ...module,
    engine,
    source,
    device,
    writes,
    submissions,
    buffers,
    textures,
    inputTensors,
    outputTensors,
    runCalls,
    sourceViewCalls: () => sourceViewCalls,
    releases: () => releases,
  };
}

test("tile planning clips boundary halos and maps scaled crops without gaps", async (t) => {
  const deps = {
    createOrtSession: async () => null,
    ensureOrt: async () => ({}),
    getOrtSessionDevice: () => null,
  };
  const { planNeuralTiles } = await loadNeuralEngine(deps);

  const one = planNeuralTiles(503, 271, {
    tileSize: 512,
    tileOverlap: 24,
    padMultiple: 8,
    scale: 2,
  });
  assert.deepEqual(one, [{
    coreX: 0, coreY: 0, coreW: 503, coreH: 271,
    inputX: 0, inputY: 0, inputW: 503, inputH: 271,
    padW: 504, padH: 272,
    cropX: 0, cropY: 0, dstX: 0, dstY: 0,
    outW: 1006, outH: 542,
  }]);

  const tiles = planNeuralTiles(1920, 1080, {
    tileSize: 512,
    tileOverlap: 24,
    padMultiple: 1,
    scale: 2,
  });
  assert.equal(tiles.length, 12, "a >720p frame is tiled, not rejected");
  assert.deepEqual(tiles[0], {
    coreX: 0, coreY: 0, coreW: 512, coreH: 512,
    inputX: 0, inputY: 0, inputW: 536, inputH: 536,
    padW: 536, padH: 536,
    cropX: 0, cropY: 0, dstX: 0, dstY: 0,
    outW: 1024, outH: 1024,
  });
  assert.deepEqual(tiles[5], {
    coreX: 512, coreY: 512, coreW: 512, coreH: 512,
    inputX: 488, inputY: 488, inputW: 560, inputH: 560,
    padW: 560, padH: 560,
    cropX: 48, cropY: 48, dstX: 1024, dstY: 1024,
    outW: 1024, outH: 1024,
  });
  assert.deepEqual(tiles.at(-1), {
    coreX: 1536, coreY: 1024, coreW: 384, coreH: 56,
    inputX: 1512, inputY: 1000, inputW: 408, inputH: 80,
    padW: 408, padH: 80,
    cropX: 48, cropY: 48, dstX: 3072, dstY: 2048,
    outW: 768, outH: 112,
  });
  for (const tile of tiles) {
    assert.equal(tile.dstX, tile.coreX * 2);
    assert.equal(tile.dstY, tile.coreY * 2);
    assert.equal(tile.cropX, (tile.coreX - tile.inputX) * 2);
    assert.equal(tile.cropY, (tile.coreY - tile.inputY) * 2);
  }
  t.after(() => { delete globalThis.__neuralTileTestDeps; });
});

test("manifest tile controls enforce a bounded, receptive-field-safe input", async (t) => {
  const deps = {
    createOrtSession: async () => null,
    ensureOrt: async () => ({}),
    getOrtSessionDevice: () => null,
  };
  const { validateNeuralManifest } = await loadNeuralEngine(deps);
  const base = { key: "model", file: "model.onnx", scale: 2 };
  assert.equal(validateNeuralManifest([{ ...base, tileSize: 512, tileOverlap: 24 }]).length, 1);
  for (const entry of [
    { ...base, tileSize: 63 },
    { ...base, tileSize: 769 },
    { ...base, tileOverlap: 17 },
    { ...base, tileOverlap: 193 },
    { ...base, tileSize: 768, tileOverlap: 65 },
  ]) {
    assert.throws(() => validateNeuralManifest([entry]), /invalid tile/);
  }
  t.after(() => { delete globalThis.__neuralTileTestDeps; });
});

test("run snapshots synchronously and keeps the one-tile path ABI exact", async (t) => {
  const harness = await createHarness(t, { width: 320, height: 180 });
  const pending = harness.engine.run(harness.source, 320, 180);
  assert.equal(harness.sourceViewCalls(), 1, "the expiring source is captured before run first yields");
  assert.equal(harness.submissions.length, 1);
  assert.equal(harness.submissions[0][0].passes.length, 2, "snapshot and first pack share one submission");

  const rendered = await pending;
  assert.deepEqual(
    { outW: rendered.outW, outH: rendered.outH },
    { outW: 640, outH: 360 },
  );
  assert.deepEqual(harness.runCalls.map((call) => call.dims), [[1, 3, 180, 320]]);
  assert.deepEqual(
    harness.writes.filter((write) => write.label === "neural-packU").map((write) => write.data),
    [[320, 180, 320, 180, 0, 0, 320, 180]],
  );
  assert.deepEqual(
    harness.writes.filter((write) => write.label === "neural-compU").map((write) => write.data),
    [[640, 230400, 0, 0, 0, 0, 640, 360]],
  );
  assert.deepEqual(harness.engine.stats(), {
    last: harness.engine.stats().last,
    mu: harness.engine.stats().mu,
    n: 1,
    skip: 0,
    fails: 0,
    temporalResetRuns: 0,
    temporalRecurrentRuns: 0,
    lastTiles: 1,
    tileRuns: 1,
    maxTileW: 320,
    maxTileH: 180,
  });
  await harness.engine.dispose();
  assert.equal(harness.releases(), 1);
});

test("multi-tile run uses bounded ONNX buffers and exact pack/crop offsets", async (t) => {
  const harness = await createHarness(t, { width: 1920, height: 1080 });
  const tiles = harness.planNeuralTiles(1920, 1080, { scale: 2 });
  const rendered = await harness.engine.run(harness.source, 1920, 1080);
  assert.deepEqual(
    { outW: rendered.outW, outH: rendered.outH },
    { outW: 3840, outH: 2160 },
  );
  assert.equal(harness.runCalls.length, 12);
  assert.deepEqual(
    harness.runCalls.map((call) => call.dims),
    tiles.map((tile) => [1, 3, tile.padH, tile.padW]),
  );

  const inputBuffers = harness.buffers.filter((buffer) => buffer.label.startsWith("neural-in-"));
  assert.equal(inputBuffers.length, 1);
  assert.ok(inputBuffers[0].size <= 560 * 560 * 3 * 4);
  assert.ok(inputBuffers[0].size < 1920 * 1080 * 3 * 4, "the ONNX input is tile-sized");
  for (const tensor of harness.outputTensors) {
    assert.ok(tensor.gpuBuffer.size <= 560 * 560 * 2 * 2 * 3 * 4);
  }

  const packWrites = harness.writes.filter((write) => write.label === "neural-packU");
  assert.deepEqual(
    packWrites.map((write) => write.data),
    tiles.map((tile) => [
      tile.padW, tile.padH, tile.inputW, tile.inputH,
      tile.inputX, tile.inputY, 1920, 1080,
    ]),
  );
  const compWrites = harness.writes.filter((write) => write.label === "neural-compU");
  assert.deepEqual(
    compWrites.map((write) => write.data.slice(2)),
    tiles.map((tile) => [
      tile.cropX, tile.cropY, tile.dstX, tile.dstY, tile.outW, tile.outH,
    ]),
  );
  assert.deepEqual(
    {
      lastTiles: harness.engine.stats().lastTiles,
      tileRuns: harness.engine.stats().tileRuns,
      maxTileW: harness.engine.stats().maxTileW,
      maxTileH: harness.engine.stats().maxTileH,
    },
    { lastTiles: 12, tileRuns: 12, maxTileW: 560, maxTileH: 560 },
  );
  await harness.engine.quiesce();
  assert.ok(harness.inputTensors.every((tensor) => tensor.disposed));
  assert.ok(harness.outputTensors.every((tensor) => tensor.disposed));
  await harness.engine.dispose();
});

test("stop cancels a tiled run after the active tile and retires its tensors", async (t) => {
  const gate = deferred();
  let runEntered;
  const entered = new Promise((resolve) => { runEntered = resolve; });
  const harness = await createHarness(t, {
    width: 130,
    height: 64,
    entry: { tileSize: 64, tileOverlap: 18 },
    runHook: async ({ makeOutput }) => {
      runEntered();
      await gate.promise;
      return { output: makeOutput() };
    },
  });
  const running = harness.engine.run(harness.source, 130, 64);
  await entered;
  const stopping = harness.engine.stop();
  gate.resolve();
  await assert.rejects(running, /neural inference cancelled by stop/);
  await stopping;
  assert.equal(harness.runCalls.length, 1, "no later tile starts after cancellation");
  await harness.engine.quiesce();
  assert.ok(harness.inputTensors.every((tensor) => tensor.disposed));
  assert.ok(harness.outputTensors.every((tensor) => tensor.disposed));
  assert.equal(harness.engine.stats().fails, 1);
  await harness.engine.dispose();
});

test("logical cancel invalidates a tiled run without concurrent GPU destruction", async (t) => {
  const gate = deferred();
  let runEntered;
  const entered = new Promise((resolve) => { runEntered = resolve; });
  const harness = await createHarness(t, {
    width: 130,
    height: 64,
    entry: { tileSize: 64, tileOverlap: 18 },
    runHook: async ({ makeOutput }) => {
      runEntered();
      await gate.promise;
      return { output: makeOutput() };
    },
  });

  const running = harness.engine.run(harness.source, 130, 64);
  await entered;
  harness.engine.cancel();
  assert.ok(harness.buffers.length > 0);
  assert.ok(harness.textures.length > 0);
  assert.ok(harness.buffers.every(({ destroyed }) => destroyed === false));
  assert.ok(harness.textures.every(({ destroyed }) => destroyed === false));

  gate.resolve();
  await assert.rejects(running, /neural inference cancelled by stop/);
  assert.equal(harness.runCalls.length, 1);
  assert.ok(
    harness.buffers.every(({ destroyed }) => destroyed === false),
    "logical cancellation leaves physical cleanup to the serialized lifecycle command",
  );
  assert.ok(harness.textures.every(({ destroyed }) => destroyed === false));

  await harness.engine.stop();
  assert.ok(harness.buffers.every(({ destroyed }) => destroyed === true));
  assert.ok(harness.textures.every(({ destroyed }) => destroyed === true));
  assert.equal(harness.releases(), 0, "stop retains the session device guard");
  await harness.engine.dispose();
  assert.equal(harness.releases(), 1);
});

test("a later tile error stops the plan and preserves completed-tile stats", async (t) => {
  const harness = await createHarness(t, {
    width: 130,
    height: 64,
    entry: { tileSize: 64, tileOverlap: 18 },
    runHook: async ({ index, makeOutput }) => {
      if (index === 1) throw new Error("tile execution failed");
      return { output: makeOutput() };
    },
  });
  await assert.rejects(
    harness.engine.run(harness.source, 130, 64),
    /tile execution failed/,
  );
  assert.equal(harness.runCalls.length, 2);
  assert.deepEqual(
    {
      lastTiles: harness.engine.stats().lastTiles,
      tileRuns: harness.engine.stats().tileRuns,
      fails: harness.engine.stats().fails,
    },
    { lastTiles: 3, tileRuns: 1, fails: 1 },
  );
  await harness.engine.quiesce();
  assert.ok(harness.inputTensors.every((tensor) => tensor.disposed));
  assert.ok(harness.outputTensors.every((tensor) => tensor.disposed));
  await harness.engine.dispose();
});

test("Neural has no policy ceiling and validates only its extension-frame device limits", async () => {
  const source = await readFile(
    new URL("../src/core/fsrcnnx-main.js", import.meta.url),
    "utf8",
  );
  const frameRuntime = await readFile(
    new URL("../src/frame/neural-frame-runtime.js", import.meta.url),
    "utf8",
  );
  const planStart = source.indexOf("function neuralPresentationPlan(");
  const presentStart = source.indexOf("function neuralFramePresentation(");
  const presentEnd = source.indexOf("function renderNeuralFrame()", presentStart);
  const present = source.slice(presentStart, presentEnd);
  const renderEnd = source.indexOf("function ensureSharpenPipeline", presentEnd);
  const render = source.slice(presentEnd, renderEnd);
  const positionStart = source.indexOf("function positionVideoCanvas(");
  const positionEnd = source.indexOf("function runtimeGpuResourcesRequested()", positionStart);
  const position = source.slice(positionStart, positionEnd);

  assert.ok(planStart >= 0 && presentStart >= 0 && presentEnd > presentStart && renderEnd > presentEnd);
  const plan = source.slice(planStart, source.indexOf("// ---- validated setting contracts", planStart));
  assert.match(plan, /const modelWidth = srcW \* modelScale/);
  assert.match(plan, /policy === "force2" \? 2/);
  assert.match(present, /neuralPresentationPlan\(/);
  assert.doesNotMatch(present, /textureSizeAllowed|MAX_(?:INPUT|PROCESSING)|pixel ceiling/i);
  assert.match(
    position,
    /textureSizeAllowed\(outW, outH, "canvas output"\)/,
  );
  assert.doesNotMatch(position, /enforceProcessingBudget/);
  assert.match(
    render,
    /runEngine\.run\(runVideo, srcW, srcH, presentation, temporal\)/,
  );
  assert.match(render, /\{ validateDimensions: false, resize: false \}/);
  assert.doesNotMatch(render, /textureSizeAllowed|storageBufferSizeAllowed/);
  assert.match(frameRuntime, /validateDeviceDimensions\(srcW, srcH, "source", runDevice\)/);
  assert.match(
    frameRuntime,
    /validateDeviceDimensions\(modelWidth, modelHeight, "neural output", runDevice\)/,
  );
  assert.match(
    frameRuntime,
    /validateDeviceDimensions\(width, height, "presentation", ownerDevice\)/,
  );
  assert.doesNotMatch(source, /maxInputPixels|MAX_PROCESSING_PIXELS|MAX_SECONDARY_SOURCE_PIXELS/);
});
