import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const MIB = 1024 * 1024;
let moduleRevision = 0;

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
    "const { createOrtSession, ensureOrt, getOrtSessionDevice } = " +
      "globalThis.__neuralTemporalAtlasRuntimeDeps;\n" +
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
  assert.notEqual(
    source,
    original,
    "temporal atlas runtime dependency injection must match",
  );
  globalThis.__neuralTemporalAtlasRuntimeDeps = deps;
  return import(
    `data:text/javascript;base64,${Buffer.from(source).toString("base64")}` +
      `#temporal-atlas-${++moduleRevision}`
  );
}

function cdaEntry({
  stateDtype = "float16",
  key,
  filePrefix = "cda-vsr",
} = {}) {
  const mixed = stateDtype === "float16";
  const stateOutput = (name, state) => ({
    [name]: {
      role: "state-out",
      state,
      dtype: stateDtype,
      channels: 64,
    },
  });
  return {
    key: key ?? (mixed ? "cda-vsr-4x-mixed" : "cda-vsr-4x-fp32"),
    label: mixed ? "CDA-VSR 4x mixed" : "CDA-VSR 4x FP32",
    scale: 4,
    contract: {
      version: 2,
      mode: "temporal",
      resetGraph: "initialize",
      recurrentGraph: "recurrent",
      tiling: {
        kind: "temporal-state-atlas-v1",
        scale: 4,
        halo: 64,
        haloDerivation: {
          motionSearchRadius: 8,
          fixedRecurrentRadius: 35,
          minimum: 64,
          alignment: 8,
        },
        largestLogicalBytesPerSourcePixel: mixed ? 512 : 776,
        preferredInputExtent: 512,
        inputAlignment: 8,
        workgroupSize: 8,
        stateAtlas: {
          stateCount: 2,
          channelsPerState: 64,
          arrayLayersPerState: 16,
          textureFormat: mixed ? "rgba16float" : "rgba32float",
        },
      },
      graphs: {
        initialize: {
          file: `${filePrefix}-initializer.onnx`,
          inputs: {
            frame: {
              role: "rgb",
              dtype: "float32",
              channels: 3,
            },
          },
          outputs: {
            output: {
              role: "rgb",
              dtype: "float32",
              channels: 3,
            },
            ...stateOutput("next_state_low", "low"),
            ...stateOutput("next_state_high", "high"),
          },
        },
        recurrent: {
          file: `${filePrefix}-recurrent.onnx`,
          inputs: {
            frame: {
              role: "rgb",
              dtype: "float32",
              channels: 3,
            },
            motion: {
              role: "motion",
              dtype: "float32",
              channels: 2,
              provider: "decoded-cda-v1",
            },
            residual: {
              role: "residual",
              dtype: "float32",
              channels: 1,
              provider: "decoded-cda-v1",
            },
            state_low: {
              role: "state-in",
              state: "low",
              reset: "required",
              dtype: stateDtype,
              channels: 64,
            },
            state_high: {
              role: "state-in",
              state: "high",
              reset: "required",
              dtype: stateDtype,
              channels: 64,
            },
          },
          outputs: {
            output: {
              role: "rgb",
              dtype: "float32",
              channels: 3,
            },
            ...stateOutput("next_state_low", "low"),
            ...stateOutput("next_state_high", "high"),
          },
        },
      },
    },
  };
}

function installGlobals(t, entries) {
  const models = Array.isArray(entries) ? entries : [entries];
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    GPUTextureUsage: globalThis.GPUTextureUsage,
    GPUShaderStage: globalThis.GPUShaderStage,
    deps: globalThis.__neuralTemporalAtlasRuntimeDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.GPUTextureUsage = previous.GPUTextureUsage;
    globalThis.GPUShaderStage = previous.GPUShaderStage;
    globalThis.__neuralTemporalAtlasRuntimeDeps = previous.deps;
  });
  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models }),
  });
  globalThis.GPUBufferUsage = {
    UNIFORM: 1,
    COPY_DST: 2,
    STORAGE: 4,
  };
  globalThis.GPUTextureUsage = {
    STORAGE_BINDING: 1,
    TEXTURE_BINDING: 2,
  };
  globalThis.GPUShaderStage = { COMPUTE: 4 };
}

function fakeDevice({ features = ["shader-f16"] } = {}) {
  const buffers = [];
  const textures = [];
  const bindGroups = [];
  const bindGroupLayouts = [];
  const pipelineLayouts = [];
  const submissions = [];
  const errorScopes = [];
  const errorScopeResults = new Map();
  const errorScopePopDelays = [];
  const events = [];
  let queueFence = Promise.resolve();
  const device = {
    features: new Set(features),
    limits: {
      maxTextureDimension2D: 8192,
      maxTextureArrayLayers: 256,
      maxBufferSize: 256 * MIB,
      maxStorageBufferBindingSize: 128 * MIB,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
    lost: new Promise(() => {}),
    queue: {
      onSubmittedWorkDone() {
        return queueFence;
      },
      writeBuffer() {},
      submit(commands) {
        submissions.push(commands);
        events.push({
          type: "submit",
          submissionCount: submissions.length,
        });
      },
    },
    pushErrorScope(filter) {
      errorScopes.push(filter);
      events.push({
        type: "scope-push",
        filter,
        submissionCount: submissions.length,
      });
    },
    async popErrorScope() {
      assert.ok(
        errorScopes.length > 0,
        "the fake device cannot pop an unopened error scope",
      );
      const filter = errorScopes.pop();
      const delay = errorScopePopDelays.shift();
      if (delay) {
        delay.entered();
        await delay.promise;
      }
      const queuedResults = errorScopeResults.get(filter);
      const result = queuedResults?.length
        ? queuedResults.shift()
        : null;
      events.push({
        type: "scope-pop",
        filter,
        result,
        submissionCount: submissions.length,
      });
      return result;
    },
    createShaderModule: ({ code }) => ({ code }),
    createBindGroupLayout(options) {
      bindGroupLayouts.push(options);
      return options;
    },
    createPipelineLayout(options) {
      pipelineLayouts.push(options);
      return options;
    },
    createComputePipeline: ({ compute, layout }) => ({
      code: compute.module.code,
      layout,
      getBindGroupLayout: (index) =>
        layout === "auto" ? {} : layout.bindGroupLayouts[index],
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
        views: [],
        destroyed: false,
        createView(viewOptions = {}) {
          const view = { texture, options: viewOptions };
          this.views.push(view);
          return view;
        },
        destroy() { this.destroyed = true; },
      };
      textures.push(texture);
      return texture;
    },
    createBindGroup(options) {
      const bindGroup = { ...options };
      bindGroups.push(bindGroup);
      return bindGroup;
    },
    createCommandEncoder() {
      const passes = [];
      return {
        beginComputePass() {
          const pass = {
            pipeline: null,
            bindGroup: null,
            dispatch: null,
          };
          passes.push(pass);
          return {
            setPipeline(pipeline) { pass.pipeline = pipeline; },
            setBindGroup(_index, bindGroup) {
              pass.bindGroup = bindGroup;
            },
            dispatchWorkgroups(x, y, z = 1) {
              pass.dispatch = [x, y, z];
            },
            end() {},
          };
        },
        finish: () => ({ passes }),
      };
    },
  };
  return {
    device,
    buffers,
    textures,
    bindGroups,
    bindGroupLayouts,
    pipelineLayouts,
    submissions,
    events,
    openErrorScopeCount() {
      return errorScopes.length;
    },
    holdQueueFence(promise) {
      queueFence = promise;
    },
    delayNextErrorScopePop() {
      let entered;
      let release;
      const enteredPromise = new Promise((resolve) => { entered = resolve; });
      const promise = new Promise((resolve) => { release = resolve; });
      errorScopePopDelays.push({ entered, promise });
      return { entered: enteredPromise, release };
    },
    queueErrorScopeResult(filter, result) {
      const queuedResults = errorScopeResults.get(filter) ?? [];
      queuedResults.push(result);
      errorScopeResults.set(filter, queuedResults);
    },
    queueValidationResult(result) {
      this.queueErrorScopeResult("validation", result);
    },
    queueOutOfMemoryResult(result) {
      this.queueErrorScopeResult("out-of-memory", result);
    },
  };
}

function dynamicMetadata(name, type, channels, output = false) {
  return {
    name,
    isTensor: true,
    type,
    shape: [
      1,
      channels,
      output ? "output_height_x4" : "height",
      output ? "output_width_x4" : "width",
    ],
  };
}

async function createHarness(
  t,
  {
    entry = cdaEntry(),
    entries,
    deviceOptions,
  } = {},
) {
  installGlobals(t, entries ?? entry);
  const gpu = fakeDevice(deviceOptions);
  const stateDtype =
    entry.contract.graphs.recurrent.inputs.state_low.dtype;
  const calls = [];
  const wrappers = [];
  const resultTensors = [];
  const releases = { initialize: 0, recurrent: 0 };
  const graphRuns = { initialize: 0, recurrent: 0 };
  const graphRunDelays = new Map();
  const sessionCreateWaiters = [];
  let sessionCreateCount = 0;
  let failRecurrentCall = null;
  let tensorId = 0;

  const makeResultTensor = (type, dims, label) => {
    const bytesPerElement = type === "float16" ? 2 : 4;
    const size = dims.reduce((product, value) => product * value, 1);
    const tensor = {
      id: ++tensorId,
      label,
      type,
      dims,
      size,
      gpuBuffer: {
        label: `${label}-gpu`,
        size: size * bytesPerElement,
      },
      disposed: false,
      dispose() { this.disposed = true; },
    };
    resultTensors.push(tensor);
    return tensor;
  };

  const makeSession = (graph) => {
    const initialize = graph === "initialize";
    return {
      device: gpu.device,
      inputNames: initialize
        ? ["frame"]
        : ["frame", "motion", "residual", "state_low", "state_high"],
      outputNames: ["output", "next_state_low", "next_state_high"],
      inputMetadata: initialize
        ? [dynamicMetadata("frame", "float32", 3)]
        : [
          dynamicMetadata("frame", "float32", 3),
          dynamicMetadata("motion", "float32", 2),
          dynamicMetadata("residual", "float32", 1),
          dynamicMetadata("state_low", stateDtype, 64),
          dynamicMetadata("state_high", stateDtype, 64),
        ],
      outputMetadata: [
        dynamicMetadata("output", "float32", 3, true),
        dynamicMetadata("next_state_low", stateDtype, 64),
        dynamicMetadata("next_state_high", stateDtype, 64),
      ],
      async run(feeds, fetches) {
        const graphIndex = graphRuns[graph]++;
        const delays = graphRunDelays.get(graph);
        const delay = delays?.shift();
        if (delay) {
          delay.entered();
          await delay.promise;
        }
        const call = {
          graph,
          graphIndex,
          feeds,
          fetches: [...fetches],
          results: null,
        };
        calls.push(call);
        gpu.events.push({
          type: "session-run",
          graph,
          graphIndex,
        });
        if (graph === "recurrent" &&
            failRecurrentCall === graphIndex) {
          failRecurrentCall = null;
          throw new Error("later temporal tile failed");
        }
        const [, , height, width] = feeds.frame.options.dims;
        const results = {
          output: makeResultTensor(
            "float32",
            [1, 3, height * 4, width * 4],
            `${graph}-${graphIndex}-rgb`,
          ),
          next_state_low: makeResultTensor(
            stateDtype,
            [1, 64, height, width],
            `${graph}-${graphIndex}-low`,
          ),
          next_state_high: makeResultTensor(
            stateDtype,
            [1, 64, height, width],
            `${graph}-${graphIndex}-high`,
          ),
        };
        call.results = results;
        return results;
      },
      async release() { releases[graph]++; },
    };
  };

  const ort = {
    Tensor: {
      fromGpuBuffer(buffer, options) {
        const wrapper = {
          buffer,
          options,
          disposed: false,
          dispose() { this.disposed = true; },
        };
        wrappers.push(wrapper);
        return wrapper;
      },
    },
  };
  const deps = {
    ensureOrt: async () => ort,
    getOrtSessionDevice: (session) => session.device,
    createOrtSession: async (url) => {
      sessionCreateCount++;
      for (const waiter of [...sessionCreateWaiters]) {
        if (sessionCreateCount >= waiter.expected) {
          sessionCreateWaiters.splice(sessionCreateWaiters.indexOf(waiter), 1);
          waiter.resolve();
        }
      }
      return makeSession(url.includes("initializer") ? "initialize" : "recurrent");
    },
  };
  const module = await loadNeuralEngine(deps);
  const engine = module.createNeuralEngine({
    log: () => {},
    warn: () => {},
  });
  await engine.init(entry.key);

  const source = {
    tex: {
      createView: () => ({ source: true }),
    },
  };
  const auxiliary = (width, height) => {
    const motionBuffer = {
      label: `provided-motion-${width}x${height}`,
      size: 1 * 2 * height * width * 4,
    };
    const residualBuffer = {
      label: `provided-residual-${width}x${height}`,
      size: 1 * 1 * height * width * 4,
    };
    return {
      motion: {
        provider: "decoded-cda-v1",
        gpuBuffer: motionBuffer,
        dataType: "float32",
        dims: [1, 2, height, width],
      },
      residual: {
        provider: "decoded-cda-v1",
        gpuBuffer: residualBuffer,
        dataType: "float32",
        dims: [1, 1, height, width],
      },
    };
  };

  return {
    ...gpu,
    engine,
    source,
    calls,
    wrappers,
    resultTensors,
    releases,
    auxiliary,
    delayNextGraphRun(graph) {
      let entered;
      let release;
      const enteredPromise = new Promise((resolve) => { entered = resolve; });
      const promise = new Promise((resolve) => { release = resolve; });
      const delays = graphRunDelays.get(graph) ?? [];
      delays.push({ entered, promise });
      graphRunDelays.set(graph, delays);
      return { entered: enteredPromise, release };
    },
    waitForSessionCreates(expected) {
      if (sessionCreateCount >= expected) return Promise.resolve();
      return new Promise((resolve) => {
        sessionCreateWaiters.push({ expected, resolve });
      });
    },
    failNextLaterRecurrentTile() {
      failRecurrentCall = graphRuns.recurrent + 1;
    },
  };
}

function atlasTextures(harness, width, height) {
  return harness.textures.filter(({ label }) =>
    label?.startsWith("neural-temporal-") &&
    label.endsWith(`-${width}x${height}`));
}

function assertTileFeedDims(call, width, height) {
  assert.deepEqual(call.feeds.frame.options.dims, [1, 3, height, width]);
  if (call.graph !== "recurrent") return;
  assert.deepEqual(call.feeds.motion.options.dims, [1, 2, height, width]);
  assert.deepEqual(call.feeds.residual.options.dims, [1, 1, height, width]);
  assert.deepEqual(call.feeds.state_low.options.dims, [1, 64, height, width]);
  assert.deepEqual(call.feeds.state_high.options.dims, [1, 64, height, width]);
}

test("mixed CDA tiles recurrent state through a transactional atlas", async (t) => {
  const harness = await createHarness(t);
  const auxiliary = harness.auxiliary(640, 360);
  const expectedTiles = [
    { width: 448, height: 360 },
    { width: 320, height: 360 },
  ];
  assert.equal(harness.device.features.has("shader-f16"), true);

  const initialized = await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  assert.deepEqual(
    { outW: initialized.outW, outH: initialized.outH },
    { outW: 2560, outH: 1440 },
  );
  assert.deepEqual(
    harness.calls.map(({ graph }) => graph),
    ["initialize", "initialize"],
  );
  const initialScopeEvents = harness.events.filter(({ type }) =>
    type.startsWith("scope-"));
  assert.deepEqual(
    initialScopeEvents.map(({ type }) => type),
    [
      "scope-push",
      "scope-push",
      "scope-push",
      "scope-pop",
      "scope-pop",
      "scope-pop",
    ],
  );
  assert.deepEqual(
    initialScopeEvents.map(({ filter }) => filter),
    [
      "out-of-memory",
      "internal",
      "validation",
      "validation",
      "internal",
      "out-of-memory",
    ],
    "WebGPU error scopes are popped in LIFO order",
  );
  assert.ok(initialScopeEvents.slice(3).every(({ result }) => result === null));
  assert.ok(
    initialScopeEvents[3].submissionCount >
      initialScopeEvents[0].submissionCount,
    "the successful scopes are popped after the frame's GPU submissions",
  );
  harness.calls.forEach((call, index) => {
    assertTileFeedDims(
      call,
      expectedTiles[index].width,
      expectedTiles[index].height,
    );
    assert.deepEqual(call.fetches, [
      "output",
      "next_state_low",
      "next_state_high",
    ]);
  });

  const initialStateBuffers = new Set(
    harness.calls.flatMap(({ results }) => [
      results.next_state_low.gpuBuffer,
      results.next_state_high.gpuBuffer,
    ]),
  );
  const firstAtlas = atlasTextures(harness, 640, 360);
  assert.deepEqual(
    firstAtlas.map(({ label }) => label).sort(),
    [
      "neural-temporal-high-bank0-640x360",
      "neural-temporal-high-bank1-640x360",
      "neural-temporal-low-bank0-640x360",
      "neural-temporal-low-bank1-640x360",
    ],
  );
  for (const texture of firstAtlas) {
    assert.equal(texture.format, "rgba16float");
    assert.deepEqual(texture.size, {
      width: 640,
      height: 360,
      depthOrArrayLayers: 16,
    });
    assert.equal(texture.views[0].options.dimension, "2d-array");
    assert.equal(texture.views[0].options.arrayLayerCount, 16);
  }
  const initialStateBindGroups = harness.bindGroups.filter(({ entries }) =>
    entries.some(({ resource }) =>
      resource?.texture?.label?.includes("neural-temporal-")));
  assert.ok(initialStateBindGroups.length > 0);
  assert.ok(
    initialStateBindGroups.every(({ entries }) =>
      entries.some(({ binding, resource }) =>
        binding === 1 && resource?.texture?.label?.includes("-bank1-"))),
    "the initializer writes every state tile into candidate bank 1",
  );

  const recurrentBindGroupStart = harness.bindGroups.length;
  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  const recurrentCalls = harness.calls.slice(2, 4);
  const recurrentStateBindGroups =
    harness.bindGroups.slice(recurrentBindGroupStart).filter(({ entries }) =>
      entries.some(({ resource }) =>
        resource?.texture?.label?.includes("neural-temporal-")));
  const recurrentPackGroups = recurrentStateBindGroups.filter(({ entries }) =>
    entries.some(({ binding, resource }) =>
      binding === 0 && resource?.texture?.label?.includes("-bank1-")));
  const recurrentUnpackGroups = recurrentStateBindGroups.filter(({ entries }) =>
    entries.some(({ binding, resource }) =>
      binding === 1 && resource?.texture?.label?.includes("-bank0-")));
  assert.equal(recurrentPackGroups.length, 4);
  assert.equal(recurrentUnpackGroups.length, 4);
  assert.equal(
    recurrentStateBindGroups.length,
    recurrentPackGroups.length + recurrentUnpackGroups.length,
    "the recurrent frame reads committed bank 1 and writes candidate bank 0",
  );
  assert.deepEqual(
    recurrentCalls.map(({ graph }) => graph),
    ["recurrent", "recurrent"],
  );
  const firstSuccessfulScopeEnd = harness.events.indexOf(
    initialScopeEvents.at(-1),
  );
  const firstRecurrentRun = harness.events.findIndex(
    ({ type, graph }) => type === "session-run" && graph === "recurrent",
  );
  assert.ok(
    firstSuccessfulScopeEnd >= 0 &&
      firstSuccessfulScopeEnd < firstRecurrentRun,
    "the successful error scopes pop before the next frame observes " +
      "committed recurrent state",
  );
  recurrentCalls.forEach((call, index) => {
    assertTileFeedDims(
      call,
      expectedTiles[index].width,
      expectedTiles[index].height,
    );
    assert.deepEqual(Object.keys(call.feeds), [
      "frame",
      "motion",
      "residual",
      "state_low",
      "state_high",
    ]);
    assert.equal(
      call.feeds.motion.buffer.label,
      "neural-temporal-aux-motion",
    );
    assert.equal(
      call.feeds.residual.buffer.label,
      "neural-temporal-aux-residual",
    );
    assert.equal(
      call.feeds.state_low.buffer.label,
      "neural-temporal-state-low",
    );
    assert.equal(
      call.feeds.state_high.buffer.label,
      "neural-temporal-state-high",
    );
    assert.notEqual(
      call.feeds.motion.buffer,
      auxiliary.motion.gpuBuffer,
      "the full-frame motion buffer is packed into tile scratch",
    );
    assert.notEqual(
      call.feeds.residual.buffer,
      auxiliary.residual.gpuBuffer,
      "the full-frame residual buffer is packed into tile scratch",
    );
    assert.equal(initialStateBuffers.has(call.feeds.state_low.buffer), false);
    assert.equal(initialStateBuffers.has(call.feeds.state_high.buffer), false);
  });
  assert.equal(
    recurrentCalls[0].feeds.state_low.buffer,
    recurrentCalls[1].feeds.state_low.buffer,
    "state scratch is bounded by the largest tile and reused",
  );
  assert.equal(
    recurrentCalls[0].feeds.motion.buffer,
    recurrentCalls[1].feeds.motion.buffer,
    "auxiliary scratch is bounded by the largest tile and reused",
  );
  assert.equal(
    recurrentCalls[0].feeds.state_low.buffer.size,
    448 * 360 * 64 * 2,
  );
  assert.equal(
    recurrentCalls[0].feeds.state_high.buffer.size,
    448 * 360 * 64 * 2,
  );
  assert.equal(
    recurrentCalls[0].feeds.motion.buffer.size,
    448 * 360 * 2 * 4,
  );
  assert.equal(
    recurrentCalls[0].feeds.residual.buffer.size,
    448 * 360 * 1 * 4,
  );

  const secondRecurrentBindGroupStart = harness.bindGroups.length;
  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  const secondRecurrentStateBindGroups =
    harness.bindGroups.slice(secondRecurrentBindGroupStart).filter(
      ({ entries }) => entries.some(({ resource }) =>
        resource?.texture?.label?.includes("neural-temporal-")),
    );
  assert.equal(
    secondRecurrentStateBindGroups.filter(({ entries }) =>
      entries.some(({ binding, resource }) =>
        binding === 0 && resource?.texture?.label?.includes("-bank0-")),
    ).length,
    4,
  );
  assert.equal(
    secondRecurrentStateBindGroups.filter(({ entries }) =>
      entries.some(({ binding, resource }) =>
        binding === 1 && resource?.texture?.label?.includes("-bank1-")),
    ).length,
    4,
  );
  assert.equal(
    secondRecurrentStateBindGroups.length,
    8,
    "the following recurrent frame reads bank 0 and writes bank 1",
  );

  await harness.engine.run(
    harness.source,
    640,
    360,
    { reset: true, auxiliary },
  );
  assert.deepEqual(
    harness.calls.slice(6, 8).map(({ graph }) => graph),
    ["initialize", "initialize"],
    "an explicit reset bypasses the committed atlas",
  );

  const resizedAuxiliary = harness.auxiliary(320, 180);
  await harness.engine.run(
    harness.source,
    320,
    180,
    { auxiliary: resizedAuxiliary },
  );
  assert.equal(harness.calls.at(-1).graph, "initialize");
  assertTileFeedDims(harness.calls.at(-1), 320, 180);
  const resizedAtlas = atlasTextures(harness, 320, 180);
  assert.equal(resizedAtlas.length, 4);
  assert.deepEqual(
    {
      temporalResetRuns: harness.engine.stats().temporalResetRuns,
      temporalRecurrentRuns: harness.engine.stats().temporalRecurrentRuns,
    },
    { temporalResetRuns: 3, temporalRecurrentRuns: 2 },
  );

  await harness.engine.dispose();
  assert.ok(harness.buffers.every(({ destroyed }) => destroyed));
  assert.ok(harness.textures.every(({ destroyed }) => destroyed));
  assert.ok(harness.wrappers.every(({ disposed }) => disposed));
  assert.ok(harness.resultTensors.every(({ disposed }) => disposed));
  assert.deepEqual(harness.releases, {
    initialize: 1,
    recurrent: 1,
  });
});

test("a failed later tile never commits its candidate atlas bank", async (t) => {
  const harness = await createHarness(t);
  const auxiliary = harness.auxiliary(640, 360);

  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  harness.failNextLaterRecurrentTile();
  await assert.rejects(
    harness.engine.run(
      harness.source,
      640,
      360,
      { auxiliary },
    ),
    /later temporal tile failed/,
  );
  assert.deepEqual(
    harness.calls.map(({ graph }) => graph),
    ["initialize", "initialize", "recurrent", "recurrent"],
  );
  assert.deepEqual(
    {
      successfulFrames: harness.engine.stats().n,
      fails: harness.engine.stats().fails,
      temporalResetRuns: harness.engine.stats().temporalResetRuns,
      temporalRecurrentRuns: harness.engine.stats().temporalRecurrentRuns,
    },
    {
      successfulFrames: 1,
      fails: 1,
      temporalResetRuns: 1,
      temporalRecurrentRuns: 0,
    },
  );

  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  assert.deepEqual(
    harness.calls.slice(4).map(({ graph }) => graph),
    ["initialize", "initialize"],
    "partial candidate writes do not make recurrent state ready",
  );
  assert.deepEqual(
    {
      temporalResetRuns: harness.engine.stats().temporalResetRuns,
      temporalRecurrentRuns: harness.engine.stats().temporalRecurrentRuns,
    },
    { temporalResetRuns: 2, temporalRecurrentRuns: 0 },
  );

  await harness.engine.dispose();
  assert.ok(harness.wrappers.every(({ disposed }) => disposed));
  assert.ok(harness.resultTensors.every(({ disposed }) => disposed));
  assert.ok(atlasTextures(harness, 640, 360).every(({ destroyed }) => destroyed));
  assert.deepEqual(harness.releases, {
    initialize: 1,
    recurrent: 1,
  });
});

test("a submitted frame with a GPU validation error is not committed", async (t) => {
  const harness = await createHarness(t);
  const auxiliary = harness.auxiliary(640, 360);

  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  const initialAtlas = atlasTextures(harness, 640, 360);
  const initialBuffers = [...harness.buffers];
  const submissionsBeforeFailure = harness.submissions.length;
  const validationError = Object.assign(
    new Error("candidate atlas write is invalid"),
    { name: "GPUValidationError" },
  );
  harness.queueValidationResult(validationError);

  await assert.rejects(
    harness.engine.run(
      harness.source,
      640,
      360,
      { auxiliary },
    ),
    (error) => {
      assert.match(
        error.message,
        /neural temporal atlas WebGPU error: validation: candidate atlas write is invalid/,
      );
      assert.equal(error.cause, validationError);
      return true;
    },
  );
  const failedPop = harness.events.find(
    ({ type, filter, result }) =>
      type === "scope-pop" &&
      filter === "validation" &&
      result === validationError,
  );
  assert.ok(failedPop, "the injected GPU validation result was observed");
  assert.ok(
    failedPop.submissionCount > submissionsBeforeFailure,
    "the validation failure is reported after the frame was submitted",
  );
  assert.deepEqual(
    harness.calls.map(({ graph }) => graph),
    ["initialize", "initialize", "recurrent", "recurrent"],
  );

  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  const rebuiltAtlas = atlasTextures(harness, 640, 360)
    .filter((texture) => !initialAtlas.includes(texture));
  assert.equal(rebuiltAtlas.length, 4);
  assert.ok(
    initialAtlas.every(({ destroyed }) => destroyed) &&
      initialBuffers.every(({ destroyed }) => destroyed),
    "a scoped GPU failure fences and rebuilds every cached GPU resource",
  );
  assert.deepEqual(
    harness.calls.slice(4).map(({ graph }) => graph),
    ["initialize", "initialize"],
    "a validation-failed candidate bank cannot select the recurrent graph",
  );
  assert.deepEqual(
    {
      successfulFrames: harness.engine.stats().n,
      fails: harness.engine.stats().fails,
      temporalResetRuns: harness.engine.stats().temporalResetRuns,
      temporalRecurrentRuns: harness.engine.stats().temporalRecurrentRuns,
    },
    {
      successfulFrames: 2,
      fails: 1,
      temporalResetRuns: 2,
      temporalRecurrentRuns: 0,
    },
  );

  await harness.engine.dispose();
});

test("an OOM retry waits for failed GPU resources to cross the queue fence", async (t) => {
  const harness = await createHarness(t);
  const auxiliary = harness.auxiliary(640, 360);

  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  let releaseFence;
  const fence = new Promise((resolve) => { releaseFence = resolve; });
  harness.holdQueueFence(fence);
  harness.queueOutOfMemoryResult(
    Object.assign(new Error("atlas allocation exhausted"), {
      name: "GPUOutOfMemoryError",
    }),
  );

  await assert.rejects(
    harness.engine.run(
      harness.source,
      640,
      360,
      { auxiliary },
    ),
    /out-of-memory: atlas allocation exhausted/,
  );
  const retiredTextureCount = harness.textures.length;
  const retiredBufferCount = harness.buffers.length;
  const retry = harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(
    harness.textures.length,
    retiredTextureCount,
    "the retry cannot allocate a replacement atlas before retirement",
  );
  assert.equal(
    harness.buffers.length,
    retiredBufferCount,
    "the retry cannot allocate replacement scratch before retirement",
  );

  releaseFence();
  await retry;
  assert.equal(
    atlasTextures(harness, 640, 360).length,
    8,
    "the retry rebuilds one atlas only after the failed atlas is destroyed",
  );

  await harness.engine.dispose();
});

test("stop cancels an initialization queued behind OOM recovery", async (t) => {
  const harness = await createHarness(t);
  const auxiliary = harness.auxiliary(640, 360);

  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  let releaseFence;
  const fence = new Promise((resolve) => { releaseFence = resolve; });
  harness.holdQueueFence(fence);
  harness.queueOutOfMemoryResult(new Error("recovery gate"));
  await assert.rejects(
    harness.engine.run(
      harness.source,
      640,
      360,
      { auxiliary },
    ),
    /out-of-memory: recovery gate/,
  );

  const queuedInit = harness.engine.init("cda-vsr-4x-mixed");
  const cancelledInit = assert.rejects(
    queuedInit,
    /neural initialization cancelled/,
  );
  const stopped = harness.engine.stop();
  releaseFence();
  await cancelledInit;
  await stopped;
  assert.equal(
    harness.releases.initialize + harness.releases.recurrent,
    0,
    "ordinary stop keeps the existing graph sessions alive",
  );

  await harness.engine.dispose();
});

test("stop during a delayed error-scope pop prevents atlas commit", async (t) => {
  const harness = await createHarness(t);
  const auxiliary = harness.auxiliary(640, 360);

  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  const delayedPop = harness.delayNextErrorScopePop();
  const scopeEventStart = harness.events.length;
  const recurrent = harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  await delayedPop.entered;
  const stopped = harness.engine.stop();
  delayedPop.release();

  await assert.rejects(recurrent, /neural inference cancelled by stop/);
  await stopped;
  assert.equal(harness.openErrorScopeCount(), 0);
  assert.deepEqual(
    harness.events.slice(scopeEventStart)
      .filter(({ type }) => type === "scope-pop")
      .map(({ filter }) => filter),
    ["validation", "internal", "out-of-memory"],
    "cancellation still unwinds every nested scope in LIFO order",
  );
  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  assert.deepEqual(
    harness.calls.slice(-2).map(({ graph }) => graph),
    ["initialize", "initialize"],
    "a candidate awaiting scope completion cannot survive stop",
  );

  await harness.engine.dispose();
});

test("a same-device model switch retires the prior temporal atlas", async (t) => {
  const first = cdaEntry();
  const second = cdaEntry({
    key: "cda-vsr-4x-mixed-alternate",
    filePrefix: "cda-vsr-alternate",
  });
  const harness = await createHarness(t, {
    entry: first,
    entries: [first, second],
  });
  const auxiliary = harness.auxiliary(320, 180);

  await harness.engine.run(
    harness.source,
    320,
    180,
    { auxiliary },
  );
  const firstAtlas = atlasTextures(harness, 320, 180);
  const firstTemporalBuffers = harness.buffers.filter(({ label }) =>
    label?.startsWith("neural-temporal-"));
  assert.equal(firstAtlas.length, 4);
  assert.ok(firstTemporalBuffers.length > 0);
  assert.ok(firstAtlas.every(({ destroyed }) => !destroyed));

  assert.equal((await harness.engine.init(second.key)).key, second.key);
  assert.ok(
    firstAtlas.every(({ destroyed }) => destroyed),
    "model publication waits for the old full-frame atlas to retire",
  );
  assert.ok(
    firstTemporalBuffers.every(({ destroyed }) => destroyed),
    "model publication also retires temporal scratch and uniforms",
  );
  assert.equal(
    atlasTextures(harness, 320, 180).length,
    4,
    "the replacement atlas remains lazy until the replacement model runs",
  );

  await harness.engine.dispose();
  assert.deepEqual(harness.releases, {
    initialize: 2,
    recurrent: 2,
  });
});

test("model publication cannot overtake a retry waiting on OOM recovery", async (t) => {
  const first = cdaEntry();
  const second = cdaEntry({
    key: "cda-vsr-4x-mixed-replacement",
    filePrefix: "cda-vsr-replacement",
  });
  const harness = await createHarness(t, {
    entry: first,
    entries: [first, second],
  });
  const auxiliary = harness.auxiliary(640, 360);

  await harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  let releaseFence;
  harness.holdQueueFence(
    new Promise((resolve) => { releaseFence = resolve; }),
  );
  harness.queueOutOfMemoryResult(new Error("serialize replacement"));
  const failedRunDelay = harness.delayNextGraphRun("recurrent");
  const failedRun = harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  await failedRunDelay.entered;

  const replacement = harness.engine.init(second.key);
  let replacementSettled = false;
  replacement.then(
    () => { replacementSettled = true; },
    () => { replacementSettled = true; },
  );
  await harness.waitForSessionCreates(4);
  await Promise.resolve();
  await Promise.resolve();
  failedRunDelay.release();
  await assert.rejects(failedRun, /out-of-memory: serialize replacement/);

  const retryDelay = harness.delayNextGraphRun("initialize");
  const retry = harness.engine.run(
    harness.source,
    640,
    360,
    { auxiliary },
  );
  releaseFence();
  await retryDelay.entered;
  assert.equal(replacementSettled, false);
  assert.deepEqual(
    harness.releases,
    { initialize: 0, recurrent: 0 },
    "the old graph group stays alive while the retry uses it",
  );

  retryDelay.release();
  await retry;
  assert.equal((await replacement).key, second.key);
  assert.deepEqual(harness.releases, {
    initialize: 1,
    recurrent: 1,
  });

  await harness.engine.dispose();
  assert.deepEqual(harness.releases, {
    initialize: 2,
    recurrent: 2,
  });
});

test("FP32 atlas packing uses an explicit unfilterable texture layout", async (t) => {
  const entry = cdaEntry({ stateDtype: "float32" });
  const harness = await createHarness(t, { entry });
  const auxiliary = harness.auxiliary(320, 180);

  await harness.engine.run(
    harness.source,
    320,
    180,
    { auxiliary },
  );

  const packBindGroupLayout = harness.bindGroupLayouts.find(
    ({ label }) =>
      label === "neural-temporal-state-pack-float32-bindings",
  );
  assert.ok(packBindGroupLayout, "the FP32 pack bind-group layout is explicit");
  assert.equal(
    packBindGroupLayout.entries.find(({ binding }) => binding === 0)
      ?.texture?.sampleType,
    "unfilterable-float",
  );
  const packPipelineLayout = harness.pipelineLayouts.find(
    ({ label }) =>
      label === "neural-temporal-state-pack-float32-layout",
  );
  assert.ok(packPipelineLayout, "the FP32 pack pipeline layout is explicit");
  assert.equal(
    packPipelineLayout.bindGroupLayouts[0],
    packBindGroupLayout,
  );
  assert.ok(
    atlasTextures(harness, 320, 180).every(
      ({ format }) => format === "rgba32float",
    ),
  );

  await harness.engine.dispose();
});
