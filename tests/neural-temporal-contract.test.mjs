import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

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
    "const { createOrtSession, ensureOrt, getOrtSessionDevice } = globalThis.__neuralTemporalTestDeps;\n" +
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
  assert.notEqual(source, original, "temporal neural test dependency injection must match");
  globalThis.__neuralTemporalTestDeps = deps;
  return import(
    `data:text/javascript;base64,${Buffer.from(source).toString("base64")}#temporal-${++moduleRevision}`
  );
}

function temporalEntry() {
  return {
    key: "temporal-2x",
    label: "Temporal 2x",
    scale: 2,
    contract: {
      version: 2,
      mode: "temporal",
      resetGraph: "initialize",
      recurrentGraph: "recurrent",
      graphs: {
        initialize: {
          file: "temporal-init.onnx",
          inputs: {
            frame: { role: "rgb", dtype: "float32", channels: 3 },
          },
          outputs: {
            image: { role: "rgb", dtype: "float32", channels: 3 },
            initial_feature: {
              role: "state-out",
              state: "feature",
              dtype: "float32",
              channels: 4,
            },
          },
        },
        recurrent: {
          file: "temporal-recurrent.onnx",
          inputs: {
            frame: { role: "rgb", dtype: "float32", channels: 3 },
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
            previous_feature: {
              role: "state-in",
              state: "feature",
              reset: "required",
              dtype: "float32",
              channels: 4,
            },
          },
          outputs: {
            image: { role: "rgb", dtype: "float32", channels: 3 },
            next_feature: {
              role: "state-out",
              state: "feature",
              dtype: "float32",
              channels: 4,
            },
          },
        },
      },
    },
  };
}

function cdaTemporalEntry(stateDtype = "float32") {
  const stateOutput = (name, state) => ({
    [name]: {
      role: "state-out",
      state,
      dtype: stateDtype,
      channels: 64,
    },
  });
  return {
    key: "cda-vsr-4x",
    label: "CDA-VSR 4x",
    scale: 4,
    contract: {
      version: 2,
      mode: "temporal",
      resetGraph: "initialize",
      recurrentGraph: "recurrent",
      graphs: {
        initialize: {
          file: "cda-vsr-initializer.onnx",
          inputs: {
            frame: { role: "rgb", dtype: "float32", channels: 3 },
          },
          outputs: {
            output: { role: "rgb", dtype: "float32", channels: 3 },
            ...stateOutput("next_state_low", "low"),
            ...stateOutput("next_state_high", "high"),
          },
        },
        recurrent: {
          file: "cda-vsr-recurrent.onnx",
          inputs: {
            frame: { role: "rgb", dtype: "float32", channels: 3 },
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
            output: { role: "rgb", dtype: "float32", channels: 3 },
            ...stateOutput("next_state_low", "low"),
            ...stateOutput("next_state_high", "high"),
          },
        },
      },
    },
  };
}

function noOpDeps() {
  return {
    createOrtSession: async () => null,
    ensureOrt: async () => ({}),
    getOrtSessionDevice: () => null,
  };
}

test("the exported CDA two-graph ABI is manifest-compatible and enumerates both assets", async (t) => {
  const previous = globalThis.__neuralTemporalTestDeps;
  t.after(() => { globalThis.__neuralTemporalTestDeps = previous; });
  const {
    neuralModelFiles,
    validateNeuralManifest,
  } = await loadNeuralEngine(noOpDeps());

  const [entry] = validateNeuralManifest({ models: [cdaTemporalEntry()] });
  assert.equal(entry.key, "cda-vsr-4x");
  assert.equal(entry.scale, 4);
  assert.deepEqual(entry.contract.states, ["low", "high"]);
  assert.deepEqual(neuralModelFiles(entry), [
    "cda-vsr-initializer.onnx",
    "cda-vsr-recurrent.onnx",
  ]);
});

test("the mixed CDA ABI keeps RGB and priors FP32 while accepting FP16 recurrent state", async (t) => {
  const previous = globalThis.__neuralTemporalTestDeps;
  t.after(() => { globalThis.__neuralTemporalTestDeps = previous; });
  const {
    validateNeuralManifest,
    validateNeuralSessionContract,
  } = await loadNeuralEngine(noOpDeps());

  const [entry] = validateNeuralManifest([cdaTemporalEntry("float16")]);
  const recurrent = entry.contract.graphs.recurrent;
  assert.equal(recurrent.inputs.frame.dtype, "float32");
  assert.equal(recurrent.inputs.motion.dtype, "float32");
  assert.equal(recurrent.inputs.residual.dtype, "float32");
  assert.equal(recurrent.inputs.state_low.dtype, "float16");
  assert.equal(recurrent.inputs.state_high.dtype, "float16");
  assert.equal(recurrent.outputs.output.dtype, "float32");
  assert.equal(recurrent.outputs.next_state_low.dtype, "float16");
  assert.equal(recurrent.outputs.next_state_high.dtype, "float16");

  const metadata = (name, type, channels, output = false) => ({
    name,
    isTensor: true,
    type,
    shape: [
      1,
      channels,
      output ? "output_height_x4" : "height",
      output ? "output_width_x4" : "width",
    ],
  });
  const resolved = validateNeuralSessionContract({
    inputNames: ["frame", "motion", "residual", "state_low", "state_high"],
    outputNames: ["output", "next_state_low", "next_state_high"],
    inputMetadata: [
      metadata("frame", "float32", 3),
      metadata("motion", "float32", 2),
      metadata("residual", "float32", 1),
      metadata("state_low", "float16", 64),
      metadata("state_high", "float16", 64),
    ],
    outputMetadata: [
      metadata("output", "float32", 3, true),
      metadata("next_state_low", "float16", 64),
      metadata("next_state_high", "float16", 64),
    ],
  }, entry, "recurrent");
  assert.equal(resolved.inputName, "frame");
  assert.equal(resolved.outputName, "output");
});

test("v2 temporal manifests normalize named graphs, providers, and paired state", async (t) => {
  const previous = globalThis.__neuralTemporalTestDeps;
  t.after(() => { globalThis.__neuralTemporalTestDeps = previous; });
  const {
    normalizeNeuralModelContract,
    resolveNeuralAuxiliaryInputs,
    validateNeuralManifest,
  } = await loadNeuralEngine(noOpDeps());

  const entry = validateNeuralManifest([temporalEntry()])[0];
  const contract = entry.contract;
  assert.equal(contract.version, 2);
  assert.equal(contract.mode, "temporal");
  assert.deepEqual(contract.states, ["feature"]);
  assert.equal(contract.graphs.recurrent.inputs.motion.provider, "decoded-cda-v1");
  assert.equal(contract.graphs.recurrent.inputs.previous_feature.reset, "required");
  assert.equal(Object.isFrozen(contract.graphs.recurrent.inputs), true);

  const motion = { gpuBuffer: {}, dataType: "float32", dims: [1, 2, 2, 2] };
  const residual = { gpuBuffer: {}, dataType: "float32", dims: [1, 1, 2, 2] };
  assert.deepEqual(
    resolveNeuralAuxiliaryInputs(contract, "recurrent", { motion, residual }),
    { motion, residual },
  );
  assert.deepEqual(
    resolveNeuralAuxiliaryInputs(contract, "recurrent", {
      motion,
      residual: undefined,
      // Exact tensor names and unique roles may be mixed.
      [contract.graphs.recurrent.inputs.residual.name]: residual,
    }),
    { motion, residual },
  );
  assert.throws(
    () => resolveNeuralAuxiliaryInputs(contract, "recurrent", { motion, typo: residual }),
    /not declared/,
  );

  const legacy = { key: "legacy", file: "legacy.onnx", scale: 2, input: "x", output: "y" };
  const normalizedLegacy = normalizeNeuralModelContract(legacy);
  assert.equal(normalizedLegacy.version, 1);
  assert.equal(normalizedLegacy.inputs[0].name, "x");
  assert.equal(normalizedLegacy.outputs[0].name, "y");
  assert.equal(Object.hasOwn(validateNeuralManifest([legacy])[0], "contract"), false);
});

test("v2 temporal manifests reject ambiguous graphs and unsafe tensor semantics", async (t) => {
  const previous = globalThis.__neuralTemporalTestDeps;
  t.after(() => { globalThis.__neuralTemporalTestDeps = previous; });
  const { validateNeuralManifest } = await loadNeuralEngine(noOpDeps());

  const mutate = (callback) => {
    const entry = structuredClone(temporalEntry());
    callback(entry);
    return [entry];
  };
  const invalid = [
    mutate((entry) => { entry.file = "legacy.onnx"; }),
    mutate((entry) => { entry.contract.graphs.recurrent.inputs.motion.provider = "unknown"; }),
    mutate((entry) => { entry.contract.graphs.recurrent.inputs.motion.channels = 3; }),
    mutate((entry) => { delete entry.contract.graphs.recurrent.inputs.residual; }),
    mutate((entry) => {
      entry.contract.graphs.recurrent.inputs.previous_feature.reset = "zeros";
    }),
    mutate((entry) => { entry.contract.graphs.recurrent.inputs.previous_feature.dtype = "float16"; }),
    mutate((entry) => { delete entry.contract.graphs.initialize.outputs.initial_feature; }),
    mutate((entry) => { entry.tileSize = 512; }),
    mutate((entry) => { entry.padMultiple = 8; }),
    mutate((entry) => {
      entry.contract.graphs.recurrent.outputs.duplicate = {
        role: "state-out",
        state: "feature",
        dtype: "float32",
        channels: 4,
      };
    }),
    mutate((entry) => { entry.contract.recurrentGraph = "initialize"; }),
  ];
  for (const manifest of invalid) {
    assert.throws(() => validateNeuralManifest(manifest), /neural manifest entry/);
  }
});

test("v2 graph/session validation checks every declared name, dtype, and channel count", async (t) => {
  const previous = globalThis.__neuralTemporalTestDeps;
  t.after(() => { globalThis.__neuralTemporalTestDeps = previous; });
  const {
    validateNeuralManifest,
    validateNeuralSessionContract,
  } = await loadNeuralEngine(noOpDeps());
  const entry = validateNeuralManifest([temporalEntry()])[0];
  const dynamic = (name, channels) => ({
    name,
    isTensor: true,
    type: "float32",
    shape: [1, channels, "height", "width"],
  });
  const valid = {
    inputNames: ["frame", "motion", "residual", "previous_feature"],
    outputNames: ["image", "next_feature"],
    inputMetadata: [
      dynamic("frame", 3),
      dynamic("motion", 2),
      dynamic("residual", 1),
      dynamic("previous_feature", 4),
    ],
    outputMetadata: [dynamic("image", 3), dynamic("next_feature", 4)],
  };
  const resolved = validateNeuralSessionContract(valid, entry, "recurrent");
  assert.equal(resolved.inputName, "frame");
  assert.equal(resolved.outputName, "image");
  assert.equal(resolved.graphName, "recurrent");

  assert.throws(
    () => validateNeuralSessionContract(
      { ...valid, inputNames: [...valid.inputNames, "undeclared"] },
      entry,
      "recurrent",
    ),
    /do not match manifest/,
  );
  assert.throws(
    () => validateNeuralSessionContract(
      {
        ...valid,
        inputMetadata: valid.inputMetadata.map((metadata) =>
          metadata.name === "motion" ? { ...metadata, shape: [1, 3, "height", "width"] } : metadata),
      },
      entry,
      "recurrent",
    ),
    /channel dimension must be 2/,
  );
  assert.throws(
    () => validateNeuralSessionContract(
      {
        ...valid,
        inputMetadata: valid.inputMetadata.map((metadata) =>
          metadata.name === "frame"
            ? { ...metadata, shape: [1, 3, 16, 16] }
            : metadata),
      },
      entry,
      "recurrent",
    ),
    /spatial dimensions must be dynamic/,
  );
});

function fakeDevice() {
  const buffers = [];
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
      writeBuffer() {},
      submit() {},
    },
    createShaderModule: ({ code }) => ({ code }),
    createComputePipeline: ({ compute }) => ({
      code: compute.module.code,
      getBindGroupLayout: () => ({}),
    }),
    createSampler: () => ({}),
    createBuffer(options) {
      const buffer = { ...options, destroy() {} };
      buffers.push(buffer);
      return buffer;
    },
    createTexture: (options) => ({
      ...options,
      createView: () => ({}),
      destroy() {},
    }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline() {},
        setBindGroup() {},
        dispatchWorkgroups() {},
        end() {},
      }),
      finish: () => ({}),
    }),
  };
  return { device, buffers };
}

test("the exact CDA descriptor is selectable and initializes both packaged graph URLs", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    GPUTextureUsage: globalThis.GPUTextureUsage,
    deps: globalThis.__neuralTemporalTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.GPUTextureUsage = previous.GPUTextureUsage;
    globalThis.__neuralTemporalTestDeps = previous.deps;
  });
  globalThis.chrome = { runtime: { getURL: (path) => `extension://${path}` } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({
      models: [
        {
          key: "legacy",
          file: "legacy.onnx",
          scale: 2,
          input: "input",
          output: "output",
        },
        cdaTemporalEntry(),
      ],
    }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };
  globalThis.GPUTextureUsage = { STORAGE_BINDING: 1, TEXTURE_BINDING: 2 };

  const { device } = fakeDevice();
  const dynamic = (name, channels, multiplier = 1) => ({
    name,
    isTensor: true,
    type: "float32",
    shape: [1, channels, `height*${multiplier}`, `width*${multiplier}`],
  });
  const releases = [];
  const sessions = [
    {
      device,
      inputNames: ["frame"],
      outputNames: ["output", "next_state_low", "next_state_high"],
      inputMetadata: [dynamic("frame", 3)],
      outputMetadata: [
        dynamic("output", 3, 4),
        dynamic("next_state_low", 64),
        dynamic("next_state_high", 64),
      ],
      async release() { releases.push("initialize"); },
    },
    {
      device,
      inputNames: ["frame", "motion", "residual", "state_low", "state_high"],
      outputNames: ["output", "next_state_low", "next_state_high"],
      inputMetadata: [
        dynamic("frame", 3),
        dynamic("motion", 2),
        dynamic("residual", 1),
        dynamic("state_low", 64),
        dynamic("state_high", 64),
      ],
      outputMetadata: [
        dynamic("output", 3, 4),
        dynamic("next_state_low", 64),
        dynamic("next_state_high", 64),
      ],
      async release() { releases.push("recurrent"); },
    },
  ];
  const createdUrls = [];
  const deps = {
    ensureOrt: async () => ({}),
    getOrtSessionDevice: (candidate) => candidate.device,
    createOrtSession: async (url) => {
      createdUrls.push(url);
      return sessions[createdUrls.length - 1];
    },
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });

  const entry = await engine.init("cda-vsr-4x");
  assert.equal(entry.key, "cda-vsr-4x");
  assert.equal(engine.activeEntry().scale, 4);
  assert.deepEqual(engine.activeContract().states, ["low", "high"]);
  assert.deepEqual(createdUrls, [
    "extension://model/neural/cda-vsr-initializer.onnx",
    "extension://model/neural/cda-vsr-recurrent.onnx",
  ]);

  await engine.dispose();
  assert.deepEqual(releases.sort(), ["initialize", "recurrent"]);
});

test("v2 execution selects static reset/recurrent graphs and retains GPU state transactionally", async (t) => {
  const previous = {
    chrome: globalThis.chrome,
    fetch: globalThis.fetch,
    GPUBufferUsage: globalThis.GPUBufferUsage,
    GPUTextureUsage: globalThis.GPUTextureUsage,
    deps: globalThis.__neuralTemporalTestDeps,
  };
  t.after(() => {
    globalThis.chrome = previous.chrome;
    globalThis.fetch = previous.fetch;
    globalThis.GPUBufferUsage = previous.GPUBufferUsage;
    globalThis.GPUTextureUsage = previous.GPUTextureUsage;
    globalThis.__neuralTemporalTestDeps = previous.deps;
  });
  globalThis.chrome = { runtime: { getURL: (path) => path } };
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [temporalEntry()] }),
  });
  globalThis.GPUBufferUsage = { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 };
  globalThis.GPUTextureUsage = { STORAGE_BINDING: 1, TEXTURE_BINDING: 2 };

  const { device } = fakeDevice();
  const calls = [];
  const disposed = { wrappers: 0, rgb: 0, states: 0 };
  const releases = { initialize: 0, recurrent: 0 };
  let stateId = 0;
  let emitWrongInitializerState = false;
  const tensor = (type, dims, kind) => {
    const size = dims.reduce((product, value) => product * value, 1);
    return {
      id: kind === "state" ? ++stateId : undefined,
      type,
      dims,
      size,
      gpuBuffer: { size: size * 4 },
      dispose() {
        if (kind === "state") disposed.states++;
        else disposed.rgb++;
      },
    };
  };
  const dynamic = (name, channels) => ({
    name,
    isTensor: true,
    type: "float32",
    shape: [1, channels, "height", "width"],
  });
  const initialize = {
    device,
    inputNames: ["frame"],
    outputNames: ["image", "initial_feature"],
    inputMetadata: [dynamic("frame", 3)],
    outputMetadata: [dynamic("image", 3), dynamic("initial_feature", 4)],
    async run(feeds, fetches) {
      calls.push({ graph: "initialize", feeds, fetches });
      const [, , height, width] = feeds.frame.dims;
      return {
        image: tensor("float32", [1, 3, height * 2, width * 2], "rgb"),
        initial_feature: tensor(
          "float32",
          [1, 4, emitWrongInitializerState ? height + 1 : height, width],
          "state",
        ),
      };
    },
    async release() { releases.initialize++; },
  };
  const recurrent = {
    device,
    inputNames: ["frame", "motion", "residual", "previous_feature"],
    outputNames: ["image", "next_feature"],
    inputMetadata: [
      dynamic("frame", 3),
      dynamic("motion", 2),
      dynamic("residual", 1),
      dynamic("previous_feature", 4),
    ],
    outputMetadata: [dynamic("image", 3), dynamic("next_feature", 4)],
    async run(feeds, fetches) {
      calls.push({ graph: "recurrent", feeds, fetches });
      const [, , height, width] = feeds.frame.dims;
      return {
        image: tensor("float32", [1, 3, height * 2, width * 2], "rgb"),
        next_feature: tensor("float32", [1, 4, height, width], "state"),
      };
    },
    async release() { releases.recurrent++; },
  };
  const ort = {
    Tensor: {
      fromGpuBuffer(gpuBuffer, options) {
        return {
          gpuBuffer,
          type: options.dataType,
          dims: options.dims,
          dispose() { disposed.wrappers++; },
        };
      },
    },
  };
  const sessions = [initialize, recurrent];
  let createIndex = 0;
  const deps = {
    ensureOrt: async () => ort,
    getOrtSessionDevice: (candidate) => candidate.device,
    createOrtSession: async () => sessions[createIndex++],
  };
  const { createNeuralEngine } = await loadNeuralEngine(deps);
  const engine = createNeuralEngine({ log: () => {}, warn: () => {} });
  await engine.init("temporal-2x");
  assert.equal(engine.activeContract().version, 2);

  const source = { tex: { createView: () => ({}) } };
  const motion = {
    provider: "decoded-cda-v1",
    gpuBuffer: { size: 1 * 2 * 2 * 2 * 4 },
    dataType: "float32",
    dims: [1, 2, 2, 2],
  };
  const residual = {
    provider: "decoded-cda-v1",
    gpuBuffer: { size: 1 * 1 * 2 * 2 * 4 },
    dataType: "float32",
    dims: [1, 1, 2, 2],
  };
  const auxiliary = { motion, residual };

  await engine.run(source, 2, 2, { reset: true, auxiliary });
  const initialState = calls[0].graph === "initialize"
    ? initialize
    : null;
  assert.ok(initialState);
  assert.deepEqual(calls[0].fetches, ["image", "initial_feature"]);

  await engine.run(source, 2, 2, { auxiliary });
  assert.equal(calls[1].graph, "recurrent");
  assert.equal(engine.stats().temporalResetRuns, 1);
  assert.equal(engine.stats().temporalRecurrentRuns, 1);
  assert.deepEqual(Object.keys(calls[1].feeds), [
    "frame",
    "motion",
    "residual",
    "previous_feature",
  ]);
  assert.equal(calls[1].feeds.motion.gpuBuffer, motion.gpuBuffer);
  assert.equal(calls[1].feeds.residual.gpuBuffer, residual.gpuBuffer);
  assert.equal(calls[1].feeds.previous_feature.id, 1);
  assert.deepEqual(calls[1].fetches, ["image", "next_feature"]);

  await assert.rejects(
    engine.run(source, 2, 2, {
      auxiliary: { motion: { ...motion, provider: undefined }, residual },
    }),
    /provider 'unknown'/,
  );

  const resizedAuxiliary = {
    motion: {
      provider: "decoded-cda-v1",
      gpuBuffer: { size: 1 * 2 * 3 * 3 * 4 },
      dataType: "float32",
      dims: [1, 2, 3, 3],
    },
    residual: {
      provider: "decoded-cda-v1",
      gpuBuffer: { size: 1 * 1 * 3 * 3 * 4 },
      dataType: "float32",
      dims: [1, 1, 3, 3],
    },
  };
  await engine.run(source, 3, 3, { auxiliary: resizedAuxiliary });
  assert.equal(
    calls[2].graph,
    "initialize",
    "a source resize must not feed stale recurrent state",
  );

  await engine.run(source, 2, 2, { temporal: { reset: true }, auxiliary });
  assert.equal(calls[3].graph, "initialize");
  emitWrongInitializerState = true;
  await assert.rejects(
    engine.run(source, 2, 2, { reset: true, auxiliary }),
    /state output 'initial_feature' spatial shape 2x3 does not match 2x2/,
  );
  emitWrongInitializerState = false;
  await engine.run(source, 2, 2, { auxiliary });
  assert.equal(
    calls[5].graph,
    "initialize",
    "a rejected initializer state must not make recurrent state ready",
  );
  assert.equal(engine.stats().temporalResetRuns, 4);
  assert.equal(engine.stats().temporalRecurrentRuns, 1);
  await engine.dispose();
  assert.deepEqual(releases, { initialize: 1, recurrent: 1 });
  assert.equal(disposed.rgb, 6);
  assert.equal(disposed.states, 6);
  assert.equal(
    disposed.wrappers,
    9,
    "successful wrappers and the rejected frame's RGB wrapper all retire",
  );
});
