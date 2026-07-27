import { ArtCnnModel } from "../src/core/fsrcnnx-artcnn-runtime.js";
import { GENERATED_MODEL_CATALOG } from "../src/core/fsrcnnx-model-catalog.js";
import { validateModelBundle } from "../src/core/fsrcnnx-model-bundle.js";
import { validateNeuralManifest } from "../src/core/fsrcnnx-neural.js";
import {
  createOrtSession,
  ensureOrt,
  getOrtSessionDevice,
  resolvePackagedAssetUrl,
} from "../src/core/fsrcnnx-rife.js";
import { FsrcnnxModel } from "../src/core/fsrcnnx-runtime.js";
import {
  formatReferenceMetrics,
  loadReferenceManifest,
  runModelReference,
  runSupportingReference,
} from "./fsrcnnx-reference-validation.js";
import {
  acquireValidationDevice,
  buildCorePipelines,
  createNeuralValidationFixture,
  createValidationPlan,
  inspectOrtFloatTensor,
  ONNX_VALIDATION_CHECKS,
  REFERENCE_VALIDATION_CHECKS,
  summarizeValidation,
  VALIDATION_TIMEOUT_MS,
  withGpuErrorScopes,
  withTimeout,
} from "./fsrcnnx-validation.js";

const plan = createValidationPlan(GENERATED_MODEL_CATALOG);
const runButton = document.getElementById("run");
const resultsBody = document.getElementById("results");
const summaryNode = document.getElementById("summary");
const rows = new Map();
let runGeneration = 0;

function errorMessage(error) {
  return error?.message || String(error);
}

function renderPlan() {
  rows.clear();
  resultsBody.replaceChildren();
  for (const check of plan) {
    const row = document.createElement("tr");
    const title = document.createElement("td");
    const status = document.createElement("td");
    const detail = document.createElement("td");
    title.textContent = check.label;
    status.textContent = "PENDING";
    status.className = "pending";
    detail.textContent = "Waiting…";
    row.dataset.checkId = check.id;
    row.append(title, status, detail);
    resultsBody.append(row);
    rows.set(check.id, { row, status, detail });
  }
}

function snapshot(runId, resultMap, done) {
  const totals = summarizeValidation(plan, resultMap);
  return Object.freeze({
    runId,
    done,
    ...totals,
    results: Object.freeze(plan.map((check) => Object.freeze({
      id: check.id,
      label: check.label,
      status: resultMap.get(check.id)?.status || "pending",
      detail: resultMap.get(check.id)?.detail || "Waiting…",
    }))),
  });
}

function publish(runId, resultMap, done = false) {
  if (runId !== runGeneration) return;
  const state = snapshot(runId, resultMap, done);
  window.__FSRCNNX_VALIDATION__ = state;
  const parts = [`${state.pass}/${state.total} checks passed`];
  if (state.fail) parts.push(`${state.fail} failed`);
  if (state.skip) parts.push(`${state.skip} skipped`);
  if (state.pending) parts.push(`${state.pending} pending`);
  summaryNode.textContent = parts.join(" · ");
  summaryNode.className = done ? (state.ok ? "pass" : "fail") : "pending";
  document.documentElement.dataset.validationState = done ? (state.ok ? "pass" : "fail") : "running";
  if (done) window.dispatchEvent(new CustomEvent("fsrcnnx-validation-complete", { detail: state }));
}

function setResult(runId, resultMap, id, status, detail) {
  if (runId !== runGeneration) return false;
  if (!rows.has(id)) throw new Error(`unknown validation check ${id}`);
  if (!new Set(["pass", "fail", "skip"]).has(status)) throw new Error(`invalid validation status ${status}`);
  if (resultMap.has(id)) throw new Error(`validation check ${id} was completed twice`);
  const result = Object.freeze({ status, detail: String(detail) });
  resultMap.set(id, result);
  const rendered = rows.get(id);
  rendered.status.textContent = status.toUpperCase();
  rendered.status.className = status;
  rendered.detail.textContent = result.detail;
  publish(runId, resultMap);
  return true;
}

function skipGpuChecks(runId, resultMap, reason) {
  setResult(runId, resultMap, "core:pipelines", "skip", reason);
  for (const check of REFERENCE_VALIDATION_CHECKS) {
    setResult(runId, resultMap, check.id, "skip", reason);
  }
  setResult(runId, resultMap, "webgpu:errors", "skip", reason);
}

async function loadBundle(spec) {
  const [manifestResponse, shaderResponse] = await withTimeout(Promise.all([
    fetch(spec.manifestPath),
    fetch(spec.shaderPath),
  ]), VALIDATION_TIMEOUT_MS, `${spec.label} fetch`);
  if (!manifestResponse.ok || !shaderResponse.ok) {
    throw new Error(`fetch failed (${manifestResponse.status}/${shaderResponse.status})`);
  }
  return {
    manifest: await manifestResponse.json(),
    source: await shaderResponse.text(),
  };
}

async function validateModel(runId, resultMap, spec, device, referenceCases, referenceError) {
  let bundle;
  try {
    bundle = await loadBundle(spec);
    const validated = validateModelBundle(spec.kind, bundle.manifest, bundle.source, {
      expectedName: spec.name,
      deviceLimits: device?.limits,
    });
    setResult(
      runId,
      resultMap,
      `${spec.name}:topology`,
      "pass",
      `${validated.manifest.passes.length} passes and ${validated.entries.size} exact shader entries`,
    );
  } catch (error) {
    setResult(runId, resultMap, `${spec.name}:topology`, "fail", errorMessage(error));
    setResult(runId, resultMap, `${spec.name}:pipelines`, "skip", "topology validation failed");
    setResult(runId, resultMap, `${spec.name}:inference`, "skip", "topology validation failed");
    return;
  }

  if (!device) {
    const reason = "no WebGPU device";
    setResult(runId, resultMap, `${spec.name}:pipelines`, "skip", reason);
    setResult(runId, resultMap, `${spec.name}:inference`, "skip", reason);
    return;
  }

  const Model = spec.kind === "artcnn" ? ArtCnnModel : FsrcnnxModel;
  let model = null;
  try {
    const pipelines = await withGpuErrorScopes(device, `${spec.label} pipeline construction`, () => {
      model = new Model(device, bundle.manifest, bundle.source, { expectedName: spec.name });
      return model.buildPipelines();
    });
    setResult(
      runId,
      resultMap,
      `${spec.name}:pipelines`,
      "pass",
      `${pipelines.length} production compute pipelines created`,
    );
  } catch (error) {
    setResult(runId, resultMap, `${spec.name}:pipelines`, "fail", errorMessage(error));
    setResult(runId, resultMap, `${spec.name}:inference`, "skip", "pipeline construction failed");
    try { model?.destroy(); } catch {}
    return;
  }

  const referenceCase = referenceCases?.get(spec.name);
  if (!referenceCase) {
    const detail = referenceError
      ? `reference fixture load failed: ${errorMessage(referenceError)}`
      : `reference fixture case '${spec.name}' is missing`;
    setResult(runId, resultMap, `${spec.name}:inference`, "fail", detail);
    try { model.destroy(); } catch {}
    return;
  }

  try {
    const stats = await withGpuErrorScopes(
      device,
      `${spec.label} numerical reference`,
      () => runModelReference(device, model, referenceCase),
    );
    setResult(
      runId,
      resultMap,
      `${spec.name}:inference`,
      "pass",
      `${stats.width}×${stats.height}; ${formatReferenceMetrics(stats)}`,
    );
  } catch (error) {
    setResult(runId, resultMap, `${spec.name}:inference`, "fail", errorMessage(error));
  } finally {
    try { model.destroy(); } catch {}
  }
}

async function validateSupportingReferences(runId, resultMap, device, referenceCases, referenceError) {
  for (const check of REFERENCE_VALIDATION_CHECKS) {
    const referenceCase = referenceCases?.get(check.id);
    if (!referenceCase) {
      const detail = referenceError
        ? `reference fixture load failed: ${errorMessage(referenceError)}`
        : `reference fixture case '${check.id}' is missing`;
      setResult(runId, resultMap, check.id, "fail", detail);
      continue;
    }
    try {
      const stats = await withGpuErrorScopes(
        device,
        check.label,
        () => runSupportingReference(device, referenceCase),
      );
      setResult(
        runId,
        resultMap,
        check.id,
        "pass",
        `${stats.width}×${stats.height}; ${formatReferenceMetrics(stats)}`,
      );
    } catch (error) {
      setResult(runId, resultMap, check.id, "fail", errorMessage(error));
    }
  }
}

function rifeInput(width, height) {
  const plane = width * height;
  const data = new Float32Array(plane * 7);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const fx = x / Math.max(1, width - 1);
      const fy = y / Math.max(1, height - 1);
      data[index] = 0.1 + 0.7 * fx;
      data[plane + index] = 0.1 + 0.7 * fy;
      data[2 * plane + index] = 0.2 + 0.5 * ((fx + fy) / 2);
      data[3 * plane + index] = 0.1 + 0.7 * (1 - fy);
      data[4 * plane + index] = 0.1 + 0.7 * (1 - fx);
      data[5 * plane + index] = 0.2 + 0.5 * ((2 - fx - fy) / 2);
    }
  }
  data.fill(0.5, 6 * plane);
  return data;
}

async function createValidationOrtSession(modelUrl, label, {
  provider,
  gpuOutput,
  enableFp16,
  inputName = null,
  outputName = null,
}) {
  let session = null;
  try {
    const options = {
      executionProviders: provider === "webgpu" ? [{ name: "webgpu" }] : [provider],
      graphOptimizationLevel: "all",
      enableGraphCapture: false,
    };
    // Dynamic-shape WebGPU graphs intentionally leave shape bookkeeping on the
    // CPU. ORT reports that expected placement notice through console.error;
    // retain actual error/fatal logging while the assertions below validate the run.
    if (provider === "webgpu") options.logSeverityLevel = 3;
    if (gpuOutput) options.preferredOutputLocation = "gpu-buffer";
    session = await createOrtSession(modelUrl, options, { enableFp16 });
    if (!session) throw new Error(`${label} session creation returned no session`);
    if (!Array.isArray(session.inputNames) || session.inputNames.length !== 1) {
      throw new Error(`${label} exposes ${session.inputNames?.length ?? 0} inputs; expected exactly one`);
    }
    if (!Array.isArray(session.outputNames) || session.outputNames.length !== 1) {
      throw new Error(`${label} exposes ${session.outputNames?.length ?? 0} outputs; expected exactly one`);
    }
    if (inputName && session.inputNames[0] !== inputName) {
      throw new Error(`${label} input is '${session.inputNames[0]}'; manifest declares '${inputName}'`);
    }
    if (outputName && session.outputNames[0] !== outputName) {
      throw new Error(`${label} output is '${session.outputNames[0]}'; manifest declares '${outputName}'`);
    }
    if (provider === "webgpu" && !getOrtSessionDevice(session)) {
      throw new Error(`${label} did not expose its ORT WebGPU device`);
    }
    return session;
  } catch (error) {
    try { await session?.release?.(); } catch {}
    throw error;
  }
}

async function runValidationOrtSession(ort, session, {
  label,
  data,
  inputDims,
  outputDims,
  gpuInput,
  gpuOutput,
}) {
  const ownerDevice = getOrtSessionDevice(session);
  let inputBuffer = null;
  let inputTensor = null;
  let outputs = null;
  try {
    if (gpuInput) {
      if (typeof ort.Tensor?.fromGpuBuffer !== "function") {
        throw new Error(`${label} ORT bundle has no Tensor.fromGpuBuffer()`);
      }
      inputBuffer = ownerDevice.createBuffer({
        label: "validation-ort-input",
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      ownerDevice.queue.writeBuffer(inputBuffer, 0, data);
      inputTensor = ort.Tensor.fromGpuBuffer(inputBuffer, { dataType: "float32", dims: inputDims });
    } else {
      inputTensor = new ort.Tensor("float32", data, inputDims);
    }

    outputs = await session.run({ [session.inputNames[0]]: inputTensor });
    const output = outputs?.[session.outputNames[0]];
    if (!output) throw new Error(`${label} did not return '${session.outputNames[0]}'`);
    if (gpuOutput && (!output.gpuBuffer || output.location !== "gpu-buffer")) {
      throw new Error(`${label} output is not at the requested gpu-buffer location`);
    }
    return await inspectOrtFloatTensor(output, outputDims, `${label} output`);
  } finally {
    // A failed output assertion can occur while GPU-backed tensors still own
    // submitted work. Fence before disposing wrappers or their user-owned input.
    try { await ownerDevice?.queue?.onSubmittedWorkDone?.(); } catch {}
    for (const output of new Set(Object.values(outputs || {}))) {
      try { output?.dispose?.(); } catch {}
    }
    try { inputTensor?.dispose?.(); } catch {}
    if (inputBuffer) {
      try { inputBuffer.destroy(); } catch {}
    }
  }
}

async function executeValidationOrtCheck(ort, check) {
  let session = null;
  let stats = null;
  let failure = null;
  let stage = "session creation";
  try {
    session = await createValidationOrtSession(check.modelUrl, check.label, check);
    stage = "inference";
    stats = await runValidationOrtSession(ort, session, check);
  } catch (error) {
    failure = new Error(`${stage}: ${errorMessage(error)}`, { cause: error });
  }

  if (session) {
    try {
      await session.release();
    } catch (error) {
      const releaseDetail = `session release: ${errorMessage(error)}`;
      failure = failure
        ? new Error(`${failure.message}; ${releaseDetail}`, { cause: failure })
        : new Error(releaseDetail, { cause: error });
    }
  }
  if (failure) throw failure;
  return stats;
}

async function loadNeuralValidationCheck() {
  const manifestUrl = resolvePackagedAssetUrl("model/neural/manifest.json");
  const response = await withTimeout(
    fetch(manifestUrl),
    VALIDATION_TIMEOUT_MS,
    "bundled neural manifest fetch",
  );
  if (!response.ok) throw new Error(`bundled neural manifest HTTP ${response.status}`);
  const entries = validateNeuralManifest(await response.json());
  if (!entries.length) throw new Error("bundled neural manifest contains no models");
  const entry = entries[0];
  const fixture = createNeuralValidationFixture(entry);
  return {
    id: "onnx:neural",
    kind: "neural",
    label: entry.label || entry.key,
    modelUrl: resolvePackagedAssetUrl(`model/neural/${entry.file}`),
    inputName: entry.input || null,
    outputName: entry.output || null,
    enableFp16: entry.fp16 === true,
    ...fixture,
  };
}

async function executeNeuralValidationOrtCheck(ort, check) {
  const failures = [];
  const canUseWebGpu = !!navigator.gpu && typeof GPUBufferUsage !== "undefined";
  if (canUseWebGpu) {
    const fp16Modes = check.enableFp16 ? [true, false] : [false];
    for (const enableFp16 of fp16Modes) {
      try {
        const stats = await executeValidationOrtCheck(ort, {
          ...check,
          provider: "webgpu",
          enableFp16,
          gpuInput: true,
          gpuOutput: true,
        });
        return {
          stats,
          provider: `WebGPU GPU-buffer (${enableFp16 ? "FP16" : "FP32"} execution)`,
          fallback: failures.length ? failures.join("; ") : null,
        };
      } catch (error) {
        failures.push(`WebGPU ${enableFp16 ? "FP16" : "FP32"}: ${errorMessage(error)}`);
      }
    }
  } else {
    failures.push("WebGPU API/GPUBufferUsage unavailable");
  }

  try {
    const stats = await executeValidationOrtCheck(ort, {
      ...check,
      provider: "wasm",
      enableFp16: false,
      gpuInput: false,
      gpuOutput: false,
    });
    return {
      stats,
      provider: "WASM CPU fallback",
      fallback: failures.join("; "),
    };
  } catch (error) {
    failures.push(`WASM: ${errorMessage(error)}`);
    throw new Error(failures.join("; "));
  }
}

async function validateOnnxModels(runId, resultMap) {
  const checks = [
    {
      id: "onnx:rife-v4.26-fp16",
      label: "RIFE 4.26 FP16",
      inputDims: [1, 7, 64, 64],
      outputDims: [1, 3, 64, 64],
      data: rifeInput(64, 64),
      provider: "wasm",
      enableFp16: false,
      gpuInput: false,
      gpuOutput: false,
      modelUrl: resolvePackagedAssetUrl("model/rife_v4.26_fp16.onnx"),
    },
    {
      id: "onnx:rife-v4.26",
      label: "default RIFE 4.26",
      inputDims: [1, 7, 64, 64],
      outputDims: [1, 3, 64, 64],
      data: rifeInput(64, 64),
      provider: "webgpu",
      enableFp16: false,
      gpuInput: true,
      gpuOutput: true,
      modelUrl: resolvePackagedAssetUrl("model/rife_v4.26.onnx"),
      // WebGPU session creation + inference runs on SwiftShader in CI, which
      // intermittently needs well over the default budget.
      timeoutMs: 120_000,
    },
  ];
  const outcomes = new Map();
  try {
    checks.push(await loadNeuralValidationCheck());
  } catch (error) {
    outcomes.set("onnx:neural", {
      status: "fail",
      detail: `manifest/model contract: ${errorMessage(error)}`,
    });
  }
  let ort;
  try {
    ort = await ensureOrt();
  } catch (error) {
    for (const check of ONNX_VALIDATION_CHECKS) {
      if (!outcomes.has(check.id)) {
        outcomes.set(check.id, { status: "fail", detail: `ORT bundle load: ${errorMessage(error)}` });
      }
    }
  }

  if (ort) {
    for (const check of checks) {
      if (outcomes.has(check.id)) continue;
      try {
        const result = await withTimeout(
          check.kind === "neural"
            ? executeNeuralValidationOrtCheck(ort, check)
            : executeValidationOrtCheck(ort, check),
          check.timeoutMs ?? VALIDATION_TIMEOUT_MS,
          `${check.label} ORT validation`,
        );
        const stats = check.kind === "neural" ? result.stats : result;
        const route = check.kind === "neural" ? result.provider : `ORT ${check.provider}`;
        const source = check.kind === "neural"
          ? `${check.sourceWidth}×${check.sourceHeight} source, ` +
            `${check.inputDims[3]}×${check.inputDims[2]} padded FP32 RGB NCHW`
          : check.inputDims.join("×");
        const fallback = check.kind === "neural" && result.fallback
          ? `; fallback reason: ${result.fallback}`
          : "";
        outcomes.set(check.id, {
          status: "pass",
          detail: `${source} → ${stats.dims.join("×")}; ${stats.elements} finite values ` +
            `in [${stats.min.toPrecision(4)}, ${stats.max.toPrecision(4)}] via ${route}${fallback}`,
        });
      } catch (error) {
        outcomes.set(check.id, { status: "fail", detail: errorMessage(error) });
      }
    }
  }
  for (const check of ONNX_VALIDATION_CHECKS) {
    const outcome = outcomes.get(check.id) || { status: "fail", detail: "validation produced no outcome" };
    setResult(runId, resultMap, check.id, outcome.status, outcome.detail);
  }
}

async function run() {
  const runId = ++runGeneration;
  const resultMap = new Map();
  renderPlan();
  runButton.disabled = true;
  publish(runId, resultMap);

  let device = null;
  let deviceDestroyed = false;
  let deviceLoss = null;
  const uncapturedErrors = [];
  const onUncapturedError = (event) => {
    event.preventDefault();
    uncapturedErrors.push(errorMessage(event.error));
  };

  try {
    try {
      const acquired = await acquireValidationDevice(navigator.gpu);
      device = acquired.device;
      device.addEventListener("uncapturederror", onUncapturedError);
      device.lost.then((info) => {
        if (!deviceDestroyed) deviceLoss = `${info.reason || "unknown"}: ${info.message || "device lost"}`;
      });
      setResult(runId, resultMap, "webgpu", "pass", acquired.detail);
    } catch (error) {
      setResult(runId, resultMap, "webgpu", "fail", errorMessage(error));
    }

    if (device) {
      try {
        const pipelines = await withGpuErrorScopes(device, "core pipeline construction", () => (
          buildCorePipelines(device, navigator.gpu.getPreferredCanvasFormat())
        ));
        setResult(runId, resultMap, "core:pipelines", "pass", `${pipelines.length} color/filter pipeline variants created`);
      } catch (error) {
        setResult(runId, resultMap, "core:pipelines", "fail", errorMessage(error));
      }

      let referenceCases = null;
      let referenceError = null;
      try {
        ({ cases: referenceCases } = await loadReferenceManifest());
      } catch (error) {
        referenceError = error;
      }
      await validateSupportingReferences(runId, resultMap, device, referenceCases, referenceError);

      for (const spec of GENERATED_MODEL_CATALOG) {
        await validateModel(runId, resultMap, spec, device, referenceCases, referenceError);
      }
    } else {
      skipGpuChecks(runId, resultMap, "no WebGPU device");
      for (const spec of GENERATED_MODEL_CATALOG) {
        await validateModel(runId, resultMap, spec, device, null, null);
      }
    }

    await validateOnnxModels(runId, resultMap);

    if (device) {
      try {
        await withTimeout(device.queue.onSubmittedWorkDone(), VALIDATION_TIMEOUT_MS, "final GPU queue fence");
        // Uncaptured GPU errors are dispatched as tasks, not promise reactions.
        // Give that channel one turn after the queue fence before declaring it clean.
        await new Promise((resolve) => setTimeout(resolve, 0));
      } catch (error) {
        uncapturedErrors.push(errorMessage(error));
      }
      const errors = [...uncapturedErrors];
      if (deviceLoss) errors.push(`device lost (${deviceLoss})`);
      setResult(
        runId,
        resultMap,
        "webgpu:errors",
        errors.length ? "fail" : "pass",
        errors.length ? errors.join("; ") : "no uncaptured errors or unexpected device loss",
      );
    }
  } catch (error) {
    for (const check of plan) {
      if (!resultMap.has(check.id)) setResult(runId, resultMap, check.id, "fail", `validation aborted: ${errorMessage(error)}`);
    }
  } finally {
    if (device) {
      device.removeEventListener("uncapturederror", onUncapturedError);
      deviceDestroyed = true;
      try { device.destroy(); } catch {}
    }
    if (runId === runGeneration) {
      for (const check of plan) {
        if (!resultMap.has(check.id)) setResult(runId, resultMap, check.id, "fail", "validation ended without a result");
      }
      publish(runId, resultMap, true);
      runButton.disabled = false;
    }
  }
}

runButton.addEventListener("click", () => void run());
window.__FSRCNNX_RUN_VALIDATION__ = run;

if (new URL(location.href).searchParams.get("autorun") === "1") {
  queueMicrotask(() => void run());
}
