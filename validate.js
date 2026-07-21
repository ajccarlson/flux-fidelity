import { ArtCnnModel } from "./fsrcnnx-artcnn-runtime.js";
import { GENERATED_MODEL_CATALOG } from "./fsrcnnx-model-catalog.js";
import { validateModelBundle } from "./fsrcnnx-model-bundle.js";
import { FsrcnnxModel } from "./fsrcnnx-runtime.js";
import {
  acquireValidationDevice,
  buildCorePipelines,
  createValidationPlan,
  runModelInference,
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

async function validateModel(runId, resultMap, spec, device) {
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

  try {
    const stats = await withGpuErrorScopes(
      device,
      `${spec.label} inference`,
      () => runModelInference(device, model),
    );
    setResult(
      runId,
      resultMap,
      `${spec.name}:inference`,
      "pass",
      `${stats.width}×${stats.height} readback; ${stats.components} finite components ` +
        `with luma [${stats.channelMin[0].toPrecision(4)}, ${stats.channelMax[0].toPrecision(4)}]`,
    );
  } catch (error) {
    setResult(runId, resultMap, `${spec.name}:inference`, "fail", errorMessage(error));
  } finally {
    try { model.destroy(); } catch {}
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
    } else {
      skipGpuChecks(runId, resultMap, "no WebGPU device");
    }

    for (const spec of GENERATED_MODEL_CATALOG) {
      await validateModel(runId, resultMap, spec, device);
    }

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
