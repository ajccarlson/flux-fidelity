const MODELS = [
  ["FSRCNNX x2", "model/FSRCNNX_x2_16-0-4-1.passes.json", "model/FSRCNNX_x2_16-0-4-1.wgsl"],
  ["FSRCNNX high x2", "model/FSRCNNX_x2_56-16-4-1.passes.json", "model/FSRCNNX_x2_56-16-4-1.wgsl"],
  ["FSRCNNX x3", "model/FSRCNNX_x3_16-0-4-1.passes.json", "model/FSRCNNX_x3_16-0-4-1.wgsl"],
  ["FSRCNNX x4", "model/FSRCNNX_x4_16-0-4-1.passes.json", "model/FSRCNNX_x4_16-0-4-1.wgsl"],
  ["ArtCNN", "model/ArtCNN_C4F32.artcnn.json", "model/ArtCNN_C4F32.artcnn.wgsl"],
  ["ArtCNN denoise", "model/ArtCNN_C4F32_DN.artcnn.json", "model/ArtCNN_C4F32_DN.artcnn.wgsl"],
  ["ArtCNN denoise/sharpen", "model/ArtCNN_C4F32_DS.artcnn.json", "model/ArtCNN_C4F32_DS.artcnn.wgsl"],
];

const runButton = document.getElementById("run");
const results = document.getElementById("results");
const summary = document.getElementById("summary");

function addResult(name, ok, detail) {
  const row = document.createElement("tr");
  const title = document.createElement("td");
  const status = document.createElement("td");
  const info = document.createElement("td");
  title.textContent = name;
  status.textContent = ok ? "PASS" : "FAIL";
  status.className = ok ? "pass" : "fail";
  info.textContent = detail;
  row.append(title, status, info);
  results.append(row);
  return ok;
}

function splitEntries(source) {
  const marker = "//==== ENTRY";
  const start = source.indexOf(marker);
  if (start < 0) return [];
  const prelude = source.slice(0, start);
  return source.slice(start).split(/(?=\/\/==== ENTRY)/).map((chunk) => prelude + chunk);
}

function validateTopology(manifest) {
  const available = new Set(["LUMA"]);
  const errors = [];
  manifest.passes.forEach((pass, index) => {
    if (pass.index !== index) errors.push(`pass ${index} index=${pass.index}`);
    for (const bind of pass.binds || []) if (!available.has(bind)) errors.push(`pass ${index} missing ${bind}`);
    if (pass.save) available.add(pass.save);
  });
  return errors;
}

function validateColorRoundTrip() {
  const samples = [
    [0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [.18, .5, .9], [.92, .34, .07], [.003, .21, .77],
  ];
  let max = 0;
  for (const [r, g, b] of samples) {
    const y = .2126 * r + .7152 * g + .0722 * b;
    const cb = (b - y) / 1.8556;
    const cr = (r - y) / 1.5748;
    const rr = y + 1.5748 * cr;
    const bb = y + 1.8556 * cb;
    const gg = (y - .2126 * rr - .0722 * bb) / .7152;
    max = Math.max(max, Math.abs(r - rr), Math.abs(g - gg), Math.abs(b - bb));
  }
  return max;
}

async function compileEntries(device, entries) {
  const errors = [];
  for (let index = 0; index < entries.length; index++) {
    const module = device.createShaderModule({ label: `validation-pass-${index}`, code: entries[index] });
    const info = await module.getCompilationInfo();
    const messages = info.messages.filter((message) => message.type === "error");
    if (messages.length) errors.push(`pass ${index}: ${messages.map((message) => message.message).join("; ")}`);
  }
  return errors;
}

async function run() {
  runButton.disabled = true;
  results.replaceChildren();
  summary.textContent = "Validating…";
  let passed = 0;
  let total = 0;
  let device = null;

  total++;
  const colorError = validateColorRoundTrip();
  if (addResult("BT.709 RGB ↔ YCbCr round trip", colorError < 1e-12, `maximum absolute error ${colorError.toExponential(3)}`)) passed++;

  total++;
  if (!navigator.gpu) {
    addResult("WebGPU", false, "navigator.gpu is unavailable");
  } else {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      device = await adapter.requestDevice();
      addResult("WebGPU", true, "adapter and device acquired");
      passed++;
    } else addResult("WebGPU", false, "no compatible adapter");
  }

  for (const [label, manifestPath, wgslPath] of MODELS) {
    try {
      const [manifest, source] = await Promise.all([
        fetch(manifestPath).then((response) => response.json()),
        fetch(wgslPath).then((response) => response.text()),
      ]);
      const topologyErrors = validateTopology(manifest);
      const entries = splitEntries(source);
      total++;
      if (addResult(`${label} topology`, topologyErrors.length === 0 && entries.length === manifest.passes.length,
        topologyErrors[0] || `${manifest.passes.length} passes and ${entries.length} WGSL entries`)) passed++;

      if (device) {
        const compilationErrors = await compileEntries(device, entries);
        total++;
        if (addResult(`${label} WGSL`, compilationErrors.length === 0,
          compilationErrors[0] || `${entries.length} compute passes compiled`)) passed++;
      }
    } catch (error) {
      total++;
      addResult(label, false, error.message);
    }
  }

  summary.textContent = `${passed}/${total} checks passed`;
  summary.className = passed === total ? "pass" : "fail";
  runButton.disabled = false;
}

runButton.addEventListener("click", () => run().catch((error) => {
  addResult("Validation harness", false, error.message);
  summary.textContent = "Validation aborted";
  summary.className = "fail";
  runButton.disabled = false;
}));
