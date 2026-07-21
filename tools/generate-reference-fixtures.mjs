import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import {
  encodeReferenceInput,
  REFERENCE_CASES,
  REFERENCE_FIXTURE_SCHEMA_VERSION,
  REFERENCE_GENERATOR_POLICY,
  REFERENCE_INPUTS,
  REFERENCE_INPUT_VERSION,
  REFERENCE_TOOLCHAIN,
} from "../reference-fixtures.js";

const root = resolve(import.meta.dirname, "..");
const metadataPath = resolve(root, "validation", "reference-fixtures.json");
const ART_CNN_MACRO_BLOCK = /#extension\s+GL_EXT_shader_explicit_arithmetic_types_float16\s*:\s*enable\r?\n#ifdef\s+GL_EXT_shader_explicit_arithmetic_types_float16\r?\n#\s*define\s+V4\s+f16vec4\r?\n#\s*define\s+M4\s+f16mat4\r?\n#\s*define\s+F\s+float16_t\r?\n#else\r?\n#\s*define\s+V4\s+vec4\r?\n#\s*define\s+M4\s+mat4\r?\n#\s*define\s+F\s+float\r?\n#endif/g;
const ART_CNN_F32_BLOCK = [
  "// Reference oracle normalization: select the upstream f32 fallback.",
  "#define V4 vec4",
  "#define M4 mat4",
  "#define F float",
].join("\n");
const MPV_BASE_OPTIONS = REFERENCE_GENERATOR_POLICY.mpvBaseOptions;
const SSIM_OPTIONS = REFERENCE_GENERATOR_POLICY.ssimOptions;

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function commandText(command, args) {
  return [command, ...args].map((value) => (
    /^[A-Za-z0-9_./:=+-]+$/.test(value) ? value : JSON.stringify(value)
  )).join(" ");
}

async function run(command, args, { cwd = root, timeoutMs = 45_000 } = {}) {
  return new Promise((resolveRun, rejectRun) => {
    const child = spawn(command, args, {
      cwd,
      env: { ...process.env, LC_ALL: "C" },
      detached: process.platform !== "win32",
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    const timer = setTimeout(() => {
      try {
        if (process.platform === "win32") child.kill("SIGKILL");
        else process.kill(-child.pid, "SIGKILL");
      } catch {}
      rejectRun(new Error(`${command} timed out after ${timeoutMs} ms`));
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(timer);
      rejectRun(new Error(`${commandText(command, args)}: ${error.message}`));
    });
    child.once("close", (code, signal) => {
      clearTimeout(timer);
      const result = {
        code,
        signal,
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8"),
      };
      if (code !== 0) {
        rejectRun(new Error(
          `${commandText(command, args)} exited ${code ?? signal}\n${result.stdout}${result.stderr}`,
        ));
      } else {
        resolveRun(result);
      }
    });
  });
}

function normalizeArtCnn(source, expectedBlocks) {
  let count = 0;
  const normalized = source.replace(ART_CNN_MACRO_BLOCK, () => {
    count++;
    return ART_CNN_F32_BLOCK;
  });
  if (count !== expectedBlocks) {
    throw new Error(`ArtCNN f32 normalization replaced ${count} macro blocks; expected ${expectedBlocks}`);
  }
  if (/GL_EXT_shader_explicit_arithmetic_types_float16|\bf16(?:vec|mat)|\bfloat16_t\b/.test(normalized)) {
    throw new Error("ArtCNN f32 normalization left a float16 feature declaration behind");
  }
  return normalized;
}

function caseMpvOptions(entry) {
  return entry.id === "filter:ssimds-reference" ? [...SSIM_OPTIONS] : [];
}

function inspectMpvLog(entry, output) {
  const failures = [
    /Failed compiling/iu,
    /Failed executing hook/iu,
    /disabling[^\n]*hook/iu,
    /shader[^\n]*(?:failed|error)/iu,
    /(?:failed|error)[^\n]*shader/iu,
  ];
  for (const pattern of failures) {
    const match = output.match(pattern);
    if (match) throw new Error(`${entry.id}: mpv reported shader failure: ${match[0]}`);
  }
}

async function toolchainVersions() {
  const [mpv, ffmpeg, glxinfo] = await Promise.all([
    run("mpv", ["--version"], { timeoutMs: 10_000 }),
    run("ffmpeg", ["-version"], { timeoutMs: 10_000 }),
    run("xvfb-run", ["-a", "glxinfo", "-B"], { timeoutMs: 10_000 }),
  ]);
  const mpvVersion = mpv.stdout.match(/^mpv\s+(v[^\s]+)/mu)?.[1];
  const libplaceboVersion = mpv.stdout.match(/^libplacebo version:\s*(v[^\s]+)/mu)?.[1];
  const ffmpegVersion = ffmpeg.stdout.match(/^ffmpeg version\s+([^\s]+)/mu)?.[1];
  const renderer = glxinfo.stdout.match(/^OpenGL renderer string:\s*(.+)$/mu)?.[1];
  const mesa = glxinfo.stdout.match(/^OpenGL core profile version string:.*\bMesa\s+([^\s]+)$/mu)?.[1];
  const acceleratedText = glxinfo.stdout.match(/^\s*Accelerated:\s*(yes|no)$/mu)?.[1];
  if (!mpvVersion || !libplaceboVersion || !ffmpegVersion || !renderer ||
      !mesa || !acceleratedText) {
    throw new Error("Could not parse the mpv, libplacebo, FFmpeg, and Mesa software-renderer versions");
  }
  return {
    mpv: mpvVersion,
    libplacebo: libplaceboVersion,
    ffmpeg: ffmpegVersion,
    renderer,
    mesa,
    accelerated: acceleratedText === "yes",
  };
}

function inputMetadata(role, id) {
  const spec = REFERENCE_INPUTS[id];
  if (!spec) throw new Error(`unknown reference input ${id}`);
  const encoded = encodeReferenceInput(id);
  return {
    role,
    id,
    width: spec.width,
    height: spec.height,
    channels: spec.channels,
    encoding: spec.encoding,
    formula: spec.formula,
    portableAnymapByteLength: encoded.byteLength,
    portableAnymapSha256: sha256(encoded),
  };
}

async function writeAtomic(path, bytes) {
  await mkdir(dirname(path), { recursive: true });
  const next = `${path}.next-${process.pid}`;
  await writeFile(next, bytes);
  await rename(next, path);
}

function inspectGrayscaleReference(entry, bytes) {
  const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  let mismatchCount = 0;
  let maxChannelDelta = 0;
  for (let pixel = 0; pixel < words.length / 3; pixel++) {
    const offset = pixel * 3;
    const minimum = Math.min(words[offset], words[offset + 1], words[offset + 2]);
    const maximum = Math.max(words[offset], words[offset + 1], words[offset + 2]);
    const delta = maximum - minimum;
    if (delta) mismatchCount++;
    maxChannelDelta = Math.max(maxChannelDelta, delta);
  }
  if (maxChannelDelta > 2) {
    throw new Error(`${entry.id}: grayscale screenshot channel spread is ${maxChannelDelta} units`);
  }
  return { mismatchCount, maxChannelDelta };
}

async function generateMpvFixture(entry, workspace, scriptPath, inputPaths) {
  const sourceBytes = await readFile(resolve(root, entry.source.path));
  const actualSourceSha = sha256(sourceBytes);
  if (actualSourceSha !== entry.source.sha256) {
    throw new Error(`${entry.source.path}: SHA-256 ${actualSourceSha}, expected ${entry.source.sha256}`);
  }

  let shaderPath = resolve(root, entry.source.path);
  if (entry.oracle.kind === "mpv-libplacebo-artcnn-f32") {
    const normalized = normalizeArtCnn(
      sourceBytes.toString("utf8"),
      entry.oracle.normalizedMacroBlocks,
    );
    shaderPath = join(workspace, `${entry.id}.f32.glsl`);
    await writeFile(shaderPath, normalized);
  }

  const screenshotPath = join(workspace, `${entry.id.replaceAll(":", "-")}.png`);
  const rawPath = join(workspace, `${entry.id.replaceAll(":", "-")}.rgb16le`);
  const inputPath = inputPaths.get(entry.inputs[0].id);
  const args = [
    ...MPV_BASE_OPTIONS,
    ...caseMpvOptions(entry),
    `--geometry=${entry.output.width}x${entry.output.height}`,
    `--script=${scriptPath}`,
    `--script-opts=reference-output=${screenshotPath},reference-delay=0.35`,
    `--glsl-shader=${shaderPath}`,
    inputPath,
  ];
  const result = await run("xvfb-run", ["-a", "mpv", ...args]);
  const mpvOutput = `${result.stdout}\n${result.stderr}`;
  inspectMpvLog(entry, mpvOutput);

  const probe = await run("ffprobe", [
    "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=width,height,pix_fmt,bits_per_raw_sample",
    "-of", "json",
    screenshotPath,
  ], { timeoutMs: 10_000 });
  const stream = JSON.parse(probe.stdout).streams?.[0];
  if (stream?.width !== entry.output.width || stream?.height !== entry.output.height) {
    throw new Error(
      `${entry.id}: screenshot is ${stream?.width ?? "?"}x${stream?.height ?? "?"}; ` +
      `expected ${entry.output.width}x${entry.output.height}`,
    );
  }
  if (!/^(?:rgb48|rgba64)(?:be|le)$/u.test(stream?.pix_fmt || "")) {
    throw new Error(
      `${entry.id}: screenshot pixel format is ${stream?.pix_fmt ?? "unknown"}; ` +
      "expected 16-bit RGB or RGBA",
    );
  }

  await run("ffmpeg", [
    "-v", "error",
    "-y",
    "-i", screenshotPath,
    "-frames:v", "1",
    "-pix_fmt", "rgb48le",
    "-f", "rawvideo",
    rawPath,
  ]);
  const bytes = await readFile(rawPath);
  const expectedLength = entry.output.width * entry.output.height * 3 * 2;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(`${entry.id}: fixture has ${bytes.byteLength} bytes; expected ${expectedLength}`);
  }
  const channelDiagnostics = /^(?:FSRCNNX|ArtCNN)/u.test(entry.id)
    ? inspectGrayscaleReference(entry, bytes)
    : null;

  return {
    bytes,
    channelDiagnostics,
    command: commandText("xvfb-run", ["-a", "mpv", ...args.map((value) => (
      value === shaderPath ? `<shader:${entry.source.path}>` :
      value === inputPath ? `<input:${entry.inputs[0].id}>` :
      value.replaceAll(workspace, "<temporary-directory>").replaceAll(root, "<repository>")
    ))]),
  };
}

const workspace = await mkdtemp(join(tmpdir(), "fsrcnnx-reference-fixtures-"));
try {
  const toolchain = await toolchainVersions();
  if (JSON.stringify(toolchain) !== JSON.stringify(REFERENCE_TOOLCHAIN)) {
    throw new Error(
      "Reference toolchain differs from the audited policy; review and update " +
      "REFERENCE_TOOLCHAIN before blessing new oracle bytes\n" +
      `detected: ${JSON.stringify(toolchain)}`,
    );
  }
  const scriptPath = join(workspace, "reference-capture.lua");
  await writeFile(scriptPath, [
    'local opts = require "mp.options"',
    'local config = { output = "reference.png", delay = 0.35 }',
    'opts.read_options(config, "reference")',
    '',
    'mp.register_event("file-loaded", function()',
    '    mp.add_timeout(config.delay, function()',
    '        mp.commandv("screenshot-to-file", config.output, "window")',
    '        mp.commandv("quit", 0)',
    '    end)',
    'end)',
    '',
  ].join("\n"));

  const inputPaths = new Map();
  for (const id of new Set(REFERENCE_CASES.flatMap((entry) => entry.inputs.map((input) => input.id)))) {
    const spec = REFERENCE_INPUTS[id];
    const extension = spec.channels === 1 ? "pgm" : "ppm";
    const path = join(workspace, `${id}.${extension}`);
    await writeFile(path, encodeReferenceInput(id));
    inputPaths.set(id, path);
  }

  const generated = new Map();
  const metadataCases = [];
  for (const entry of REFERENCE_CASES) {
    const source = entry.source ? { ...entry.source } : null;
    const inputs = entry.inputs.map(({ role, id }) => inputMetadata(role, id));
    if (entry.output.kind === "fixture") {
      const { sha256: auditedSha256, ...output } = entry.output;
      const result = await generateMpvFixture(entry, workspace, scriptPath, inputPaths);
      const digest = sha256(result.bytes);
      if (digest !== auditedSha256) {
        throw new Error(
          `${entry.id}: generated SHA-256 ${digest}, expected audited ${auditedSha256}; ` +
          "inspect the numerical change before updating the canonical digest",
        );
      }
      generated.set(entry.output.path, result.bytes);
      metadataCases.push({
        id: entry.id,
        label: entry.label,
        source,
        inputs,
        output: {
          ...output,
          channels: 3,
          byteLength: result.bytes.byteLength,
          sha256: digest,
          ...(result.channelDiagnostics ? { channelDiagnostics: result.channelDiagnostics } : {}),
        },
        oracle: {
          ...entry.oracle,
          capture: "window",
          options: caseMpvOptions(entry),
          command: result.command,
        },
        tolerances: { ...entry.tolerances },
      });
    } else {
      metadataCases.push({
        id: entry.id,
        label: entry.label,
        source,
        inputs,
        output: { ...entry.output },
        oracle: { ...entry.oracle },
        tolerances: { ...entry.tolerances },
      });
    }
  }

  const metadata = {
    schemaVersion: REFERENCE_FIXTURE_SCHEMA_VERSION,
    inputVersion: REFERENCE_INPUT_VERSION,
    generator: { ...REFERENCE_GENERATOR_POLICY },
    toolchain: {
      ...toolchain,
    },
    cases: metadataCases,
  };

  for (const [path, bytes] of generated) await writeAtomic(resolve(root, path), bytes);
  await writeAtomic(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
  console.log(`Reference fixtures: generated ${generated.size} binaries for ${metadataCases.length} cases`);
} finally {
  await rm(workspace, { recursive: true, force: true });
}
