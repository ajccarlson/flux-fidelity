#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(import.meta.dirname, "..");
const mediaDirectory = resolve(root, "tests/fixtures/browser/media");
const EXPECTED_FFMPEG_VERSION = "n7.1.1";
const EXPECTED_LIBVPX_VERSION = "1.15.0";

export const BROWSER_VIDEO_GENERATION_SPECS = Object.freeze([
  Object.freeze({
    id: "bt709-a",
    file: "bt709-a.webm",
    input: "testsrc2=size=160x90:rate=24",
    filter: null,
    width: 160,
    height: 90,
    profile: "0",
    pixelFormat: "yuv420p",
    primaries: "bt709",
    transfer: "bt709",
    matrix: "bt709",
  }),
  Object.freeze({
    id: "bt709-b",
    file: "bt709-b.webm",
    input: "testsrc2=size=128x72:rate=24",
    filter: "hue=h=45:s=1.1",
    width: 128,
    height: 72,
    profile: "0",
    pixelFormat: "yuv420p",
    primaries: "bt709",
    transfer: "bt709",
    matrix: "bt709",
  }),
  Object.freeze({
    id: "bt2020-pq",
    file: "bt2020-pq.webm",
    input: "testsrc2=size=128x72:rate=24",
    filter: "hue=h=90:s=0.9",
    width: 128,
    height: 72,
    profile: "2",
    pixelFormat: "yuv420p10le",
    primaries: "bt2020",
    transfer: "smpte2084",
    matrix: "bt2020nc",
  }),
  Object.freeze({
    id: "bt2020-sdr",
    file: "bt2020-sdr.webm",
    input: "testsrc2=size=128x72:rate=24",
    filter: "hue=h=135:s=1.0",
    width: 128,
    height: 72,
    profile: "0",
    pixelFormat: "yuv420p",
    primaries: "bt2020",
    transfer: "bt709",
    matrix: "bt2020nc",
  }),
]);

function command(name, args, { encoding = "utf8" } = {}) {
  const result = spawnSync(name, args, {
    cwd: root,
    encoding,
    stdio: encoding ? ["ignore", "pipe", "pipe"] : ["ignore", "pipe", "pipe"],
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const detail = String(result.stderr || result.stdout || "").trim();
    throw new Error(`${name} exited with ${result.status}${detail ? `: ${detail}` : ""}`);
  }
  return String(result.stdout || "").trim();
}

function toolchain() {
  const ffmpeg = command("ffmpeg", ["-hide_banner", "-version"]).split("\n", 1)[0];
  const ffmpegVersion = ffmpeg.match(/^ffmpeg version (\S+)/)?.[1] || null;
  const libvpxVersion = command("pkg-config", ["--modversion", "vpx"]);
  return { ffmpegVersion, libvpxVersion };
}

function generate(spec) {
  const output = resolve(mediaDirectory, spec.file);
  const frameFilters = [
    spec.filter,
    `format=${spec.pixelFormat}`,
    `setparams=range=limited:color_primaries=${spec.primaries}:` +
      `color_trc=${spec.transfer}:colorspace=${spec.matrix}`,
  ].filter(Boolean).join(",");
  const args = [
    "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
    "-f", "lavfi", "-i", spec.input,
    "-vf", frameFilters,
    "-frames:v", "48", "-map", "0:v:0", "-an", "-sn", "-dn",
    "-c:v", "libvpx-vp9", "-profile:v", spec.profile,
    "-pix_fmt", spec.pixelFormat, "-b:v", "0", "-crf", "32",
    "-deadline", "good", "-cpu-used", "0", "-threads", "1", "-row-mt", "0",
    "-tile-columns", "0", "-frame-parallel", "0", "-lag-in-frames", "0",
    "-auto-alt-ref", "0", "-g", "24",
    "-color_primaries", spec.primaries, "-color_trc", spec.transfer,
    "-colorspace", spec.matrix, "-color_range", "tv",
    "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
    "-f", "webm", output,
  ];
  command("ffmpeg", args);
  const probe = JSON.parse(command("ffprobe", [
    "-v", "error", "-count_frames", "-select_streams", "v:0",
    "-show_entries",
    "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_read_frames,color_range,color_space,color_transfer,color_primaries:format=duration",
    "-of", "json", output,
  ]));
  const stream = probe.streams?.[0];
  const expected = {
    codec_name: "vp9",
    profile: `Profile ${spec.profile}`,
    width: spec.width,
    height: spec.height,
    pix_fmt: spec.pixelFormat,
    color_range: "tv",
    color_space: spec.matrix,
    color_transfer: spec.transfer,
    color_primaries: spec.primaries,
    r_frame_rate: "24/1",
    nb_read_frames: "48",
  };
  for (const [field, value] of Object.entries(expected)) {
    if (stream?.[field] !== value) {
      throw new Error(`${spec.id}: ffprobe ${field} is ${stream?.[field] ?? "missing"}, expected ${value}`);
    }
  }
  if (probe.format?.duration !== "2.000000") {
    throw new Error(`${spec.id}: duration is ${probe.format?.duration ?? "missing"}, expected 2.000000`);
  }
  const data = readFileSync(output);
  return {
    id: spec.id,
    path: `media/${spec.file}`,
    bytes: data.length,
    sha256: createHash("sha256").update(data).digest("hex"),
    probe,
    command: ["ffmpeg", ...args.slice(0, -1), `tests/fixtures/browser/media/${spec.file}`],
  };
}

function main() {
  if (!process.argv.includes("--write")) {
    throw new Error("Refusing to overwrite tracked fixtures without --write");
  }
  const versions = toolchain();
  if (versions.ffmpegVersion !== EXPECTED_FFMPEG_VERSION ||
      versions.libvpxVersion !== EXPECTED_LIBVPX_VERSION) {
    throw new Error(
      `Exact fixture reproduction requires ffmpeg ${EXPECTED_FFMPEG_VERSION} and ` +
      `libvpx ${EXPECTED_LIBVPX_VERSION}; found ${versions.ffmpegVersion || "unknown"} and ` +
      `${versions.libvpxVersion || "unknown"}`,
    );
  }
  mkdirSync(mediaDirectory, { recursive: true });
  const results = BROWSER_VIDEO_GENERATION_SPECS.map(generate);
  process.stdout.write(`${JSON.stringify({ toolchain: versions, clips: results }, null, 2)}\n`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); }
  catch (error) {
    console.error(error.message || String(error));
    process.exitCode = 1;
  }
}
