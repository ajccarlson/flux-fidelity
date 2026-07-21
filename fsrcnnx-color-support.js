// fsrcnnx-color-support.js — fail-closed decoded-video color policy.
//
// WebGPU converts imported video frames into the requested target color space,
// but the enhancement models and presentation surfaces are intentionally limited
// to sRGB SDR.  WebCodecs is the only browser API that exposes the decoded
// frame's source color metadata.  Its fields are nullable and decoder detection
// is best-effort, so incomplete metadata must remain on the native video path.

export const SRGB_COLOR_SPACE = "srgb";

const SDR_TRANSFERS = new Set(["bt709", "iec61966-2-1"]);
const SUPPORTED_MATRICES = new Set(["bt709", "rgb"]);
// `pq` and `hlg` are the current WebCodecs enum spellings. Retain the H.273
// names exposed by some earlier implementations and test doubles so an HDR
// frame can never become fail-open because of vocabulary drift.
const HDR_TRANSFERS = new Set(["pq", "hlg", "smpte2084", "arib-std-b67"]);
const WIDE_GAMUT_PRIMARIES = new Set(["bt2020", "smpte432"]);

function field(value, name) {
  const candidate = value?.[name];
  return typeof candidate === "string" && candidate ? candidate : null;
}

function snapshot(value) {
  return Object.freeze({
    primaries: field(value, "primaries"),
    transfer: field(value, "transfer"),
    matrix: field(value, "matrix"),
    fullRange: typeof value?.fullRange === "boolean" ? value.fullRange : null,
  });
}

function result(supported, code, detail, colorSpace) {
  return Object.freeze({ supported, code, detail, colorSpace });
}

export function classifyVideoColorSpace(value) {
  const colorSpace = snapshot(value);

  if (HDR_TRANSFERS.has(colorSpace.transfer)) {
    return result(
      false,
      "color-hdr-unsupported",
      `HDR ${colorSpace.transfer.toUpperCase()} video is outside the sRGB SDR processing boundary.`,
      colorSpace,
    );
  }
  if (WIDE_GAMUT_PRIMARIES.has(colorSpace.primaries)) {
    return result(
      false,
      "color-wide-gamut-unsupported",
      `${colorSpace.primaries} video cannot be enhanced without losing wide-gamut color.`,
      colorSpace,
    );
  }

  if (!colorSpace.primaries || !colorSpace.transfer || !colorSpace.matrix ||
      colorSpace.fullRange === null) {
    return result(
      false,
      "color-metadata-unavailable",
      "Decoded-frame color metadata is incomplete, so color-safe enhancement cannot be verified.",
      colorSpace,
    );
  }

  if (colorSpace.primaries !== "bt709" ||
      !SDR_TRANSFERS.has(colorSpace.transfer) ||
      !SUPPORTED_MATRICES.has(colorSpace.matrix)) {
    return result(
      false,
      "color-space-unsupported",
      `Video color ${colorSpace.primaries}/${colorSpace.transfer}/${colorSpace.matrix} is outside the validated BT.709 SDR boundary.`,
      colorSpace,
    );
  }

  return result(
    true,
    "color-supported",
    `BT.709 SDR (${colorSpace.fullRange ? "full" : "limited"} range)`,
    colorSpace,
  );
}

export function probeVideoColorSupport(video, {
  VideoFrame: VideoFrameConstructor = globalThis.VideoFrame,
} = {}) {
  if (typeof VideoFrameConstructor !== "function") {
    return result(
      false,
      "color-metadata-unavailable",
      "This browser context does not expose decoded-frame color metadata.",
      snapshot(null),
    );
  }

  let frame = null;
  try {
    frame = new VideoFrameConstructor(video);
    return classifyVideoColorSpace(frame.colorSpace);
  } catch (error) {
    const name = typeof error?.name === "string" && error.name ? error.name : null;
    return Object.freeze({
      ...result(
        false,
        "color-metadata-unavailable",
        "Decoded-frame color metadata could not be read from this video.",
        snapshot(null),
      ),
      ...(name ? { errorName: name } : {}),
    });
  } finally {
    try { frame?.close?.(); } catch {}
  }
}
