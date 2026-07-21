#!/usr/bin/env python3
"""Export a community SR checkpoint (.pth/.safetensors) to the extension's
neural engine format: ONNX (dynamic dims, fp16 weights / fp32 IO) plus a
model/neural/manifest.json entry.

Usage:
  python export.py --pth 4xBHI_dat2_real.pth --key dat2_real --label "DAT2 4x real-web"

Any spandrel-supported architecture works (SPAN, RealPLKSR, DAT, ATD, DRCT,
ESRGAN, Compact/MoSR, HAT, ...). The pad-multiple (window-size constraint for
transformer archs) is auto-detected by probing awkward input dims.
"""
import argparse, json, os, sys

def log(*a): print("[export]", *a)

def load_pth(path):
    from spandrel import ModelLoader
    d = ModelLoader().load_from_file(path)
    if d.input_channels != 3 or d.output_channels != 3:
        sys.exit(f"need RGB 3->3 model, got {d.input_channels}->{d.output_channels}")
    log(f"arch={d.architecture.name} scale={d.scale}x")
    return d.model.eval(), d.scale, d.architecture.name

def export_onnx(model, scale, out_path, opset=17, fp16=True, dynamo=False):
    import torch
    dummy = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        # Legacy (TorchScript) exporter by default: torch 2.12's dynamo path
        # segfaults in decompositions on archs that assign fused weights in
        # forward (SPAN's Conv3XC). Legacy handles conv nets + dynamic_axes
        # cleanly and honors opset 17. --dynamo opts into the new path.
        torch.onnx.export(model, dummy, out_path, input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "oh", 3: "ow"}},
                          opset_version=opset, dynamo=dynamo)
    if fp16:
        import onnx
        from onnxconverter_common import float16
        m = onnx.load(out_path)
        m = float16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m, out_path)
    log(f"wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB, {'fp16 weights / fp32 IO' if fp16 else 'fp32'})")

def detect_pad_multiple(onnx_path, scale):
    """Probe awkward dims to find the arch's dim-divisibility requirement."""
    import numpy as np, onnxruntime as rt
    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    probes = [(1, 63, 47), (2, 50, 46), (4, 52, 44), (8, 56, 48), (16, 64, 48)]
    for mult, h, w in probes:
        try:
            x = np.random.rand(1, 3, h, w).astype(np.float32)
            y = sess.run(None, {"input": x})[0]
            assert y.shape == (1, 3, h * scale, w * scale), f"shape {y.shape}"
            assert np.isfinite(y).all(), "non-finite output (fp16 overflow?)"
            log(f"verify OK at {w}x{h} -> pad multiple {mult}; output shape {y.shape}")
            return mult
        except Exception as e:
            log(f"dims {w}x{h} rejected ({type(e).__name__}) — trying next multiple")
    sys.exit("model rejected all probe dims — export manually with a known padMultiple")

def upsert_manifest(out_dir, entry):
    path = os.path.join(out_dir, "manifest.json")
    lst = []
    if os.path.exists(path):
        j = json.load(open(path)); lst = j if isinstance(j, list) else j.get("models", [])
    lst = [m for m in lst if m.get("key") != entry["key"]] + [entry]
    json.dump(lst, open(path, "w"), indent=2)
    log(f"manifest updated: {path} ({len(lst)} models)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth"); ap.add_argument("--key"); ap.add_argument("--label")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "model", "neural"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--dynamo", action="store_true", help="use the new dynamo ONNX exporter (default: legacy TorchScript)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.pth: sys.exit("--pth required")
    model, scale, arch = load_pth(a.pth)
    key = a.key or os.path.splitext(os.path.basename(a.pth))[0].lower().replace("-", "_")
    label = a.label or f"{arch} {scale}x {key}"
    fname = f"{key}.fp16.onnx" if not a.no_fp16 else f"{key}.onnx"
    out_path = os.path.join(a.out, fname)
    export_onnx(model, scale, out_path, a.opset, fp16=not a.no_fp16, dynamo=a.dynamo)
    mult = detect_pad_multiple(out_path, scale)
    upsert_manifest(a.out, {"key": key, "label": label, "file": fname, "scale": scale,
                            "padMultiple": mult, "arch": arch})
    log("done — reload the extension; the model appears under Engine: Neural (ONNX)")

if __name__ == "__main__":
    main()
