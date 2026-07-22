# RIFE FP16 reproduction

`reproduce-rife-fp16.py` derives the bundled FP16 model from the verified
`model/rife_v4.26.onnx` parent. It checks the parent hash, exact dependency
versions, initializer and tensor-attribute conversion, exact graph surgery,
graph validity, deterministic output hash, and numerical agreement with the
FP32 parent on a structured CPU inference vector.

The authoritative environment is CPython 3.11.15 on Linux x86_64 in the
following digest-pinned image. CI runs this same image. From the repository
root, reproduce and verify the checked-in artifact with:

```sh
docker run --rm \
  -v "$PWD:/workspace" -w /workspace \
  python@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 \
  sh tools/model-reproduction/check-rife-fp16.sh
```

The requirements file locks every wheel by hash. The script also fixes
`PYTHONHASHSEED`, timezone, and locale before conversion. Use `--write` only
inside this environment when deliberately regenerating the tracked artifact.
