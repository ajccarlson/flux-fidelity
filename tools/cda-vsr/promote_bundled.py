#!/usr/bin/env python3
"""Promote the exact reviewed CDA-VSR exports into shipping ONNX artifacts.

The conversion receipt intentionally describes arbitrary exports as local-only.
This separate, deterministic step changes only model metadata for the two
canonical graphs selected for the extension package. It does not claim that
upstream supplied a license or redistribution permission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_GRAPHS = {
    "initializer": {
        "file": "cda-vsr-initializer.onnx",
        "sha256": "7d688658a2acdf249d5224dac7c6d4cdad8764ecc02e34084bc0306cabf3ac0d",
        "promoted_bytes": 4_174_053,
        "promoted_sha256":
            "7773490658a7cad663e9b4f7e9cc8269b3d0c7a9a8e5840ec3151e895c1161f1",
    },
    "recurrent": {
        "file": "cda-vsr-recurrent.onnx",
        "sha256": "c1c69f1163f2d83bfa8af40ed69edc9cfc962d50e86c79c31f2019cee7c7af24",
        "promoted_bytes": 2_758_968,
        "promoted_sha256":
            "442be6f8d356889070ed70acdb49f9d2d77f24b6947e51e823404ca5a6d66a05",
    },
}

EXPECTED_SOURCE_METADATA = {
    "fsrcnnx.architecture": "CDA-VSR",
    "fsrcnnx.checkpoint_sha256":
        "afc8745b890289ae421c500279d9ccf2a27c92cf3e71133b20840c7816e86d3e",
    "fsrcnnx.precision_profile": "mixed-fp16",
    "fsrcnnx.shipping_catalog": "false",
    "fsrcnnx.source_sha256":
        "0defb80e5fcbaa2abd0eb9cbc4f4f2050a68e94fa6f743aa48a785cc734fd87b",
    "fsrcnnx.spatial_shape": "dynamic",
}

PROMOTED_METADATA = {
    "fsrcnnx.distribution_status": "repository-owner-accepted-risk",
    "fsrcnnx.shipping_catalog": "true",
    "fsrcnnx.upstream_license": "not-specified",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata(model) -> dict[str, str]:
    values = {}
    for item in model.metadata_props:
        if item.key in values:
            raise RuntimeError(f"duplicate ONNX metadata key: {item.key}")
        values[item.key] = item.value
    return values


def set_metadata(model, values: dict[str, str]) -> None:
    del model.metadata_props[:]
    for key, value in sorted(values.items()):
        item = model.metadata_props.add()
        item.key = key
        item.value = value


def graph_identity(model) -> bytes:
    clone = type(model)()
    clone.CopyFrom(model)
    del clone.metadata_props[:]
    return clone.SerializeToString(deterministic=True)


def promote_graph(onnx, role: str, source_path: Path) -> bytes:
    source_bytes = source_path.read_bytes()
    expected = SOURCE_GRAPHS[role]
    actual_hash = sha256(source_bytes)
    if actual_hash != expected["sha256"]:
        raise RuntimeError(
            f"{source_path}: SHA-256 {actual_hash}, expected {expected['sha256']}"
        )

    model = onnx.load_model_from_string(source_bytes)
    onnx.checker.check_model(model, full_check=True)
    source_identity = graph_identity(model)
    values = metadata(model)
    for key, expected_value in {
        **EXPECTED_SOURCE_METADATA,
        "fsrcnnx.graph_role": role,
    }.items():
        if values.get(key) != expected_value:
            raise RuntimeError(
                f"{source_path}: metadata {key}={values.get(key)!r}, "
                f"expected {expected_value!r}"
            )

    values.update(PROMOTED_METADATA)
    values["fsrcnnx.parent_export_sha256"] = expected["sha256"]
    set_metadata(model, values)
    onnx.checker.check_model(model, full_check=True)
    if graph_identity(model) != source_identity:
        raise RuntimeError(f"{source_path}: promotion changed graph content")

    output = model.SerializeToString(deterministic=True)
    reloaded = onnx.load_model_from_string(output)
    onnx.checker.check_model(reloaded, full_check=True)
    if graph_identity(reloaded) != source_identity:
        raise RuntimeError(f"{source_path}: serialized promotion changed graph content")
    promoted = metadata(reloaded)
    for key, expected_value in {
        **PROMOTED_METADATA,
        "fsrcnnx.parent_export_sha256": expected["sha256"],
    }.items():
        if promoted.get(key) != expected_value:
            raise RuntimeError(
                f"{source_path}: promoted metadata {key} is not {expected_value!r}"
            )
    if (
        len(output) != expected["promoted_bytes"]
        or sha256(output) != expected["promoted_sha256"]
    ):
        raise RuntimeError(
            f"{source_path}: promoted bytes differ from the reviewed shipping artifact"
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote the exact canonical CDA-VSR export metadata for bundling."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the promoted graphs; otherwise only verify and report them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if source_dir == output_dir:
        raise RuntimeError("source and output directories must be different")
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "onnx is required; install tools/cda-vsr/requirements.txt"
        ) from error

    records = {}
    for role, expected in SOURCE_GRAPHS.items():
        source = source_dir / expected["file"]
        output = promote_graph(onnx, role, source)
        target = output_dir / expected["file"]
        if args.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(output)
        records[role] = {
            "file": expected["file"],
            "parent_sha256": expected["sha256"],
            "bytes": len(output),
            "sha256": sha256(output),
            "written": args.write,
        }
    print(json.dumps({"graphs": records}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
