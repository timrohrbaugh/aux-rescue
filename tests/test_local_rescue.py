"""Synthetic smoke tests covering both aux layouts.

Builds a fake "source" model directory with a known orphan layout, then a
fake "dest" representing what save_pretrained would have written. Runs
rescue_local and asserts the orphans land in model-auxiliary.safetensors
and the index registers them.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from aux_rescue import diff_orphans, rescue_local
from aux_rescue.core import AUX_OUT_NAME, INDEX_NAME


def _write_shard(path: Path, tensors: dict) -> None:
    from safetensors.torch import save_file
    save_file(tensors, str(path))


def _write_index(dir: Path, weight_map: dict) -> None:
    (dir / INDEX_NAME).write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2)
    )


def _t(*shape) -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.float32)


def test_separate_aux_file_layout(tmp_path: Path):
    """Ling-style: aux weights live in their own file."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    _write_shard(src / "model-00001-of-00002.safetensors", {
        "model.layers.0.weight": _t(4, 4),
    })
    _write_shard(src / "model-00002-of-00002.safetensors", {
        "model.layers.1.weight": _t(4, 4),
    })
    _write_shard(src / "model-mtp-layer.safetensors", {
        "mtp.layer.weight": _t(4, 4),
        "mtp.fc.weight": _t(4, 4),
    })
    _write_index(src, {
        "model.layers.0.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.weight": "model-00002-of-00002.safetensors",
        "mtp.layer.weight": "model-mtp-layer.safetensors",
        "mtp.fc.weight": "model-mtp-layer.safetensors",
    })

    _write_shard(dst / "model-00001-of-00002.safetensors", {
        "model.layers.0.weight": _t(4, 4),
    })
    _write_shard(dst / "model-00002-of-00002.safetensors", {
        "model.layers.1.weight": _t(4, 4),
    })
    _write_index(dst, {
        "model.layers.0.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.weight": "model-00002-of-00002.safetensors",
    })

    report = diff_orphans(str(src), str(dst))
    assert sorted(report.orphan_keys) == ["mtp.fc.weight", "mtp.layer.weight"]

    rescue_local(str(src), str(dst))

    aux = dst / AUX_OUT_NAME
    assert aux.exists()

    from safetensors import safe_open
    with safe_open(str(aux), framework="pt") as f:
        assert sorted(f.keys()) == ["mtp.fc.weight", "mtp.layer.weight"]

    idx = json.loads((dst / INDEX_NAME).read_text())["weight_map"]
    assert idx["mtp.layer.weight"] == AUX_OUT_NAME
    assert idx["mtp.fc.weight"] == AUX_OUT_NAME
    assert idx["model.layers.0.weight"] == "model-00001-of-00002.safetensors"


def test_embedded_aux_keys_layout(tmp_path: Path):
    """Qwen3.6-style: aux keys live inside main shards mixed with main weights.

    Critical case: dst received heretic-modified versions of model.layers.*
    tensors. Rescue must extract ONLY the mtp.* orphans, never overwrite
    the modified main weights.
    """
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    _write_shard(src / "model-00001-of-00002.safetensors", {
        "model.layers.0.weight": torch.ones(4, 4),
        "mtp.embed.weight": torch.full((4, 4), 7.0),
    })
    _write_shard(src / "model-00002-of-00002.safetensors", {
        "model.layers.1.weight": torch.ones(4, 4),
        "mtp.head.weight": torch.full((4, 4), 9.0),
    })
    _write_index(src, {
        "model.layers.0.weight": "model-00001-of-00002.safetensors",
        "mtp.embed.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.weight": "model-00002-of-00002.safetensors",
        "mtp.head.weight": "model-00002-of-00002.safetensors",
    })

    modified = torch.full((4, 4), 42.0)
    _write_shard(dst / "model-00001-of-00002.safetensors", {
        "model.layers.0.weight": modified,
    })
    _write_shard(dst / "model-00002-of-00002.safetensors", {
        "model.layers.1.weight": modified,
    })
    _write_index(dst, {
        "model.layers.0.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.weight": "model-00002-of-00002.safetensors",
    })

    rescue_local(str(src), str(dst))

    from safetensors import safe_open
    with safe_open(str(dst / "model-00001-of-00002.safetensors"), framework="pt") as f:
        assert torch.equal(f.get_tensor("model.layers.0.weight"), modified), \
            "Rescue must NOT overwrite heretic's modifications to main weights"
        assert "mtp.embed.weight" not in f.keys(), \
            "mtp.* must be moved to aux file, not duplicated in main shard"

    aux = dst / AUX_OUT_NAME
    assert aux.exists()
    with safe_open(str(aux), framework="pt") as f:
        assert sorted(f.keys()) == ["mtp.embed.weight", "mtp.head.weight"]
        assert torch.equal(f.get_tensor("mtp.embed.weight"), torch.full((4, 4), 7.0))
        assert torch.equal(f.get_tensor("mtp.head.weight"), torch.full((4, 4), 9.0))


def test_no_orphans_is_noop(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    tensors = {"model.layers.0.weight": _t(4, 4)}
    _write_shard(src / "model.safetensors", tensors)
    _write_shard(dst / "model.safetensors", tensors)

    report = diff_orphans(str(src), str(dst))
    assert not report.has_orphans

    rescue_local(str(src), str(dst))
    assert not (dst / AUX_OUT_NAME).exists()
    assert not (dst / INDEX_NAME).exists()
