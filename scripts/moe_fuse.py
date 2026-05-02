"""moe-fuse: repair fused-MoE artifacts that lost their fused layout on save.

Problem this fixes
------------------
Some MoE models (Qwen3.5MoE, Qwen3.6MoE) ship fused 3D expert tensors on
disk:

    experts.gate_up_proj  shape [num_experts, 2*intermediate, hidden]
    experts.down_proj     shape [num_experts, hidden,         intermediate]

When heretic loads such a model, transformers exposes the experts as a
ModuleList of per-expert ``nn.Linear`` modules.  PEFT/LoRA wraps each
Linear, ``merge_and_unload`` bakes the LoRA in, ``state_dict()`` returns
per-expert keys (``experts.{N}.gate_proj.weight``, …), and
``save_pretrained`` writes them.

The same model class on the *same* transformers version cannot reload
that artifact: it expects fused 3D parameters and finds none.  Result:
80+ fused tensors initialise with random weights, the LM body is
effectively reset, and inference outputs garbage.

What this tool does
-------------------
Fuses the per-expert tensors **as found in the dest** (preserving
heretic's abliteration) into the source's expected fused layout, writes
them to a single new safetensors file, and patches the dest's
``model.safetensors.index.json`` so the loader finds them.  The
unfused per-expert keys remain in the dest's main shards as harmless
"unexpected" keys at load time.

Usage
-----

    # Diagnose
    python moe_fuse.py --source Qwen/Qwen3.5-35B-A3B \\
                       --dest trohrbaugh/Qwen3.5-35B-A3B-heretic --check

    # Fuse on Hub
    python moe_fuse.py --source Qwen/Qwen3.5-35B-A3B \\
                       --dest trohrbaugh/Qwen3.5-35B-A3B-heretic

    # Fuse a local directory
    python moe_fuse.py --source Qwen/Qwen3.5-35B-A3B \\
                       --dest /data/Qwen3-5-35B-A3B-Heretic
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FUSED_OUT_NAME = "model-moe-fused.safetensors"
INDEX_NAME = "model.safetensors.index.json"


@dataclass
class FusionPlan:
    """One layer's worth of fusion work."""
    parent_prefix: str  # e.g. "model.language_model.layers.0.mlp.experts."
    num_experts: int
    expected_gate_up_shape: tuple
    expected_down_shape: tuple
    gate_up_key: str  # e.g. "...experts.gate_up_proj"
    down_key: str     # e.g. "...experts.down_proj"


def _is_local(spec: str) -> bool:
    return Path(spec).is_dir()


def _hub_repo(spec: str) -> str:
    return spec[5:] if spec.startswith("hf://") else spec


def _read_index(spec: str, token: str | None = None) -> dict:
    if _is_local(spec):
        idx = Path(spec) / INDEX_NAME
        if idx.exists():
            return json.loads(idx.read_text())
        single = Path(spec) / "model.safetensors"
        if single.exists():
            with safe_open(str(single), framework="pt") as f:
                return {
                    "weight_map": {k: "model.safetensors" for k in f.keys()},
                    "metadata": {},
                }
        return {"weight_map": {}, "metadata": {}}
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(_hub_repo(spec), INDEX_NAME, token=token)
    return json.loads(Path(p).read_text())


def _resolve_shard(spec: str, fname: str, token: str | None = None) -> str | None:
    if _is_local(spec):
        p = Path(spec) / fname
        return str(p) if p.exists() else None
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(_hub_repo(spec), fname, token=token)
    except Exception:
        return None


def build_fusion_plans(
    source: str,
    dest: str,
    token: str | None = None,
    verify_shapes: bool = True,
) -> list[FusionPlan]:
    """Identify every layer that needs fusing.

    For every ``...experts.gate_up_proj`` (and matching ``.down_proj``) in
    source's index that's missing from dest's, walk back to the parent
    prefix and (when ``verify_shapes`` is True) read ONE example tensor
    from source to learn the fused layout.  All MoE layers are assumed
    to share that layout — true for every architecture we've encountered
    (Qwen3.5MoE, Qwen3.6MoE, …).

    ``verify_shapes=False`` is for fast --check: returns plans with empty
    expected shapes (still good enough for diagnosis).
    """
    src_idx = _read_index(source, token=token).get("weight_map", {})
    dst_idx = _read_index(dest, token=token).get("weight_map", {})

    fused_keys = sorted(
        k for k in src_idx
        if k.endswith(".experts.gate_up_proj") or k.endswith(".experts.down_proj")
    )
    if not fused_keys:
        return []

    layer_re = re.compile(r"^(.+\.experts\.)(gate_up_proj|down_proj)$")

    plans: dict[str, dict] = {}
    for k in fused_keys:
        m = layer_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        which = m.group(2)
        if k in dst_idx:
            continue
        plans.setdefault(prefix, {})[which] = k

    if not plans:
        return []

    sample_gate_up_shape: tuple = ()
    sample_down_shape: tuple = ()
    if verify_shapes:
        first_prefix = next(iter(plans))
        for which, target in (("gate_up_proj", None), ("down_proj", None)):
            full = plans[first_prefix].get(which)
            if not full:
                continue
            shard = src_idx.get(full)
            local = _resolve_shard(source, shard, token=token)
            if local is None:
                print(f"warning: cannot locate source shard {shard}; "
                      f"shape verification will be skipped")
                continue
            with safe_open(local, framework="pt") as f:
                if full in f.keys():
                    t = f.get_tensor(full)
                    if which == "gate_up_proj":
                        sample_gate_up_shape = tuple(t.shape)
                    else:
                        sample_down_shape = tuple(t.shape)

    num_experts = sample_gate_up_shape[0] if sample_gate_up_shape else 0

    out: list[FusionPlan] = []
    for prefix, tk in plans.items():
        gu = tk.get("gate_up_proj")
        dp = tk.get("down_proj")
        if not gu or not dp:
            print(f"warning: layer {prefix} has only one of gate_up_proj/down_proj "
                  f"in source; skipping (unusual)")
            continue
        out.append(FusionPlan(
            parent_prefix=prefix,
            num_experts=num_experts,
            expected_gate_up_shape=sample_gate_up_shape,
            expected_down_shape=sample_down_shape,
            gate_up_key=gu,
            down_key=dp,
        ))
    return out


def _open_dest_tensor(
    dest: str, dst_idx: dict, key: str, token: str | None = None,
) -> torch.Tensor | None:
    """Open one tensor from a multi-shard local or Hub dest."""
    shard = dst_idx.get(key)
    if shard is None:
        return None
    local = _resolve_shard(dest, shard, token=token)
    if local is None:
        return None
    with safe_open(local, framework="pt") as f:
        if key in f.keys():
            return f.get_tensor(key)
    return None


def fuse_one_layer(
    plan: FusionPlan,
    dest: str,
    dst_idx: dict,
    token: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build (gate_up_fused, down_fused) for one layer from per-expert tensors.

    Returns None if the dest does not contain the per-expert tensors for this
    layer.  This is not fatal: the caller skips the layer.  Common case: the
    layer is an MTP draft head whose source has fused experts but heretic
    didn't save it (it's an aux-rescue case, not a fusion case) — those will
    have to be aux-rescued separately.
    """
    gates: list[torch.Tensor] = []
    ups: list[torch.Tensor] = []
    downs: list[torch.Tensor] = []
    for e in range(plan.num_experts):
        g = _open_dest_tensor(
            dest, dst_idx,
            f"{plan.parent_prefix}{e}.gate_proj.weight", token=token,
        )
        u = _open_dest_tensor(
            dest, dst_idx,
            f"{plan.parent_prefix}{e}.up_proj.weight", token=token,
        )
        d = _open_dest_tensor(
            dest, dst_idx,
            f"{plan.parent_prefix}{e}.down_proj.weight", token=token,
        )
        if g is None or u is None or d is None:
            print(
                f"  SKIP: per-expert tensor missing at {plan.parent_prefix}{e} "
                f"in dest; this layer can't be fused (likely an aux-rescue "
                f"target — try aux-rescue --include-prefix {plan.parent_prefix.split('.')[0]}.)"
            )
            return None
        gates.append(g)
        ups.append(u)
        downs.append(d)

    # Per-expert gate_proj.weight shape [intermediate, hidden].
    # Fused gate_up_proj per expert is cat([gate, up], dim=0) shape
    # [2*intermediate, hidden].  Stack across experts -> [E, 2*I, H].
    gate_up = torch.stack(
        [torch.cat([gates[e], ups[e]], dim=0) for e in range(plan.num_experts)],
        dim=0,
    )

    # Per-expert down_proj.weight shape [hidden, intermediate].
    # Fused down_proj is just stack -> [E, H, I].
    down = torch.stack(downs, dim=0)

    if tuple(gate_up.shape) != plan.expected_gate_up_shape:
        print(
            f"  ERROR: gate_up shape mismatch at {plan.parent_prefix}: "
            f"got {tuple(gate_up.shape)}, expected {plan.expected_gate_up_shape}"
        )
        return None
    if tuple(down.shape) != plan.expected_down_shape:
        print(
            f"  ERROR: down shape mismatch at {plan.parent_prefix}: "
            f"got {tuple(down.shape)}, expected {plan.expected_down_shape}"
        )
        return None

    return gate_up, down


def patch_local_index(dest_dir: str, fused_file: str, new_keys: list[str]) -> None:
    idx_path = Path(dest_dir) / INDEX_NAME
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"weight_map": {}, "metadata": {}}
    wm = idx.setdefault("weight_map", {})
    for k in new_keys:
        wm[k] = fused_file
    idx_path.write_text(json.dumps(idx, indent=2))


def write_marker(
    dest_dir: str | None,
    repo_id: str | None,
    source: str,
    plans: list[FusionPlan],
    out_name: str,
    token: str | None = None,
) -> None:
    """Drop a .moe_fuse.json marker so the repair is programmatically detectable."""
    from datetime import datetime, timezone
    marker = {
        "tool": "moe-fuse",
        "homepage": "https://github.com/timrohrbaugh/aux-rescue",
        "fused_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "out_file": out_name,
        "layers_fused": len(plans),
        "tensors_fused": sum(2 for _ in plans),  # gate_up + down per layer
        "sample_keys": [p.gate_up_key for p in plans[:5]],
    }
    payload = json.dumps(marker, indent=2)
    if dest_dir is not None:
        (Path(dest_dir) / ".moe_fuse.json").write_text(payload)
    else:
        from huggingface_hub import upload_file
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(payload)
            p = f.name
        upload_file(
            path_or_fileobj=p,
            path_in_repo=".moe_fuse.json",
            repo_id=repo_id,
            token=token,
            commit_message=f"moe-fuse: write .moe_fuse.json marker ({len(plans)} layers)",
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="moe-fuse")
    p.add_argument("--source", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--check", action="store_true")
    p.add_argument("--token", default=None)
    p.add_argument("--out-name", default=FUSED_OUT_NAME)
    p.add_argument("--chunk-layers", type=int, default=10,
                   help="Layers per output file (memory ceiling). "
                        "Default: %(default)s. Larger MoE models need lower "
                        "values to avoid OOM (e.g. 5 for 122B-A10B).")
    args = p.parse_args(argv)

    from huggingface_hub import get_token
    token = args.token or get_token()

    src_local = _is_local(args.source)
    dst_local = _is_local(args.dest)
    print(f"source: {args.source}  ({'local' if src_local else 'hub'})")
    print(f"dest:   {args.dest}  ({'local' if dst_local else 'hub'})")
    print()

    print("Building fusion plan...")
    plans = build_fusion_plans(
        args.source, args.dest, token=token,
        verify_shapes=not args.check,
    )
    if not plans:
        print("OK: no fused-MoE keys missing in dest. Nothing to do.")
        return 0

    print(f"Found {len(plans)} layer(s) needing fusion:")
    print(f"  total tensors to fuse: {len(plans) * 2} "
          f"({len(plans)} gate_up_proj + {len(plans)} down_proj)")
    if plans[0].expected_gate_up_shape:
        print(f"  sample: {plans[0].parent_prefix}* "
              f"({plans[0].num_experts} experts, "
              f"gate_up shape={plans[0].expected_gate_up_shape})")
    else:
        print(f"  sample: {plans[0].parent_prefix}* (run without --check "
              f"to verify shapes against source)")

    if args.check:
        print("\n--check mode: no writes.")
        return 1

    dst_idx = _read_index(args.dest, token=token).get("weight_map", {})

    n_chunks = (len(plans) + args.chunk_layers - 1) // args.chunk_layers
    print(f"Will write {n_chunks} chunk file(s), {args.chunk_layers} layers each "
          f"(--chunk-layers={args.chunk_layers}).")

    if dst_local:
        from huggingface_hub import upload_file as _unused  # noqa
    else:
        from huggingface_hub import upload_file

    written_keys_to_file: dict[str, str] = {}
    skipped_layers: list[str] = []
    t0 = time.time()
    for chunk_idx in range(n_chunks):
        chunk_plans = plans[chunk_idx * args.chunk_layers : (chunk_idx + 1) * args.chunk_layers]
        chunk_name = (
            args.out_name if n_chunks == 1
            else args.out_name.replace(".safetensors", f"-{chunk_idx:03d}.safetensors")
        )
        print(f"\n[chunk {chunk_idx + 1}/{n_chunks}] {chunk_name}: "
              f"{len(chunk_plans)} layers")

        chunk_tensors: dict[str, torch.Tensor] = {}
        for i, plan in enumerate(chunk_plans, 1):
            print(f"  [{i}/{len(chunk_plans)}] fusing {plan.parent_prefix}* "
                  f"({plan.num_experts} experts)...", flush=True)
            out = fuse_one_layer(plan, args.dest, dst_idx, token=token)
            if out is None:
                skipped_layers.append(plan.parent_prefix)
                continue
            gate_up, down = out
            chunk_tensors[plan.gate_up_key] = gate_up.contiguous()
            chunk_tensors[plan.down_key] = down.contiguous()
            for k in (plan.gate_up_key, plan.down_key):
                written_keys_to_file[k] = chunk_name

        if not chunk_tensors:
            print(f"  (chunk has no fuseable layers; skipping write)")
            continue

        if dst_local:
            out_path = Path(args.dest) / chunk_name
            print(f"  writing {out_path}...")
            save_file(chunk_tensors, str(out_path))
        else:
            print(f"  uploading {chunk_name} to {args.dest}...")
            with tempfile.TemporaryDirectory() as td:
                tmp_path = Path(td) / chunk_name
                save_file(chunk_tensors, str(tmp_path))
                upload_file(
                    path_or_fileobj=str(tmp_path),
                    path_in_repo=chunk_name,
                    repo_id=_hub_repo(args.dest),
                    token=token,
                    commit_message=f"moe-fuse: chunk {chunk_idx + 1}/{n_chunks} "
                                   f"({len(chunk_tensors)} tensors)",
                )
        chunk_tensors.clear()
        import gc; gc.collect()

    if skipped_layers:
        print(f"\nSkipped {len(skipped_layers)} layer(s) (per-expert tensors "
              f"missing in dest):")
        for p in skipped_layers[:5]:
            print(f"  {p}")
        if len(skipped_layers) > 5:
            print(f"  ... and {len(skipped_layers) - 5} more")
        print("These need aux-rescue (not moe-fuse) to be restored.")

    print(f"\nFused all {len(written_keys_to_file)} tensors in "
          f"{time.time() - t0:.1f}s across {n_chunks} chunk(s)")

    print("Patching index...")
    if dst_local:
        idx_path = Path(args.dest) / INDEX_NAME
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"weight_map": {}, "metadata": {}}
        wm = idx.setdefault("weight_map", {})
        for k, fname in written_keys_to_file.items():
            wm[k] = fname
        idx_path.write_text(json.dumps(idx, indent=2))
        write_marker(
            dest_dir=args.dest, repo_id=None, source=args.source,
            plans=plans, out_name=args.out_name,
        )
        print(f"DONE. Patched {INDEX_NAME}, wrote .moe_fuse.json")
    else:
        idx = _read_index(args.dest, token=token)
        wm = idx.setdefault("weight_map", {})
        for k, fname in written_keys_to_file.items():
            wm[k] = fname
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(idx, indent=2))
            ip = f.name
        upload_file(
            path_or_fileobj=ip,
            path_in_repo=INDEX_NAME,
            repo_id=_hub_repo(args.dest),
            token=token,
            commit_message=f"moe-fuse: register {len(written_keys_to_file)} fused keys",
        )
        write_marker(
            dest_dir=None, repo_id=_hub_repo(args.dest), source=args.source,
            plans=plans, out_name=args.out_name, token=token,
        )
        print(f"DONE. Patched {INDEX_NAME}, wrote .moe_fuse.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
