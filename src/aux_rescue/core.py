"""Source-of-truth logic for detecting and extracting orphan safetensors tensors.

The two model layouts this handles:

  1. **Separate aux file** — e.g. ``model-mtp-layer.safetensors`` shipped
     alongside the main shards (Ling-2.6-flash, some EAGLE drafters).
  2. **Embedded keys** — e.g. Qwen3.6-27B keeps ``mtp.*`` tensors INSIDE
     ``model-00013-of-00015.safetensors`` and
     ``model-00015-of-00015.safetensors``, mixed with main weights.

Both layouts produce the same failure mode: ``AutoModelForCausalLM.from_pretrained``
maps tensor keys to parameters in the resolved model class, drops the keys it
can't place, and ``save_pretrained`` then writes only what's in ``state_dict()``.
The aux weights vanish on round-trip with no error.

This module computes the orphan set (source keys with no home in dest) and
extracts the corresponding tensors from source shards on demand.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


AUX_OUT_NAME = "model-auxiliary.safetensors"
INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
MARKER_NAME = ".aux_rescue.json"


def build_marker(
    source: str,
    rescued_keys: list[str],
    include_prefix: list[str] | None,
    exclude_prefix: list[str] | None,
    out_name: str,
) -> dict:
    """Build the contents of the ``.aux_rescue.json`` marker file.

    Written into rescued repos so downstream users (and tools) can detect
    that a rescue was applied, what was restored, and from where.
    """
    from datetime import datetime, timezone
    from aux_rescue import __version__

    return {
        "tool": "aux-rescue",
        "version": __version__,
        "homepage": "https://github.com/timrohrbaugh/aux-rescue",
        "rescued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "out_file": out_name,
        "tensor_count": len(rescued_keys),
        "include_prefix": include_prefix or [],
        "exclude_prefix": exclude_prefix or [],
        "sample_keys": rescued_keys[:10],
    }


@dataclass
class OrphanReport:
    """Summary of what's missing from dest relative to source.

    ``orphans_by_shard`` is the actionable structure: for each source shard
    that contains orphan keys, a list of the keys to extract.

    ``dest_only_keys`` are tensor names present in dest but not in source.
    These usually indicate a rename (e.g. ``NemotronHForCausalLM`` renaming
    ``backbone.embeddings.weight`` → ``model.embed_tokens.weight`` on load),
    in which case some "orphans" aren't truly missing — they were just
    renamed. The CLI surfaces this and recommends ``--include-prefix``.
    """
    source: str
    dest: str
    source_total: int = 0
    dest_total: int = 0
    orphan_keys: list[str] = field(default_factory=list)
    orphans_by_shard: dict[str, list[str]] = field(default_factory=dict)
    aux_files_in_source: list[str] = field(default_factory=list)
    dest_only_keys: list[str] = field(default_factory=list)

    @property
    def has_orphans(self) -> bool:
        return bool(self.orphan_keys)

    @property
    def likely_rename_present(self) -> bool:
        """True when dest has keys not in source — strong rename signal."""
        return bool(self.dest_only_keys)

    def prefix_groups(self, depth: int = 2) -> dict[str, int]:
        """Group orphan keys by their first ``depth`` dotted segments.

        Useful for showing users a high-level view (``mtp.layers``,
        ``model.vision_tower.encoder``, …) and for suggesting
        ``--include-prefix`` values.
        """
        groups: dict[str, int] = {}
        for k in self.orphan_keys:
            parts = k.split(".")
            prefix = ".".join(parts[:depth]) if len(parts) >= depth else k
            groups[prefix] = groups.get(prefix, 0) + 1
        return dict(sorted(groups.items(), key=lambda kv: -kv[1]))

    def summary(self) -> str:
        if not self.has_orphans:
            return (
                f"OK: dest contains all {self.source_total} source tensor keys; "
                f"no rescue needed."
            )
        lines = [
            f"Source: {self.source}",
            f"Dest:   {self.dest}",
            f"Source tensors: {self.source_total}",
            f"Dest tensors:   {self.dest_total}",
            f"Orphan tensors: {len(self.orphan_keys)} "
            f"(present in source, missing in dest)",
        ]
        if self.aux_files_in_source:
            lines.append(
                f"Source-only aux files: {', '.join(self.aux_files_in_source)}"
            )
        groups = self.prefix_groups(depth=2)
        if groups:
            lines.append("Orphan groups (by name prefix):")
            for prefix, count in groups.items():
                lines.append(f"  {prefix}.*  -> {count} tensor(s)")
        if self.orphans_by_shard:
            lines.append("By source shard:")
            for shard, keys in sorted(self.orphans_by_shard.items()):
                sample = ", ".join(keys[:3])
                more = f" ... (+{len(keys) - 3} more)" if len(keys) > 3 else ""
                lines.append(f"  {shard}: {len(keys)} key(s)  [{sample}{more}]")
        if self.dest_only_keys:
            lines.append("")
            lines.append(
                f"Note: dest has {len(self.dest_only_keys)} key(s) NOT in "
                f"source. This usually means the loader renamed tensors "
                f"on load (e.g. backbone.* -> model.*). Some 'orphans' may "
                f"be the same weights under their original name — rescuing "
                f"them would duplicate the embeddings. Use --include-prefix "
                f"to rescue only the genuinely-missing components."
            )
            for k in self.dest_only_keys[:5]:
                lines.append(f"  dest-only: {k}")
            if len(self.dest_only_keys) > 5:
                lines.append(
                    f"  ... (+{len(self.dest_only_keys) - 5} more dest-only)"
                )
        return "\n".join(lines)


def safetensors_keys(local_path: str | Path) -> set[str]:
    """Tensor names contained in a single safetensors file."""
    from safetensors import safe_open

    with safe_open(str(local_path), framework="pt") as f:
        return set(f.keys())


def _is_local(spec: str) -> bool:
    """A spec is local if it's an existing directory path."""
    return Path(spec).is_dir()


def _hub_repo(spec: str) -> str:
    """Strip ``hf://`` scheme prefix if present."""
    if spec.startswith("hf://"):
        return spec[len("hf://"):]
    return spec


def source_weight_map(source: str, token: str | None = None) -> dict[str, str]:
    """Return ``{tensor_name: shard_filename}`` for ``source``.

    ``source`` is either a local directory or an HF repo id (with optional
    ``hf://`` prefix). Reads ``model.safetensors.index.json`` if present;
    falls back to a single ``model.safetensors``.
    """
    if _is_local(source):
        src = Path(source)
        idx = src / INDEX_NAME
        if idx.exists():
            return dict(json.loads(idx.read_text()).get("weight_map", {}))
        single = src / SINGLE_NAME
        if single.exists():
            return {k: SINGLE_NAME for k in safetensors_keys(single)}
        return {}

    from huggingface_hub import hf_hub_download

    repo = _hub_repo(source)
    try:
        idx_local = hf_hub_download(repo, INDEX_NAME, token=token)
        return dict(json.loads(Path(idx_local).read_text()).get("weight_map", {}))
    except Exception:
        try:
            single = hf_hub_download(repo, SINGLE_NAME, token=token)
            return {k: SINGLE_NAME for k in safetensors_keys(single)}
        except Exception:
            return {}


def dest_keys(dest: str, token: str | None = None) -> set[str]:
    """Set of tensor keys in ``dest``'s safetensors files.

    For local: opens each file listed in the index and reads its keys.
    For Hub: reads the index's ``weight_map`` keys directly (no download
    of weight files needed).
    """
    if _is_local(dest):
        dst = Path(dest)
        idx = dst / INDEX_NAME
        if idx.exists():
            files = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        else:
            single = dst / SINGLE_NAME
            files = {SINGLE_NAME} if single.exists() else set()
        keys: set[str] = set()
        for fname in files:
            fpath = dst / fname
            if fpath.exists():
                keys.update(safetensors_keys(fpath))
        return keys

    from huggingface_hub import hf_hub_download

    repo = _hub_repo(dest)
    try:
        idx_local = hf_hub_download(repo, INDEX_NAME, token=token)
        return set(
            json.loads(Path(idx_local).read_text()).get("weight_map", {}).keys()
        )
    except Exception:
        try:
            single = hf_hub_download(repo, SINGLE_NAME, token=token)
            return safetensors_keys(single)
        except Exception:
            return set()


def _list_source_safetensors_files(source: str, token: str | None = None) -> list[str]:
    """List ``*.safetensors`` filenames at the top level of ``source``."""
    if _is_local(source):
        return sorted(p.name for p in Path(source).glob("*.safetensors"))
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo = _hub_repo(source)
    try:
        files = api.list_repo_files(repo)
    except Exception:
        return []
    return sorted(
        f for f in files
        if f.endswith(".safetensors") and "/" not in f
    )


def _passes_filters(
    name: str,
    include_prefix: list[str] | None,
    exclude_prefix: list[str] | None,
) -> bool:
    if include_prefix and not any(name.startswith(p) for p in include_prefix):
        return False
    if exclude_prefix and any(name.startswith(p) for p in exclude_prefix):
        return False
    return True


def diff_orphans(
    source: str,
    dest: str,
    token: str | None = None,
    include_prefix: list[str] | None = None,
    exclude_prefix: list[str] | None = None,
) -> OrphanReport:
    """Compute the orphan tensor set: keys in source but not in dest.

    Pure read operation. Use this for ``--check`` mode to see what would be
    rescued without making any changes.

    Both ``source`` and ``dest`` accept a local path or an HF repo id (with
    optional ``hf://`` scheme).

    ``include_prefix`` (if given) restricts rescue to keys starting with any
    of the listed prefixes. ``exclude_prefix`` removes keys starting with any
    listed prefix. ``include`` is applied first, then ``exclude``.

    The full unfiltered diff is always surfaced via ``dest_only_keys`` and
    the report's prefix groups so users can pick the right filter.
    """
    src_map = source_weight_map(source, token=token)
    dst_keys = dest_keys(dest, token=token)

    raw_orphans = sorted(set(src_map.keys()) - dst_keys)
    dest_only = sorted(dst_keys - set(src_map.keys()))

    orphans = [
        k for k in raw_orphans
        if _passes_filters(k, include_prefix, exclude_prefix)
    ]
    by_shard: dict[str, list[str]] = {}
    for k in orphans:
        by_shard.setdefault(src_map[k], []).append(k)

    src_files = set(_list_source_safetensors_files(source, token=token))
    dst_files_in_src = set(src_map.values())
    aux_only = sorted(src_files - dst_files_in_src)

    return OrphanReport(
        source=source,
        dest=dest,
        source_total=len(src_map),
        dest_total=len(dst_keys),
        orphan_keys=orphans,
        orphans_by_shard=by_shard,
        aux_files_in_source=aux_only,
        dest_only_keys=dest_only,
    )


def resolve_shard_path(
    source: str, shard_name: str, token: str | None = None
) -> str | None:
    """Return a local path to ``shard_name`` from ``source``.

    Downloads on demand if ``source`` is an HF repo id.
    """
    if _is_local(source):
        p = Path(source) / shard_name
        return str(p) if p.exists() else None
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(_hub_repo(source), shard_name, token=token)
    except Exception:
        return None


def extract_orphans(
    source: str,
    orphans_by_shard: dict[str, list[str]],
    token: str | None = None,
) -> dict:
    """Materialize orphan tensors from source shards.

    Returns ``{tensor_name: torch.Tensor}``. Each shard is opened at most once.
    Skips (with warning) any shard that fails to resolve or read.
    """
    from safetensors import safe_open

    extracted: dict = {}
    for shard_name, keys in orphans_by_shard.items():
        shard_path = resolve_shard_path(source, shard_name, token=token)
        if shard_path is None:
            print(
                f"warning: cannot locate source shard {shard_name}; "
                f"skipping {len(keys)} tensor(s)"
            )
            continue
        try:
            with safe_open(shard_path, framework="pt") as f:
                for k in keys:
                    extracted[k] = f.get_tensor(k)
        except Exception as e:
            print(f"warning: failed reading {shard_name}: {e}")
            continue
    return extracted


def write_aux_file(extracted: dict, out_path: str | Path) -> None:
    """Write extracted orphan tensors to a single safetensors file."""
    from safetensors.torch import save_file

    save_file(extracted, str(out_path))


def patch_local_index(
    dest_dir: str | Path,
    aux_file: str,
    new_keys: list[str],
) -> None:
    """Register ``new_keys`` → ``aux_file`` in the dest's index.

    If no index exists (single-file dest), synthesizes one covering both the
    existing ``model.safetensors`` and the new aux file so loaders that scan
    the index find everything.
    """
    dst = Path(dest_dir)
    idx_path = dst / INDEX_NAME
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        wm = idx.setdefault("weight_map", {})
        for k in new_keys:
            wm[k] = aux_file
        idx_path.write_text(json.dumps(idx, indent=2))
        return

    wm: dict[str, str] = {}
    single = dst / SINGLE_NAME
    if single.exists():
        for k in safetensors_keys(single):
            wm[k] = SINGLE_NAME
    for k in new_keys:
        wm[k] = aux_file
    idx_path.write_text(json.dumps({"metadata": {}, "weight_map": wm}, indent=2))
