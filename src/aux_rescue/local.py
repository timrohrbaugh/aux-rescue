"""Rescue auxiliary tensors into a local model directory."""
from __future__ import annotations

from pathlib import Path

from aux_rescue.core import (
    AUX_OUT_NAME,
    INDEX_NAME,
    OrphanReport,
    diff_orphans,
    extract_orphans,
    patch_local_index,
    write_aux_file,
)


def rescue_local(
    source: str,
    dest_dir: str,
    token: str | None = None,
    out_name: str = AUX_OUT_NAME,
    dry_run: bool = False,
    include_prefix: list[str] | None = None,
    exclude_prefix: list[str] | None = None,
) -> OrphanReport:
    """Restore orphan tensors from ``source`` into local ``dest_dir``.

    Writes ``out_name`` (default ``model-auxiliary.safetensors``) into
    ``dest_dir`` and registers the new keys in
    ``model.safetensors.index.json``. If the destination already contains
    ``out_name``, the call is a no-op.

    ``source`` may be a local directory or an HF repo id (optionally with
    ``hf://`` prefix). Source shards are downloaded on demand.

    ``include_prefix`` / ``exclude_prefix`` filter which orphan keys are
    rescued — useful when the loader renames a tensor and ``diff_orphans``
    flags the renamed copy as missing.

    ``dry_run=True`` runs the diff and returns the report without writing.
    """
    report = diff_orphans(
        source, dest_dir, token=token,
        include_prefix=include_prefix, exclude_prefix=exclude_prefix,
    )
    if not report.has_orphans:
        return report
    if dry_run:
        return report

    dst = Path(dest_dir)
    out_path = dst / out_name
    if out_path.exists():
        print(f"note: {out_name} already present in {dest_dir}; skipping")
        return report

    extracted = extract_orphans(source, report.orphans_by_shard, token=token)
    if not extracted:
        return report

    write_aux_file(extracted, out_path)
    patch_local_index(dst, out_name, list(extracted.keys()))

    print(
        f"rescued {len(extracted)} tensor(s) into {out_path} "
        f"and registered them in {INDEX_NAME}"
    )
    return report
