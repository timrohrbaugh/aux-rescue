"""Rescue auxiliary tensors into an HF Hub repo."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from aux_rescue.core import (
    AUX_OUT_NAME,
    INDEX_NAME,
    MARKER_NAME,
    OrphanReport,
    _hub_repo,
    build_marker,
    diff_orphans,
    extract_orphans,
    write_aux_file,
)


def rescue_hub(
    source: str,
    dest_repo: str,
    token: str | None = None,
    out_name: str = AUX_OUT_NAME,
    dry_run: bool = False,
    commit_message: str | None = None,
    include_prefix: list[str] | None = None,
    exclude_prefix: list[str] | None = None,
) -> OrphanReport:
    """Restore orphan tensors from ``source`` into HF Hub repo ``dest_repo``.

    Uploads ``out_name`` to the repo and patches its
    ``model.safetensors.index.json`` so loaders that scan the index find the
    new tensors.

    Both ``source`` and ``dest_repo`` accept ``hf://`` prefixes. ``source``
    may also be a local directory.

    ``include_prefix`` / ``exclude_prefix`` filter which orphan keys are
    rescued — useful when the loader renames a tensor and ``diff_orphans``
    flags the renamed copy as missing.

    ``dry_run=True`` runs the diff and returns the report without uploading.
    """
    from huggingface_hub import get_token, hf_hub_download, upload_file

    if token is None:
        token = get_token()

    report = diff_orphans(
        source, dest_repo, token=token,
        include_prefix=include_prefix, exclude_prefix=exclude_prefix,
    )
    if not report.has_orphans:
        return report
    if dry_run:
        return report

    extracted = extract_orphans(source, report.orphans_by_shard, token=token)
    if not extracted:
        return report

    repo = _hub_repo(dest_repo)
    msg = commit_message or (
        f"aux-rescue: restore {len(extracted)} auxiliary tensor(s) "
        f"(MTP/draft/encoder heads)"
    )

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / out_name
        write_aux_file(extracted, out_path)

        upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=out_name,
            repo_id=repo,
            token=token,
            commit_message=msg,
        )

    try:
        idx_local = hf_hub_download(repo, INDEX_NAME, token=token)
        idx = json.loads(Path(idx_local).read_text())
    except Exception:
        idx = {"metadata": {}, "weight_map": {}}

    wm = idx.setdefault("weight_map", {})
    for k in extracted:
        wm[k] = out_name

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False,
    ) as fout:
        fout.write(json.dumps(idx, indent=2))
        new_idx_path = fout.name

    upload_file(
        path_or_fileobj=new_idx_path,
        path_in_repo=INDEX_NAME,
        repo_id=repo,
        token=token,
        commit_message=f"aux-rescue: register {len(extracted)} aux key(s) in index",
    )

    marker = build_marker(
        source=source,
        rescued_keys=list(extracted.keys()),
        include_prefix=include_prefix,
        exclude_prefix=exclude_prefix,
        out_name=out_name,
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False,
    ) as fout:
        fout.write(json.dumps(marker, indent=2))
        marker_path = fout.name
    upload_file(
        path_or_fileobj=marker_path,
        path_in_repo=MARKER_NAME,
        repo_id=repo,
        token=token,
        commit_message=f"aux-rescue: write {MARKER_NAME} marker",
    )

    print(
        f"rescued {len(extracted)} tensor(s) onto {repo} as {out_name}, "
        f"patched {INDEX_NAME}, wrote {MARKER_NAME}"
    )
    return report
