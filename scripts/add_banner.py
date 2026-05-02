"""Insert (or refresh) an aux-rescue banner in a Hub repo's README.md.

Reads the rescue context from the repo's ``.aux_rescue.json`` marker
(written automatically by aux-rescue) so the banner is always consistent
with what was actually rescued. If the marker is absent, you can pass
``--source`` and ``--prefix`` explicitly.

The banner is delimited by HTML comment markers so re-runs replace
in-place rather than stacking. Existing READMEs without the banner
get one inserted just after the YAML frontmatter (or at the top if no
frontmatter exists).

Usage:
    python scripts/add_banner.py --repo trohrbaugh/my-model
    python scripts/add_banner.py --repo trohrbaugh/my-model --dry-run
    python scripts/add_banner.py --repo trohrbaugh/my-model --remove
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path


BANNER_BEGIN = "<!-- aux-rescue-banner: begin -->"
BANNER_END = "<!-- aux-rescue-banner: end -->"


def render_banner(
    marker: dict,
    symptom: str | None = None,
    marker_source: str | None = None,
) -> str:
    """Render the banner from a marker dict.

    Accepts both ``.aux_rescue.json`` (lost-aux-tensor rescue) and
    ``.moe_fuse.json`` (fused-MoE round-trip repair) shapes.
    """
    is_moe = marker.get("tool") == "moe-fuse" or marker_source == ".moe_fuse.json"
    homepage = marker.get("homepage", "https://github.com/timrohrbaugh/aux-rescue")
    source = marker.get("source", "<unknown source>")
    out_file = marker.get("out_file", "model-auxiliary.safetensors")

    if is_moe:
        date = marker.get("fused_at", "")[:10]
        n_layers = marker.get("layers_fused", 0)
        n_tensors = marker.get("tensors_fused", 0)
        if symptom is None:
            symptom = (
                "The LM body's MoE expert weights had been saved as "
                "per-expert nn.Linear tensors and could not be reloaded by "
                "the model class — the fused 3D parameters re-initialised "
                "to random at load time, producing garbage output."
            )
        body = [
            f"> ## ⚠️ Update {date} — please re-pull this model",
            ">",
            f"> An earlier upload of this repo had its MoE expert weights in",
            "> per-expert (unfused) form: separate `experts.{N}.gate_proj.weight`,",
            "> `up_proj.weight`, `down_proj.weight` instead of the fused 3D",
            "> `experts.gate_up_proj` / `experts.down_proj` tensors the loader",
            "> expects. PEFT/LoRA wrapping during abliteration converts fused",
            "> 3D tensors into per-expert Linear modules; `merge_and_unload` +",
            "> `save_pretrained` then writes them in that wrong layout.",
            ">",
            f"> **Symptom:** {symptom}",
            ">",
            f"> **Status:** fixed on {date} by re-fusing the per-expert tensors",
            f"> back into 3D form ({n_layers} layers, {n_tensors} fused tensors)",
            "> and patching `model.safetensors.index.json`. The repo now carries",
            "> a `.moe_fuse.json` marker file so this fact is programmatically",
            "> verifiable. The original per-expert keys remain in the main shards",
            "> as harmless 'unexpected' keys at load time.",
            ">",
            f"> Source used for shape reference: [`{source}`](https://huggingface.co/{source})",
            "> (heretic's abliteration is preserved — fusion uses the dest's",
            "> per-expert weights, not source's).",
            ">",
            f"> If you cloned this repo before {date}, please pull again.",
            f"> Repaired with [`moe-fuse`]({homepage}).",
        ]
    else:
        date = marker.get("rescued_at", "")[:10]
        count = marker.get("tensor_count", 0)
        prefixes = marker.get("include_prefix") or []
        prefix_desc = ", ".join(f"`{p}*`" for p in prefixes) if prefixes else "auxiliary"
        if symptom is None:
            symptom = (
                "Affected workflows that depend on these tensors will fail to "
                "load. Other workflows were unaffected."
            )
        body = [
            f"> ## ⚠️ Update {date} — please re-pull this model",
            ">",
            f"> An earlier upload of this repo was missing {prefix_desc} weights",
            f"> ({count} tensors) due to a bug in `save_pretrained`: tensors not",
            "> registered in the loaded model class's `state_dict()` were silently",
            "> dropped on save.",
            ">",
            f"> **Symptom:** {symptom}",
            ">",
            f"> **Status:** fixed on {date} by uploading `{out_file}` and",
            "> patching `model.safetensors.index.json`. The repo now carries an",
            "> `.aux_rescue.json` marker file so this fact is programmatically",
            "> verifiable.",
            ">",
            f"> Source used for restoration: [`{source}`](https://huggingface.co/{source})",
            ">",
            f"> If you cloned this repo before {date}, please pull again.",
            f"> Repaired with [`aux-rescue`]({homepage}).",
        ]

    return "\n".join([BANNER_BEGIN, ""] + body + ["", BANNER_END]) + "\n"


def insert_or_replace(readme: str, banner_block: str) -> str:
    """Insert banner just after frontmatter, or replace existing banner."""
    if BANNER_BEGIN in readme and BANNER_END in readme:
        pattern = re.compile(
            re.escape(BANNER_BEGIN) + r".*?" + re.escape(BANNER_END) + r"\n?",
            flags=re.DOTALL,
        )
        return pattern.sub(banner_block, readme, count=1)

    parts = readme.split("---\n", 2)
    if len(parts) == 3 and parts[0].strip() == "":
        return "---\n" + parts[1] + "---\n\n" + banner_block + "\n" + parts[2].lstrip("\n")
    return banner_block + "\n" + readme


def remove_banner(readme: str) -> str:
    pattern = re.compile(
        r"\n*" + re.escape(BANNER_BEGIN) + r".*?" + re.escape(BANNER_END) + r"\n*",
        flags=re.DOTALL,
    )
    return pattern.sub("\n", readme, count=1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="add_banner",
        description="Insert/refresh an aux-rescue banner in a Hub repo README.",
    )
    p.add_argument("--repo", required=True,
                   help="Hub repo id (e.g. trohrbaugh/my-model)")
    p.add_argument("--token", default=None,
                   help="HF token (default: cached login)")
    p.add_argument("--symptom", default=None,
                   help="Custom symptom paragraph (one or two sentences). "
                        "If absent, a generic one is used.")
    p.add_argument("--source", default=None,
                   help="Override source repo when no marker exists.")
    p.add_argument("--include-prefix", action="append", default=None,
                   help="Override prefixes when no marker exists.")
    p.add_argument("--tensor-count", type=int, default=None,
                   help="Override tensor count when no marker exists.")
    p.add_argument("--rescued-at", default=None,
                   help="Override rescued_at ISO timestamp when no marker exists.")
    p.add_argument("--remove", action="store_true",
                   help="Remove an existing banner instead of inserting one.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be uploaded; don't push.")
    args = p.parse_args(argv)

    from huggingface_hub import (
        get_token, hf_hub_download, upload_file,
    )
    token = args.token or get_token()

    try:
        readme_path = hf_hub_download(args.repo, "README.md", token=token)
        readme = Path(readme_path).read_text()
    except Exception as e:
        print(f"warning: could not fetch existing README ({e}); starting fresh")
        readme = "---\n---\n"

    if args.remove:
        new_readme = remove_banner(readme)
        commit_msg = "aux-rescue: remove banner"
    else:
        marker: dict | None = None
        marker_source: str | None = None
        for marker_name in (".aux_rescue.json", ".moe_fuse.json"):
            try:
                marker_path = hf_hub_download(
                    args.repo, marker_name, token=token,
                )
                marker = json.loads(Path(marker_path).read_text())
                marker_source = marker_name
                break
            except Exception:
                continue
        if marker is None:
            from datetime import datetime, timezone
            if not (args.source and args.include_prefix and args.tensor_count):
                print(
                    "no .aux_rescue.json on repo; pass --source / "
                    "--include-prefix / --tensor-count to override",
                    file=sys.stderr,
                )
                return 2
            marker = {
                "source": args.source,
                "include_prefix": args.include_prefix,
                "tensor_count": args.tensor_count,
                "rescued_at": (
                    args.rescued_at
                    or datetime.now(timezone.utc).isoformat(timespec="seconds")
                ),
                "out_file": "model-auxiliary.safetensors",
                "homepage": "https://github.com/timrohrbaugh/aux-rescue",
            }
        banner = render_banner(marker, symptom=args.symptom, marker_source=marker_source)
        new_readme = insert_or_replace(readme, banner)
        n_for_msg = (
            marker.get("tensors_fused")
            or marker.get("tensor_count")
            or "?"
        )
        tool = "moe-fuse" if marker_source == ".moe_fuse.json" else "aux-rescue"
        commit_msg = (
            f"{tool}: refresh banner ({n_for_msg} tensors)"
            if BANNER_BEGIN in readme
            else f"{tool}: add banner ({n_for_msg} tensors)"
        )

    if new_readme == readme:
        print("no change; nothing to upload")
        return 0

    if args.dry_run:
        print("--- new README (preview, first 80 lines) ---")
        for line in new_readme.splitlines()[:80]:
            print(line)
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(new_readme)
        path = f.name

    upload_file(
        path_or_fileobj=path,
        path_in_repo="README.md",
        repo_id=args.repo,
        token=token,
        commit_message=commit_msg,
    )
    print(f"updated README on {args.repo}: {commit_msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
