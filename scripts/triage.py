"""Batch triage: scan a Hugging Face user/org's repos for save_pretrained drops.

For each repo:
  1. Identify the source/base model (cardData.base_model -> README link
     -> repo-name guesser).
  2. If a .aux_rescue.json marker is present, mark as already rescued.
  3. Otherwise run diff_orphans against the source and bucket the result:
       OK         - dest matches source, nothing missing
       NEED FIX   - has clean (no rename-overlap) orphans -> rescuable
       RENAME     - all orphans overlap with dest-only -> not auto-rescuable
       NO SOURCE  - couldn't identify the source model
       ERROR      - HF API or read failure

Prints a single-page table.

Usage:
    python scripts/triage.py --user trohrbaugh
    python scripts/triage.py --user trohrbaugh --output triage.tsv
    python scripts/triage.py --repos repo1 repo2 repo3
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from aux_rescue.core import diff_orphans


SUFFIX_PATTERNS = [
    r"-heretic-ara-uncensored$",
    r"-heretic-ara-FP8$",
    r"-heretic-ara-v\d+$",
    r"-heretic-ara$",
    r"-heretic-uncensored$",
    r"-heretic-V\d+$",
    r"-heretic-v\d+$",
    r"-heretic-nvfp4$",
    r"-heretic$",
    r"-uncensored$",
    r"-abliterated$",
    r"-decensored$",
    r"-FP8$",
    r"-nvfp4$",
]


def strip_known_suffixes(name: str) -> str:
    """Iteratively strip known heretic / quant suffixes from a model name."""
    prev = None
    cur = name
    while prev != cur:
        prev = cur
        for pat in SUFFIX_PATTERNS:
            cur = re.sub(pat, "", cur)
    return cur


KNOWN_PREFIXES: dict[str, list[str]] = {
    # repo-name prefix -> candidate orgs to try, in priority order
    "gemma": ["google"],
    "Gemma": ["google"],
    "Qwen": ["Qwen"],
    "qwen": ["Qwen"],
    "Llama": ["meta-llama"],
    "Mistral": ["mistralai"],
    "Phi": ["microsoft"],
    "granite": ["ibm-granite"],
    "Granite": ["ibm-granite"],
    "GLM": ["zai-org", "THUDM"],
    "DeepSeek": ["deepseek-ai"],
    "ERNIE": ["baidu"],
    "Seed-OSS": ["ByteDance"],
    "OmniCoder": ["allenai", "allenai-org"],
    "LFM2": ["LiquidAI"],
    "Ling": ["inclusionAI"],
    "NVIDIA-Nemotron": ["nvidia"],
    "Nemotron": ["nvidia"],
    "Stable-DiffCoder": ["stabilityai"],
}


def guess_source_from_name(repo_name: str) -> list[str]:
    """Generate ordered candidate ``org/model`` strings for a stripped name."""
    base = strip_known_suffixes(repo_name)
    candidates: list[str] = []
    for prefix, orgs in KNOWN_PREFIXES.items():
        if base.lower().startswith(prefix.lower()):
            for org in orgs:
                candidates.append(f"{org}/{base}")
    return candidates


URL_RE = re.compile(r"huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)")


def extract_source_from_readme(readme: str) -> list[str]:
    """Return all huggingface.co/X/Y references in order of appearance."""
    seen: set[str] = set()
    out: list[str] = []
    for m in URL_RE.finditer(readme):
        repo = m.group(1).strip(".,)")
        if repo not in seen and not repo.endswith("-heretic"):
            seen.add(repo)
            out.append(repo)
    return out


def hub_repo_exists(api, repo_id: str) -> bool:
    try:
        api.model_info(repo_id)
        return True
    except Exception:
        return False


@dataclass
class TriageResult:
    repo: str
    source: str | None
    status: str
    orphan_clean: int
    orphan_rename: int
    dest_only: int
    src_total: int
    dst_total: int
    has_marker: bool
    note: str = ""

    def status_label(self) -> str:
        if self.has_marker:
            return "RESCUED"
        return self.status


def resolve_source(api, info, readme_text: str | None) -> str | None:
    """Best-effort source resolution. Returns the repo id of source, or None."""
    cd = info.cardData or {}
    base = cd.get("base_model")
    candidates: list[str] = []

    if isinstance(base, str):
        candidates.append(base)
    elif isinstance(base, list):
        candidates.extend(b for b in base if isinstance(b, str))

    if readme_text:
        candidates.extend(extract_source_from_readme(readme_text))

    name_only = info.id.split("/", 1)[-1]
    candidates.extend(guess_source_from_name(name_only))

    if "ara" in name_only.lower() or "v2" in name_only.lower():
        for stripped in [
            re.sub(r"-Base$", "", c) for c in candidates if c.endswith("-Base")
        ]:
            candidates.append(stripped)

    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if hub_repo_exists(api, c):
            return c
    return None


def triage_one(api, repo_id: str, token: str | None = None) -> TriageResult:
    try:
        info = api.model_info(repo_id)
    except Exception as e:
        return TriageResult(
            repo=repo_id, source=None, status="ERROR",
            orphan_clean=0, orphan_rename=0, dest_only=0,
            src_total=0, dst_total=0, has_marker=False,
            note=f"model_info: {e}",
        )

    files = {s.rfilename for s in info.siblings or []}
    has_marker = ".aux_rescue.json" in files

    readme_text = None
    if "README.md" in files:
        try:
            from huggingface_hub import hf_hub_download
            readme_path = hf_hub_download(repo_id, "README.md", token=token)
            readme_text = Path(readme_path).read_text()
        except Exception:
            pass

    source = resolve_source(api, info, readme_text)
    if source is None:
        return TriageResult(
            repo=repo_id, source=None, status="NO SOURCE",
            orphan_clean=0, orphan_rename=0, dest_only=0,
            src_total=0, dst_total=0, has_marker=has_marker,
            note="could not identify base model",
        )

    try:
        report = diff_orphans(source, repo_id, token=token)
    except Exception as e:
        return TriageResult(
            repo=repo_id, source=source, status="ERROR",
            orphan_clean=0, orphan_rename=0, dest_only=0,
            src_total=0, dst_total=0, has_marker=has_marker,
            note=f"diff_orphans: {e}",
        )

    dest_only_tops = {k.split(".", 1)[0] for k in report.dest_only_keys}
    clean = sum(
        1 for k in report.orphan_keys
        if k.split(".", 1)[0] not in dest_only_tops
    )
    rename = len(report.orphan_keys) - clean

    if not report.has_orphans:
        status = "OK"
    elif clean > 0:
        status = "NEED FIX"
    else:
        status = "RENAME ONLY"

    return TriageResult(
        repo=repo_id, source=source, status=status,
        orphan_clean=clean, orphan_rename=rename,
        dest_only=len(report.dest_only_keys),
        src_total=report.source_total, dst_total=report.dest_total,
        has_marker=has_marker,
    )


def render_table(results: list[TriageResult]) -> str:
    name_w = min(48, max(len(r.repo) for r in results))
    src_w = min(40, max((len(r.source or "?") for r in results), default=10))
    header = (
        f"{'STATUS':<11}  {'REPO':<{name_w}}  {'SOURCE':<{src_w}}  "
        f"{'CLEAN':>5}  {'RENAM':>5}  {'EXTRA':>6}  {'NOTE'}"
    )
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda x: (
        {"NEED FIX": 0, "RENAME ONLY": 1, "OK": 2,
         "RESCUED": 3, "NO SOURCE": 4, "ERROR": 5}.get(x.status_label(), 6),
        x.repo,
    )):
        status = r.status_label()
        repo_short = r.repo if len(r.repo) <= name_w else r.repo[: name_w - 1] + "…"
        src_short = (r.source or "?")
        src_short = src_short if len(src_short) <= src_w else src_short[: src_w - 1] + "…"
        lines.append(
            f"{status:<11}  {repo_short:<{name_w}}  {src_short:<{src_w}}  "
            f"{r.orphan_clean:>5}  {r.orphan_rename:>5}  {r.dest_only:>6}  {r.note}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="triage")
    p.add_argument("--user", default=None,
                   help="List models from this HF user/org and triage all of them")
    p.add_argument("--repos", nargs="+", default=None,
                   help="Explicit list of repos to triage (alternative to --user)")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--token", default=None)
    p.add_argument("--output", default=None,
                   help="Also write results as TSV to this path")
    args = p.parse_args(argv)

    from huggingface_hub import HfApi, get_token
    token = args.token or get_token()
    api = HfApi(token=token)

    if args.repos:
        repo_ids = list(args.repos)
    elif args.user:
        models = list(api.list_models(author=args.user, limit=args.limit, token=token))
        repo_ids = [m.id for m in models]
    else:
        print("provide --user or --repos", file=sys.stderr)
        return 2

    print(f"Triaging {len(repo_ids)} repo(s)...")
    print()
    results: list[TriageResult] = []
    for i, rid in enumerate(repo_ids, start=1):
        print(f"  [{i}/{len(repo_ids)}] {rid}", flush=True)
        results.append(triage_one(api, rid, token=token))

    print()
    print(render_table(results))

    if args.output:
        rows = [
            "\t".join([
                "status", "repo", "source", "src_total", "dst_total",
                "clean_orphans", "rename_orphans", "dest_only", "note",
            ])
        ]
        for r in results:
            rows.append("\t".join([
                r.status_label(), r.repo, r.source or "",
                str(r.src_total), str(r.dst_total),
                str(r.orphan_clean), str(r.orphan_rename), str(r.dest_only),
                r.note,
            ]))
        Path(args.output).write_text("\n".join(rows) + "\n")
        print(f"\nTSV written to {args.output}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status_label()] = counts.get(r.status_label(), 0) + 1
    print()
    print("Summary:", "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
