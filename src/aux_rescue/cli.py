"""Command-line entry point for aux-rescue.

Usage:
    aux-rescue --source <src> --dest <dst> [--check] [--token TOKEN]

Where ``<src>`` and ``<dst>`` are each one of:
  - a local directory path           (e.g. ./my-saved-model)
  - an HF Hub repo id                (e.g. Qwen/Qwen3.6-27B)
  - an HF Hub repo with explicit scheme (e.g. hf://Qwen/Qwen3.6-27B)

A spec is treated as Hub if the path doesn't exist as a local directory and
contains ``/`` (the standard HF ``user/model`` form), or if it has the
``hf://`` scheme. Otherwise it's treated as local.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from aux_rescue.core import OrphanReport


def _is_hub_spec(spec: str) -> bool:
    """True if ``spec`` should be treated as an HF repo id."""
    if spec.startswith("hf://"):
        return True
    if Path(spec).is_dir():
        return False
    return "/" in spec


def _quote(s: str) -> str:
    return shlex.quote(s)


def _suggest_next_steps(
    args: argparse.Namespace,
    report: OrphanReport,
    src_is_hub: bool,
    dst_is_hub: bool,
) -> str:
    """Build a friendly 'here's what to do' block for --check output."""
    if not report.has_orphans:
        return (
            "\nNothing to rescue. Your model has every tensor the source "
            "shipped — you're good."
        )

    fix_cmd = ["aux-rescue", "--source", _quote(args.source),
               "--dest", _quote(args.dest)]
    if args.token:
        fix_cmd += ["--token", "$HF_TOKEN"]

    suggestions: list[str] = []
    suggestions.append("")
    suggestions.append("=" * 78)
    suggestions.append("WHAT TO DO NEXT")
    suggestions.append("=" * 78)

    if report.likely_rename_present:
        # Compute prefix overlap between orphan groups and dest-only groups.
        # If they share a prefix, the orphans under that prefix are NOT
        # missing — they're renamed.
        orphan_groups = report.prefix_groups(depth=2)
        dest_only_groups: dict[str, int] = {}
        for k in report.dest_only_keys:
            parts = k.split(".")
            pfx = ".".join(parts[:2]) if len(parts) >= 2 else k
            dest_only_groups[pfx] = dest_only_groups.get(pfx, 0) + 1

        renamed_prefixes = set(orphan_groups) & set(dest_only_groups)
        clean_prefixes = [p for p in orphan_groups if p not in renamed_prefixes]

        suggestions.append(
            "\nThe dest has tensor names that aren't in the source. This "
            "usually means one of two things:\n"
            "  (a) The loader renamed tensors on load (e.g. backbone.* -> "
            "model.*) — some 'orphans' are the same weights under their\n"
            "      original names, and rescuing would duplicate them.\n"
            "  (b) The source and dest use different module wrappers "
            "(e.g. .linear.weight vs .weight) — same as (a) but at every\n"
            "      level of a subtree."
        )

        if renamed_prefixes:
            suggestions.append("")
            suggestions.append(
                f"Detected likely rename in: "
                f"{', '.join(sorted(renamed_prefixes))}.*"
            )
            suggestions.append(
                "Do NOT rescue these prefixes — the weights are already in "
                "dest under different names."
            )

        if not clean_prefixes:
            suggestions.append("")
            suggestions.append(
                "All orphan groups overlap with renamed dest keys. There "
                "is nothing safe to auto-rescue here."
            )
            suggestions.append(
                "Recommended: try loading the dest model with "
                "transformers and observe whether it warns about "
                "missing or unexpected keys."
            )
            suggestions.append(
                "If it loads cleanly, no action needed. If not, you'll "
                "want to re-run your save with a fixed pipeline that "
                "preserves the original wrapping, rather than rescuing."
            )
        else:
            suggestions.append("")
            suggestions.append(
                "Safe to rescue ONLY the prefixes below (no dest-only "
                "overlap):"
            )
            cmd_parts = []
            for p in clean_prefixes[:3]:
                cmd_parts.append(f"--include-prefix {p}.")
            cmd_with_filter = list(fix_cmd) + cmd_parts
            suggestions.append("")
            suggestions.append("    " + " ".join(cmd_with_filter))
    else:
        suggestions.append("\nRun this to fix it:\n")
        suggestions.append("    " + " ".join(fix_cmd))

    suggestions.append("")
    suggestions.append(
        "Tip: if you'd rather work on a local copy first, download the dest "
        "to your machine,\nrun aux-rescue against the local directory, "
        "verify, then re-upload:\n"
    )
    if dst_is_hub:
        suggestions.append(
            "    huggingface-cli download "
            f"{_quote(args.dest.removeprefix('hf://'))} "
            "--local-dir ./local-copy"
        )
        local_cmd = ["aux-rescue", "--source", _quote(args.source),
                     "--dest", "./local-copy"]
        if report.likely_rename_present:
            for p in list(report.prefix_groups(depth=2).keys())[:1]:
                local_cmd += [f"--include-prefix {p}."]
        suggestions.append("    " + " ".join(local_cmd))
        suggestions.append(
            "    # then push back: huggingface-cli upload "
            f"{_quote(args.dest.removeprefix('hf://'))} ./local-copy"
        )
    else:
        suggestions.append(
            "    # already running locally — just drop --check and re-run."
        )

    suggestions.append("")
    suggestions.append(
        "Don't have your model on Hugging Face? You can run aux-rescue "
        "purely locally:\n"
    )
    suggestions.append(
        "    aux-rescue --source <path-or-repo-of-original> "
        "--dest /path/to/your/saved/model"
    )
    suggestions.append("")
    suggestions.append(
        "Always re-test after rescue: load the model, run a forward pass "
        "(plus image/audio/MTP if applicable),\nand confirm it produces "
        "sensible output before publishing."
    )
    return "\n".join(suggestions)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="aux-rescue",
        description=(
            "Restore auxiliary safetensors weights (MTP, EAGLE, Medusa, "
            "vision/audio encoders) that AutoModelForCausalLM.save_pretrained "
            "silently dropped."
        ),
        epilog=(
            "Examples:\n"
            "  # Diagnose a Hub repo against its source:\n"
            "  aux-rescue --source Qwen/Qwen3.6-27B "
            "--dest myuser/my-finetune --check\n\n"
            "  # Repair a local directory you saved with save_pretrained:\n"
            "  aux-rescue --source Qwen/Qwen3.6-27B "
            "--dest ./my-saved-model\n\n"
            "  # Repair a Hub repo, scoped to the MTP head only:\n"
            "  aux-rescue --source nvidia/X --dest myuser/X-finetune "
            "--include-prefix mtp.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", required=True,
        help="Original model: local dir or HF repo id (e.g. Qwen/Qwen3.6-27B). "
             "Use hf:// prefix to force hub interpretation.",
    )
    p.add_argument(
        "--dest", required=True,
        help="Modified model to repair: local dir or HF repo id.",
    )
    p.add_argument(
        "--check", action="store_true",
        help="Read-only: report what's missing without writing anything. "
             "Exits 0 if clean, 1 if orphans found.",
    )
    p.add_argument(
        "--include-prefix", action="append", default=None, metavar="PREFIX",
        help="Rescue only tensor names starting with PREFIX (e.g. 'mtp.'). "
             "May be repeated.",
    )
    p.add_argument(
        "--exclude-prefix", action="append", default=None, metavar="PREFIX",
        help="Skip tensor names starting with PREFIX. May be repeated. "
             "Applied after --include-prefix.",
    )
    p.add_argument(
        "--token", default=None,
        help="HF token (otherwise uses HF_TOKEN env or cached login).",
    )
    p.add_argument(
        "--out-name", default="model-auxiliary.safetensors",
        help="Filename for the rescued aux tensors (default: %(default)s).",
    )
    p.add_argument(
        "--commit-message", default=None,
        help="Commit message for hub uploads (default: auto-generated).",
    )
    args = p.parse_args(argv)

    src_is_hub = _is_hub_spec(args.source)
    dst_is_hub = _is_hub_spec(args.dest)

    print(f"source: {args.source}  ({'hub' if src_is_hub else 'local'})")
    print(f"dest:   {args.dest}  ({'hub' if dst_is_hub else 'local'})")
    print()

    if args.check:
        from aux_rescue.core import diff_orphans

        report = diff_orphans(
            args.source, args.dest, token=args.token,
            include_prefix=args.include_prefix,
            exclude_prefix=args.exclude_prefix,
        )
        print(report.summary())
        print(_suggest_next_steps(args, report, src_is_hub, dst_is_hub))
        return 0 if not report.has_orphans else 1

    if dst_is_hub:
        from aux_rescue.hub import rescue_hub

        report = rescue_hub(
            args.source,
            args.dest,
            token=args.token,
            out_name=args.out_name,
            commit_message=args.commit_message,
            include_prefix=args.include_prefix,
            exclude_prefix=args.exclude_prefix,
        )
    else:
        from aux_rescue.local import rescue_local

        report = rescue_local(
            args.source,
            args.dest,
            token=args.token,
            out_name=args.out_name,
            include_prefix=args.include_prefix,
            exclude_prefix=args.exclude_prefix,
        )

    if not report.has_orphans:
        print("OK: no orphan tensors detected; nothing to rescue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
