"""Restore auxiliary safetensors weights that ``save_pretrained`` silently drops.

Public API:

  - :func:`aux_rescue.local.rescue_local` — fix a local model directory
  - :func:`aux_rescue.hub.rescue_hub` — fix an HF Hub repo
  - :func:`aux_rescue.core.diff_orphans` — read-only check; returns orphan tensor names
"""
from aux_rescue.core import (
    diff_orphans,
    OrphanReport,
)
from aux_rescue.local import rescue_local
from aux_rescue.hub import rescue_hub

__all__ = [
    "diff_orphans",
    "OrphanReport",
    "rescue_local",
    "rescue_hub",
]
__version__ = "0.1.0"
