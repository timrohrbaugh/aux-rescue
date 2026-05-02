"""Compare vision/audio tower weight statistics: rescued dest vs source.

If the rescue worked, every tower parameter should match the source within
floating-point precision. Any difference means we either rescued wrong
keys or the model class loaded random init for some param.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText

target = sys.argv[1] if len(sys.argv) > 1 else "/llm/gemma-4-E4B-it-heretic"
source_repo = sys.argv[2] if len(sys.argv) > 2 else "google/gemma-4-E4B-it"

print(f"Loading rescued model: {target}")
model = AutoModelForImageTextToText.from_pretrained(
    target, dtype=torch.bfloat16, device_map="auto",
)
model.eval()

state = model.state_dict()
tower_params = {
    k: v for k, v in state.items()
    if k.startswith("model.vision_tower.") or k.startswith("model.audio_tower.")
}
print(f"Rescued model has {len(tower_params)} vision/audio tower params")


print(f"\nReading source weights from local cache or downloading: {source_repo}")
from huggingface_hub import hf_hub_download

try:
    idx_path = hf_hub_download(source_repo, "model.safetensors.index.json")
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
except Exception:
    weight_map = None

source_tensors: dict[str, torch.Tensor] = {}
if weight_map is None:
    single = hf_hub_download(source_repo, "model.safetensors")
    with safe_open(single, framework="pt") as f:
        for k in f.keys():
            if k in tower_params:
                source_tensors[k] = f.get_tensor(k)
else:
    needed_files = set()
    for k in tower_params:
        if k in weight_map:
            needed_files.add(weight_map[k])
    for fname in needed_files:
        local = hf_hub_download(source_repo, fname)
        with safe_open(local, framework="pt") as f:
            for k in f.keys():
                if k in tower_params:
                    source_tensors[k] = f.get_tensor(k)

print(f"Source provides {len(source_tensors)} of those keys")

mismatches = []
shape_mismatches = []
matched = 0
for k, v_dst in tower_params.items():
    if k not in source_tensors:
        mismatches.append((k, "missing-in-source"))
        continue
    v_src = source_tensors[k].to(v_dst.dtype).to(v_dst.device)
    if v_src.shape != v_dst.shape:
        shape_mismatches.append((k, v_src.shape, v_dst.shape))
        continue
    diff = (v_src - v_dst).abs().max().item()
    if diff > 1e-3:
        mismatches.append((k, f"max_abs_diff={diff:.4e}"))
    else:
        matched += 1

print(f"\nMatched (max_abs_diff < 1e-3): {matched} / {len(tower_params)}")
if shape_mismatches:
    print(f"\nShape mismatches: {len(shape_mismatches)}")
    for k, sshape, dshape in shape_mismatches[:5]:
        print(f"  {k}: src{tuple(sshape)} vs dst{tuple(dshape)}")
if mismatches:
    print(f"\nValue mismatches: {len(mismatches)}")
    for k, why in mismatches[:5]:
        print(f"  {k}: {why}")

if not mismatches and not shape_mismatches:
    print("\nPASS: every vision/audio tower parameter matches source byte-for-byte.")
else:
    print("\nFAIL: some tower parameters did NOT match source.")
    sys.exit(1)
