# aux-rescue

Restore auxiliary safetensors weights that `AutoModelForCausalLM.save_pretrained` silently drops.

## The bug this fixes

`AutoModelForCausalLM.from_pretrained` reads every `*.safetensors` file in a model directory, but only loads tensors whose keys map to parameters in the resolved model class. Keys that don't match (Multi-Token Prediction draft heads, EAGLE/Medusa drafters, vision/audio encoders on multimodal models loaded as CausalLM) are silently discarded. `save_pretrained` then writes only what's in `state_dict()`, so those tensors are lost on round-trip with no warning.

The damage shows up later — vLLM speculative decoding fails to start, multimodal inputs return garbage, etc.

Affected workflows:

- Abliteration (heretic, ablit)
- Fine-tuning that saves a full model
- Mergekit / model surgery
- PEFT `merge_and_unload().save_pretrained()`
- Any tool that runs `from_pretrained → save_pretrained` on a model with non-LM tensors

## Two layouts handled

1. **Separate aux file** — e.g. Ling-2.6-flash's `model-mtp-layer.safetensors` shipped alongside main shards.
2. **Embedded keys** — e.g. Qwen3.6-27B keeps `mtp.*` tensors inside `model-00013-of-00015.safetensors` and `model-00015-of-00015.safetensors`, mixed with main weights.

For embedded layout the tool extracts orphan tensors only (never overwrites your modifications) and writes them to a single `model-auxiliary.safetensors`, registered in the destination's `model.safetensors.index.json`.

## Install

```bash
pip install aux-rescue
# or, from a clone:
pip install -e .
```

## CLI

```bash
# Check what's missing without writing anything
aux-rescue --source Qwen/Qwen3.6-27B --dest ./my-fine-tuned-model --check

# Repair a local model directory
aux-rescue --source Qwen/Qwen3.6-27B --dest ./my-fine-tuned-model

# Repair a model already pushed to the Hub
aux-rescue --source Qwen/Qwen3.6-27B --dest myuser/my-fine-tuned-model --token $HF_TOKEN

# Force hub interpretation when ambiguous
aux-rescue --source hf://Qwen/Qwen3.6-27B --dest hf://myuser/my-model
```

`--check` exits 0 if there's nothing to rescue, 1 if orphans were found — drop it into a CI step before publishing models.

## Library

```python
from aux_rescue import rescue_local, rescue_hub, diff_orphans

# Read-only diagnostic
report = diff_orphans("Qwen/Qwen3.6-27B", "./my-model")
print(report.summary())

# Repair local
rescue_local("Qwen/Qwen3.6-27B", "./my-model")

# Repair hub
rescue_hub("Qwen/Qwen3.6-27B", "myuser/my-model", token=hf_token)
```

## What it does NOT touch

- Your modifications to the main model weights
- Tokenizer / processor / chat template files
- `config.json` / `generation_config.json`
- Anything outside top-level `*.safetensors`

Stays narrow on purpose. If you need to rescue tokenizer or processor configs, those are different problems with different solutions.

## License

Apache-2.0.
