#!/usr/bin/env python
"""Run Qwen/RocketPPA latency and throughput inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_device(torch_module, requested: str):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is not available")
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def prompt_from_features(features: dict[str, object]) -> str:
    from rocket_ppa.data import build_serving_prompt

    string_features = {key: str(value) for key, value in features.items()}
    try:
        return build_serving_prompt(string_features)
    except ValueError:
        return "Predict serving performance from these features. " + "; ".join(f"{key}: {value}" for key, value in features.items())


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    from rocket_ppa.checkpoint import load_checkpoint
    from rocket_ppa.config_loader import load_config, require_keys

    args = load_config("INFER_CONFIG")
    require_keys(args, ("checkpoint", "prompt", "features", "max_length", "device"))
    model, normalizer = load_checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = args.prompt
    if prompt is None and args.features is not None:
        features = json.loads(args.features) if isinstance(args.features, str) else args.features
        prompt = prompt_from_features(features)
    if prompt is None:
        raise ValueError("set INFER_CONFIG['prompt'] or INFER_CONFIG['features']")
    device = resolve_device(torch, args.device)
    encoded = tokenizer([prompt], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
    model.to(device).eval()
    with torch.no_grad():
        raw = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    result = {name: normalizer.denormalize_prediction(name, raw[name].cpu()).item() for name in model.config.output_names}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
