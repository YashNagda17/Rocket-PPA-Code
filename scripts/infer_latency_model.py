#!/usr/bin/env python
"""Run Qwen/RocketPPA direct latency inference."""

from __future__ import annotations

import json
import os
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


def configure_gpu_environment(gpu_memory_fraction: float) -> None:
    """Configure CUDA allocator behavior before torch imports."""

    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be > 0 and <= 1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def effective_cpu_count() -> int:
    """Return the CPUs this process can actually run on, honoring affinity masks."""

    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def configure_cpu_environment(cpu_threads: int) -> None:
    """Configure common CPU backend environment variables before torch imports."""

    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    thread_count = str(cpu_threads)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count
    os.environ["TORCH_NUM_THREADS"] = thread_count
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"


def configure_gpu_memory(torch_module, gpu_memory_fraction: float) -> None:
    """Cap this process at the requested fraction of each visible GPU."""

    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be > 0 and <= 1")
    if not torch_module.cuda.is_available():
        return
    for device_index in range(torch_module.cuda.device_count()):
        torch_module.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=device_index)
    print(f"gpu_memory_fraction={gpu_memory_fraction:.2f} visible_gpus={torch_module.cuda.device_count()}")


def configure_cpu_parallelism(torch_module, cpu_threads: int) -> None:
    """Configure PyTorch to use all requested cores for this one inference process."""

    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    torch_module.set_num_threads(cpu_threads)
    torch_module.set_num_interop_threads(cpu_threads)
    print(
        "cpu_parallelism="
        f"threads={torch_module.get_num_threads()} "
        f"interop_threads={torch_module.get_num_interop_threads()}"
    )


def prompt_from_features(features: dict[str, object]) -> str:
    from rocket_ppa.data import SERVING_FIELDS, build_serving_prompt

    string_features = {key: str(value) for key, value in features.items()}
    if set(SERVING_FIELDS).issubset(string_features):
        return build_serving_prompt(string_features)
    return "Predict serving performance from these features. " + "; ".join(f"{key}: {value}" for key, value in features.items())


def main() -> None:
    from rocket_ppa.config_loader import load_config, require_keys

    args = load_config("INFER_CONFIG")
    require_keys(args, ("checkpoint", "prompt", "features", "max_length", "device"))
    cpu_threads = int(getattr(args, "cpu_threads", effective_cpu_count()))
    gpu_memory_fraction = float(getattr(args, "gpu_memory_fraction", 0.95))
    configure_cpu_environment(cpu_threads)
    configure_gpu_environment(gpu_memory_fraction)

    import torch
    from rocket_ppa.checkpoint import load_checkpoint
    from rocket_ppa.local_hf import load_auto_tokenizer_prefer_local

    configure_cpu_parallelism(torch, cpu_threads)
    configure_gpu_memory(torch, gpu_memory_fraction)
    model, normalizer = load_checkpoint(args.checkpoint)
    tokenizer = load_auto_tokenizer_prefer_local(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt = args.prompt
    if prompt is None and args.features is not None:
        features = json.loads(args.features) if isinstance(args.features, str) else args.features
        prompt = prompt_from_features(features)
    if prompt is None:
        raise ValueError("set INFER_CONFIG['prompt'] or INFER_CONFIG['features']")
    device = resolve_device(torch, args.device)
    encoded = tokenizer([prompt], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    active_token_ids = [
        int(token_id)
        for token_id, keep in zip(encoded["input_ids"][0].tolist(), encoded["attention_mask"][0].tolist(), strict=True)
        if keep
    ]
    active_tokens = (
        tokenizer.convert_ids_to_tokens(active_token_ids)
        if hasattr(tokenizer, "convert_ids_to_tokens")
        else [str(token_id) for token_id in active_token_ids]
    )
    mlp_input_text = (
        tokenizer.convert_tokens_to_string(active_tokens)
        if hasattr(tokenizer, "convert_tokens_to_string")
        else " ".join(active_tokens)
    )
    encoded = encoded.to(device)
    model.to(device).eval()
    with torch.no_grad():
        raw = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    result = {name: normalizer.denormalize_prediction(name, raw[name].cpu()).item() for name in model.config.output_names}
    result["debug"] = {
        "input_token_count": len(active_token_ids),
        "mlp_input_token_count": raw["mlp_input_token_count"].detach().cpu().tolist(),
        "input_token_ids": active_token_ids,
        "input_tokens": active_tokens,
        "mlp_input_text": mlp_input_text,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
