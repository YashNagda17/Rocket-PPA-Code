#!/usr/bin/env python
"""Fine-tune Qwen LoRA adapters plus RocketPPA MoE MLP experts."""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_collate(tokenizer, max_length: int):
    def collate(batch: list[dict[str, object]]) -> dict[str, object]:
        import torch

        prompts = [str(item["prompt"]) for item in batch]
        encoded = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        for name in ("first_token_latency", "throughput"):
            encoded[name] = torch.stack([item[name] for item in batch])
        return encoded

    return collate


def resolve_device(torch_module, requested: str):
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is not available")
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(requested)


def run_epoch(model, loader, optimizer, device) -> float:
    import torch
    from torch import nn

    training = optimizer is not None
    model.train(training)
    loss_fn = nn.SmoothL1Loss()
    total = 0.0
    count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = {name: batch[name].to(device) for name in model.config.output_names}
        with torch.set_grad_enabled(training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = sum(loss_fn(outputs[name], targets[name]) for name in model.config.output_names)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total += loss.item() * input_ids.shape[0]
        count += input_ids.shape[0]
    return total / max(count, 1)


def main() -> None:
    import torch
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer

    from rocket_ppa.checkpoint import save_checkpoint
    from rocket_ppa.config_loader import load_config, require_keys
    from rocket_ppa.data import LatencyDataset, load_rows
    from rocket_ppa.model import RocketPPAConfig, RocketPPAQwenModel

    args = load_config("TRAIN_CONFIG")
    require_keys(
        args,
        (
            "data",
            "output",
            "base_model",
            "epochs",
            "batch_size",
            "max_length",
            "lr",
            "val_fraction",
            "seed",
            "experts",
            "top_k",
            "expert_hidden_size",
            "lora_rank",
            "lora_alpha",
            "device",
            "bf16",
        ),
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(args.data)
    dataset = LatencyDataset(rows)
    val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator) if val_size else (dataset, None)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate = build_collate(tokenizer, args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate) if val_set else None
    config = RocketPPAConfig(
        base_model_name=args.base_model,
        num_experts=args.experts,
        top_k=args.top_k,
        expert_hidden_size=args.expert_hidden_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    device = resolve_device(torch, args.device)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else None
    model = RocketPPAQwenModel.from_pretrained(config, torch_dtype=dtype).to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device) if val_loader else train_loss
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss <= best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items() if "lora_" in key or "heads" in key or "final_norm" in key}
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, model, dataset.normalizer)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
