#!/usr/bin/env python
"""Fine-tune Qwen LoRA adapters plus one RocketPPA MoE MLP latency head."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class LatencyCollator:
    """Picklable DataLoader collator for tokenized latency batches."""

    tokenizer: Any
    max_length: int

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        import torch

        from rocket_ppa.model import LATENCY_TARGET

        prompts = [str(item["prompt"]) for item in batch]
        encoded = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids_for_log = encoded["input_ids"].tolist()
        token_ids_for_log = []
        tokens_for_log = []
        text_for_log = []
        for row_ids, row_mask in zip(input_ids_for_log, encoded["attention_mask"].tolist(), strict=True):
            active_ids = [int(token_id) for token_id, keep in zip(row_ids, row_mask, strict=True) if keep]
            token_ids_for_log.append(active_ids)
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                active_tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
            else:
                active_tokens = [str(token_id) for token_id in active_ids]
            tokens_for_log.append(active_tokens)
            if hasattr(self.tokenizer, "convert_tokens_to_string"):
                text_for_log.append(self.tokenizer.convert_tokens_to_string(active_tokens))
            else:
                text_for_log.append(" ".join(active_tokens))
        encoded[LATENCY_TARGET] = torch.stack([item[LATENCY_TARGET] for item in batch])
        encoded["input_token_ids"] = token_ids_for_log
        encoded["input_tokens"] = tokens_for_log
        encoded["mlp_input_text"] = text_for_log
        encoded["prompts"] = prompts
        encoded["prompt_char_counts"] = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.long)
        return encoded


def build_collate(tokenizer, max_length: int):
    return LatencyCollator(tokenizer=tokenizer, max_length=max_length)


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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def configure_gpu_memory(torch_module, gpu_memory_fraction: float) -> None:
    """Cap this process at the requested fraction of each visible GPU."""

    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("gpu_memory_fraction must be > 0 and <= 1")
    if not torch_module.cuda.is_available():
        return
    for device_index in range(torch_module.cuda.device_count()):
        torch_module.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=device_index)
    print(f"gpu_memory_fraction={gpu_memory_fraction:.2f} visible_gpus={torch_module.cuda.device_count()}")


def maybe_wrap_multi_gpu(torch_module, model, device):
    """Use all visible GPUs from one training process when CUDA is selected."""

    if device.type == "cuda" and torch_module.cuda.device_count() > 1:
        device_ids = list(range(torch_module.cuda.device_count()))
        print(f"multi_gpu_training=DataParallel device_ids={device_ids}")
        return torch_module.nn.DataParallel(model, device_ids=device_ids)
    return model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def configure_cpu_parallelism(torch_module, cpu_threads: int) -> None:
    """Configure PyTorch to use the requested CPU threads for one model process."""

    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least 1")
    torch_module.set_num_threads(cpu_threads)
    torch_module.set_num_interop_threads(cpu_threads)
    print(
        "cpu_parallelism="
        f"threads={torch_module.get_num_threads()} "
        f"interop_threads={torch_module.get_num_interop_threads()}"
    )


class TrainingCsvLogger:
    """Append-and-flush CSV logger for per-sample, per-batch, and per-epoch metrics."""

    fieldnames = (
        "record_type",
        "model",
        "epoch",
        "phase",
        "iteration",
        "batch_index",
        "sample_index",
        "batch_size",
        "prompt_chars",
        "prompt_tokens",
        "mlp_input_token_count",
        "input_token_ids",
        "input_tokens",
        "mlp_input_text",
        "loss",
        "epoch_loss",
        "elapsed_seconds",
        "real_latency",
        "predicted_latency",
        "normalized_real_latency",
        "normalized_predicted_latency",
        "prompt",
    )

    def __init__(self, path: str | Path, model_name: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.handle.flush()
        self.model_name = model_name

    def log(self, **row: Any) -> None:
        record = {field: row.get(field, "") for field in self.fieldnames}
        record["model"] = record["model"] or self.model_name
        self.writer.writerow(record)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def run_epoch(model, loader, optimizer, device, normalizer, logger: TrainingCsvLogger, epoch: int, phase: str, iteration: int) -> tuple[float, int]:
    import torch
    from torch import nn

    from rocket_ppa.model import LATENCY_TARGET

    training = optimizer is not None
    model.train(training)
    loss_fn = nn.SmoothL1Loss()
    total = 0.0
    count = 0
    epoch_started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        iteration += 1
        batch_started = time.perf_counter()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch[LATENCY_TARGET].to(device)
        with torch.set_grad_enabled(training):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = outputs[LATENCY_TARGET]
            loss = loss_fn(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = input_ids.shape[0]
        prompt_tokens = attention_mask.detach().sum(dim=1).cpu().tolist()
        mlp_input_token_counts = outputs["mlp_input_token_count"].detach().cpu().tolist()
        prompt_chars = batch["prompt_char_counts"].cpu().tolist()
        normalized_real = target.detach().cpu()
        normalized_predicted = prediction.detach().cpu()
        real_latency = normalizer.denormalize_prediction(LATENCY_TARGET, normalized_real).tolist()
        predicted_latency = normalizer.denormalize_prediction(LATENCY_TARGET, normalized_predicted).tolist()
        batch_loss = loss.item()
        batch_elapsed = time.perf_counter() - batch_started
        for sample_index, prompt in enumerate(batch["prompts"]):
            logger.log(
                record_type="sample",
                epoch=epoch,
                phase=phase,
                iteration=iteration,
                batch_index=batch_index,
                sample_index=sample_index,
                batch_size=batch_size,
                prompt_chars=prompt_chars[sample_index],
                prompt_tokens=prompt_tokens[sample_index],
                mlp_input_token_count=mlp_input_token_counts[sample_index],
                input_token_ids=json.dumps(batch["input_token_ids"][sample_index]),
                input_tokens=json.dumps(batch["input_tokens"][sample_index], ensure_ascii=False),
                mlp_input_text=batch["mlp_input_text"][sample_index],
                loss=batch_loss,
                elapsed_seconds=batch_elapsed,
                real_latency=real_latency[sample_index],
                predicted_latency=predicted_latency[sample_index],
                normalized_real_latency=normalized_real[sample_index].item(),
                normalized_predicted_latency=normalized_predicted[sample_index].item(),
                prompt=prompt,
            )
        logger.log(
            record_type="batch",
            epoch=epoch,
            phase=phase,
            iteration=iteration,
            batch_index=batch_index,
            batch_size=batch_size,
            prompt_chars=sum(prompt_chars),
            prompt_tokens=sum(prompt_tokens),
            mlp_input_token_count=sum(mlp_input_token_counts),
            loss=batch_loss,
            elapsed_seconds=batch_elapsed,
        )
        total += batch_loss * batch_size
        count += batch_size
    epoch_loss = total / max(count, 1)
    epoch_elapsed = time.perf_counter() - epoch_started
    logger.log(
        record_type="epoch",
        epoch=epoch,
        phase=phase,
        iteration=iteration,
        batch_size=count,
        loss=epoch_loss,
        epoch_loss=epoch_loss,
        elapsed_seconds=epoch_elapsed,
    )
    return epoch_loss, iteration


def ensure_one_loss_trains_all_required_components(model) -> None:
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    required = {
        "LoRA A adapters": ("lora_A",),
        "LoRA B adapters": ("lora_B",),
        "MoE expert MLP layers": ("latency_head.experts",),
        "top-k routing weights": ("latency_head.gate",),
    }
    missing = [
        label
        for label, needles in required.items()
        if not any(any(needle in name for needle in needles) for name in trainable_names)
    ]
    if missing:
        raise RuntimeError(f"single latency loss cannot train missing components: {', '.join(missing)}")


def main() -> None:
    from rocket_ppa.config_loader import load_config, require_keys

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
            "save_base_model",
        ),
    )
    cpu_threads = int(getattr(args, "cpu_threads", effective_cpu_count()))
    num_workers = int(getattr(args, "num_workers", 0))
    gpu_memory_fraction = float(getattr(args, "gpu_memory_fraction", 0.95))
    if num_workers < 0:
        raise ValueError("num_workers must be at least 0")
    configure_cpu_environment(cpu_threads)
    configure_gpu_environment(gpu_memory_fraction)

    import torch
    from torch.utils.data import DataLoader, random_split

    from rocket_ppa.checkpoint import save_checkpoint
    from rocket_ppa.data import LatencyDataset, load_rows
    from rocket_ppa.local_hf import load_auto_tokenizer_prefer_local
    from rocket_ppa.model import RocketPPAConfig, RocketPPAQwenModel

    configure_cpu_parallelism(torch, cpu_threads)
    configure_gpu_memory(torch, gpu_memory_fraction)
    print(f"dataloader_workers={num_workers} (0 keeps data loading in the single model process)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(args.data)
    dataset = LatencyDataset(rows)
    val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator) if val_size else (dataset, None)
    tokenizer = load_auto_tokenizer_prefer_local(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate = build_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            collate_fn=collate,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        if val_set
        else None
    )
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
    model = RocketPPAQwenModel.from_pretrained(config, dtype=dtype).to(device)
    ensure_one_loss_trains_all_required_components(model)
    model = maybe_wrap_multi_gpu(torch, model, device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    metrics_csv = Path(getattr(args, "metrics_csv", None) or Path(args.output) / "training_metrics.csv")
    logger = TrainingCsvLogger(metrics_csv, args.base_model)
    best_val = float("inf")
    best_state = None
    iteration = 0
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, iteration = run_epoch(model, train_loader, optimizer, device, dataset.normalizer, logger, epoch, "train", iteration)
            if val_loader:
                val_loss, iteration = run_epoch(model, val_loader, None, device, dataset.normalizer, logger, epoch, "val", iteration)
            else:
                val_loss = train_loss
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} metrics_csv={metrics_csv}")
            if val_loss <= best_val:
                best_val = val_loss
                best_state = {
                    key: value.detach().cpu()
                    for key, value in unwrap_model(model).state_dict().items()
                    if "lora_" in key or "latency_head" in key or "final_norm" in key
                }
    finally:
        logger.close()
    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state, strict=False)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, unwrap_model(model), dataset.normalizer, save_base_model=args.save_base_model)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
