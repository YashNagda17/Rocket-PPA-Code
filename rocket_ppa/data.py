"""Dataset, prompt construction, and normalization utilities for RocketPPA fine-tuning."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

TARGETS = ("Latency",)
PROMPT_FIELD = "prompt"
MODEL_FIELD = "Model"
ACCELERATOR_FIELD = "Accelerator"
NUM_CHIPS_FIELD = "Num_Chips"
BATCH_FIELD = "Batch"
INPUT_SEQUENCE_FIELD = "Input_Sequence"
OUTPUT_SEQUENCE_FIELD = "Out_Seq"
SUPPORTED_MODELS = ("LLaMA3_8B", "Qwen2.5_14B")
SUPPORTED_ACCELERATORS = ("A100", "V100", "H100")
SUPPORTED_NUM_CHIPS = ("1", "2")
SERVING_FIELDS = (MODEL_FIELD, ACCELERATOR_FIELD, NUM_CHIPS_FIELD, BATCH_FIELD, INPUT_SEQUENCE_FIELD, OUTPUT_SEQUENCE_FIELD)
POSITIVE_INTEGER_FIELDS = (BATCH_FIELD, INPUT_SEQUENCE_FIELD, OUTPUT_SEQUENCE_FIELD)

MODEL_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "LLaMA3_8B": {
        "name": "LLaMA 3 8B",
        "description": "An 8B-parameter decoder-only transformer optimized for efficient general-purpose text generation and instruction following.",
        "layers": 32,
        "vocab_size": 128256,
        "attention_heads": 32,
        "kv_heads": 8,
        "embedding_dim": 4096,
        "context_length": 8192,
        "speciality": "Grouped-query attention reduces KV-cache bandwidth pressure during autoregressive decoding.",
        "optimizations": "Use paged KV cache, continuous batching, fused attention kernels, tensor cores with bf16, and graph/cuda-kernel capture where supported.",
    },
    "Qwen2.5_14B": {
        "name": "Qwen2.5 14B",
        "description": "A 14B-parameter decoder-only Qwen model family member designed for multilingual, coding, math, and long-context workloads.",
        "layers": 48,
        "vocab_size": 152064,
        "attention_heads": 40,
        "kv_heads": 8,
        "embedding_dim": 5120,
        "context_length": 131072,
        "speciality": "Large vocabulary and long-context training make prompt length and KV-cache size especially important for serving performance.",
        "optimizations": "Use bf16 tensor cores, FlashAttention-style kernels, paged KV cache, prefill/decode disaggregation when available, and continuous batching.",
    },
}

HARDWARE_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "A100": {
        "name": "NVIDIA A100 40GB on GCP",
        "memory": "40 GB HBM2",
        "compute_bandwidth": "up to 312 TFLOPS bf16 tensor-core throughput",
        "memory_bandwidth": "about 1.6 TB/s HBM2 bandwidth",
        "bf16_perfect_max_bandwidth": "312 TFLOPS bf16 dense tensor-core peak under perfect utilization",
        "interconnect_bandwidth": "single-chip configuration does not require inter-chip communication; multi-chip A100 deployments should use the available NVLink or host fabric bandwidth",
        "max_storage": "VM-attached local SSD or persistent disk capacity; accelerator memory is 40 GB per chip",
        "dtype": "bf16",
        "optimizations": "Exploit tensor cores, NVLink where available, CUDA graphs, fused attention/MLP kernels, and memory-aware KV-cache paging.",
    },
    "V100": {
        "name": "NVIDIA V100 16GB on GCP",
        "memory": "16 GB HBM2",
        "compute_bandwidth": "up to 125 TFLOPS tensor-core mixed-precision throughput; bf16 is treated as the configured serving dtype for prompt context",
        "memory_bandwidth": "about 900 GB/s HBM2 bandwidth",
        "bf16_perfect_max_bandwidth": "bf16 is modeled as the serving dtype, but V100 tensor cores are not native bf16; use the configured mixed-precision peak as the perfect-utilization ceiling",
        "interconnect_bandwidth": "single-chip configuration does not require inter-chip communication; multi-chip V100 deployments should use the available NVLink or PCIe fabric bandwidth",
        "max_storage": "VM-attached local SSD or persistent disk capacity; accelerator memory is 16 GB per chip",
        "dtype": "bf16",
        "optimizations": "Keep batches small enough for 16 GB memory, use fused kernels, quantized or sharded KV cache if needed, and minimize host-device transfers.",
    },
    "H100": {
        "name": "NVIDIA H100 80GB on GCP",
        "memory": "80 GB HBM3",
        "compute_bandwidth": "up to about 1,979 TFLOPS bf16 tensor-core throughput with sparsity, about 989 TFLOPS without sparsity",
        "memory_bandwidth": "about 3.35 TB/s HBM3 bandwidth",
        "bf16_perfect_max_bandwidth": "about 989 TFLOPS bf16 dense tensor-core peak, or about 1,979 TFLOPS with sparsity, under perfect utilization",
        "interconnect_bandwidth": "single-chip configuration does not require inter-chip communication; multi-chip H100 deployments should use the available NVLink/NVSwitch fabric bandwidth",
        "max_storage": "VM-attached local SSD or persistent disk capacity; accelerator memory is 80 GB per chip",
        "dtype": "bf16",
        "optimizations": "Use Transformer Engine/bf16 tensor cores, FlashAttention-style kernels, CUDA graphs, larger continuous batches, and efficient KV-cache paging.",
    },
}


@dataclass
class Normalizer:
    target_mean: dict[str, float]
    target_std: dict[str, float]
    epsilon: float = 1e-6

    def normalize_targets(self, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for name, value in targets.items():
            mean = torch.tensor(self.target_mean[name], device=value.device, dtype=value.dtype)
            std = torch.tensor(self.target_std[name], device=value.device, dtype=value.dtype)
            out[name] = (torch.log(value.clamp_min(0) + self.epsilon) - mean) / std.clamp_min(self.epsilon)
        return out

    def denormalize_prediction(self, name: str, value: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.target_mean[name], device=value.device, dtype=value.dtype)
        std = torch.tensor(self.target_std[name], device=value.device, dtype=value.dtype)
        return torch.exp(value * std + mean) - self.epsilon

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class LatencyDataset(Dataset):
    def __init__(self, rows: list[dict[str, str | float]], normalizer: Normalizer | None = None):
        self.rows = rows
        self.normalizer = normalizer or build_normalizer(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str | torch.Tensor]:
        row = self.rows[idx]
        raw_targets = {name: torch.tensor(float(row[name]), dtype=torch.float32) for name in TARGETS}
        targets = self.normalizer.normalize_targets(raw_targets)
        return {"prompt": str(row[PROMPT_FIELD]), **targets}


def _require_supported(record: dict[str, str], field: str, supported: tuple[str, ...]) -> str:
    value = str(record.get(field, "")).strip()
    if value not in supported:
        raise ValueError(f"unsupported {field}={value!r}; expected one of: {', '.join(supported)}")
    return value


def _require_positive_integer(record: dict[str, str], field: str) -> str:
    value = str(record.get(field, "")).strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {field}={value!r}; expected a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"unsupported {field}={value!r}; expected a positive integer")
    return value


def validate_serving_record(record: dict[str, str]) -> None:
    _require_supported(record, MODEL_FIELD, SUPPORTED_MODELS)
    _require_supported(record, ACCELERATOR_FIELD, SUPPORTED_ACCELERATORS)
    _require_supported(record, NUM_CHIPS_FIELD, SUPPORTED_NUM_CHIPS)
    for field in POSITIVE_INTEGER_FIELDS:
        _require_positive_integer(record, field)


def build_serving_prompt(record: dict[str, str]) -> str:
    validate_serving_record(record)
    model_key = _require_supported(record, MODEL_FIELD, SUPPORTED_MODELS)
    accelerator_key = _require_supported(record, ACCELERATOR_FIELD, SUPPORTED_ACCELERATORS)
    num_chips = _require_supported(record, NUM_CHIPS_FIELD, SUPPORTED_NUM_CHIPS)
    model = MODEL_DESCRIPTIONS[model_key]
    hardware = HARDWARE_DESCRIPTIONS[accelerator_key]
    batch = record.get(BATCH_FIELD, "")
    input_sequence = record.get(INPUT_SEQUENCE_FIELD, "")
    output_sequence = record.get(OUTPUT_SEQUENCE_FIELD, "")
    return (
        "Predict LLM serving performance for this configuration.\n"
        f"Model: {model['name']} ({model_key}). {model['description']} "
        f"Architecture: layers={model['layers']}, vocab_size={model['vocab_size']}, attention_heads={model['attention_heads']}, "
        f"kv_heads={model['kv_heads']}, embedding_dim={model['embedding_dim']}, context_length={model['context_length']}. "
        f"Model optimizations for faster performance: {model['optimizations']} "
        f"Model speciality: {model['speciality']}\n"
        f"Hardware: {hardware['name']} with num_chips={num_chips}. Accelerator memory={hardware['memory']}; "
        f"compute bandwidth={hardware['compute_bandwidth']}; memory bandwidth={hardware['memory_bandwidth']}; "
        f"bf16 perfect max bandwidth={hardware['bf16_perfect_max_bandwidth']}; "
        f"interconnect bandwidth={hardware['interconnect_bandwidth']}; "
        f"max storage={hardware['max_storage']}; dtype={hardware['dtype']}. "
        f"Hardware optimizations: {hardware['optimizations']}\n"
        f"Workload: batch={batch}; input_sequence_tokens={input_sequence}; output_sequence_tokens={output_sequence}."
    )


def _row_to_prompt(record: dict[str, str]) -> str:
    if PROMPT_FIELD in record and record[PROMPT_FIELD]:
        return record[PROMPT_FIELD]
    if set(SERVING_FIELDS).issubset(record):
        return build_serving_prompt(record)
    feature_parts = [f"{key}: {value}" for key, value in record.items() if key not in TARGETS]
    return "Predict serving performance from these features. " + "; ".join(feature_parts)


def load_rows(path: str | Path) -> list[dict[str, str | float]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("dataset is empty")
    rows: list[dict[str, str | float]] = []
    for record in records:
        string_record = {key: str(value) for key, value in record.items()}
        if PROMPT_FIELD in string_record and string_record[PROMPT_FIELD]:
            prompt = string_record[PROMPT_FIELD]
        elif set(SERVING_FIELDS).issubset(string_record):
            try:
                prompt = build_serving_prompt(string_record)
            except ValueError:
                continue
        else:
            prompt = _row_to_prompt(string_record)
        row: dict[str, str | float] = {PROMPT_FIELD: prompt}
        for target in TARGETS:
            if target not in record:
                raise ValueError(f"missing required target column: {target}")
            row[target] = float(record[target])
        rows.append(row)
    if not rows:
        raise ValueError("no rows remained after dropping out-of-bounds serving configurations")
    return rows


def build_normalizer(rows: Iterable[dict[str, str | float]], epsilon: float = 1e-6) -> Normalizer:
    materialized = list(rows)
    target_mean: dict[str, float] = {}
    target_std: dict[str, float] = {}
    for name in TARGETS:
        values = torch.log(torch.tensor([float(row[name]) for row in materialized], dtype=torch.float32).clamp_min(0) + epsilon)
        target_mean[name] = values.mean().item()
        target_std[name] = values.std(unbiased=False).clamp_min(epsilon).item()
    return Normalizer(target_mean=target_mean, target_std=target_std, epsilon=epsilon)
