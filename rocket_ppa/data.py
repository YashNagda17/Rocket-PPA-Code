"""Dataset and normalization utilities for Qwen/RocketPPA fine-tuning."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

TARGETS = ("first_token_latency", "throughput")
PROMPT_FIELD = "prompt"


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


def _row_to_prompt(record: dict[str, str]) -> str:
    if PROMPT_FIELD in record and record[PROMPT_FIELD]:
        return record[PROMPT_FIELD]
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
        row: dict[str, str | float] = {PROMPT_FIELD: _row_to_prompt({key: str(value) for key, value in record.items()})}
        for target in TARGETS:
            if target not in record:
                raise ValueError(f"missing required target column: {target}")
            row[target] = float(record[target])
        rows.append(row)
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
