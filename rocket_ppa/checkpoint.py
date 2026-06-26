"""Checkpoint helpers for Qwen LoRA adapters and the RocketPPA latency head."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from .data import Normalizer
from .local_hf import load_auto_model_prefer_local
from .model import RocketPPAConfig, RocketPPAQwenModel


def save_checkpoint(path: str | Path, model: RocketPPAQwenModel, normalizer: Normalizer, save_base_model: bool = True) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(path / "qwen_lora")
    if save_base_model:
        base = model.base_model.get_base_model() if hasattr(model.base_model, "get_base_model") else model.base_model
        base.save_pretrained(path / "base_model")
    torch.save(
        {"latency_head": model.latency_head.state_dict(), "final_norm": model.final_norm.state_dict()},
        path / "rocket_ppa_heads.pt",
    )
    (path / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2), encoding="utf-8")
    normalizer.save(path / "normalizer.json")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[RocketPPAQwenModel, Normalizer]:
    path = Path(path)
    config = RocketPPAConfig.from_dict(json.loads((path / "config.json").read_text(encoding="utf-8")))
    base_model_path = path / "base_model"
    base_source = str(base_model_path) if base_model_path.exists() else config.base_model_name
    base = load_auto_model_prefer_local(base_source, trust_remote_code=True, device_map=None)
    base_model = PeftModel.from_pretrained(base, path / "qwen_lora")
    model = RocketPPAQwenModel(base_model, config)
    head_state = torch.load(path / "rocket_ppa_heads.pt", map_location=map_location)
    latency_head_state = head_state.get("latency_head", head_state.get("heads"))
    if latency_head_state is None:
        raise ValueError("checkpoint is missing latency head weights")
    if any(key.startswith("Latency.") for key in latency_head_state):
        latency_head_state = {key.removeprefix("Latency."): value for key, value in latency_head_state.items()}
    model.latency_head.load_state_dict(latency_head_state)
    model.final_norm.load_state_dict(head_state["final_norm"])
    return model, Normalizer.load(path / "normalizer.json")
