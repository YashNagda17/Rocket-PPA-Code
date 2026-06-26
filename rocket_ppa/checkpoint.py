"""Checkpoint helpers for Qwen LoRA adapters and RocketPPA heads."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel

from .data import Normalizer
from .model import RocketPPAConfig, RocketPPAQwenModel


def save_checkpoint(path: str | Path, model: RocketPPAQwenModel, normalizer: Normalizer, save_base_model: bool = True) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(path / "qwen_lora")
    if save_base_model:
        base = model.base_model.get_base_model() if hasattr(model.base_model, "get_base_model") else model.base_model
        base.save_pretrained(path / "base_model")
    torch.save({"heads": model.heads.state_dict(), "final_norm": model.final_norm.state_dict()}, path / "rocket_ppa_heads.pt")
    (path / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2), encoding="utf-8")
    normalizer.save(path / "normalizer.json")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[RocketPPAQwenModel, Normalizer]:
    path = Path(path)
    config = RocketPPAConfig.from_dict(json.loads((path / "config.json").read_text(encoding="utf-8")))
    base_model_path = path / "base_model"
    base_source = str(base_model_path) if base_model_path.exists() else config.base_model_name
    base = AutoModel.from_pretrained(base_source, trust_remote_code=True, device_map=None)
    base_model = PeftModel.from_pretrained(base, path / "qwen_lora")
    model = RocketPPAQwenModel(base_model, config)
    head_state = torch.load(path / "rocket_ppa_heads.pt", map_location=map_location)
    model.heads.load_state_dict(head_state["heads"])
    model.final_norm.load_state_dict(head_state["final_norm"])
    return model, Normalizer.load(path / "normalizer.json")
