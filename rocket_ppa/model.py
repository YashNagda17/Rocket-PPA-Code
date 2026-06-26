"""Qwen + RocketPPA direct latency regression fine-tuning.

This module keeps the base language model as Qwen and adds one RocketPPA
regression stack on top: final hidden-state average pooling and a top-k
Mixture-of-Experts MLP head that predicts end-to-end Latency directly. LoRA is
applied to the Qwen backbone through PEFT while the routing matrix and MoE MLP
experts remain trainable, so one loss backpropagation updates LoRA A/B adapter
weights, expert MLP layers, and routing weights together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, PreTrainedModel


def _base_hidden_size(model: PreTrainedModel) -> int:
    candidate = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(candidate.config, "hidden_size"):
        return int(candidate.config.hidden_size)
    if hasattr(candidate.config, "text_config") and hasattr(candidate.config.text_config, "hidden_size"):
        return int(candidate.config.text_config.hidden_size)
    raise ValueError("could not infer hidden_size from the Qwen base model config")


DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
LATENCY_TARGET = "Latency"


@dataclass
class RocketPPAConfig:
    """Configuration for the RocketPPA latency head and Qwen LoRA setup."""

    base_model_name: str = DEFAULT_BASE_MODEL
    hidden_size: int | None = None
    dropout: float = 0.1
    num_experts: int = 6
    top_k: int = 3
    expert_hidden_size: int = 1024
    expert_layers: int = 3
    output_name: str = LATENCY_TARGET
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"

    def __post_init__(self) -> None:
        if self.output_name != LATENCY_TARGET:
            raise ValueError(f"RocketPPAQwenModel predicts only {LATENCY_TARGET!r}")
        if self.num_experts < 1:
            raise ValueError("num_experts must be at least 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.expert_layers < 1:
            raise ValueError("expert_layers must be at least 1")

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.output_name,)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RocketPPAConfig":
        payload = dict(data)
        if "output_names" in payload:
            output_names = tuple(payload.pop("output_names"))
            if output_names != (LATENCY_TARGET,):
                raise ValueError(f"checkpoint must contain only {LATENCY_TARGET!r}, got {output_names!r}")
            payload["output_name"] = LATENCY_TARGET
        return cls(**payload)


class ExpertMLP(nn.Module):
    """RocketPPA MLP expert used inside the latency MoE head."""

    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(max(layers - 1, 1)):
            modules.extend([nn.Linear(current, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current = hidden_dim
        modules.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TopKMoERegressor(nn.Module):
    """Top-k gated mixture of RocketPPA MLP experts."""

    def __init__(self, hidden_size: int, config: RocketPPAConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = min(config.top_k, config.num_experts)
        self.gate = nn.Linear(hidden_size, config.num_experts)
        self.experts = nn.ModuleList(
            ExpertMLP(hidden_size, config.expert_hidden_size, config.expert_layers, config.dropout)
            for _ in range(config.num_experts)
        )

    def forward(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_probs = F.softmax(self.gate(pooled), dim=-1)
        top_values, top_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_weights = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=-1)
        chosen_outputs = expert_outputs.gather(dim=-1, index=top_indices)
        prediction = (chosen_outputs * top_weights).sum(dim=-1)
        return prediction, gate_probs


class RocketPPAQwenModel(nn.Module):
    """Qwen backbone with LoRA and one RocketPPA MoE latency head."""

    def __init__(self, base_model: PreTrainedModel, config: RocketPPAConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        hidden_size = config.hidden_size or _base_hidden_size(base_model)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.latency_head = TopKMoERegressor(hidden_size, config)

    @classmethod
    def from_pretrained(
        cls,
        config: RocketPPAConfig,
        torch_dtype: torch.dtype | None = None,
        device_map: str | dict[str, int] | None = None,
    ) -> "RocketPPAQwenModel":
        base_model = AutoModel.from_pretrained(
            config.base_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=config.lora_target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        base_model = get_peft_model(base_model, lora_config)
        return cls(base_model, config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden = outputs.last_hidden_state
        mask = torch.ones(input_ids.shape, dtype=hidden.dtype, device=hidden.device) if attention_mask is None else attention_mask.to(hidden.dtype)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = self.final_norm(pooled)
        latency, gate_probs = self.latency_head(pooled)
        return {"pooled_embedding": pooled, LATENCY_TARGET: latency, f"{LATENCY_TARGET}_gate": gate_probs}
