"""Editable runtime configuration for training and inference scripts.

Change the dictionaries below, or point ROCKET_PPA_CONFIG_MODULE at another
Python module that defines TRAIN_CONFIG and/or INFER_CONFIG.
"""

TRAIN_CONFIG = {
    "data": "data/latency.csv",
    "output": "checkpoints/rocket_ppa_qwen",
    "base_model": "Qwen/Qwen3.5-4B",
    "epochs": 3,
    "batch_size": 2,
    "max_length": 512,
    "lr": 2e-4,
    "val_fraction": 0.2,
    "seed": 7,
    "experts": 6,
    "top_k": 3,
    "expert_hidden_size": 1024,
    "lora_rank": 16,
    "lora_alpha": 32.0,
    "device": "auto",
    "bf16": False,
}

INFER_CONFIG = {
    "checkpoint": "checkpoints/rocket_ppa_qwen",
    "prompt": None,
    "features": {
        "prompt_tokens": 128,
        "batch_size": 1,
        "gpu_memory_gb": 80,
    },
    "max_length": 512,
    "device": "auto",
}
