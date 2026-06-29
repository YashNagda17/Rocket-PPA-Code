"""Editable runtime configuration for training and inference scripts.

Change the dictionaries below, or point ROCKET_PPA_CONFIG_MODULE at another
Python module that defines TRAIN_CONFIG and/or INFER_CONFIG.
"""

import os

TRAIN_CONFIG = {
    "data": "data/latency.csv",
    "output": "models/rocket_ppa_qwen",
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
    "save_base_model": True,
    # Use all CPU cores inside one model process. Keep DataLoader workers at 0
    # to avoid spawning parallel model-serving/training processes that duplicate RAM.
    "cpu_threads": os.cpu_count() or 1,
    "num_workers": 0,
    # Limit this training/inference process to 95% of each visible GPU
    # so CUDA keeps a small safety margin for drivers and display processes.
    "gpu_memory_fraction": 0.95,
}

INFER_CONFIG = {
    "checkpoint": "models/rocket_ppa_qwen",
    "prompt": None,
    "features": {
        "Model": "LLaMA3_8B",
        "Accelerator": "H100",
        "Num_Chips": 1,
        "Batch": 1,
        "Input_Sequence": 128,
        "Out_Seq": 128,
    },
    "max_length": 512,
    "device": "auto",
    "cpu_threads": os.cpu_count() or 1,
    "gpu_memory_fraction": 0.95,
}
