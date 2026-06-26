# Qwen + RocketPPA Latency Predictor

This repository contains a PyTorch implementation of the RocketPPA fine-tuning
recipe adapted for LLM-serving performance prediction.

The model uses **Qwen/Qwen3.5-4B** as the default base model, applies **LoRA across
all linear layers of the Qwen backbone**, and trains **RocketPPA-style top-k MoE
MLP expert heads** to predict:

- `first_token_latency`
- `throughput`

The heads use the final Qwen hidden states, masked average pooling, one gated MoE
MLP stack per target, log + z-score target normalization, and Smooth L1 training.
The scripts read Python variables from `rocket_ppa/run_config.py` instead of CLI
flags. The CLIs support `device = "auto"`, `"cpu"`, and `"cuda"`; CPU uses
float32 for compatibility, while GPU training can opt into `bf16 = True`.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Edit the Python dictionaries in `rocket_ppa/run_config.py`:

```python
TRAIN_CONFIG = {
    "data": "data/latency.csv",
    "output": "checkpoints/rocket_ppa_qwen",
    "base_model": "Qwen/Qwen3.5-4B",
    "device": "auto",
    "bf16": False,
    # ...training hyperparameters...
}

INFER_CONFIG = {
    "checkpoint": "checkpoints/rocket_ppa_qwen",
    "features": {"prompt_tokens": 128, "batch_size": 1, "gpu_memory_gb": 80},
    "prompt": None,
    "device": "auto",
}
```

To keep local changes out of the package, create another Python module with the
same variables and run with `ROCKET_PPA_CONFIG_MODULE=my_config_module`.

## Training data format

Use CSV or JSONL. Every row must include both targets:

```csv
prompt,first_token_latency,throughput
"prompt_tokens: 128; batch_size: 1; gpu: H100; memory_gb: 80",0.12,42.5
```

If `prompt` is omitted, all non-target columns are converted into a simple prompt:

```csv
prompt_tokens,batch_size,gpu_memory_gb,first_token_latency,throughput
128,1,80,0.12,42.5
```

## Commands

Train with the settings in `rocket_ppa/run_config.py`:

```bash
python scripts/train_latency_model.py
```

Train with an alternate config module:

```bash
ROCKET_PPA_CONFIG_MODULE=my_project.latency_config python scripts/train_latency_model.py
```

Run inference with the settings in `rocket_ppa/run_config.py`:

```bash
python scripts/infer_latency_model.py
```

Run inference with an alternate config module:

```bash
ROCKET_PPA_CONFIG_MODULE=my_project.latency_config python scripts/infer_latency_model.py
```
