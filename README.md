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
Checkpoints are written under `models/` by default, including the LoRA adapter,
RocketPPA heads, tokenizer, normalizer, and a `base_model/` snapshot when
`save_base_model = True`.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Edit the Python dictionaries in `rocket_ppa/run_config.py`:

```python
TRAIN_CONFIG = {
    "data": "data/latency.csv",
    "output": "models/rocket_ppa_qwen",
    "base_model": "Qwen/Qwen3.5-4B",
    "device": "auto",
    "bf16": False,
    "save_base_model": True,
    # ...training hyperparameters...
}

INFER_CONFIG = {
    "checkpoint": "models/rocket_ppa_qwen",
    "features": {"Model": "LLaMA3_8B", "Accelerator": "H100", "Num_Chips": 1, "Batch": 1, "Input_Sequence": 128, "Out_Seq": 128},
    "prompt": None,
    "device": "auto",
}
```

To keep local changes out of the package, create another Python module with the
same variables and run with `ROCKET_PPA_CONFIG_MODULE=my_config_module`.

## Training data format

Use CSV or JSONL. Every row must include both targets. For dynamic prompt
construction, provide the serving configuration columns below:

```csv
Model,Accelerator,Num_Chips,Batch,Input_Sequence,Out_Seq,first_token_latency,throughput
LLaMA3_8B,H100,1,1,128,128,0.12,42.5
Qwen2.5_14B,A100,2,4,1024,256,0.31,88.0
```

Supported values are `Model in {LLaMA3_8B, Qwen2.5_14B}`, `Accelerator in
{A100, V100, H100}`, and `Num_Chips in {1, 2}`. The prompt template injects
model architecture details, model optimizations, GCP accelerator memory/compute
and memory-bandwidth context, bf16 dtype, hardware optimizations, and the row
workload (`Batch`, `Input_Sequence`, and `Out_Seq`).

If a `prompt` column is present, it is used as-is. If the serving columns are
missing, all non-target columns are converted into a simple fallback prompt.

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
