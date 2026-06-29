# Qwen + RocketPPA Latency Predictor

This repository contains a PyTorch implementation of the RocketPPA fine-tuning
recipe adapted for LLM-serving latency prediction.

The model uses **Qwen/Qwen3.5-4B** as the default base model, applies **LoRA across
all linear layers of the Qwen backbone**, and trains one **RocketPPA-style top-k
MoE MLP head** to predict the target column directly:

- `Latency`

The head uses the final Qwen hidden states, masked average pooling, a gated MoE
MLP stack, log + z-score target normalization, and Smooth L1 training. Training
uses one latency loss, so the same backpropagation step updates the LoRA adapter
A/B weights, the expert MLP layers, and the routing/gating weight matrix at once.
This follows the RocketPPA paper's implementation pattern: an LLM embedding is
pooled, a linear router produces expert weights, the top-k experts are
renormalized, and the final scalar is the weighted sum of the selected expert
MLP outputs. This repo applies that pattern to one serving metric, `Latency`.
The scripts read Python variables from `rocket_ppa/run_config.py` instead of CLI
flags. The CLIs support `device = "auto"`, `"cpu"`, and `"cuda"`; CPU uses
float32 for compatibility, while GPU training can opt into `bf16 = True`.
For CPU runs, `cpu_threads` controls the PyTorch/BLAS threads used by one
model process, while `num_workers` controls only DataLoader subprocesses. Keep
`num_workers = 0` when you want all cores applied to a single model run without
parallel workers duplicating RAM. For CUDA training on A100, V100, or H100
systems, `gpu_memory_fraction = 0.95` caps this one process at 95% of each
visible GPU, and multi-GPU training uses all visible CUDA devices from the same
training process instead of separate serving processes.
Checkpoints are written under `models/` by default, including the LoRA adapter,
RocketPPA head, tokenizer, normalizer, and a `base_model/` snapshot when
`save_base_model = True`. Training metrics also include each sample's active
input token IDs, tokenizer token strings, reconstructed input text for the
tokens that produced the pre-mean hidden states, prompt token count, and the
token count used to pool hidden states immediately before the RocketPPA MLP
head. The mean pooled embedding itself is not written to debug logs. Inference
returns the same token-debug payload under `debug`.

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
    "cpu_threads": os.cpu_count() or 1,
    "num_workers": 0,
    "gpu_memory_fraction": 0.95,
    # ...training hyperparameters...
}

INFER_CONFIG = {
    "checkpoint": "models/rocket_ppa_qwen",
    "features": {"Model": "LLaMA3_8B", "Accelerator": "H100", "Num_Chips": 1, "Batch": 1, "Input_Sequence": 128, "Out_Seq": 128},
    "prompt": None,
    "device": "auto",
    "cpu_threads": os.cpu_count() or 1,
    "gpu_memory_fraction": 0.95,
}
```

To keep local changes out of the package, create another Python module with the
same variables and run with `ROCKET_PPA_CONFIG_MODULE=my_config_module`.

## Training data format

Give the fine-tuning script a CSV or JSONL file directly by setting
`TRAIN_CONFIG["data"]` to its path. The required target column is `Latency`.
For dynamic prompt construction, provide these serving configuration columns:

```csv
Model,Accelerator,Num_Chips,Batch,Input_Sequence,Out_Seq,Latency
LLaMA3_8B,H100,1,1,128,128,3.13
Qwen2.5_14B,A100,2,4,1024,256,3.22
```

Supported values are `Model in {LLaMA3_8B, Qwen2.5_14B}`, `Accelerator in
{A100, V100, H100}`, and `Num_Chips in {1, 2}`. Rows whose serving configuration
is outside those supported bounds are skipped during loading instead of stopping
training. The prompt template injects model architecture details, model
optimizations, GCP accelerator memory/compute and memory-bandwidth context,
bf16 dtype, bf16 perfect-max-bandwidth context, multi-chip interconnect-bandwidth
context, hardware optimizations, and the row workload (`Batch`, `Input_Sequence`,
and `Out_Seq`).

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
