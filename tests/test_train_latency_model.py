import multiprocessing as mp
import pickle

import pytest

torch = pytest.importorskip("torch")

from rocket_ppa.model import LATENCY_TARGET
from scripts.train_latency_model import build_collate, configure_gpu_memory


class SimpleTokenizer:
    def convert_ids_to_tokens(self, input_ids):
        return [f"tok_{token_id}" for token_id in input_ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def __call__(self, prompts, padding, truncation, max_length, return_tensors):
        assert padding is True
        assert truncation is True
        assert return_tensors == "pt"
        rows = []
        for prompt in prompts:
            token_ids = [min(ord(char), max_length) for char in prompt[:max_length]] or [0]
            rows.append(token_ids)
        width = max(len(row) for row in rows)
        input_ids = [row + [0] * (width - len(row)) for row in rows]
        attention_mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _run_collate_in_child(collate):
    batch = [
        {"prompt": "abc", LATENCY_TARGET: torch.tensor(1.0)},
        {"prompt": "de", LATENCY_TARGET: torch.tensor(2.0)},
    ]
    return collate(batch)["prompt_char_counts"].tolist()


def test_build_collate_returns_picklable_callable_for_multiprocessing_dataloader():
    collate = build_collate(SimpleTokenizer(), max_length=8)

    pickle.loads(pickle.dumps(collate))

    context = mp.get_context("spawn")
    with context.Pool(1) as pool:
        assert pool.apply(_run_collate_in_child, (collate,)) == [3, 2]


def test_collate_logs_active_token_ids_and_tokens_for_debugging():
    collate = build_collate(SimpleTokenizer(), max_length=8)
    batch = [
        {"prompt": "abc", LATENCY_TARGET: torch.tensor(1.0)},
        {"prompt": "de", LATENCY_TARGET: torch.tensor(2.0)},
    ]

    encoded = collate(batch)

    assert encoded["input_token_ids"] == [[8, 8, 8], [8, 8]]
    assert encoded["input_tokens"] == [["tok_8", "tok_8", "tok_8"], ["tok_8", "tok_8"]]
    assert encoded["mlp_input_text"] == ["tok_8tok_8tok_8", "tok_8tok_8"]


def test_run_config_uses_cpu_threads_without_dataloader_workers():
    from rocket_ppa.run_config import INFER_CONFIG, TRAIN_CONFIG

    assert TRAIN_CONFIG["cpu_threads"] >= 1
    assert TRAIN_CONFIG["num_workers"] == 0
    assert INFER_CONFIG["cpu_threads"] >= 1
    assert TRAIN_CONFIG["gpu_memory_fraction"] == 0.95
    assert INFER_CONFIG["gpu_memory_fraction"] == 0.95


class FakeCuda:
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def device_count(self):
        return 3

    def set_per_process_memory_fraction(self, fraction, device):
        self.calls.append((fraction, device))


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def test_configure_gpu_memory_sets_95_percent_for_each_visible_gpu():
    fake_torch = FakeTorch()

    configure_gpu_memory(fake_torch, 0.95)

    assert fake_torch.cuda.calls == [(0.95, 0), (0.95, 1), (0.95, 2)]


def test_model_reports_mlp_input_token_count_from_attention_mask():
    from rocket_ppa.model import RocketPPAConfig, RocketPPAQwenModel

    class FakeConfig:
        hidden_size = 4

    class FakeBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeConfig()

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
            del attention_mask, output_hidden_states, return_dict
            hidden = torch.ones((*input_ids.shape, 4), dtype=torch.float32)
            return type("Output", (), {"last_hidden_state": hidden})()

    model = RocketPPAQwenModel(
        FakeBase(),
        RocketPPAConfig(base_model_name="fake", num_experts=1, top_k=1, expert_hidden_size=4, expert_layers=1),
    )

    outputs = model(
        input_ids=torch.tensor([[1, 2, 0], [3, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
    )

    assert outputs["mlp_input_token_count"].tolist() == [2, 1]
    assert "pooled_embedding" not in outputs
