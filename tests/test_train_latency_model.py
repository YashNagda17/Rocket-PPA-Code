import multiprocessing as mp
import pickle

import pytest

torch = pytest.importorskip("torch")

from rocket_ppa.model import LATENCY_TARGET
from scripts.train_latency_model import build_collate


class SimpleTokenizer:
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
