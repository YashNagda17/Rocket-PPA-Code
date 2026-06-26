from unittest.mock import patch

from rocket_ppa.local_hf import _load_prefer_local


def test_invalid_saved_snapshot_downloads_and_loads_from_managed_dir(tmp_path):
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    calls = []

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        if len(calls) == 1:
            raise ValueError(
                "Unrecognized model in models/hf/Qwen--Qwen3.5-4B. "
                "Should have a `model_type` key in its config.json."
            )
        return {"path": path, "kwargs": kwargs}

    with patch("rocket_ppa.local_hf._download_snapshot") as download_snapshot:
        artifact = _load_prefer_local(
            loader,
            "Qwen/Qwen3.5-4B",
            local_dir=managed_dir,
            trust_remote_code=True,
        )

    download_snapshot.assert_called_once_with("Qwen/Qwen3.5-4B", managed_dir)
    assert artifact == {"path": str(managed_dir), "kwargs": {"trust_remote_code": True}}
    assert calls == [
        (str(managed_dir), {"trust_remote_code": True}),
        (str(managed_dir), {"trust_remote_code": True}),
    ]


def test_missing_cache_downloads_snapshot_before_loading(tmp_path):
    managed_dir = tmp_path / "managed"
    calls = []

    def loader(path, **kwargs):
        calls.append((path, kwargs))
        if kwargs.get("local_files_only"):
            raise OSError("not cached")
        return {"path": path, "kwargs": kwargs}

    with patch("rocket_ppa.local_hf._download_snapshot") as download_snapshot:
        artifact = _load_prefer_local(
            loader,
            "Qwen/Qwen3.5-4B",
            local_dir=managed_dir,
            trust_remote_code=True,
        )

    download_snapshot.assert_called_once_with("Qwen/Qwen3.5-4B", managed_dir)
    assert artifact == {"path": str(managed_dir), "kwargs": {"trust_remote_code": True}}
    assert calls == [
        ("Qwen/Qwen3.5-4B", {"local_files_only": True, "trust_remote_code": True}),
        (str(managed_dir), {"trust_remote_code": True}),
    ]
