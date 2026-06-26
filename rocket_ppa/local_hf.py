"""Hugging Face loading helpers that persist and prefer local model snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _managed_model_dir(source: str | Path, local_dir: str | Path | None = None) -> Path:
    if local_dir is not None:
        return Path(local_dir).expanduser()
    safe_name = str(source).strip("/").replace("/", "--")
    return Path("models") / "hf" / safe_name


def _is_recoverable_load_error(error: Exception) -> bool:
    """Return whether a local HF load error should fall back to a fresh download."""

    if isinstance(error, OSError):
        return True
    if isinstance(error, ValueError):
        message = str(error)
        return "Unrecognized model" in message or "model_type" in message
    return False


def _download_snapshot(source: str, destination: Path, force_download: bool = False) -> None:
    """Download a complete Hub snapshot into ``destination`` for direct local loads."""

    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=source,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=force_download,
    )
    print(f"downloaded model snapshot to {destination}")


def _load_prefer_local(loader: Callable[..., T], source: str | Path, local_dir: str | Path | None = None, **kwargs: Any) -> T:
    """Load a HF artifact from a complete project-local Hub snapshot.

    Explicit local paths are loaded directly. Hub model IDs are downloaded as
    complete snapshots into ``models/hf/<repo--model>`` and then loaded from
    that directory. This avoids re-saving an already-instantiated model, which
    can omit tokenizer/config files and leave the managed directory unusable for
    a later model load.
    """

    source_path = Path(source).expanduser()
    if source_path.exists():
        print(f"loading local model files from {source_path}")
        return loader(str(source_path), **kwargs)

    source_name = str(source)
    managed_dir = _managed_model_dir(source_name, local_dir)
    if managed_dir.exists():
        try:
            print(f"loading saved model files from {managed_dir}")
            return loader(str(managed_dir), **kwargs)
        except Exception as error:
            if not _is_recoverable_load_error(error):
                raise
            print(
                f"saved model directory {managed_dir} is not loadable ({error}); "
                "downloading a fresh Hugging Face snapshot"
            )
            _download_snapshot(source_name, managed_dir, force_download=True)
            return loader(str(managed_dir), **kwargs)

    print(f"downloading Hugging Face snapshot for {source_name} to {managed_dir}")
    _download_snapshot(source_name, managed_dir)
    return loader(str(managed_dir), **kwargs)


def load_auto_model_prefer_local(source: str | Path, local_dir: str | Path | None = None, **kwargs: Any) -> Any:
    """Load ``transformers.AutoModel`` while preferring saved local files/cache."""

    from transformers import AutoModel

    return _load_prefer_local(AutoModel.from_pretrained, source, local_dir=local_dir, **kwargs)


def load_auto_tokenizer_prefer_local(source: str | Path, local_dir: str | Path | None = None, **kwargs: Any) -> Any:
    """Load ``transformers.AutoTokenizer`` while preferring saved local files/cache."""

    from transformers import AutoTokenizer

    return _load_prefer_local(AutoTokenizer.from_pretrained, source, local_dir=local_dir, **kwargs)
