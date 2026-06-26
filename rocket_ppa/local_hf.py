"""Hugging Face loading helpers that persist and prefer local model files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def _managed_model_dir(source: str | Path, local_dir: str | Path | None = None) -> Path:
    if local_dir is not None:
        return Path(local_dir).expanduser()
    safe_name = str(source).strip("/").replace("/", "--")
    return Path("models") / "hf" / safe_name


def _save_pretrained(artifact: T, destination: Path) -> None:
    save_pretrained = getattr(artifact, "save_pretrained", None)
    if save_pretrained is None:
        return
    destination.mkdir(parents=True, exist_ok=True)
    save_pretrained(str(destination))
    print(f"saved model files to {destination}")


def _is_recoverable_load_error(error: Exception) -> bool:
    """Return whether a local HF load error should fall back to a fresh download."""

    if isinstance(error, OSError):
        return True
    if isinstance(error, ValueError):
        message = str(error)
        return "Unrecognized model" in message or "model_type" in message
    return False


def _download_snapshot(source: str, destination: Path) -> None:
    """Download a complete Hub snapshot into ``destination`` for direct local loads."""

    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=source,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"downloaded model snapshot to {destination}")


def _load_prefer_local(loader: Callable[..., T], source: str | Path, local_dir: str | Path | None = None, **kwargs: Any) -> T:
    """Load a HF artifact from a project-local directory/cache before downloading.

    Explicit local paths are loaded directly. Hub model IDs are loaded from
    ``models/hf/<repo--model>`` when present, then from the Hugging Face cache
    with ``local_files_only=True``. If files are missing or a saved local
    snapshot is invalid, the Hub snapshot is downloaded into the project-local
    model directory so the next run can use that directory directly.
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
            _download_snapshot(source_name, managed_dir)
            return loader(str(managed_dir), **kwargs)

    try:
        print(f"checking local Hugging Face cache for {source_name}")
        artifact = loader(source_name, local_files_only=True, **kwargs)
    except Exception as error:
        if not _is_recoverable_load_error(error):
            raise
        print(f"{source_name} is not fully cached locally ({error}); downloading model snapshot")
        _download_snapshot(source_name, managed_dir)
        return loader(str(managed_dir), **kwargs)

    _save_pretrained(artifact, managed_dir)
    return artifact


def load_auto_model_prefer_local(source: str | Path, local_dir: str | Path | None = None, **kwargs: Any) -> Any:
    """Load ``transformers.AutoModel`` while preferring saved local files/cache."""

    from transformers import AutoModel

    return _load_prefer_local(AutoModel.from_pretrained, source, local_dir=local_dir, **kwargs)


def load_auto_tokenizer_prefer_local(source: str | Path, local_dir: str | Path | None = None, **kwargs: Any) -> Any:
    """Load ``transformers.AutoTokenizer`` while preferring saved local files/cache."""

    from transformers import AutoTokenizer

    return _load_prefer_local(AutoTokenizer.from_pretrained, source, local_dir=local_dir, **kwargs)
