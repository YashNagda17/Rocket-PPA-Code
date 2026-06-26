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


def _load_prefer_local(loader: Callable[..., T], source: str | Path, local_dir: str | Path | None = None, **kwargs: Any) -> T:
    """Load a HF artifact from a project-local directory/cache before downloading.

    Explicit local paths are loaded directly. Hub model IDs are loaded from
    ``models/hf/<repo--model>`` when present, then from the Hugging Face cache
    with ``local_files_only=True``. If files are missing, the artifact is
    downloaded once and saved into the project-local model directory so the next
    run can use that directory directly.
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
        except OSError:
            print(f"saved model directory {managed_dir} is incomplete; checking Hugging Face cache")

    try:
        print(f"checking local Hugging Face cache for {source_name}")
        artifact = loader(source_name, local_files_only=True, **kwargs)
    except OSError:
        print(f"{source_name} is not fully cached locally; downloading missing files")
        artifact = loader(source_name, **kwargs)

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
