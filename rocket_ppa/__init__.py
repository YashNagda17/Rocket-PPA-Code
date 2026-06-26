"""RocketPPA-style Qwen direct latency predictor."""

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"

__all__ = ["DEFAULT_BASE_MODEL", "RocketPPAConfig", "RocketPPAQwenModel"]


def __getattr__(name: str):
    if name in {"RocketPPAConfig", "RocketPPAQwenModel"}:
        from .model import RocketPPAConfig, RocketPPAQwenModel

        return {"RocketPPAConfig": RocketPPAConfig, "RocketPPAQwenModel": RocketPPAQwenModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
