"""Environment settings and YAML config (model catalog, router thresholds)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

load_dotenv(ROOT / ".env")


class Settings:
    """Provider keys and base URLs from the environment. A provider with no key
    is unavailable; its models resolve to the mock provider instead."""

    def __init__(self) -> None:
        self.openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai") or ""
        self.openai_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        self.nvidia_key = os.getenv("nvidia") or os.getenv("NVIDIA_API_KEY") or ""
        self.nvidia_base = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.groq_base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        self.openrouter_key = os.getenv("openrouter") or os.getenv("OPENROUTER_API_KEY") or ""
        self.openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        self.force_mock = (os.getenv("TRIAGE_FORCE_MOCK") or "0") in ("1", "true", "True")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def load_models() -> list[dict[str, Any]]:
    with open(CONFIG_DIR / "models.yaml") as f:
        return yaml.safe_load(f)["models"]


@lru_cache(maxsize=1)
def load_router_config() -> dict[str, Any]:
    with open(CONFIG_DIR / "router.yaml") as f:
        return yaml.safe_load(f)


def get_model(model_id: str) -> dict[str, Any] | None:
    for m in load_models():
        if m["id"] == model_id:
            return m
    return None


def default_small_model() -> dict[str, Any]:
    """The router's auto-pick: the smallest model flagged `default_small`."""
    pool = [m for m in load_models() if m.get("default_small")] or load_models()
    return min(pool, key=lambda m: m.get("size_b", 1e9))
