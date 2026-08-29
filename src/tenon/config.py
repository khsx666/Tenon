"""Configuration loading.

Values come from environment variables, optionally backed by a local `.env`
file (gitignored). The API key is never printed, logged, or persisted —
error messages in this module deliberately never include it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass
class Config:
    api_key: str            # from TENON_API_KEY; never print or persist
    base_url: str           # any OpenAI-compatible endpoint
    model: str
    max_turns: int = 50             # hard cap on loop turns (termination layer)
    timeout: float = 120.0          # default bash timeout (s), capped at 600
    permission_mode: str = "ask"    # ask | auto-edit | auto | read-only
    context_token_budget: int = 100_000  # trigger for context compression


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal KEY=VALUE .env file (no interpolation).

    Blank lines and `#` comments are ignored; an optional `export ` prefix
    is tolerated. Real environment variables always win over file values.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            raise ConfigError(f"{path}:{lineno}: expected KEY=VALUE, got {line!r}")
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_config(env_file: str = ".env") -> Config:
    file_values = _read_env_file(Path(env_file))

    def get(key: str) -> str | None:
        return os.environ.get(key) or file_values.get(key)

    api_key = get("TENON_API_KEY")
    if not api_key:
        raise ConfigError(
            "TENON_API_KEY is not set. Copy .env.example to .env and fill in "
            "your OpenAI-compatible endpoint credentials, or export the "
            "variable in your shell. Never commit the key."
        )
    base_url, model = get("TENON_BASE_URL"), get("TENON_MODEL")
    missing = [k for k, v in (("TENON_BASE_URL", base_url), ("TENON_MODEL", model)) if not v]
    if missing:
        raise ConfigError(
            f"Missing required setting(s): {', '.join(missing)} — see .env.example."
        )
    return Config(api_key=api_key, base_url=base_url, model=model)
