from __future__ import annotations

import logging
import os
from pathlib import Path
import random
import shlex

try:
    import torch
except Exception:  # pragma: no cover - optional during lightweight tests
    torch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def resolve_device(requested: str | None) -> str:
    if requested and requested.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        logger.warning("CUDA requested (%s) but unavailable; falling back to cpu.", requested)
        return "cpu"
    if requested:
        return requested
    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


def resolve_torch_dtype(dtype_name: str | None):
    if torch is None:
        return None
    if not dtype_name:
        return None
    key = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def configure_huggingface_cache(cache_dir: str | None, token: str | None = None) -> str | None:
    if not cache_dir:
        return None

    root = Path(cache_dir).expanduser().resolve()
    hub = root / "hub"
    transformers = root / "transformers"
    datasets = root / "datasets"

    hub.mkdir(parents=True, exist_ok=True)
    transformers.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers)
    os.environ["HF_DATASETS_CACHE"] = str(datasets)
    # Disable xet backend for more stable large-model downloads on flaky networks.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    logger.info(
        "Configured Hugging Face cache: HF_HOME=%s TRANSFORMERS_CACHE=%s HF_DATASETS_CACHE=%s token=%s",
        root,
        transformers,
        datasets,
        "set" if token else "unset",
    )
    return str(root)


def resolve_huggingface_token(explicit_token: str | None, token_env_name: str | None = "HF_TOKEN") -> str | None:
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()

    candidate_envs: list[str] = []
    if token_env_name and token_env_name.strip():
        candidate_envs.append(token_env_name.strip())
    candidate_envs.extend(["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])

    seen: set[str] = set()
    for env_name in candidate_envs:
        if env_name in seen:
            continue
        seen.add(env_name)
        value = os.environ.get(env_name)
        if value and value.strip():
            logger.info("Using Hugging Face token from env var: %s", env_name)
            return value.strip()
    return None


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def collect_huggingface_worker_env() -> dict[str, str]:
    keys = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "HF_HUB_ETAG_TIMEOUT",
    )
    out: dict[str, str] = {}
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        text = str(value)
        if not text:
            continue
        out[key] = text
    return out


def merge_env_overrides(
    base: dict[str, str] | None,
    extra: dict[str, str] | None,
) -> dict[str, str] | None:
    merged: dict[str, str] = {}
    if base:
        merged.update({str(k): str(v) for k, v in base.items()})
    if extra:
        merged.update({str(k): str(v) for k, v in extra.items()})
    return merged or None


def build_worker_launch_command(
    *,
    python_executable: str,
    worker_script: str | Path,
    worker_args: list[str] | None = None,
    remote_host: str | None = None,
    remote_workdir: str | None = None,
    remote_env: dict[str, str] | None = None,
) -> list[str]:
    script_path = str(worker_script)
    args = [str(v) for v in (worker_args or [])]
    host = str(remote_host).strip() if remote_host else ""
    if not host:
        return [str(python_executable), script_path, *args]

    parts: list[str] = []
    workdir = str(remote_workdir).strip() if remote_workdir else ""
    if workdir:
        parts.append(f"cd {shlex.quote(workdir)}")
        parts.append("&&")

    if remote_env:
        env_tokens = [
            f"{str(key)}={shlex.quote(str(value))}"
            for key, value in sorted(remote_env.items(), key=lambda kv: str(kv[0]))
        ]
        if env_tokens:
            parts.append(" ".join(env_tokens))

    parts.append("exec")
    parts.append(shlex.quote(str(python_executable)))
    parts.append(shlex.quote(script_path))
    parts.extend(shlex.quote(arg) for arg in args)
    remote_cmd = " ".join(parts)
    return ["ssh", host, remote_cmd]
