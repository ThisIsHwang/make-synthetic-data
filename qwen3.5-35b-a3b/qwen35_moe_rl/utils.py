from __future__ import annotations

import logging
import os
from pathlib import Path
import random

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


# ---------------------------------------------------------------------------
# Distributed training utilities
# ---------------------------------------------------------------------------


def is_distributed() -> bool:
    if torch is None:
        return False
    return torch.distributed.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return int(torch.distributed.get_rank())
    return 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    if is_distributed():
        return int(torch.distributed.get_world_size())
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl") -> None:
    if torch is None:
        raise RuntimeError("torch is required for distributed training")
    if torch.distributed.is_initialized():
        logger.info("torch.distributed already initialized; skipping.")
        return
    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=backend)
    logger.info(
        "Initialized distributed: rank=%s local_rank=%s world_size=%s backend=%s",
        torch.distributed.get_rank(),
        local_rank,
        torch.distributed.get_world_size(),
        backend,
    )


def cleanup_distributed() -> None:
    if torch is not None and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Destroyed distributed process group.")


def barrier() -> None:
    if is_distributed():
        torch.distributed.barrier()


def all_reduce_scalar(value: float, op: str = "mean") -> float:
    if not is_distributed():
        return value
    tensor = torch.tensor(value, device="cuda", dtype=torch.float64)
    if op == "sum":
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    elif op == "mean":
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor = tensor / get_world_size()
    elif op == "max":
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    elif op == "min":
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    else:
        raise ValueError(f"Unsupported reduce op: {op}")
    return float(tensor.item())
