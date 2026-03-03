from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def _ensure_repo_import_path() -> None:
    repo_pkg_root = Path(__file__).resolve().parents[1]
    repo_pkg_root_text = str(repo_pkg_root)
    if repo_pkg_root_text not in sys.path:
        sys.path.insert(0, repo_pkg_root_text)


_ensure_repo_import_path()

from gemma27_rl.rollout import compute_completion_logprobs, compute_completion_logprobs_batch  # noqa: E402
from gemma27_rl.utils import resolve_torch_dtype  # noqa: E402


def _reply(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reference model worker")
    _ = parser.parse_args(argv)

    model: AutoModelForCausalLM | None = None
    device = "cpu"

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")

            req_type = str(req.get("type", "")).strip().lower()
            if req_type == "close":
                _reply({"ok": True})
                return 0

            if req_type == "init":
                cfg_payload = req.get("config") or {}
                if not isinstance(cfg_payload, dict):
                    raise ValueError("init.config must be an object")

                model_name_or_path = str(cfg_payload.get("model_name_or_path") or "").strip()
                if not model_name_or_path:
                    raise ValueError("init.config.model_name_or_path is required")

                trust_remote_code = bool(cfg_payload.get("trust_remote_code", False))
                dtype_raw = str(cfg_payload.get("dtype", "float32"))
                attn_implementation = cfg_payload.get("attn_implementation")
                device = str(cfg_payload.get("device", "cpu"))

                kwargs: dict[str, Any] = {
                    "trust_remote_code": trust_remote_code,
                }
                dtype = resolve_torch_dtype(dtype_raw)
                if dtype is not None:
                    kwargs["torch_dtype"] = dtype
                if attn_implementation:
                    kwargs["attn_implementation"] = attn_implementation

                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
                model.to(device)
                cfg_obj = getattr(model, "config", None)
                if cfg_obj is not None and getattr(cfg_obj, "use_cache", None):
                    cfg_obj.use_cache = False
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                _reply({"ok": True})
                continue

            if model is None:
                raise RuntimeError("worker is not initialized. send init first.")

            if req_type == "score":
                prompt_ids = req.get("prompt_ids") or []
                completion_ids = req.get("completion_ids") or []
                if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
                    raise ValueError("score.prompt_ids and score.completion_ids must be lists")
                prompt_ids_int = [int(v) for v in prompt_ids]
                completion_ids_int = [int(v) for v in completion_ids]
                logprobs = compute_completion_logprobs(
                    model=model,
                    prompt_input_ids=prompt_ids_int,
                    completion_token_ids=completion_ids_int,
                    device=device,
                ).tolist()
                _reply({"ok": True, "logprobs": logprobs})
                continue

            if req_type == "score_batch":
                items = req.get("items") or []
                if not isinstance(items, list):
                    raise ValueError("score_batch.items must be a list")
                parsed_items: list[tuple[list[int], list[int]]] = []
                for item in items:
                    if not isinstance(item, dict):
                        raise ValueError("score_batch.items[*] must be objects")
                    prompt_ids = item.get("prompt_ids") or []
                    completion_ids = item.get("completion_ids") or []
                    if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
                        raise ValueError("score_batch.items[*].prompt_ids and completion_ids must be lists")
                    prompt_ids_int = [int(v) for v in prompt_ids]
                    completion_ids_int = [int(v) for v in completion_ids]
                    parsed_items.append((prompt_ids_int, completion_ids_int))

                micro_batch = max(1, len(parsed_items))
                while True:
                    try:
                        rows_t = compute_completion_logprobs_batch(
                            model=model,
                            items=parsed_items,
                            device=device,
                            micro_batch_size=micro_batch,
                        )
                        break
                    except Exception as exc:
                        msg = str(exc).lower()
                        is_oom = ("out of memory" in msg) or ("cuda oom" in msg)
                        if (not is_oom) or micro_batch <= 1:
                            raise
                        micro_batch = max(1, micro_batch // 2)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                rows = [[float(v) for v in row.tolist()] for row in rows_t]
                _reply({"ok": True, "logprobs_rows": rows})
                continue

            raise ValueError(f"unsupported request type: {req_type}")
        except Exception as exc:
            _reply(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
