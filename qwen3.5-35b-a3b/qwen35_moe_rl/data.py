from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import DataConfig
from .types import Example


logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL row line=%s err=%s", line_no, exc)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [row for row in payload["data"] if isinstance(row, dict)]
        return [payload]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - dependency/runtime issue
        raise RuntimeError(
            "Parquet input requires datasets package. Install datasets>=2.21.0."
        ) from exc

    ds = load_dataset("parquet", data_files=str(path), split="train")
    return [dict(row) for row in ds]


def _load_records(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".jsonl", ".jsonlines"}:
        return _read_jsonl(file_path)
    if suffix == ".json":
        return _read_json(file_path)
    if suffix == ".parquet":
        return _read_parquet(file_path)
    raise ValueError(f"Unsupported data file extension: {suffix}")


def _load_records_from_hf_dataset(
    dataset_name: str,
    dataset_config_name: str | None,
    split_name: str,
    revision: str | None,
    streaming: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - dependency/runtime issue
        raise RuntimeError(
            "HF dataset input requires datasets package. Install datasets>=2.21.0."
        ) from exc

    kwargs: dict[str, Any] = {"split": split_name, "streaming": streaming}
    if revision:
        kwargs["revision"] = revision

    ds = load_dataset(dataset_name, dataset_config_name, **kwargs)
    rows: list[dict[str, Any]] = []

    if streaming:
        for idx, row in enumerate(ds):
            if isinstance(row, dict):
                rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    # Non-streaming path.
    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))
    return [dict(row) for row in ds]


def _pick_text(row: dict[str, Any], field: str, default: str | None = None) -> str | None:
    if field not in row or row[field] is None:
        return default
    value = str(row[field]).strip()
    if not value:
        return default
    return value


def _resolve_split(records: list[dict[str, Any]], split_field: str | None, split_name: str | None) -> list[dict[str, Any]]:
    if not split_field or not split_name:
        return records
    return [row for row in records if str(row.get(split_field, "")).strip() == split_name]


def load_examples(cfg: DataConfig, split: str, limit: int | None = None) -> list[Example]:
    use_hf_dataset = bool(cfg.hf_dataset_name and str(cfg.hf_dataset_name).strip())
    if use_hf_dataset:
        if split == "train":
            hf_split = cfg.hf_train_split
        else:
            hf_split = cfg.hf_eval_split or cfg.hf_train_split
        records = _load_records_from_hf_dataset(
            dataset_name=cfg.hf_dataset_name or "",
            dataset_config_name=cfg.hf_dataset_config_name,
            split_name=hf_split,
            revision=cfg.hf_revision,
            streaming=cfg.hf_streaming,
            limit=limit,
        )
    else:
        if split == "train":
            data_file = cfg.train_file
        else:
            data_file = cfg.eval_file or cfg.train_file
        if not data_file:
            raise ValueError(f"No data file configured for split={split}")
        records = _load_records(data_file)
        split_name = cfg.train_split if split == "train" else cfg.eval_split
        records = _resolve_split(records, cfg.split_field, split_name)

    examples: list[Example] = []
    for idx, row in enumerate(records):
        if cfg.skip_bad_source and bool(row.get(cfg.is_bad_source_field, False)):
            continue

        src = _pick_text(row, cfg.src_text_field)
        if not src:
            continue

        src_lang = _pick_text(row, cfg.src_lang_field, cfg.default_src_lang) or cfg.default_src_lang
        tgt_lang = _pick_text(row, cfg.tgt_lang_field, cfg.default_tgt_lang) or cfg.default_tgt_lang
        src_code = _pick_text(row, cfg.src_lang_code_field, cfg.default_src_lang_code)
        tgt_code = _pick_text(row, cfg.tgt_lang_code_field, cfg.default_tgt_lang_code)
        ref_text = _pick_text(row, cfg.ref_text_field, None)
        ex_id = _pick_text(row, cfg.id_field, str(idx)) or str(idx)

        examples.append(
            Example(
                example_id=ex_id,
                src_text=src,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_lang_code=src_code,
                tgt_lang_code=tgt_code,
                ref_text=ref_text,
            )
        )
        if limit is not None and len(examples) >= limit:
            break

    logger.info("Loaded %s examples for split=%s", len(examples), split)
    return examples
