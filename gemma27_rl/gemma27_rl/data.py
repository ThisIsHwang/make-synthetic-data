from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import logging
from pathlib import Path
from typing import Any

from .config import DataConfig
from .rl_types import Example


logger = logging.getLogger(__name__)
_BAD_SOURCE_FLAG_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_BAD_SOURCE_FLAG_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}
_WARNED_UNKNOWN_BAD_SOURCE_FLAGS: set[str] = set()
_CACHE_INPUT_FILE_PATH_FIELD = "__gemma27_input_file_path__"
_SPLIT_CACHE_VERSION = 1


@dataclass(frozen=True)
class _LoadedRecord:
    row: dict[str, Any]
    input_file_path: str | None = None


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


def _discover_data_files(directory: str, pattern: str) -> list[Path]:
    base_dir = Path(directory)
    files = sorted(path.resolve() for path in base_dir.glob(pattern) if path.is_file())
    if files:
        return files
    raise FileNotFoundError(f"No data files matched pattern={pattern!r} under directory={directory!r}")


def _load_records_from_file(path: str) -> list[_LoadedRecord]:
    file_path = Path(path).resolve()
    records: list[_LoadedRecord] = []
    for row in _load_records(str(file_path)):
        row_copy = dict(row)
        cached_input_file_path = row_copy.pop(_CACHE_INPUT_FILE_PATH_FIELD, None)
        input_file_path = (
            str(cached_input_file_path).strip()
            if isinstance(cached_input_file_path, str) and str(cached_input_file_path).strip()
            else str(file_path)
        )
        records.append(_LoadedRecord(row=row_copy, input_file_path=input_file_path))
    return records


def _load_records_from_dir(directory: str, pattern: str) -> list[_LoadedRecord]:
    records: list[_LoadedRecord] = []
    for path in _discover_data_files(directory, pattern):
        records.extend(_load_records_from_file(str(path)))
    return records


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


def _wrap_hf_records(records: list[dict[str, Any]]) -> list[_LoadedRecord]:
    return [_LoadedRecord(row=row, input_file_path=None) for row in records]


def _pick_text(row: dict[str, Any], field: str, default: str | None = None) -> str | None:
    if field not in row or row[field] is None:
        return default
    value = str(row[field]).strip()
    if not value:
        return default
    return value


def _pick_nested_text(row: dict[str, Any], field_path: str | None, default: str | None = None) -> str | None:
    path_text = str(field_path or "").strip()
    if not path_text:
        return default

    current: Any = row
    for part in path_text.split("."):
        key = str(part).strip()
        if not key:
            return default
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    if current is None:
        return default
    text = str(current).strip()
    if not text:
        return default
    return text


def _derive_domain(teacher_path: str | None, input_file_path: str | None) -> str:
    teacher_path_text = str(teacher_path or "").strip()
    if teacher_path_text:
        normalized = teacher_path_text.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        for idx, part in enumerate(parts[:-1]):
            if part == "translation" and idx + 1 < len(parts):
                dataset_id = str(parts[idx + 1]).strip()
                if dataset_id:
                    return dataset_id
        teacher_stem = Path(teacher_path_text).stem.strip()
        if teacher_stem:
            return teacher_stem

    input_path_text = str(input_file_path or "").strip()
    if input_path_text:
        input_stem = Path(input_path_text).stem.strip()
        if input_stem:
            return input_stem
    return "unknown-domain"


def _parse_bad_source_flag(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if not text:
        return False
    if text in _BAD_SOURCE_FLAG_FALSE_VALUES:
        return False
    if text in _BAD_SOURCE_FLAG_TRUE_VALUES:
        return True
    if text not in _WARNED_UNKNOWN_BAD_SOURCE_FLAGS:
        _WARNED_UNKNOWN_BAD_SOURCE_FLAGS.add(text)
        logger.warning(
            "Unrecognized bad-source flag value %r; treating it as False. "
            "Use an explicit boolean or one of %s/%s.",
            value,
            sorted(_BAD_SOURCE_FLAG_TRUE_VALUES),
            sorted(_BAD_SOURCE_FLAG_FALSE_VALUES),
        )
    return False


def _append_example_with_optional_reverse(
    *,
    examples: list[Example],
    example_id: str,
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    src_lang_code: str | None,
    tgt_lang_code: str | None,
    ref_text: str | None,
    domain: str | None,
    teacher_path: str | None,
    input_file_path: str | None,
    bidirectional_with_ref: bool,
    limit: int | None,
) -> bool:
    examples.append(
        Example(
            example_id=example_id,
            src_text=src_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_lang_code=src_lang_code,
            tgt_lang_code=tgt_lang_code,
            ref_text=ref_text,
            domain=domain,
            teacher_path=teacher_path,
            input_file_path=input_file_path,
        )
    )
    if limit is not None and len(examples) >= limit:
        return True

    reverse_src = str(ref_text or "").strip()
    if (not bidirectional_with_ref) or (not reverse_src):
        return False

    examples.append(
        Example(
            example_id=f"{example_id}::reverse",
            src_text=reverse_src,
            src_lang=tgt_lang,
            tgt_lang=src_lang,
            src_lang_code=tgt_lang_code,
            tgt_lang_code=src_lang_code,
            ref_text=src_text,
            domain=domain,
            teacher_path=teacher_path,
            input_file_path=input_file_path,
        )
    )
    return limit is not None and len(examples) >= limit


def _resolve_split(
    records: list[_LoadedRecord],
    split_field: str | None,
    split_name: str | None,
) -> list[_LoadedRecord]:
    if not split_field or not split_name:
        return records
    return [record for record in records if str(record.row.get(split_field, "")).strip() == split_name]


def _build_split_key(row: dict[str, Any], id_field: str, input_file_path: str | None = None) -> str:
    raw = row.get(id_field)
    if raw is not None:
        text = str(raw).strip()
        if text:
            if input_file_path:
                return f"{input_file_path}::{text}"
            return text
    # Fallback for rows without stable IDs.
    payload_text = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if input_file_path:
        return f"{input_file_path}::{payload_text}"
    return payload_text


def _stable_split_hash(seed: int, key: str) -> int:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def _apply_eval_sampling_split(
    records: list[_LoadedRecord],
    split: str,
    id_field: str,
    eval_ratio: float | None,
    eval_count: int | None,
    seed: int,
    min_eval_samples: int,
) -> list[_LoadedRecord]:
    if split not in {"train", "eval"}:
        return records

    total = len(records)
    if total <= 1:
        return records if split == "train" else []

    if eval_count is not None:
        eval_size = int(eval_count)
    else:
        # Ratio-based split fallback.
        ratio = float(eval_ratio or 0.0)
        eval_size = int(round(total * ratio))
        eval_size = max(min_eval_samples, eval_size)
    eval_size = min(eval_size, total - 1)
    eval_size = max(1, eval_size)

    scored: list[tuple[int, int]] = []
    for idx, record in enumerate(records):
        key = _build_split_key(record.row, id_field=id_field, input_file_path=record.input_file_path)
        score = _stable_split_hash(seed=seed, key=key)
        scored.append((idx, score))

    scored.sort(key=lambda item: (item[1], item[0]))
    eval_idx = {idx for idx, _score in scored[:eval_size]}

    if split == "eval":
        sampled = [record for idx, record in enumerate(records) if idx in eval_idx]
    else:
        sampled = [record for idx, record in enumerate(records) if idx not in eval_idx]

    logger.info(
        "Applied eval_sampling split=%s total=%s eval_size=%s train_size=%s seed=%s mode=%s ratio=%s count=%s",
        split,
        total,
        eval_size,
        total - eval_size,
        seed,
        "count" if eval_count is not None else "ratio",
        eval_ratio,
        eval_count,
    )
    return sampled


def _record_domain(record: _LoadedRecord, domain_field_path: str) -> str:
    teacher_path = _pick_nested_text(record.row, domain_field_path, None)
    return _derive_domain(teacher_path, record.input_file_path)


def _compute_eval_size_for_ratio(*, total: int, eval_ratio: float | None, min_eval_samples: int) -> int:
    if total <= 1:
        return 0
    ratio = float(eval_ratio or 0.0)
    eval_size = int(round(total * ratio))
    eval_size = max(int(min_eval_samples), eval_size)
    eval_size = min(eval_size, total - 1)
    return max(1, eval_size)


def _allocate_eval_counts_by_domain(
    *,
    domain_sizes: dict[str, int],
    total_eval_count: int,
) -> dict[str, int]:
    eligible = [
        (domain, int(size))
        for domain, size in sorted(domain_sizes.items())
        if int(size) > 1
    ]
    if not eligible:
        return {domain: 0 for domain in domain_sizes}

    max_total = sum(size - 1 for _, size in eligible)
    target = min(max(0, int(total_eval_count)), max_total)
    if target <= 0:
        return {domain: 0 for domain in domain_sizes}

    total_weight = sum(size for _, size in eligible)
    allocations: dict[str, int] = {domain: 0 for domain in domain_sizes}
    remainders: list[tuple[float, str]] = []

    assigned = 0
    for domain, size in eligible:
        ideal = (float(target) * float(size)) / float(max(1, total_weight))
        base = min(size - 1, int(ideal))
        allocations[domain] = base
        assigned += base
        remainders.append((ideal - float(base), domain))

    remainders.sort(key=lambda item: (-item[0], item[1]))
    remainder_idx = 0
    while assigned < target and remainders:
        _, domain = remainders[remainder_idx % len(remainders)]
        capacity = int(domain_sizes.get(domain, 0)) - 1
        if allocations[domain] < capacity:
            allocations[domain] += 1
            assigned += 1
        remainder_idx += 1
        if remainder_idx > (len(remainders) * max(1, target + len(remainders))):
            break

    return allocations


def _apply_domain_stratified_eval_sampling_split(
    *,
    records: list[_LoadedRecord],
    split: str,
    id_field: str,
    eval_ratio: float | None,
    eval_count: int | None,
    seed: int,
    min_eval_samples: int,
    domain_field_path: str,
) -> list[_LoadedRecord]:
    if split not in {"train", "eval"}:
        return records

    domain_to_indices: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        domain = _record_domain(record, domain_field_path)
        domain_to_indices.setdefault(domain, []).append(idx)

    if eval_count is not None:
        eval_counts_by_domain = _allocate_eval_counts_by_domain(
            domain_sizes={domain: len(indices) for domain, indices in domain_to_indices.items()},
            total_eval_count=int(eval_count),
        )
    else:
        eval_counts_by_domain = {
            domain: _compute_eval_size_for_ratio(
                total=len(indices),
                eval_ratio=eval_ratio,
                min_eval_samples=min_eval_samples,
            )
            for domain, indices in domain_to_indices.items()
        }

    selected_eval_indices: set[int] = set()
    summary_parts: list[str] = []
    for domain, indices in sorted(domain_to_indices.items()):
        eval_size = max(0, int(eval_counts_by_domain.get(domain, 0)))
        if len(indices) <= 1 or eval_size <= 0:
            summary_parts.append(f"{domain}:train={len(indices)} eval=0")
            continue

        scored: list[tuple[int, int]] = []
        for idx in indices:
            record = records[idx]
            key = _build_split_key(record.row, id_field=id_field, input_file_path=record.input_file_path)
            score = _stable_split_hash(seed=seed, key=key)
            scored.append((idx, score))
        scored.sort(key=lambda item: (item[1], item[0]))

        domain_eval_indices = {idx for idx, _ in scored[:eval_size]}
        selected_eval_indices.update(domain_eval_indices)
        summary_parts.append(f"{domain}:train={len(indices) - len(domain_eval_indices)} eval={len(domain_eval_indices)}")

    if split == "eval":
        sampled = [record for idx, record in enumerate(records) if idx in selected_eval_indices]
    else:
        sampled = [record for idx, record in enumerate(records) if idx not in selected_eval_indices]

    logger.info(
        "Applied domain-stratified eval_sampling split=%s total=%s seed=%s mode=%s ratio=%s count=%s summary=%s",
        split,
        len(records),
        seed,
        "count" if eval_count is not None else "ratio",
        eval_ratio,
        eval_count,
        ", ".join(summary_parts),
    )
    return sampled


def _default_split_cache_dir(train_dir: str) -> Path:
    train_dir_path = Path(train_dir).resolve()
    return train_dir_path.parent / ".gemma27_split_cache" / train_dir_path.name


def _resolve_split_cache_dir(cfg: DataConfig, train_dir: str) -> Path:
    if cfg.split_cache_dir:
        return Path(cfg.split_cache_dir).resolve()
    return _default_split_cache_dir(train_dir)


def _build_dir_split_cache_key(
    *,
    cfg: DataConfig,
    train_dir: str,
    pattern: str,
    source_files: list[Path],
) -> str:
    payload = {
        "version": _SPLIT_CACHE_VERSION,
        "train_dir": str(Path(train_dir).resolve()),
        "pattern": str(pattern),
        "id_field": str(cfg.id_field),
        "domain_field_path": str(cfg.domain_field_path),
        "eval_sampling_ratio": cfg.eval_sampling_ratio,
        "eval_sampling_count": cfg.eval_sampling_count,
        "eval_sampling_seed": int(cfg.eval_sampling_seed),
        "eval_sampling_min_samples": int(cfg.eval_sampling_min_samples),
        "files": [
            {
                "path": str(path),
                "size": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
            }
            for path in source_files
        ],
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:20]


def _cache_payload_for_record(record: _LoadedRecord) -> dict[str, Any]:
    row_copy = dict(record.row)
    if record.input_file_path:
        row_copy[_CACHE_INPUT_FILE_PATH_FIELD] = str(record.input_file_path)
    return row_copy


def _write_cached_split_records(path: Path, records: list[_LoadedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_cache_payload_for_record(record), ensure_ascii=False) + "\n")


def _load_cached_or_build_dir_split_records(
    *,
    cfg: DataConfig,
    split: str,
    train_dir: str,
    pattern: str,
) -> tuple[list[_LoadedRecord], bool]:
    source_files = _discover_data_files(train_dir, pattern)
    cache_dir = _resolve_split_cache_dir(cfg, train_dir)
    cache_key = _build_dir_split_cache_key(
        cfg=cfg,
        train_dir=train_dir,
        pattern=pattern,
        source_files=source_files,
    )
    cache_root = cache_dir / cache_key
    train_cache_path = cache_root / "train.jsonl"
    eval_cache_path = cache_root / "eval.jsonl"

    if train_cache_path.exists() and eval_cache_path.exists():
        cache_path = train_cache_path if split == "train" else eval_cache_path
        logger.info("Using cached directory split for split=%s from %s", split, cache_path)
        return _load_records_from_file(str(cache_path)), True

    records = _load_records_from_dir(train_dir, pattern)
    train_records = _apply_domain_stratified_eval_sampling_split(
        records=records,
        split="train",
        id_field=cfg.id_field,
        eval_ratio=float(cfg.eval_sampling_ratio) if cfg.eval_sampling_ratio is not None else None,
        eval_count=int(cfg.eval_sampling_count) if cfg.eval_sampling_count is not None else None,
        seed=int(cfg.eval_sampling_seed),
        min_eval_samples=int(cfg.eval_sampling_min_samples),
        domain_field_path=cfg.domain_field_path,
    )
    eval_records = _apply_domain_stratified_eval_sampling_split(
        records=records,
        split="eval",
        id_field=cfg.id_field,
        eval_ratio=float(cfg.eval_sampling_ratio) if cfg.eval_sampling_ratio is not None else None,
        eval_count=int(cfg.eval_sampling_count) if cfg.eval_sampling_count is not None else None,
        seed=int(cfg.eval_sampling_seed),
        min_eval_samples=int(cfg.eval_sampling_min_samples),
        domain_field_path=cfg.domain_field_path,
    )

    cache_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": _SPLIT_CACHE_VERSION,
        "train_dir": str(Path(train_dir).resolve()),
        "pattern": pattern,
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "domain_field_path": cfg.domain_field_path,
        "eval_sampling_ratio": cfg.eval_sampling_ratio,
        "eval_sampling_count": cfg.eval_sampling_count,
        "eval_sampling_seed": cfg.eval_sampling_seed,
        "eval_sampling_min_samples": cfg.eval_sampling_min_samples,
    }
    (cache_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_cached_split_records(train_cache_path, train_records)
    _write_cached_split_records(eval_cache_path, eval_records)

    cache_path = train_cache_path if split == "train" else eval_cache_path
    logger.info(
        "Created cached directory split for split=%s cache=%s train_count=%s eval_count=%s",
        split,
        cache_path,
        len(train_records),
        len(eval_records),
    )
    return (train_records if split == "train" else eval_records), True


def _should_use_cached_directory_split(
    *,
    cfg: DataConfig,
    source_kind: str | None,
) -> bool:
    return bool(
        source_kind == "dir"
        and cfg.split_cache_enabled
        and cfg.eval_file is None
        and cfg.eval_dir is None
        and (cfg.eval_sampling_count is not None or cfg.eval_sampling_ratio is not None)
        and not (cfg.split_field and (cfg.train_split or cfg.eval_split))
    )


def _resolve_file_or_dir_override(
    cfg: DataConfig,
    split: str,
) -> tuple[str | None, str | None, str | None]:
    if split == "train":
        if cfg.train_dir:
            return "dir", cfg.train_dir, cfg.train_glob
        if cfg.train_file:
            return "file", cfg.train_file, None
        return None, None, None

    if cfg.eval_dir:
        return "dir", cfg.eval_dir, cfg.eval_glob or cfg.train_glob
    if cfg.eval_file:
        return "file", cfg.eval_file, None
    if cfg.train_dir:
        return "dir", cfg.train_dir, cfg.train_glob
    if cfg.train_file:
        return "file", cfg.train_file, None
    return None, None, None


def load_examples(cfg: DataConfig, split: str, limit: int | None = None) -> list[Example]:
    use_hf_dataset = bool(cfg.hf_dataset_name and str(cfg.hf_dataset_name).strip())
    bidirectional_with_ref = bool(cfg.bidirectional_with_ref)
    if split == "eval" and cfg.eval_bidirectional_with_ref is not None:
        bidirectional_with_ref = bool(cfg.eval_bidirectional_with_ref)

    source_kind, source_value, source_glob = _resolve_file_or_dir_override(cfg, split)
    used_cached_split = False

    if source_kind == "dir" and source_value and _should_use_cached_directory_split(cfg=cfg, source_kind=source_kind):
        records, used_cached_split = _load_cached_or_build_dir_split_records(
            cfg=cfg,
            split=split,
            train_dir=source_value,
            pattern=source_glob or cfg.train_glob,
        )
    elif source_kind == "dir" and source_value:
        records = _load_records_from_dir(source_value, source_glob or cfg.train_glob)
        split_name = cfg.train_split if split == "train" else cfg.eval_split
        records = _resolve_split(records, cfg.split_field, split_name)
    elif source_kind == "file" and source_value:
        records = _load_records_from_file(source_value)
        split_name = cfg.train_split if split == "train" else cfg.eval_split
        records = _resolve_split(records, cfg.split_field, split_name)
    elif use_hf_dataset:
        if split == "train":
            hf_split = cfg.hf_train_split
        else:
            hf_split = cfg.hf_eval_split or cfg.hf_train_split
        records = _wrap_hf_records(
            _load_records_from_hf_dataset(
                dataset_name=cfg.hf_dataset_name or "",
                dataset_config_name=cfg.hf_dataset_config_name,
                split_name=hf_split,
                revision=cfg.hf_revision,
                streaming=cfg.hf_streaming,
                limit=limit,
            )
        )
    else:
        if not source_value:
            raise ValueError(f"No data file configured for split={split}")
        records = _load_records_from_file(source_value)
        split_name = cfg.train_split if split == "train" else cfg.eval_split
        records = _resolve_split(records, cfg.split_field, split_name)

    if (
        source_kind in {"file", "dir"}
        and (not used_cached_split)
        and cfg.eval_file is None
        and cfg.eval_dir is None
        and (cfg.eval_sampling_count is not None or cfg.eval_sampling_ratio is not None)
        and not (cfg.split_field and (cfg.train_split or cfg.eval_split))
    ):
        records = _apply_eval_sampling_split(
            records=records,
            split=split,
            id_field=cfg.id_field,
            eval_ratio=float(cfg.eval_sampling_ratio) if cfg.eval_sampling_ratio is not None else None,
            eval_count=int(cfg.eval_sampling_count) if cfg.eval_sampling_count is not None else None,
            seed=int(cfg.eval_sampling_seed),
            min_eval_samples=int(cfg.eval_sampling_min_samples),
        )

    examples: list[Example] = []
    skipped_bad_source = 0
    skipped_empty_source = 0
    reverse_examples_added = 0
    for idx, record in enumerate(records):
        row = record.row
        if cfg.skip_bad_source and _parse_bad_source_flag(row.get(cfg.is_bad_source_field, False)):
            skipped_bad_source += 1
            continue

        src = _pick_text(row, cfg.src_text_field)
        if not src:
            skipped_empty_source += 1
            continue

        src_lang = _pick_text(row, cfg.src_lang_field, cfg.default_src_lang) or cfg.default_src_lang
        tgt_lang = _pick_text(row, cfg.tgt_lang_field, cfg.default_tgt_lang) or cfg.default_tgt_lang
        src_code = _pick_text(row, cfg.src_lang_code_field, cfg.default_src_lang_code)
        tgt_code = _pick_text(row, cfg.tgt_lang_code_field, cfg.default_tgt_lang_code)
        ref_text = _pick_text(row, cfg.ref_text_field, None)
        teacher_path = _pick_nested_text(row, cfg.domain_field_path, None)
        domain = _derive_domain(teacher_path, record.input_file_path)
        ex_id = _pick_text(row, cfg.id_field, str(idx)) or str(idx)

        before_count = len(examples)
        reached_limit = _append_example_with_optional_reverse(
            examples=examples,
            example_id=ex_id,
            src_text=src,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_lang_code=src_code,
            tgt_lang_code=tgt_code,
            ref_text=ref_text,
            domain=domain,
            teacher_path=teacher_path,
            input_file_path=record.input_file_path,
            bidirectional_with_ref=bidirectional_with_ref,
            limit=limit,
        )
        reverse_examples_added += max(0, len(examples) - before_count - 1)
        if reached_limit:
            break

    logger.info(
        "Loaded %s examples for split=%s (records=%s source=%s cached_split=%s skipped_bad_source=%s skipped_empty_source=%s reverse_examples=%s)",
        len(examples),
        split,
        len(records),
        source_kind or ("hf" if use_hf_dataset else "file"),
        used_cached_split,
        skipped_bad_source,
        skipped_empty_source,
        reverse_examples_added,
    )
    return examples
