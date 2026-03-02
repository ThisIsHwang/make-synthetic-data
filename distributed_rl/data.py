"""Data loading pipeline: raw data → TRL-compatible HuggingFace Dataset.

=== What TRL's GRPOTrainer Expects ===

TRL's GRPOTrainer requires a HuggingFace ``Dataset`` with a ``"prompt"`` column.
Each prompt is a list of chat messages:

    [{"role": "user", "content": "Translate the following English text..."}]

TRL then applies the model's chat template to convert this into the actual
token sequence.  For example, Qwen's template wraps it in ``<|im_start|>``
tags, and Gemma uses ``<start_of_turn>``.

=== Additional Columns for Reward Functions ===

Besides ``"prompt"``, we also include ``"src_text"`` and ``"ref_text"`` columns.
TRL passes ALL dataset columns as ``**kwargs`` to each reward function:

    reward_func(completions=["translation1", ...], src_text=["source1", ...], ...)

This is how our reward models (MetricX, XComet) access the source text and
reference translation — they extract them from kwargs.

=== Data Loading Flow ===

  1. Load raw records from file (JSONL/JSON/Parquet) or HuggingFace Hub
  2. For each record:
     a. Extract source text, language codes, reference translation
     b. Skip bad/empty sources if configured
     c. Format the translation prompt using the template from config
     d. Wrap the prompt in chat message format: [{"role": "user", "content": ...}]
  3. Return as HuggingFace Dataset with columns: prompt, src_text, ref_text

Ref: Adapted from qwen3.5-35b-a3b/qwen35_moe_rl/data.py (JSONL/HF loading)
Ref: TRL dataset format — https://huggingface.co/docs/trl/grpo_trainer#dataset-format
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import DataConfig, PromptConfig
from .prompting import format_translation_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw record loading — handles multiple file formats
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL (JSON Lines) file where each line is a JSON object.

    JSONL is the most common format for translation datasets:
      {"id": "1", "src_text": "Hello world", "ref_text": "안녕하세요", ...}
      {"id": "2", "src_text": "Good morning", "ref_text": "좋은 아침", ...}

    Invalid lines are skipped with a warning (robust to partial corruption).
    """
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
    """Read a JSON file — expects either a list of objects or an object with a "data" key."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return [row for row in payload["data"] if isinstance(row, dict)]
        return [payload]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet file using HuggingFace datasets library."""
    from datasets import load_dataset as _load_dataset

    ds = _load_dataset("parquet", data_files=str(path), split="train")
    return [dict(row) for row in ds]


def _load_records_from_file(path: str) -> list[dict[str, Any]]:
    """Dispatch to the correct file reader based on file extension."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".jsonl", ".jsonlines"}:
        return _read_jsonl(file_path)
    if suffix == ".json":
        return _read_json(file_path)
    if suffix == ".parquet":
        return _read_parquet(file_path)
    raise ValueError(f"Unsupported data file extension: {suffix}")


def _load_records_from_hf(
    dataset_name: str,
    config_name: str | None,
    split_name: str,
    revision: str | None,
    streaming: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Load records from a HuggingFace Hub dataset.

    Example:
      dataset_name = "google/wmt24pp"
      config_name = "en-ko_KR"   (language pair)
      split_name = "train"

    When ``streaming=True``, records are loaded lazily (one at a time) to avoid
    downloading the entire dataset.  This is useful for very large datasets.
    """
    from datasets import load_dataset as _load_dataset

    kwargs: dict[str, Any] = {"split": split_name, "streaming": streaming}
    if revision:
        kwargs["revision"] = revision
    ds = _load_dataset(dataset_name, config_name, **kwargs)

    rows: list[dict[str, Any]] = []
    if streaming:
        # In streaming mode, iterate until we reach the limit.
        for row in ds:
            if isinstance(row, dict):
                rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
        return rows

    # Non-streaming: dataset is fully loaded in memory.
    if limit is not None and limit < len(ds):
        ds = ds.select(range(limit))
    return [dict(row) for row in ds]


def _pick(row: dict[str, Any], field: str, default: str | None = None) -> str | None:
    """Extract a string field from a row, falling back to a default.

    Returns None if the field is missing, None-valued, or empty after stripping.
    This is used to safely handle datasets with inconsistent schemas.
    """
    if field not in row or row[field] is None:
        return default
    value = str(row[field]).strip()
    return value if value else default


# ---------------------------------------------------------------------------
# Public API: load dataset in TRL-compatible format
# ---------------------------------------------------------------------------

def load_dataset_for_trl(
    data_cfg: DataConfig,
    prompt_cfg: PromptConfig,
    split: str = "train",
) -> Any:
    """Load data and convert to TRL GRPOTrainer format.

    This is the main entry point for data preparation.  It:
      1. Loads raw records from the configured source (file or HuggingFace)
      2. Filters out bad/empty sources
      3. Formats each record into a chat-style prompt
      4. Returns a HuggingFace Dataset with the columns TRL needs

    Returns:
        HuggingFace ``Dataset`` with columns:

        - ``prompt``: list of chat messages, e.g.
          ``[{"role": "user", "content": "Translate..."}]``
          TRL feeds this to the model and generates ``num_generations``
          completions per prompt.

        - ``src_text``: source text (passed to reward functions via ``**kwargs``)
          MetricX/XComet need this to score translation quality.

        - ``ref_text``: reference translation (for reference-based scoring)
          Can be empty if using QE (quality estimation) mode.
    """
    from datasets import Dataset

    # Decide which data source and limit to use based on the split.
    use_hf = bool(data_cfg.hf_dataset_name and str(data_cfg.hf_dataset_name).strip())
    limit = data_cfg.limit if split == "train" else data_cfg.eval_limit

    if use_hf:
        hf_split = data_cfg.hf_train_split if split == "train" else (data_cfg.hf_eval_split or data_cfg.hf_train_split)
        records = _load_records_from_hf(
            dataset_name=data_cfg.hf_dataset_name or "",
            config_name=data_cfg.hf_dataset_config_name,
            split_name=hf_split,
            revision=data_cfg.hf_revision,
            streaming=data_cfg.hf_streaming,
            limit=limit,
        )
    else:
        data_file = data_cfg.train_file if split == "train" else (data_cfg.eval_file or data_cfg.train_file)
        if not data_file:
            raise ValueError(f"No data file configured for split={split}")
        records = _load_records_from_file(data_file)
        if limit is not None:
            records = records[:limit]

    # --- Convert raw records to TRL format ---
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        # Skip records flagged as bad source (e.g. garbled text in the dataset).
        if data_cfg.skip_bad_source and bool(record.get(data_cfg.is_bad_source_field, False)):
            continue

        # Extract source text (the text to translate).
        # Skip if source is empty — can't translate nothing.
        src_text = _pick(record, data_cfg.src_text_field)
        if not src_text:
            continue

        # Extract language names and codes, falling back to defaults.
        # The defaults are set in DataConfig (e.g. English → Korean).
        src_lang = _pick(record, data_cfg.src_lang_field, data_cfg.default_src_lang) or data_cfg.default_src_lang
        tgt_lang = _pick(record, data_cfg.tgt_lang_field, data_cfg.default_tgt_lang) or data_cfg.default_tgt_lang
        src_code = _pick(record, data_cfg.src_lang_code_field, data_cfg.default_src_lang_code) or ""
        tgt_code = _pick(record, data_cfg.tgt_lang_code_field, data_cfg.default_tgt_lang_code) or ""
        ref_text = _pick(record, data_cfg.ref_text_field) or ""

        # Format the translation prompt with the template from config.
        # e.g. "You are a professional English (en) to Korean (ko) translator..."
        prompt_text = format_translation_prompt(
            src_text=src_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_lang_code=src_code,
            tgt_lang_code=tgt_code,
            template=prompt_cfg.template,
        )

        # Wrap the prompt in TRL's expected chat format.
        # TRL applies the model's chat template to this list.
        # For Qwen: [{"role": "user", "content": "..."}] → "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
        # The model then generates the completion (translation) from there.
        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            # These extra columns are passed as **kwargs to reward functions.
            # TRL doesn't use them directly, but our reward wrappers read them.
            "src_text": src_text,
            "ref_text": ref_text,
        })

    logger.info("Loaded %d examples for split=%s (TRL format)", len(rows), split)
    return Dataset.from_list(rows)
