#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _default_output_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "runs" / "evals" / "wmt24pp_enko_eval.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WMT24++ en-ko eval JSONL for gemma27b_sft."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="google/wmt24pp",
        help="HF dataset id. Default: google/wmt24pp",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="en-ko_KR",
        help="HF config name. Default: en-ko_KR",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HF split name. Default: train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="Output JSONL path. Default: runs/evals/wmt24pp_enko_eval.jsonl",
    )
    parser.add_argument(
        "--source-field",
        type=str,
        default="source_text",
        help="Output source field name. Default: source_text",
    )
    parser.add_argument(
        "--target-field",
        type=str,
        default="target_text",
        help="Output target field name. Default: target_text",
    )
    parser.add_argument(
        "--source-lang-code",
        type=str,
        default="en",
        help="Output source language code. Default: en",
    )
    parser.add_argument(
        "--target-lang-code",
        type=str,
        default="ko",
        help="Output target language code. Default: ko",
    )
    parser.add_argument(
        "--source-lang-name",
        type=str,
        default="English",
        help="Output source language name. Default: English",
    )
    parser.add_argument(
        "--target-lang-name",
        type=str,
        default="Korean",
        help="Output target language name. Default: Korean",
    )
    parser.add_argument(
        "--keep-bad-source",
        action="store_true",
        help="Keep rows where is_bad_source=true. Default drops them.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debug. Default: all rows",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def main() -> int:
    args = _parse_args()
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {out_path}")

    dataset = load_dataset(args.dataset, args.config, split=args.split)

    total = 0
    bad_source = 0
    empty = 0
    kept = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for row in dataset:
            if args.max_samples is not None and kept >= args.max_samples:
                break
            total += 1

            is_bad_source = bool(row.get("is_bad_source", False))
            if is_bad_source:
                bad_source += 1
                if not args.keep_bad_source:
                    continue

            source = _safe_str(row.get("source"))
            target = _safe_str(row.get("target"))
            if not source or not target:
                empty += 1
                continue

            record = {
                args.source_field: source,
                args.target_field: target,
                "source_lang_code": args.source_lang_code,
                "target_lang_code": args.target_lang_code,
                "source_lang_name": args.source_lang_name,
                "target_lang_name": args.target_lang_name,
                "wmt24pp_lp": _safe_str(row.get("lp")),
                "wmt24pp_domain": _safe_str(row.get("domain")),
                "wmt24pp_document_id": _safe_str(row.get("document_id")),
                "wmt24pp_segment_id": int(row.get("segment_id", -1)),
                "wmt24pp_is_bad_source": is_bad_source,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"dataset={args.dataset} config={args.config} split={args.split}")
    print(f"output={out_path}")
    print(f"rows_total={total} rows_kept={kept} rows_bad_source={bad_source} rows_empty={empty}")
    print(
        "format="
        f"{args.source_field}/{args.target_field}"
        f" src={args.source_lang_code} tgt={args.target_lang_code}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
