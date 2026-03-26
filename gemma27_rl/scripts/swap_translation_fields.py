#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Swap source/target-style fields in JSONL or JSON files.")
    parser.add_argument("inputs", nargs="+", help="Input file paths")
    parser.add_argument(
        "--swap",
        nargs=2,
        metavar=("FIELD_A", "FIELD_B"),
        action="append",
        default=[],
        help="Field pair to swap. Repeatable. Example: --swap source_text target_text",
    )
    parser.add_argument(
        "--reverse-arrow-field",
        action="append",
        default=[],
        metavar="FIELD",
        help="Reverse arrow-style values like en->ko to ko->en for the given field. Repeatable.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--output-dir",
        help="Write transformed files into this directory, preserving input filenames.",
    )
    group.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files in place.",
    )
    parser.add_argument(
        "--suffix",
        default=".swapped",
        help="Suffix to insert before the extension when not using --in-place or --output-dir. Default: .swapped",
    )
    return parser


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jsonlines"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}: line {line_no} is not a JSON object")
                rows.append(payload)
        return rows, "jsonl"

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
            if len(rows) != len(payload):
                raise ValueError(f"{path}: JSON list contains non-object rows")
            return rows, "json"
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            rows = [row for row in payload["data"] if isinstance(row, dict)]
            if len(rows) != len(payload["data"]):
                raise ValueError(f"{path}: payload['data'] contains non-object rows")
            return rows, "json_data"
        raise ValueError(f"{path}: unsupported JSON structure")

    raise ValueError(f"{path}: unsupported file type {path.suffix}")


def _write_rows(path: Path, rows: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if fmt == "json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    if fmt == "json_data":
        path.write_text(json.dumps({"data": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    raise ValueError(f"Unsupported output format: {fmt}")


def _reverse_arrow_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if "->" not in value:
        return value
    left, right = value.split("->", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return value
    return f"{right}->{left}"


def _swap_row(
    row: dict[str, Any],
    *,
    swaps: list[tuple[str, str]],
    reverse_arrow_fields: list[str],
) -> dict[str, Any]:
    out = dict(row)
    for left, right in swaps:
        out[left], out[right] = out.get(right), out.get(left)
    for field in reverse_arrow_fields:
        if field in out:
            out[field] = _reverse_arrow_value(out[field])
    return out


def _default_output_path(path: Path, suffix: str) -> Path:
    if not suffix:
        raise ValueError("--suffix must not be empty")
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _resolve_output_path(path: Path, *, output_dir: str | None, in_place: bool, suffix: str) -> Path:
    if in_place:
        return path
    if output_dir:
        return Path(output_dir).resolve() / path.name
    return _default_output_path(path, suffix)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    swap_pairs = [(str(left), str(right)) for left, right in args.swap]
    if not swap_pairs:
        parser.error("at least one --swap FIELD_A FIELD_B pair is required")

    output_dir = str(args.output_dir).strip() if args.output_dir else None
    reverse_arrow_fields = [str(field).strip() for field in args.reverse_arrow_field if str(field).strip()]

    for raw_path in args.inputs:
        input_path = Path(raw_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        rows, fmt = _read_rows(input_path)
        swapped_rows = [
            _swap_row(row, swaps=swap_pairs, reverse_arrow_fields=reverse_arrow_fields)
            for row in rows
        ]
        output_path = _resolve_output_path(
            input_path,
            output_dir=output_dir,
            in_place=bool(args.in_place),
            suffix=str(args.suffix),
        )
        _write_rows(output_path, swapped_rows, fmt)
        print(f"{input_path} -> {output_path} rows={len(swapped_rows)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
