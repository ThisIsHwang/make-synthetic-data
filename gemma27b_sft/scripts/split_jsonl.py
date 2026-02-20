#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into train/eval JSONL files."
    )
    parser.add_argument("input_jsonl", help="Path to input JSONL file")
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="Eval split ratio in (0, 1). Default: 0.05",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split. Default: 42",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: input file directory",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Explicit train output path (overrides auto naming)",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="Explicit eval output path (overrides auto naming)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def _ratio_to_label(ratio: float) -> str:
    pct = ratio * 100.0
    rounded = round(pct)
    if abs(pct - rounded) < 1e-9:
        return str(int(rounded))
    return f"{pct:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _derive_output_paths(
    input_path: Path,
    eval_ratio: float,
    output_dir: Path | None,
    train_output: Path | None,
    eval_output: Path | None,
) -> tuple[Path, Path]:
    base_name = input_path.name
    stem = base_name[:-6] if base_name.endswith(".jsonl") else input_path.stem
    out_dir = output_dir if output_dir is not None else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_name = f"{stem}_train_{_ratio_to_label(1.0 - eval_ratio)}.jsonl"
    eval_name = f"{stem}_eval_{_ratio_to_label(eval_ratio)}.jsonl"

    train_path = train_output if train_output is not None else out_dir / train_name
    eval_path = eval_output if eval_output is not None else out_dir / eval_name
    return train_path.resolve(), eval_path.resolve()


def _count_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _compute_eval_count(total: int, eval_ratio: float) -> int:
    if total <= 1:
        return 0
    raw = int(round(total * eval_ratio))
    return max(1, min(total - 1, raw))


def _normalize_line(line: str) -> str:
    if line.endswith("\n"):
        return line
    return f"{line}\n"


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    if args.eval_ratio <= 0.0 or args.eval_ratio >= 1.0:
        raise ValueError(f"--eval-ratio must be in (0, 1), got {args.eval_ratio}")

    train_path, eval_path = _derive_output_paths(
        input_path=input_path,
        eval_ratio=args.eval_ratio,
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        train_output=args.train_output.expanduser().resolve() if args.train_output else None,
        eval_output=args.eval_output.expanduser().resolve() if args.eval_output else None,
    )
    if train_path == eval_path:
        raise ValueError("Train and eval output paths are identical.")
    if input_path in {train_path, eval_path}:
        raise ValueError("Output path cannot be the same as input path.")
    for path in (train_path, eval_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output already exists (use --force): {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    total_records = _count_records(input_path)
    if total_records == 0:
        raise ValueError(f"No non-empty JSONL records found in: {input_path}")
    eval_count = _compute_eval_count(total_records, args.eval_ratio)
    rng = random.Random(args.seed)
    eval_indices = set(rng.sample(range(total_records), eval_count)) if eval_count > 0 else set()

    seen = 0
    train_written = 0
    eval_written = 0
    with (
        input_path.open("r", encoding="utf-8") as fin,
        train_path.open("w", encoding="utf-8") as ftrain,
        eval_path.open("w", encoding="utf-8") as feval,
    ):
        for raw_line in fin:
            if not raw_line.strip():
                continue
            if seen in eval_indices:
                feval.write(_normalize_line(raw_line))
                eval_written += 1
            else:
                ftrain.write(_normalize_line(raw_line))
                train_written += 1
            seen += 1

    if seen != total_records:
        raise RuntimeError(f"Split record count mismatch: seen={seen}, expected={total_records}")
    if train_written + eval_written != total_records:
        raise RuntimeError(
            "Written record count mismatch: "
            f"train={train_written}, eval={eval_written}, total={total_records}"
        )

    print(f"Input: {input_path}")
    print(f"Total records: {total_records}")
    print(f"Train: {train_written} -> {train_path}")
    print(f"Eval:  {eval_written} -> {eval_path}")
    print(f"Seed: {args.seed}  Eval ratio: {args.eval_ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
