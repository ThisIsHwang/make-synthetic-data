#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemma27b_sft.config import DataConfig, ModelConfig, SFTConfig, _coerce_dataclass
from gemma27b_sft.data import _messages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with XCOMET on eval JSONL.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_8xh100_deepspeed.yaml"))
    parser.add_argument("--model-dir", type=Path, default=None, help="Checkpoint or output dir to evaluate")
    parser.add_argument("--eval-file", type=Path, default=None, help="Eval JSONL path override")
    parser.add_argument("--source-field", type=str, default=None)
    parser.add_argument("--target-field", type=str, default=None)
    parser.add_argument("--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gen-device", type=str, default="cuda")
    parser.add_argument("--xcomet-model", type=str, default="Unbabel/XCOMET-XL")
    parser.add_argument("--xcomet-device", type=str, default="cpu")
    parser.add_argument("--xcomet-batch-size", type=int, default=4)
    parser.add_argument("--use-reference", action="store_true", help="Use references for XCOMET scoring")
    parser.add_argument("--skip-xcomet", action="store_true", help="Run generation only")
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Output summary json path (default: <model_dir>/xcomet_eval_summary.json)",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=None,
        help="Output per-sample jsonl path (default: <model_dir>/xcomet_eval_predictions.jsonl)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_path(path: str | Path | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _load_partial_config(config_path: Path) -> tuple[DataConfig, ModelConfig, Path | None]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    cfg = _coerce_dataclass(SFTConfig, payload, path="config")

    data_cfg = cfg.data
    model_cfg = cfg.model
    train_output = cfg.train.output_dir
    base_dir = config_path.parent.resolve()
    data_cfg.train_file = str(_resolve_path(data_cfg.train_file, base_dir))
    data_cfg.eval_file = str(_resolve_path(data_cfg.eval_file, base_dir)) if data_cfg.eval_file else None
    train_output_path = _resolve_path(train_output, base_dir) if train_output else None
    return data_cfg, model_cfg, train_output_path


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_eval_rows(eval_file: Path, source_field: str, target_field: str, max_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with eval_file.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            src = _safe_string(row.get(source_field))
            tgt = _safe_string(row.get(target_field))
            if not src or not tgt:
                continue
            rows.append(row)
            if len(rows) >= max_samples:
                break
    return rows


def _load_tokenizer(tokenizer_name_or_path: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _render_prompt(tokenizer, prompt_messages: list[dict[str, str]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(rendered, str):
            return rendered
    except Exception:
        pass

    # Fallback for tokenizer without chat template.
    user_text = prompt_messages[0]["content"] if prompt_messages else ""
    return f"USER: {user_text}\n\nASSISTANT:"


def _prepare_generation_device(requested: str) -> str:
    req = requested.strip().lower()
    if req == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        if req.startswith("cuda"):
            return req
        return "cuda:0"
    return "cpu"


def _move_to_device(batch: Any, device: str) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device) for v in batch)
    if isinstance(batch, list):
        return [_move_to_device(v, device) for v in batch]
    return batch


def _generation_stop_ids(tokenizer) -> list[int]:
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    if (
        isinstance(eot_id, int)
        and eot_id >= 0
        and (unk_id is None or eot_id != unk_id)
        and eot_id not in stop_ids
    ):
        stop_ids.append(int(eot_id))
    return stop_ids


def _generate_translations(
    model,
    tokenizer,
    data_cfg: DataConfig,
    rows: list[dict[str, Any]],
    generation_batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    hypotheses: list[str] = []
    stop_ids = _generation_stop_ids(tokenizer)

    for start in range(0, len(rows), generation_batch_size):
        batch_rows = rows[start : start + generation_batch_size]
        prompts: list[str] = []
        for row in batch_rows:
            src = _safe_string(row.get(data_cfg.source_field))
            tgt = _safe_string(row.get(data_cfg.target_field))
            prompt_messages, _ = _messages(data_cfg, row, src, tgt)
            prompts.append(_render_prompt(tokenizer, prompt_messages))

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_ids if stop_ids else None,
                pad_token_id=tokenizer.pad_token_id,
            )

        # With left padding, generation starts after the full padded prompt width.
        prompt_width = int(encoded["input_ids"].shape[1])
        for row_idx in range(outputs.shape[0]):
            gen_ids = outputs[row_idx, prompt_width:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            hypotheses.append(text)

        print(f"[generate] {min(start + len(batch_rows), len(rows))}/{len(rows)}")

    return hypotheses


def _load_xcomet(model_name: str, device: str):
    try:
        from comet import download_model, load_from_checkpoint
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "XCOMET requires unbabel-comet. Install with: pip install 'unbabel-comet>=2.2.0'"
        ) from exc

    model_path = download_model(model_name)
    scorer = load_from_checkpoint(model_path)
    if device.startswith("cuda") and torch.cuda.is_available():
        scorer.to(torch.device(device))
    scorer.eval()
    return scorer


def _score_xcomet(
    scorer,
    rows: list[dict[str, Any]],
    hypotheses: list[str],
    source_field: str,
    target_field: str,
    use_reference: bool,
    batch_size: int,
    device: str,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_hyp = hypotheses[start : start + batch_size]
        payload: list[dict[str, str]] = []
        for row, hyp in zip(batch_rows, batch_hyp):
            item = {"src": _safe_string(row.get(source_field)), "mt": hyp}
            if use_reference:
                item["ref"] = _safe_string(row.get(target_field))
            payload.append(item)

        batch_inputs = scorer.prepare_for_inference(payload)
        batch_inputs = _move_to_device(batch_inputs, device)
        with torch.no_grad():
            pred = scorer.predict_step(batch_inputs)
        score_tensor = pred.get("scores") if isinstance(pred, dict) else getattr(pred, "scores", None)
        if score_tensor is None:
            raise RuntimeError("XCOMET returned no scores.")
        batch_scores = torch.as_tensor(score_tensor).detach().float().cpu().tolist()
        scores.extend(float(v) for v in batch_scores)
        print(f"[xcomet] {min(start + len(batch_rows), len(rows))}/{len(rows)}")
    return scores


def main() -> int:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    data_cfg, model_cfg, train_output_dir = _load_partial_config(config_path)
    if args.source_field:
        data_cfg.source_field = args.source_field
    if args.target_field:
        data_cfg.target_field = args.target_field

    eval_file = args.eval_file.expanduser().resolve() if args.eval_file else None
    if eval_file is None:
        if not data_cfg.eval_file:
            raise ValueError("No eval file. Set --eval-file or data.eval_file in config.")
        eval_file = Path(data_cfg.eval_file).expanduser().resolve()
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    model_dir = args.model_dir.expanduser().resolve() if args.model_dir else train_output_dir
    if model_dir is None:
        raise ValueError("No model dir. Set --model-dir or train.output_dir in config.")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    output_summary = (
        args.output_summary.expanduser().resolve()
        if args.output_summary
        else (model_dir / "xcomet_eval_summary.json")
    )
    output_predictions = (
        args.output_predictions.expanduser().resolve()
        if args.output_predictions
        else (model_dir / "xcomet_eval_predictions.jsonl")
    )
    if (output_summary.exists() or output_predictions.exists()) and not args.overwrite:
        raise FileExistsError("Output exists. Use --overwrite.")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_eval_rows(
        eval_file=eval_file,
        source_field=data_cfg.source_field,
        target_field=data_cfg.target_field,
        max_samples=max(1, args.max_samples),
    )
    if not rows:
        raise ValueError(f"No usable eval rows in {eval_file}")

    tokenizer_name_or_path = (
        args.tokenizer_name_or_path
        or model_cfg.tokenizer_name_or_path
        or str(model_dir)
    )
    tokenizer = _load_tokenizer(tokenizer_name_or_path)

    gen_device = _prepare_generation_device(args.gen_device)
    dtype = torch.float32
    if gen_device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(gen_device)
    model.eval()

    hypotheses = _generate_translations(
        model=model,
        tokenizer=tokenizer,
        data_cfg=data_cfg,
        rows=rows,
        generation_batch_size=max(1, args.generation_batch_size),
        max_input_tokens=max(32, args.max_input_tokens),
        max_new_tokens=max(8, args.max_new_tokens),
        device=gen_device,
    )

    # Release generation model first to avoid overlapping GPU memory with XCOMET.
    del model
    if gen_device.startswith("cuda"):
        torch.cuda.empty_cache()

    xcomet_scores: list[float] = []
    if not args.skip_xcomet:
        xcomet_device = _prepare_generation_device(args.xcomet_device)
        scorer = _load_xcomet(args.xcomet_model, xcomet_device)
        xcomet_scores = _score_xcomet(
            scorer=scorer,
            rows=rows,
            hypotheses=hypotheses,
            source_field=data_cfg.source_field,
            target_field=data_cfg.target_field,
            use_reference=bool(args.use_reference),
            batch_size=max(1, args.xcomet_batch_size),
            device=xcomet_device,
        )

    with output_predictions.open("w", encoding="utf-8") as f:
        for idx, (row, hyp) in enumerate(zip(rows, hypotheses)):
            out = {
                "idx": idx,
                "source_text": _safe_string(row.get(data_cfg.source_field)),
                "reference_text": _safe_string(row.get(data_cfg.target_field)),
                "hypothesis_text": hyp,
            }
            if idx < len(xcomet_scores):
                out["xcomet_score"] = float(xcomet_scores[idx])
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary: dict[str, Any] = {
        "model_dir": str(model_dir),
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "eval_file": str(eval_file),
        "num_samples": len(rows),
        "config_data": asdict(data_cfg),
        "generation": {
            "device": gen_device,
            "batch_size": int(args.generation_batch_size),
            "max_input_tokens": int(args.max_input_tokens),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "xcomet": {
            "enabled": not args.skip_xcomet,
            "model": args.xcomet_model,
            "device": args.xcomet_device,
            "batch_size": int(args.xcomet_batch_size),
            "use_reference": bool(args.use_reference),
        },
        "outputs": {
            "summary": str(output_summary),
            "predictions": str(output_predictions),
        },
    }
    if xcomet_scores:
        summary["xcomet"]["mean"] = float(statistics.fmean(xcomet_scores))
        summary["xcomet"]["stdev"] = float(statistics.pstdev(xcomet_scores)) if len(xcomet_scores) > 1 else 0.0
        summary["xcomet"]["min"] = float(min(xcomet_scores))
        summary["xcomet"]["max"] = float(max(xcomet_scores))

    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"model_dir={model_dir}")
    print(f"eval_file={eval_file}")
    print(f"num_samples={len(rows)}")
    if xcomet_scores:
        print(f"xcomet_mean={summary['xcomet']['mean']:.6f} stdev={summary['xcomet']['stdev']:.6f}")
    else:
        print("xcomet_skipped=true")
    print(f"summary={output_summary}")
    print(f"predictions={output_predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
