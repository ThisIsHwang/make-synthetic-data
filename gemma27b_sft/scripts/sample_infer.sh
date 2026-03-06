#!/usr/bin/env bash
set -euo pipefail

# Run from gemma27b_sft directory.
# Optional:
#   MODEL_DIR=/path/to/checkpoint ./scripts/sample_infer.sh
#   SRC_TEXT="..." ./scripts/sample_infer.sh
#   CONFIG_PATH=configs/train_8xh100_deepspeed.yaml ./scripts/sample_infer.sh

MODEL_DIR="${MODEL_DIR:-}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-}"

REQUESTED_CONFIG_PATH="${CONFIG_PATH:-}"
if [[ -n "${REQUESTED_CONFIG_PATH}" ]]; then
  if [[ ! -f "${REQUESTED_CONFIG_PATH}" ]]; then
    echo "Config file not found: ${REQUESTED_CONFIG_PATH}" >&2
    exit 1
  fi
  CONFIG_PATH="${REQUESTED_CONFIG_PATH}"
else
  CONFIG_PATH="configs/train_8xh100_deepspeed.yaml"
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    CONFIG_PATH="configs/train_8xh100_fsdp.yaml"
  fi
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    CONFIG_PATH=""
  fi
fi

SRC_TEXT="${SRC_TEXT:-The weather is lovely today. Let us go for a walk by the river.}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python interpreter not found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

echo "MODEL_DIR=${MODEL_DIR}"
echo "TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "SRC_TEXT=${SRC_TEXT}"
echo "PYTHON_BIN=${PYTHON_BIN}"

MODEL_DIR="${MODEL_DIR}" TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH}" CONFIG_PATH="${CONFIG_PATH}" SRC_TEXT="${SRC_TEXT}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma27b_sft.config import SFTConfig, _coerce_dataclass

requested_model_dir = os.environ["MODEL_DIR"].strip()
requested_tokenizer_name_or_path = os.environ["TOKENIZER_NAME_OR_PATH"].strip()
requested_config_path = os.environ["CONFIG_PATH"].strip()
config_path = Path(requested_config_path).expanduser().resolve() if requested_config_path else None
src = os.environ["SRC_TEXT"]

def _resolve_path(path: str | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _resolve_local_model_ref(path: str | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    if raw.startswith(("./", "../", "~")):
        return str((base_dir / p).resolve())
    return raw


def _load_partial_config(path: Path | None) -> tuple[SFTConfig, Path | None]:
    if path is None:
        return SFTConfig(), None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        payload = {}
    cfg = _coerce_dataclass(SFTConfig, payload, path="config")
    base_dir = path.parent.resolve()
    cfg.train.output_dir = _resolve_path(cfg.train.output_dir, base_dir) or cfg.train.output_dir
    cfg.model.name_or_path = _resolve_local_model_ref(cfg.model.name_or_path, base_dir) or cfg.model.name_or_path
    tokenizer_name_or_path = _resolve_local_model_ref(cfg.model.tokenizer_name_or_path, base_dir)
    if tokenizer_name_or_path is not None:
        cfg.model.tokenizer_name_or_path = tokenizer_name_or_path
    resume_from_checkpoint = _resolve_path(cfg.train.resume_from_checkpoint, base_dir)
    if resume_from_checkpoint is not None:
        cfg.train.resume_from_checkpoint = resume_from_checkpoint
    return cfg, path.resolve()


def _is_loadable_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists() and any(
        path.glob(pattern)
        for pattern in (
            "model.safetensors*",
            "pytorch_model.bin*",
            "adapter_model.safetensors*",
            "adapter_model.bin*",
        )
    )


def _has_tokenizer_artifacts(path: Path) -> bool:
    if not path.is_dir():
        return False
    candidates = (
        "tokenizer.json",
        "tokenizer_config.json",
        "spiece.model",
        "tokenizer.model",
        "vocab.json",
    )
    return any((path / name).exists() for name in candidates)


def _pick_latest_checkpoint(path: Path) -> Path | None:
    checkpoints = [p for p in path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return checkpoints[0].resolve()


def _find_run_config_path(base_cfg: SFTConfig, fallback_path: Path | None, model_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if model_dir is not None:
        candidates.append(model_dir / "resolved_config.yaml")
        if model_dir.name.startswith("checkpoint-"):
            candidates.append(model_dir.parent / "resolved_config.yaml")
    output_dir = str(base_cfg.train.output_dir or "").strip()
    if output_dir:
        candidates.append(Path(output_dir).expanduser().resolve() / "resolved_config.yaml")
    if fallback_path is not None:
        candidates.append(fallback_path)
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return fallback_path


def _default_model_dir_from_cfg(cfg: SFTConfig) -> Path | None:
    output_dir_raw = str(cfg.train.output_dir or "").strip()
    if output_dir_raw:
        output_dir = Path(output_dir_raw).expanduser().resolve()
        if _is_loadable_model_dir(output_dir):
            return output_dir
        checkpoint_dir = _pick_latest_checkpoint(output_dir)
        if checkpoint_dir is not None:
            return checkpoint_dir
    for fallback in (
        Path("../outputs/gemma3-27b-it-sft-deepspeed"),
        Path("../outputs/gemma3-27b-it-sft-fsdp"),
    ):
        resolved = fallback.expanduser().resolve()
        if _is_loadable_model_dir(resolved):
            return resolved
        checkpoint_dir = _pick_latest_checkpoint(resolved)
        if checkpoint_dir is not None:
            return checkpoint_dir
    return None


cfg, selected_config_path = _load_partial_config(config_path)
model_dir = Path(requested_model_dir).expanduser().resolve() if requested_model_dir else _default_model_dir_from_cfg(cfg)
selected_config_path = _find_run_config_path(cfg, selected_config_path, model_dir)
cfg, selected_config_path = _load_partial_config(selected_config_path)
if model_dir is None and not requested_model_dir:
    model_dir = _default_model_dir_from_cfg(cfg)
if model_dir is None:
    raise FileNotFoundError("Could not determine a model directory. Set MODEL_DIR explicitly or check train.output_dir.")
if not model_dir.exists():
    raise FileNotFoundError(f"Model directory not found: {model_dir}")

tokenizer_name_or_path = requested_tokenizer_name_or_path
if not tokenizer_name_or_path:
    tokenizer_name_or_path = str(cfg.model.tokenizer_name_or_path or "").strip()
if not tokenizer_name_or_path:
    if _has_tokenizer_artifacts(model_dir):
        tokenizer_name_or_path = str(model_dir)
    else:
        output_dir_raw = str(cfg.train.output_dir or "").strip()
        if output_dir_raw:
            output_dir = Path(output_dir_raw).expanduser().resolve()
            if _has_tokenizer_artifacts(output_dir):
                tokenizer_name_or_path = str(output_dir)
if not tokenizer_name_or_path:
    tokenizer_name_or_path = str(cfg.model.name_or_path).strip() or str(model_dir)

trust_remote_code = bool(cfg.model.trust_remote_code)
print(f"USING_CONFIG_PATH={selected_config_path or '<none>'}")
print(f"USING_MODEL_DIR={model_dir}")
print(f"USING_TOKENIZER_NAME_OR_PATH={tokenizer_name_or_path}")
print(f"TRUST_REMOTE_CODE={trust_remote_code}")

tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, **tokenizer_kwargs)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=False, **tokenizer_kwargs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float32
if torch.cuda.is_available():
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

attn_implementation = str(cfg.model.attn_implementation or "auto").strip().lower()
if attn_implementation in {"", "auto"}:
    attn_implementation = "sdpa"

model = AutoModelForCausalLM.from_pretrained(
    str(model_dir),
    torch_dtype=dtype,
    device_map="auto",
    attn_implementation=attn_implementation,
    trust_remote_code=trust_remote_code,
)
model.eval()

data_cfg = cfg.data

prompt_template = str(data_cfg.prompt_template)
src_lang_code = str(os.environ.get("SRC_LANG_CODE") or data_cfg.source_lang_code).strip()
tgt_lang_code = str(os.environ.get("TGT_LANG_CODE") or data_cfg.target_lang_code).strip()
src_lang_name_cfg = str(data_cfg.source_lang_name).strip()
tgt_lang_name_cfg = str(data_cfg.target_lang_name).strip()
src_lang_name = str(os.environ.get("SRC_LANG_NAME") or src_lang_name_cfg).strip()
tgt_lang_name = str(os.environ.get("TGT_LANG_NAME") or tgt_lang_name_cfg).strip()

try:
    from gemma27b_sft.data import _normalize_code, _resolve_language_name  # pylint: disable=import-error
except Exception:  # pylint: disable=broad-except
    def _normalize_code(code: str) -> str:
        return code.strip().replace("_", "-").lower()

    def _resolve_language_name(name: str, code: str) -> str:
        if name and name.strip() and name.strip().lower() != "auto":
            return name.strip()
        return code

src_lang_code = _normalize_code(src_lang_code)
tgt_lang_code = _normalize_code(tgt_lang_code)
source_lang = _resolve_language_name(src_lang_name, src_lang_code)
target_lang = _resolve_language_name(tgt_lang_name, tgt_lang_code)

try:
    prompt = prompt_template.format(
        source_lang=source_lang,
        src_lang_code=src_lang_code,
        target_lang=target_lang,
        tgt_lang_code=tgt_lang_code,
        text=src,
    )
except KeyError as exc:
    missing = exc.args[0]
    raise ValueError(
        f"Unknown placeholder in prompt_template: {missing}. "
        "Allowed: source_lang, src_lang_code, target_lang, tgt_lang_code, text."
    ) from exc

with torch.inference_mode():
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        input_ids = enc["input_ids"]
        unk_id = tokenizer.unk_token_id
        eos_id = tokenizer.eos_token_id
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        stop_ids = []
        if isinstance(eos_id, int) and eos_id >= 0:
            stop_ids.append(eos_id)
        if (
            isinstance(eot_id, int)
            and eot_id >= 0
            and (unk_id is None or eot_id != unk_id)
            and eot_id not in stop_ids
        ):
            stop_ids.append(eot_id)
        gen_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,
        }
        if stop_ids:
            gen_kwargs["eos_token_id"] = stop_ids if len(stop_ids) > 1 else stop_ids[0]
        out = model.generate(
            **enc,
            **gen_kwargs,
        )
        gen = out[0][input_ids.shape[1] :]
    else:
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,
        }
        if isinstance(tokenizer.eos_token_id, int) and tokenizer.eos_token_id >= 0:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        out = model.generate(**enc, **gen_kwargs)
        gen = out[0][enc["input_ids"].shape[1] :]

print("\n=== Model Output ===")
print(tokenizer.decode(gen, skip_special_tokens=True).strip())
PY
