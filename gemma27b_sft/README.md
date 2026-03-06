# Gemma 3 27B IT SFT

Full-parameter SFT project for `google/gemma-3-27b-it` on the synthetic translation dataset.

This project enforces:
- Optimizer: `Adafactor`
- Learning rate: `1e-4`
- Global batch size: configurable (default config uses `16`)
- Trainable params: configurable (defaults keep embeddings frozen; can unfreeze for sanity overfit checks)

## 1) Setup (uv)

```bash
cd gemma27b_sft
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

For 8x H100 DeepSpeed full fine-tuning, also install DeepSpeed on the training node:

```bash
uv pip install "deepspeed>=0.14.0"
```

For XCOMET eval, install optional dependency:

```bash
uv pip install "unbabel-comet>=2.2.0"
```

Training preprocessing uses Hugging Face `datasets`
(`>=2.21.0,<5.0.0`; supports 4.x).
When using `datasets` 4.x, use `pyarrow>=21.0.0`.
Gemma 3 training/serving compatibility requires
`transformers>=4.50.0,<5.0.0`.

## 2) Prepare Data

Expected JSONL fields:
- `source_text`
- `target_text`

Example input:
- `../runs/exp001/final_dataset.jsonl`

Set it in `configs/train_example.yaml`:
- `data.train_file`

Prompt formatting is configurable with `data.prompt_template`.
Available placeholders:
- `{source_lang}`, `{src_lang_code}`, `{target_lang}`, `{tgt_lang_code}`, `{text}`
Only `{text}` is required; the others are optional.

### Optional: Split `final_dataset.jsonl` into Train/Eval (95:5)

If you only have one file (for example `final_dataset.jsonl`), use:

```bash
cd gemma27b_sft
python3 scripts/split_jsonl.py /path/to/final_dataset.jsonl
```

Defaults:
- `eval_ratio=0.05`
- `seed=42`
- output names:
  - `*_train_95.jsonl`
  - `*_eval_5.jsonl`

Example output:
- `/path/to/final_dataset_train_95.jsonl`
- `/path/to/final_dataset_eval_5.jsonl`

Useful options:

```bash
python3 scripts/split_jsonl.py /path/to/final_dataset.jsonl --eval-ratio 0.05 --seed 42 --output-dir /path/to/out --force
```

Then set both files in config:

```yaml
data:
  train_file: /path/to/final_dataset_train_95.jsonl
  eval_file: /path/to/final_dataset_eval_5.jsonl
```

If `data.eval_file` is empty/null:
- training still runs normally
- eval is disabled (`evaluation_strategy=no`)
- `eval_steps` is effectively ignored

If `data.eval_file` is set but the file does not exist:
- startup fails with `FileNotFoundError`
- when eval is enabled, training tracks `eval_loss` and keeps the best checkpoint
  (`load_best_model_at_end=true`, `metric_for_best_model=eval_loss`).
  Final `trainer.save_model()` writes the best-eval model weights.

### Optional: Use WMT24++ en-ko as Eval Set

Prepare eval JSONL in the same format (`source_text`, `target_text`):

```bash
cd gemma27b_sft
python3 scripts/prepare_wmt24pp_eval.py --overwrite
```

Default output:
- `../runs/evals/wmt24pp_enko_eval.jsonl`

The generated JSONL also includes:
- `source_lang_code=en`, `target_lang_code=ko`
- language names and WMT24++ metadata fields

`configs/train_8xh100_deepspeed.yaml` is configured to use this file as `data.eval_file`.

## 3) Batch Size Rule

Global batch is configurable and checked at startup.

`gradient_accumulation_steps` is auto-computed:

`gradient_accumulation_steps = global_batch_size / (per_device_train_batch_size * WORLD_SIZE)`

So `global_batch_size` must be divisible by `per_device_train_batch_size * WORLD_SIZE`.

## 4) Run Training

### 4.1) 8x H100 DeepSpeed (Recommended)

Use this path for full fine-tuning on one node with 8x H100.

1) Install runtime deps in the same environment:

```bash
cd gemma27b_sft
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv pip install "deepspeed>=0.14.0"
```

2) Preflight check:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
print("bf16_supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
```

Expected:
- `gpu_count: 8`
- `bf16_supported: True`

3) Confirm config:
- training config: `configs/train_8xh100_deepspeed.yaml`
- DeepSpeed ZeRO-3 config: `configs/deepspeed_zero3_8xh100.json`

4) Launch:

```bash
cd gemma27b_sft
source .venv/bin/activate
torchrun --nproc_per_node=8 -m gemma27b_sft.cli --config configs/train_8xh100_deepspeed.yaml
```

This config currently uses:
- `per_device_train_batch_size: 1`
- `global_batch_size: 16` (gradient accumulation auto-derived)
- `bf16: true`
- `max_seq_length: 1024`
- `train.deepspeed: ./deepspeed_zero3_8xh100.json`
- `train.expected_world_size: 8`

TRL SFTTrainer variant (same config/params, separate entrypoint):

```bash
cd gemma27b_sft
source .venv/bin/activate
torchrun --nproc_per_node=8 -m gemma27b_sft.cli_trl --config configs/train_8xh100_deepspeed.yaml
```

5) Resume from a checkpoint:

```yaml
train:
  resume_from_checkpoint: /abs/path/to/checkpoint-XXXX
```

Then run the same `torchrun` command again.

6) Recommended launch env flags:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
```

### 4.2) Other launch modes

Single-node multi-GPU generic run:

```bash
torchrun --nproc_per_node=8 -m gemma27b_sft.cli --config configs/train_example.yaml
```

Legacy FSDP config:

```bash
accelerate launch --num_processes 8 -m gemma27b_sft.cli --config configs/train_8xh100_fsdp.yaml
```

Single-process debug run:

```bash
python -m gemma27b_sft.cli --config configs/train_example.yaml
```

### 4.3) DeepSpeed runtime checks

At startup, CLI validates:
- `train.deepspeed` file exists.
- `deepspeed` package is installed.
- `train.deepspeed` and `train.fsdp` are not enabled together.
- `WORLD_SIZE` matches launch intent (`train.expected_world_size` warning).
- multi-GPU launch is used when DeepSpeed is enabled.

If any of these fail, training exits early with a clear error message.

## 5) Quick Inference

From `gemma27b_sft` directory:

```bash
bash scripts/sample_infer.sh
```
This uses the same `data.prompt_template` as training (from config).

Optional overrides:

```bash
MODEL_DIR=../outputs/gemma3-27b-it-sft-deepspeed/checkpoint-1000 bash scripts/sample_infer.sh
SRC_TEXT="I love this project." bash scripts/sample_infer.sh
CONFIG_PATH=configs/train_8xh100_deepspeed.yaml bash scripts/sample_infer.sh
```

## 5.1) XCOMET Eval (Loss + Quality)

To evaluate a trained checkpoint on `data.eval_file` with XCOMET:

```bash
cd gemma27b_sft
python3 scripts/eval_xcomet.py \
  --config configs/train_8xh100_deepspeed.yaml \
  --model-dir ../outputs/gemma3-27b-it-sft-deepspeed \
  --max-samples 256 \
  --use-reference \
  --overwrite
```

Outputs:
- `<model-dir>/xcomet_eval_summary.json`
- `<model-dir>/xcomet_eval_predictions.jsonl`

Memory safety note:
- This script generates translations first, then releases the translation model and runs XCOMET.
- To minimize OOM risk, keep `--xcomet-device cpu` (default) or use a different free GPU.
- Running XCOMET during training steps on the same GPU as 27B SFT is not recommended.

## 6) Config Notes

- `model.freeze_input_embeddings: true`
  - Default keeps input embeddings frozen.
  - If train-set fit is poor, try `false` as a sanity check.
- `model.freeze_output_embeddings: true`
  - Keeps output embeddings frozen too (if not tied to input embeddings).
- `model.freeze_vision_encoder: true`
  - Recommended for text-only SFT to reduce memory use.
  - Set `false` only when training with real image inputs.
- `train.gradient_checkpointing: true`
  - Recommended default for Gemma 3 full fine-tuning runs in this project.
- `train.max_seq_length: 1024` (default in provided configs)
  - 27B full SFT at 2048 can easily OOM even on 8x H100 depending on stack version.
  - Increase to 1536/2048 only after 1024 is stable.
- `model.attn_implementation: auto` (recommended default)
  - Uses `flash_attention_2` automatically when CUDA + `flash_attn` are available.
  - Falls back to `sdpa` automatically if FlashAttention is unavailable.
  - Install with `uv pip install flash-attn --no-build-isolation` when your CUDA toolchain supports it.
  - For Gemma 3 FSDP legacy mode, this project disables FSDP activation checkpointing and keeps
    HF gradient checkpointing to avoid known checkpoint/mask mismatch failures.
- `train.deepspeed`
  - Path to DeepSpeed config JSON (example: `configs/deepspeed_zero3_8xh100.json`).
  - Use this for 8x H100 full fine-tuning.
- `train.expected_world_size`
  - Use `8` for 8x H100. The CLI warns on mismatch and raises if DeepSpeed/FSDP is enabled with `WORLD_SIZE=1`.
- `train.fsdp`
  - Optional legacy backend. Do not set together with `train.deepspeed`.
- `train.fsdp_transformer_layer_cls_to_wrap`
  - FSDP-only setting. Set `auto` (recommended): class name is auto-detected from the loaded model.
- `train.fsdp_limit_all_gathers`, `train.fsdp_activation_checkpointing`
  - FSDP-only memory/activation checkpoint controls.
  - Generic rule: when `train.fsdp_activation_checkpointing=true`, HF `gradient_checkpointing` must be `false`.
    The CLI enforces this automatically.
  - Gemma 3 rule: `train.fsdp_activation_checkpointing` is forced to `false` at runtime for stability.
- `data.source_lang_code`, `data.target_lang_code`
  - Set fixed language codes (example: `en`, `ko`).
  - WMT-style codes are supported directly; unknown codes are still accepted.
- `data.source_lang_code_field`, `data.target_lang_code_field`
  - Optional per-row language-code columns for multilingual mixed datasets.
- `data.source_lang_name`, `data.target_lang_name`
  - Use `auto` to infer language names from WMT-style codes.
- `data.log_text_samples`, `data.log_text_max_chars`
  - Logs pre-tokenization training text previews:
    `SOURCE`, rendered `PROMPT`, `TARGET`,
    `CHAT_TEMPLATE_PROMPT`, and `CHAT_TEMPLATE_FULL`.
    This lets you inspect the exact chat-template-applied text before token IDs are built.
  - Set `data.log_text_max_chars: 0` to disable truncation completely.
  - CLI also emits tokenization risk warnings when many samples are truncated.
- Qwen3 compatibility
  - Qwen3 tokenizer inputs are `input_ids` + `attention_mask`; this pipeline
    keeps Qwen3 behavior unchanged.
  - `token_type_ids` is emitted only when required by tokenizer/model runtime
    (and forced for Gemma3 safety), so mixed-model configs do not break.
  - Tokenizer loading is `use_fast=True` first, then automatic `use_fast=False`
    fallback when fast tokenizer backend initialization fails.
  - You can set `model.tokenizer_name_or_path` if model checkpoint path and
    tokenizer path are different.
  - `sample_infer.sh` uses EOS stop by default and adds `<end_of_turn>` stop
    only when that token exists in the tokenizer vocab.

## 7) Output

The trainer writes to:
- `train.output_dir`

Also writes:
- `resolved_config.yaml`
- Hugging Face checkpoints (`checkpoint-*`)
- Final model + tokenizer on `trainer.save_model()`
- Processor artifacts (`preprocessor_config.json`, `processor_config.json`) for vLLM/serving compatibility

## 8) If You Still See CUDA OOM

Check these first:
- Ensure multi-process launch is actually used:
  - `torchrun --nproc_per_node=8 ...` (DeepSpeed)
- Verify runtime:
  - `WORLD_SIZE=8`
  - `train.max_seq_length=1024`
- Keep `per_device_train_batch_size=1` for 27B full SFT.
- If you need faster recovery from OOM/debug instability, lower `global_batch_size` (for example `16`).

DeepSpeed-specific checks:
- Verify `train.deepspeed` points to `configs/deepspeed_zero3_8xh100.json`.
- Do not enable `train.fsdp` together with `train.deepspeed`.
- Keep ZeRO stage at `3` for 27B full fine-tuning.
- If OOM persists, reduce `train.max_seq_length` first (for example `1024 -> 768`) before changing model settings.

Common DeepSpeed launch errors:
- `deepspeed is not installed`
  - Install in the active env: `uv pip install "deepspeed>=0.14.0"`.
- `DeepSpeed is enabled but WORLD_SIZE=1`
  - Use `torchrun --nproc_per_node=8 ...`.
- `global_batch_size must be divisible by per_device_train_batch_size * WORLD_SIZE`
  - Adjust `train.global_batch_size` or launch world size.

## 9) vLLM Error: `GemmaTokenizerFast has no attribute image_token_id`

This is usually an environment mismatch (older `transformers` in the serving env).

Fix on the serving server:

```bash
pip install -U "transformers>=4.50.0,<5.0.0" "tokenizers>=0.21.0"
```

Then verify:

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/path/to/your/sft_model_dir", use_fast=True)
print("has image_token_id:", hasattr(tok, "image_token_id"))
print("image_token_id:", getattr(tok, "image_token_id", None))
PY
```

Expected: `has image_token_id: True`.
