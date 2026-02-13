# Gemma 27B IT SFT

Full-parameter SFT project for `google/gemma-2-27b-it` on the synthetic translation dataset.

This project enforces:
- Optimizer: `Adafactor`
- Learning rate: `1e-4`
- Global batch size: `64`
- Trainable params: all parameters **except embeddings** (input embeddings frozen, output embeddings optionally frozen)

## 1) Setup (uv)

```bash
cd gemma27b_sft
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

This project uses a pure JSONL loader for training data preprocessing
(no `datasets` / `pyarrow` dependency in the training path).

## 2) Prepare Data

Expected JSONL fields:
- `source_text`
- `target_text`

Example input:
- `../runs/exp001/final_dataset.jsonl`

Set it in `configs/train_example.yaml`:
- `data.train_file`

Prompt formatting is configurable with `data.prompt_template`.
Default template matches:
- `{source_lang}`, `{src_lang_code}`, `{target_lang}`, `{tgt_lang_code}`, `{text}`

## 3) Batch Size Rule

Global batch is fixed to `64` and is checked at startup.

`gradient_accumulation_steps` is auto-computed:

`gradient_accumulation_steps = 64 / (per_device_train_batch_size * WORLD_SIZE)`

So `64` must be divisible by `per_device_train_batch_size * WORLD_SIZE`.

## 4) Run Training

Single-node multi-GPU (recommended via `accelerate`):

```bash
accelerate launch --num_processes 8 -m gemma27b_sft.cli --config configs/train_example.yaml
```

Single process debug run:

```bash
python -m gemma27b_sft.cli --config configs/train_example.yaml
```

## 5) Config Notes

- `model.freeze_output_embeddings: true`
  - Keeps output embeddings frozen too (if not tied to input embeddings)
- `train.gradient_checkpointing: true`
  - Recommended for Gemma 27B full SFT memory
- `model.attn_implementation: auto` (recommended default)
  - Uses `flash_attention_2` automatically when CUDA + `flash_attn` are available.
  - Falls back to `sdpa` automatically if FlashAttention is unavailable.
  - Install with `uv pip install flash-attn --no-build-isolation` when your CUDA toolchain supports it.
- `data.source_lang_code`, `data.target_lang_code`
  - Set fixed language codes (example: `en`, `ko`).
  - WMT-style codes are supported directly; unknown codes are still accepted.
- `data.source_lang_code_field`, `data.target_lang_code_field`
  - Optional per-row language-code columns for multilingual mixed datasets.
- `data.source_lang_name`, `data.target_lang_name`
  - Use `auto` to infer language names from WMT-style codes.

## 6) Output

The trainer writes to:
- `train.output_dir`

Also writes:
- `resolved_config.yaml`
- Hugging Face checkpoints (`checkpoint-*`)
- Final model + tokenizer on `trainer.save_model()`
