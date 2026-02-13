# Gemma 3 27B IT SFT

Full-parameter SFT project for `google/gemma-3-27b-it` on the synthetic translation dataset.

This project enforces:
- Optimizer: `Adafactor`
- Learning rate: `1e-4`
- Global batch size: configurable (default config uses `16`)
- Trainable params: all parameters **except embeddings** (input embeddings frozen, output embeddings optionally frozen)

## 1) Setup (uv)

```bash
cd gemma27b_sft
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

Training preprocessing uses Hugging Face `datasets` (version-pinned in
`requirements.txt` and `pyproject.toml`).

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

Global batch is configurable and checked at startup.

`gradient_accumulation_steps` is auto-computed:

`gradient_accumulation_steps = global_batch_size / (per_device_train_batch_size * WORLD_SIZE)`

So `global_batch_size` must be divisible by `per_device_train_batch_size * WORLD_SIZE`.

## 4) Run Training

Single-node multi-GPU (recommended via `accelerate`):

```bash
accelerate launch --num_processes 8 -m gemma27b_sft.cli --config configs/train_example.yaml
```

27B full SFT on 1 node / 8x H100 (FSDP full shard):

```bash
accelerate launch --num_processes 8 -m gemma27b_sft.cli --config configs/train_8xh100_fsdp.yaml
```

Single process debug run:

```bash
python -m gemma27b_sft.cli --config configs/train_example.yaml
```

## 5) Quick Inference

From `gemma27b_sft` directory:

```bash
bash scripts/sample_infer.sh
```
This uses the same `data.prompt_template` as training (from config).

Optional overrides:

```bash
MODEL_DIR=../outputs/gemma3-27b-it-sft-fsdp/checkpoint-1000 bash scripts/sample_infer.sh
SRC_TEXT="I love this project." bash scripts/sample_infer.sh
CONFIG_PATH=configs/train_8xh100_fsdp.yaml bash scripts/sample_infer.sh
```

## 6) Config Notes

- `model.freeze_output_embeddings: true`
  - Keeps output embeddings frozen too (if not tied to input embeddings)
- `train.gradient_checkpointing: true`
  - Recommended default for Gemma 3 FSDP runs in this project.
- `train.max_seq_length: 1024` (default in provided configs)
  - 27B full SFT at 2048 can easily OOM even on 8x H100 depending on stack version.
  - Increase to 1536/2048 only after 1024 is stable.
- `model.attn_implementation: auto` (recommended default)
  - Uses `flash_attention_2` automatically when CUDA + `flash_attn` are available.
  - Falls back to `sdpa` automatically if FlashAttention is unavailable.
  - Install with `uv pip install flash-attn --no-build-isolation` when your CUDA toolchain supports it.
  - For Gemma 3 FSDP, this project disables FSDP activation checkpointing and keeps
    HF gradient checkpointing to avoid known checkpoint/mask mismatch failures.
- `train.fsdp`
  - Set to `full_shard auto_wrap` for 27B full fine-tuning on 8x H100.
- `train.expected_world_size`
  - Use `8` for 8x H100. The CLI warns on mismatch and raises if FSDP is enabled with `WORLD_SIZE=1`.
- `train.fsdp_transformer_layer_cls_to_wrap`
  - Set `auto` (recommended): class name is auto-detected from the loaded model.
- `train.fsdp_limit_all_gathers`, `train.fsdp_activation_checkpointing`
  - Memory/activation checkpoint controls for FSDP.
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

## 7) Output

The trainer writes to:
- `train.output_dir`

Also writes:
- `resolved_config.yaml`
- Hugging Face checkpoints (`checkpoint-*`)
- Final model + tokenizer on `trainer.save_model()`

## 8) If You Still See CUDA OOM

Check these first:
- Ensure multi-process launch is actually used:
  - `accelerate launch --num_processes 8 ...`
- Verify runtime:
  - `WORLD_SIZE=8`
  - `train.max_seq_length=1024`
- Keep `per_device_train_batch_size=1` for 27B full SFT.
- If you need faster recovery from OOM/debug instability, lower `global_batch_size` (for example `16`).
