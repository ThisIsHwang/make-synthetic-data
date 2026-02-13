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

## 2) Prepare Data

Expected JSONL fields:
- `source_text`
- `target_text`

Example input:
- `../runs/exp001/final_dataset.jsonl`

Set it in `configs/train_example.yaml`:
- `data.train_file`

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
- `model.attn_implementation: flash_attention_2`
  - Use if your environment supports it

## 6) Output

The trainer writes to:
- `train.output_dir`

Also writes:
- `resolved_config.yaml`
- Hugging Face checkpoints (`checkpoint-*`)
- Final model + tokenizer on `trainer.save_model()`
