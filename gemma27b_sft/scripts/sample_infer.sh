#!/usr/bin/env bash
set -euo pipefail

# Run from gemma27b_sft directory.
# Optional:
#   MODEL_DIR=/path/to/checkpoint ./scripts/sample_infer.sh
#   SRC_TEXT="..." ./scripts/sample_infer.sh

MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  MODEL_DIR="$(ls -dt ../outputs/gemma3-27b-it-sft-fsdp/checkpoint-* 2>/dev/null | head -1 || true)"
fi
if [[ -z "${MODEL_DIR}" ]]; then
  MODEL_DIR="../outputs/gemma3-27b-it-sft-fsdp"
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

SRC_TEXT="${SRC_TEXT:-The weather is lovely today. Let us go for a walk by the river.}"

echo "MODEL_DIR=${MODEL_DIR}"
echo "SRC_TEXT=${SRC_TEXT}"

MODEL_DIR="${MODEL_DIR}" SRC_TEXT="${SRC_TEXT}" python - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = os.environ["MODEL_DIR"]
src = os.environ["SRC_TEXT"]

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()

prompt = (
    "You are a professional English (en) to Korean (ko) translator. "
    "Produce only the Korean translation.\n\n"
    + src
)

with torch.inference_mode():
    if getattr(tokenizer, "chat_template", None):
        chat_inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(chat_inputs, torch.Tensor):
            input_ids = chat_inputs.to(model.device)
            model_inputs = {"input_ids": input_ids}
        else:
            # Some tokenizer versions return BatchEncoding here.
            chat_inputs = chat_inputs.to(model.device)
            if "input_ids" not in chat_inputs:
                raise ValueError("chat template output has no input_ids")
            model_inputs = dict(chat_inputs)
            input_ids = model_inputs["input_ids"]
        out = model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
        gen = out[0][input_ids.shape[1] :]
    else:
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=256, do_sample=False)
        gen = out[0][enc["input_ids"].shape[1] :]

print("\n=== Model Output ===")
print(tokenizer.decode(gen, skip_special_tokens=True).strip())
PY
