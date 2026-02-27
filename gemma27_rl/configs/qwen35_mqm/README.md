# Qwen3.5-27B + MetricX-XXL + GEMBA-MQM (OpenAI-Compatible)

This folder contains isolated configs for RL experiments with:
- policy: `Qwen/Qwen3.5-27B-Instruct`
- sequence rewards: `MetricX-24-XXL` + `GEMBA-MQM`
- **xCOMET disabled**
- **DeepSpeed backend enabled** (no `device_map=auto`)
- scorer runtime isolation: MetricX/xCOMET can run on separate uv envs via
  `reward.metricx.python_executable` / `reward.xcomet.python_executable`.

MQM prompt/scoring behavior is aligned with:
`/home/seungyoonee/initial_translation/configs/metrics/gemba_mqm.yaml`
and the message/scoring logic in:
`/home/seungyoonee/initial_translation/evalmt/metrics/gemba_mqm_metric.py`

## Configs

- `train_wmt24pp_enko_qwen35_27b_mqm_dev4gpu.yaml`
  - environment check / coding run
  - policy on 3 GPUs (`[0,1,2]`), MetricX on 1 GPU (`cuda:3`)
  - reference model enabled on CPU (`reference_device: cpu`)
  - `rl.backend: deepspeed`, `zero_stage: 2`

- `train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml`
  - scale-up run
  - policy on 6 GPUs (`[0..5]`), reference on 1 GPU (`[6]`), MetricX on 1 GPU (`cuda:7`)
  - larger sampling and training batch settings
  - `rl.backend: deepspeed`, `zero_stage: 2`

## Run

Create split uv envs (example):

```bash
cd /home/seungyoonee/make-synthetic-data/gemma27_rl
./scripts/setup_split_uv_envs.sh
```

```bash
export HF_TOKEN=...
export OPENAI_API_KEY=...
.venv_train/bin/deepspeed --num_gpus 4 .venv_train/bin/gemma27_rl --config configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_dev4gpu.yaml
```

or

```bash
.venv_train/bin/deepspeed --num_gpus 8 .venv_train/bin/gemma27_rl --config configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml
```
