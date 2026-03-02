# Qwen3.5-27B + MetricX/XCOMET (+ optional MQM)

This folder contains isolated configs for RL experiments with:
- policy: `Qwen/Qwen3.5-27B-Instruct`
- sequence rewards: `MetricX-24-XXL` (+ optional `XCOMET-XL`, `GEMBA-MQM`)
- **DeepSpeed backend enabled** (no `device_map=auto`)
- scorer runtime isolation: MetricX/xCOMET can run on separate uv envs via
  `reward.metricx.python_executable` / `reward.xcomet.python_executable`.
- remote worker routing: reference/MetricX/xCOMET can run on a dedicated aux node
  via `misc.aux_worker_host` (or per-component `*_worker_host`)

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

- `train_wmt24pp_enko_qwen35_27b_metricx_xcomet_multinode8p1aux.yaml`
  - multi-node run template
  - **8 policy nodes** use all GPUs for Qwen policy (`policy_gpu_ids: [0..7]` per node)
  - **1 aux node** runs reference + MetricX + xCOMET via SSH worker launch
  - set `misc.aux_worker_host` / `misc.aux_worker_remote_workdir` to your cluster values

## Run

Create split uv envs (example):

```bash
cd /home/seungyoonee/make-synthetic-data/gemma27_rl
./scripts/setup_split_uv_envs.sh
```

Note: xCOMET venv is pinned to `setuptools<81` (for `pkg_resources` compatibility).

```bash
export HF_TOKEN=...
export OPENAI_API_KEY=...
.venv_train/bin/deepspeed --num_gpus 4 .venv_train/bin/gemma27_rl --config configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_dev4gpu.yaml
```

or

```bash
.venv_train/bin/deepspeed --num_gpus 8 .venv_train/bin/gemma27_rl --config configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml
```

8+1 multi-node example (policy nodes are only in hostfile/include; aux node is **not** in DeepSpeed ranks):

```bash
HOSTFILE=/path/to/policy_8nodes.hostfile \
INCLUDE='policy-node-01:0,1,2,3,4,5,6,7@policy-node-02:0,1,2,3,4,5,6,7@policy-node-03:0,1,2,3,4,5,6,7@policy-node-04:0,1,2,3,4,5,6,7@policy-node-05:0,1,2,3,4,5,6,7@policy-node-06:0,1,2,3,4,5,6,7@policy-node-07:0,1,2,3,4,5,6,7@policy-node-08:0,1,2,3,4,5,6,7' \
CONFIG=configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_metricx_xcomet_multinode8p1aux.yaml \
./scripts/run_deepspeed_from_venv.sh
```
