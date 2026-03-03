# distributed_rl

TRL GRPOTrainer + DeepSpeed 기반 Multi-Node Distributed RL Post-Training.

## Changelog (vs `f127263`)

### MiniRL Stability (arxiv 2512.01374) — Phase 1

| 파일 | 변경 | 설명 |
|------|------|------|
| `config.py` | Modified | `StabilityConfig` dataclass 추가, `DistributedRLConfig.stability` 필드, validation 로직 |
| `stability.py` | **New** | `StabilityMonitorCallback` — entropy floor / KL ceiling 감시, trend 분석, collapse 시 학습 자동 중단 |
| `rewards/base.py` | Modified | `make_trl_reward_func()`에 `clip_value` 파라미터 추가 (reward outlier clipping) |
| `trainer.py` | Modified | 비대칭 epsilon (`epsilon_high`/`epsilon_low`) TRL 전달, `StabilityMonitorCallback` 주입, reward clip 전달, TRL 버전 경고 |
| `configs/qwen35_moe_8gpu.yaml` | Modified | `stability:` 섹션 추가 (entropy_floor=0.5, kl_ceiling=10.0, halt_on_collapse_steps=20, monitor_router_entropy=true) |
| `configs/gemma27b_8gpu.yaml` | Modified | `stability:` 섹션 추가 (entropy_floor=0.3, kl_ceiling=15.0) |
| `tests/test_stability.py` | **New** | 20개 테스트 (StabilityConfig, StabilityMonitorCallback, reward clipping, validation) |

### Docs

| 파일 | 변경 | 설명 |
|------|------|------|
| `README.md` | Modified | WMT24pp En→Ko 번역 RL 학습 가이드 추가, 본 changelog 추가 |

MetricX / XComet reward 모델을 사용하여 MoE (Qwen3.5-35B-A3B) 및 Dense (Gemma-27B) 모델을 GRPO/DAPO/DR-GRPO/SAPO/GSPO로 학습합니다.

## GPU Allocation

```
8-GPU Node (Single-Node 또는 Master Node):
┌─────────────────────────────────────────┐  ┌──────────┐  ┌──────────┐
│  GPU 0  GPU 1  GPU 2  GPU 3  GPU 4  GPU 5  │  │  GPU 6   │  │  GPU 7   │
│         Policy Training (DeepSpeed)      │  │  MetricX │  │  XComet  │
└─────────────────────────────────────────┘  └──────────┘  └──────────┘

Worker Node (Multi-Node 시):
┌──────────────────────────────────────────────────────────────────────┐
│  GPU 0  GPU 1  GPU 2  GPU 3  GPU 4  GPU 5  GPU 6  GPU 7            │
│                   Policy Training (DeepSpeed)                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Installation

> 모든 명령어는 **프로젝트 루트** (`hwang-post-training/`)에서 실행합니다.

```
hwang-post-training/            ← 프로젝트 루트 (여기서 실행)
├── .env                       ← HF_HOME, HF_TOKEN 등 환경변수 (gitignored)
├── distributed_rl/
│   ├── configs/                ← YAML config 파일들
│   └── ...
├── .venv-metrics/              ← MetricX 전용 venv (Step 2에서 생성)
└── .venv-xcomet/               ← XComet 전용 venv (Step 3에서 생성)
```

### 0. 환경 변수 (.env)

프로젝트 루트에 `.env` 파일을 생성합니다. 학습 시작 시 자동으로 로드됩니다.

```bash
cp .env.example .env
# .env 파일을 편집하여 실제 값 입력
```

`.env` 예시:
```
HF_HOME=/group-volume/huggingface
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 1. Training 환경

```bash
# 현재 활성 venv에 distributed_rl 패키지 설치
uv pip install -e distributed_rl
# flash-attn 설치 (--no-build-isolation 필수)
uv pip install flash-attn --no-build-isolation
```

주요 의존성:
- `torch >= 2.6.0`
- `transformers >= 4.57.0`
- `trl >= 0.15.0`
- `accelerate >= 0.34.0`
- `deepspeed >= 0.15.4`
- `datasets >= 2.21.0`

### 2. MetricX 환경 (별도 venv)

MetricX는 의존성 충돌 방지를 위해 별도 Python 환경을 사용합니다.

```bash
# 프로젝트 루트에서 실행
uv venv .venv-metrics
uv pip install --python .venv-metrics/bin/python -r distributed_rl/requirements-metrics.txt
uv pip install --python .venv-metrics/bin/python --no-deps -e distributed_rl
```

Config의 `python_executable`은 YAML 파일 위치 기준 상대 경로입니다:
```yaml
# distributed_rl/configs/*.yaml 기준 → ../../ = 프로젝트 루트
reward:
  metricx:
    python_executable: ../../.venv-metrics/bin/python
```

### 3. XComet 환경 (별도 venv)

```bash
# 프로젝트 루트에서 실행
uv venv .venv-xcomet
uv pip install --python .venv-xcomet/bin/python -r distributed_rl/requirements-xcomet.txt
uv pip install --python .venv-xcomet/bin/python --no-deps -e distributed_rl
```

Config에서 `reward.xcomet.python_executable`로 경로를 지정합니다:
```yaml
# distributed_rl/configs/*.yaml 기준 → ../../ = 프로젝트 루트
reward:
  xcomet:
    python_executable: ../../.venv-xcomet/bin/python
```

## Quick Start (Toy Config)

단일 GPU에서 작은 모델로 동작을 확인합니다:

```bash
python3 -m distributed_rl --config distributed_rl/configs/train_toy.yaml
```

## Single-Node Multi-GPU (8 GPUs)

8-GPU 노드에서 GPU 0-5는 학습, GPU 6은 MetricX, GPU 7은 XComet에 할당합니다.

### Qwen3.5-35B MoE

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
    --nproc_per_node=6 \
    -m distributed_rl \
    --config distributed_rl/configs/qwen35_moe_8gpu.yaml
```

### Gemma-27B Dense

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
    --nproc_per_node=6 \
    -m distributed_rl \
    --config distributed_rl/configs/gemma27b_8gpu.yaml
```

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`로 학습 프로세스의 GPU를 제한합니다. Reward 모델은 subprocess로 실행되며, 별도로 GPU 6/7에 접근합니다 (config의 `reward.metricx.gpu_id`, `reward.xcomet.gpu_id`).

## Multi-Node Multi-GPU (2 Nodes x 8 GPUs)

2개 노드, 총 16 GPUs. Node 0 (Master)는 6 GPU 학습 + 2 GPU reward, Node 1 (Worker)는 8 GPU 전부 학습.

### Node 0 (Master)

```bash
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
    --nproc_per_node=6 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m distributed_rl \
    --config distributed_rl/configs/qwen35_moe_8gpu.yaml
```

### Node 1 (Worker)

Worker 노드에는 reward 모델이 없으므로 8 GPU 전부 학습에 사용합니다:

```bash
export MASTER_ADDR=<master-node-ip>
export MASTER_PORT=29500

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m distributed_rl \
    --config distributed_rl/configs/qwen35_moe_8gpu.yaml
```

총 학습 rank 수: 6 (Node 0) + 8 (Node 1) = **14 ranks**. Effective batch size = 14 x `per_device_train_batch_size` x `gradient_accumulation_steps`.

### 주의사항

- `MASTER_ADDR`는 Node 0의 IP 주소 또는 hostname. 모든 노드에서 접근 가능해야 합니다.
- `MASTER_PORT`는 모든 노드에서 동일해야 합니다 (기본 29500).
- Reward 모델은 rank 0 (Master 노드)에서만 실행됩니다. Worker 노드에는 MetricX/XComet venv가 없어도 됩니다.
- NCCL timeout은 config의 `distributed.nccl_timeout_minutes`로 설정합니다 (기본 120분). Reward scoring이 오래 걸릴 수 있으므로 넉넉히 설정하세요.

## WMT24pp En→Ko 번역 RL 학습 가이드

[google/wmt24pp](https://huggingface.co/datasets/google/wmt24pp) 데이터셋의 `en-ko_KR` config를 사용하여 번역 모델을 GRPO로 학습하는 End-to-End 가이드입니다.

### 사전 준비

```bash
# 프로젝트 루트 (hwang-post-training/) 에서 실행

# 1. .env 파일 설정 (위 Installation > 환경 변수 참조)
cp .env.example .env && vi .env

# 2. Training 환경 설치
uv pip install -e distributed_rl
uv pip install flash-attn --no-build-isolation

# 3. MetricX 별도 venv (위 Installation 참조)
# 4. XComet 별도 venv (위 Installation 참조)
```

### 데이터셋 구조

WMT24pp `en-ko_KR` split의 필드 매핑:

| WMT24pp 필드 | Config 매핑 | 설명 |
|--------------|------------|------|
| `segment_id` | `id_field` | 고유 ID |
| `source` | `src_text_field` | 영어 원문 |
| `target` | `ref_text_field` | 한국어 참조 번역 |
| `src_lang` | `src_lang_field` | 소스 언어명 |
| `tgt_lang` | `tgt_lang_field` | 타겟 언어명 |
| `src_lang_code` | `src_lang_code_field` | 소스 언어 코드 (en) |
| `tgt_lang_code` | `tgt_lang_code_field` | 타겟 언어 코드 (ko) |
| `is_bad_source` | `is_bad_source_field` | 불량 소스 플래그 |

Config에서 HuggingFace 데이터셋으로 지정:

```yaml
data:
  hf_dataset_name: google/wmt24pp
  hf_dataset_config_name: en-ko_KR
  hf_train_split: train
  hf_eval_split: train      # WMT24pp는 train split만 존재
  eval_limit: 32             # eval은 소량 샘플링

  id_field: segment_id
  src_text_field: source
  ref_text_field: target
  src_lang_field: src_lang
  tgt_lang_field: tgt_lang
  src_lang_code_field: src_lang_code
  tgt_lang_code_field: tgt_lang_code
  is_bad_source_field: is_bad_source
  skip_bad_source: true
  default_src_lang: English
  default_tgt_lang: Korean
  default_src_lang_code: en
  default_tgt_lang_code: ko
```

### Step 1: Toy Config으로 동작 확인

단일 GPU에서 작은 모델(Qwen2-0.5B)로 파이프라인이 정상 동작하는지 확인합니다. `distributed_rl/configs/train_toy.yaml`은 WMT24pp 64개 샘플로 10 step만 학습합니다.

```bash
python3 -m distributed_rl --config distributed_rl/configs/train_toy.yaml
```

확인 사항:
- WMT24pp 데이터 로딩 및 필드 매핑 정상 동작
- 프롬프트 생성 및 rollout 생성 확인
- MetricX reward scoring 동작 (in-process 모드, `python_executable` 미설정)
- Loss 감소 추이 확인

### Step 2: Qwen3.5-35B MoE — Single Node 8-GPU

MoE 모델은 `output_router_logits: true`와 `router_aux_loss_coef`로 expert collapse를 방지합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
    --nproc_per_node=6 \
    -m distributed_rl \
    --config distributed_rl/configs/qwen35_moe_8gpu.yaml
```

주요 하이퍼파라미터 (`configs/qwen35_moe_8gpu.yaml`):

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `training.loss_type` | `grpo` | GRPO (GSPO는 `gspo`로 변경) |
| `training.beta` | `0.01` | KL penalty (0이면 reference model 비활성) |
| `training.lr` | `1e-6` | RL fine-tuning용 낮은 LR |
| `generation.num_generations` | `4` | 프롬프트당 completions (group size) |
| `generation.temperature` | `0.4` | 번역은 낮은 temperature 권장 |
| `training.per_device_train_batch_size` | `2` | GPU당 프롬프트 수 |
| `training.gradient_accumulation_steps` | `4` | Effective batch = 6 × 2 × 4 = 48 |
| `training.output_router_logits` | `true` | MoE 필수: router logits 출력 |
| `training.router_aux_loss_coef` | `0.001` | MoE 필수: load balancing loss |
| `reward.metricx.model_name` | `metricx-24-hybrid-xxl-v2p6` | XXL 모델 (더 정확) |

### Step 3: Gemma-27B Dense — Single Node 8-GPU

Dense 모델은 MoE 설정이 불필요합니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
    --nproc_per_node=6 \
    -m distributed_rl \
    --config distributed_rl/configs/gemma27b_8gpu.yaml
```

Gemma-27B와 Qwen3.5-35B의 config 차이:

| 설정 | Qwen3.5 MoE | Gemma-27B Dense |
|------|-------------|-----------------|
| `model.trust_remote_code` | `true` | `false` |
| `training.output_router_logits` | `true` | `false` |
| `training.router_aux_loss_coef` | `0.001` | `0.0` |

### Step 4: Multi-Node 확장 (2 Nodes)

위 "Multi-Node Multi-GPU" 섹션을 참조하세요. Config 파일은 동일하게 사용하며, `torchrun` 인자만 변경합니다.

### Reward 모델 조합

MetricX와 XComet을 함께 사용하려면 가중치를 조정합니다:

```yaml
reward:
  w_metricx: 0.7     # MetricX 가중치
  w_xcomet: 0.3      # XComet 가중치
  metricx:
    enabled: true
    gpu_id: 6         # 전용 GPU
    python_executable: ../../.venv-metrics/bin/python
  xcomet:
    enabled: true
    gpu_id: 7         # 전용 GPU (MetricX와 다른 GPU)
    python_executable: ../../.venv-xcomet/bin/python
```

MetricX만 사용 (기본):

```yaml
reward:
  w_metricx: 1.0
  w_xcomet: 0.0
  xcomet:
    enabled: false    # 비활성화하면 GPU 7 불필요
```

### 학습 모니터링

```yaml
misc:
  report_to: [wandb]          # W&B 로깅 활성화

training:
  log_completions: true        # 생성된 번역문 로깅
  logging_steps: 1             # 매 step 로그
  eval_steps: 50               # 50 step마다 eval
```

주요 관찰 지표:
- `loss`: RL loss (감소 추이)
- `reward/mean`: 평균 reward (증가 추이)
- `reward/std`: reward 분산 (너무 작으면 diversity 부족)
- `kl`: KL divergence (beta > 0일 때, 지나치게 커지면 불안정)

### 출력물

```
outputs/<run-name>/
├── checkpoint-200/     # 주기적 체크포인트 (save_steps마다)
├── checkpoint-400/
├── final/              # 학습 완료 후 최종 모델 (rank 0만 저장)
├── runs/               # TensorBoard 로그 (report_to에 tensorboard 포함 시)
└── ...
```

## Config 구조

```yaml
model:        # 모델 경로, attention 구현, gradient checkpointing
data:         # 데이터셋 (JSONL/HF), 필드 매핑, 언어 기본값
prompt:       # 번역 프롬프트 템플릿
generation:   # 생성 파라미터 (temperature, top_p, num_generations)
reward:       # MetricX/XComet 설정, 가중치, GPU 할당
training:     # Loss type, LR, batch size, MoE 설정
distributed:  # DeepSpeed config 경로, NCCL timeout
misc:         # Seed, dtype, output 경로
```

### Loss Types

| Loss | Config값 | 설명 |
|------|----------|------|
| GRPO | `grpo` | Group Relative Policy Optimization (기본) |
| DAPO | `dapo` | Dynamic Advantage Policy Optimization |
| DR-GRPO | `dr_grpo` | Doubly-Robust GRPO |
| SAPO | `sapo` | Self-Advantage Policy Optimization |
| GSPO | `gspo` | `trl.experimental.gspo_token` 사용 |

```yaml
training:
  loss_type: grpo  # grpo | dapo | dr_grpo | sapo | gspo
```

## Tests

```bash
uv pip install pytest pyyaml
pytest distributed_rl/tests/ -v
```
