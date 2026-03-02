# distributed_rl

TRL GRPOTrainer + DeepSpeed 기반 Multi-Node Distributed RL Post-Training.

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

### 1. Training 환경

```bash
cd distributed_rl
pip install -e .
# 또는 requirements만 설치:
pip install -r requirements.txt
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
python -m venv .venv-metrics
source .venv-metrics/bin/activate
pip install torch transformers sentencepiece
# MetricX 모델은 MT5 기반이므로 추가 패키지 불필요
deactivate
```

Config에서 `reward.metricx.python_executable`로 경로를 지정합니다:
```yaml
reward:
  metricx:
    python_executable: ../../.venv-metrics/bin/python
```

### 3. XComet 환경 (별도 venv)

```bash
python -m venv .venv-xcomet
source .venv-xcomet/bin/activate
pip install torch unbabel-comet
deactivate
```

Config에서 `reward.xcomet.python_executable`로 경로를 지정합니다:
```yaml
reward:
  xcomet:
    python_executable: ../../.venv-xcomet/bin/python
```

## Quick Start (Toy Config)

단일 GPU에서 작은 모델로 동작을 확인합니다:

```bash
python3 -m distributed_rl --config configs/train_toy.yaml
```

## Single-Node Multi-GPU (8 GPUs)

8-GPU 노드에서 GPU 0-5는 학습, GPU 6은 MetricX, GPU 7은 XComet에 할당합니다.

### Qwen3.5-35B MoE

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
    --nproc_per_node=6 \
    -m distributed_rl \
    --config configs/qwen35_moe_8gpu.yaml
```

### Gemma-27B Dense

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun \
    --nproc_per_node=6 \
    -m distributed_rl \
    --config configs/gemma27b_8gpu.yaml
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
    --config configs/qwen35_moe_8gpu.yaml
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
    --config configs/qwen35_moe_8gpu.yaml
```

총 학습 rank 수: 6 (Node 0) + 8 (Node 1) = **14 ranks**. Effective batch size = 14 x `per_device_train_batch_size` x `gradient_accumulation_steps`.

### 주의사항

- `MASTER_ADDR`는 Node 0의 IP 주소 또는 hostname. 모든 노드에서 접근 가능해야 합니다.
- `MASTER_PORT`는 모든 노드에서 동일해야 합니다 (기본 29500).
- Reward 모델은 rank 0 (Master 노드)에서만 실행됩니다. Worker 노드에는 MetricX/XComet venv가 없어도 됩니다.
- NCCL timeout은 config의 `distributed.nccl_timeout_minutes`로 설정합니다 (기본 120분). Reward scoring이 오래 걸릴 수 있으므로 넉넉히 설정하세요.

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
cd distributed_rl
pip install pytest pyyaml
pytest tests/ -v
```
