# Gemma 27B RL Post-Training (GRPO)

`gemma27b_sft`로 학습된 체크포인트를 시작점으로, `SPEC.MD` 요구사항에 맞춘
TranslateGemma 스타일 RL post-training 파이프라인입니다.

구현 포함 항목:
- rollout 수집 (completion + old/ref logprobs + token char offsets)
- MetricX-QE sequence reward (`5.0 - score`)
- XCOMET-XL sentence score + error spans
- OpenAI-compatible GEMBA-MQM sequence reward
- error span -> token reward 매핑
- sequence reward broadcast + token reward additive + batch normalize
- GRPO/PPO-clip 스타일 업데이트 (value head 없음)
- metric-only eval 및 toy RL loop

## 1) 설치

```bash
cd gemma27_rl
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt`는 `pyproject.toml`의 `.[full]` extras를 설치하는 thin wrapper입니다.

최소 editable install만 원하면:

```bash
uv pip install -e .
```

일반적인 RL 실행 환경은 보통 다음 extras가 필요합니다:

```bash
uv pip install -e ".[train,reward]"
```

TensorBoard/W&B 모니터링까지 쓰려면:

```bash
uv pip install -e ".[train,reward,monitor]"
```

DeepSpeed multi-GPU 학습까지 포함하면:

```bash
uv pip install -e ".[train,reward,deepspeed]"
```

분리 환경(학습/MetricX/xCOMET)을 쓰려면:

```bash
./scripts/setup_split_uv_envs.sh
```

참고: `xCOMET` 환경은 `pkg_resources` 호환을 위해 `setuptools<81`을 사용합니다.

## 2) 설정

기본 예시: `configs/train_toy.yaml`
WMT24pp 빠른 테스트 예시: `configs/train_wmt24pp_enko_toy.yaml`
27B LoRA + colocated reference 예시: `configs/train_27b_8gpu_lora_colocate.yaml`
27B LoRA + colocated reference + single-node MQM 예시: `configs/train_27b_8gpu_lora_colocate_single_node_mqm.yaml`
Qwen3.5 + MQM 전용 예시: `configs/qwen35_mqm/`

핵심값:
- `model.policy_name_or_path`: SFT 결과 체크포인트 경로
- `model.policy_gpu_ids`: policy 모델에 할당할 GPU 인덱스 목록 (예: `[0,1,2]`)
- `model.reference_gpu_ids`: reference 모델에 할당할 GPU 인덱스 목록 (예: `[3,4,5]`)
- `model.lora.*`: 27B 같은 대형 모델을 adapter-only로 RL 업데이트할 때 사용
- `model.reference_runtime`: `worker|in_process|cpu|colocate`
  - `colocate`: LoRA policy의 adapter를 잠시 끄고 같은 policy base로 reference logprob를 계산
- `data.train_file`: RL 학습용 JSONL/JSON/Parquet
- 또는 `data.hf_dataset_name` + `data.hf_dataset_config_name` + `data.hf_train_split`
- SFT eval set을 쓰려면 `data.eval_file`(권장) 또는 `data.hf_eval_split`을
  train과 다른 값으로 설정
- `data.eval_sampling_count`: `eval_file`이 없을 때 dev(eval) 절대 샘플 수 (우선 적용)
- `data.eval_sampling_ratio`: `eval_sampling_count`가 없을 때 dev(eval) 분할 비율
  (`data.eval_sampling_seed` + `data.id_field` 기반 해시로 고정 분할)
- `generation.num_samples_per_prompt`: GRPO group 크기
- `reward.metricx.*`, `reward.xcomet.*`, `reward.mqm.*`, `reward.esa.*`
- `reward.esa.*`: OpenAI-compatible LLM 기반 GEMBA-ESA scalar 점수(0~100) 보상
- ESA를 MQM과 비슷한 기여도로 시작하려면 `reward.w_esa_seq: 0.2`,
  `reward.esa_seq_scale: 0.25` 권장 (실효: `ESA*0.05`, 즉 0~100 -> 0~5)
- `reward.metricx.python_executable`: MetricX를 별도 uv 환경 파이썬으로 실행할 때 지정
- `logging.tensorboard_enabled`: 기본 `true`, `output_dir/tensorboard`에 scalar 기록
- `logging.wandb_enabled`: `true`면 W&B로 같은 metric을 함께 기록
- `logging.wandb_project`, `logging.wandb_run_name`, `logging.wandb_mode`
- `reward.xcomet.python_executable`: xCOMET을 별도 uv 환경 파이썬으로 실행할 때 지정
- `misc.aux_worker_host`: reference/MetricX/xCOMET을 올릴 전용 aux 노드 host (SSH 접속 가능해야 함)
- `misc.aux_worker_remote_workdir`: aux 노드에서 worker 실행 전 `cd`할 경로
- `model.reference_worker_host`, `reward.metricx.worker_host`, `reward.xcomet.worker_host`:
  컴포넌트별 host override (미설정 시 `misc.aux_worker_host` 사용)
- `rl.*` (clip, kl, batch, updates)
- `misc.huggingface_cache_dir`: HF 캐시 루트 (예: `/media/sdd3`)
- `misc.huggingface_token`: (권장 비활성) 직접 토큰 입력값
- `misc.huggingface_token_env`: 토큰을 읽을 환경변수 이름 (기본 `HF_TOKEN`)

GPU 배치(자동):
- 기본값(`misc.device: cuda`, `reward.metricx.device: cuda`, `reward.xcomet.device: cuda`)이면
  실행 시 자동으로 `policy -> cuda:0`, `metricx -> cuda:1`, `xcomet -> cuda:2` 순으로
  가능한 한 서로 다른 GPU를 배정합니다.
- GPU 개수가 부족하면 가능한 범위에서 배정하고 경고 로그를 출력합니다.
- `xcomet`은 Lightning `Trainer` 재생성을 피하고 모델을 메모리에 상주시켜,
  반복 스코어링 시 초기화 오버헤드를 줄입니다.
- `mqm`은 외부 OpenAI-compatible API judge를 호출하므로 로컬 GPU를 점유하지 않습니다.
- `reward.metricx.python_executable`/`reward.xcomet.python_executable`가 설정되면
  각 scorer는 학습 프로세스와 분리된 서브프로세스(해당 Python)에서 모델을 로드/추론합니다.

GPU 배치(명시적 8-GPU 분할):
- `model.policy_gpu_ids`/`model.reference_gpu_ids`를 설정하면 자동 배치보다 우선합니다.
- `device_map=auto` 경로는 비활성화되어 있습니다.
- policy를 여러 GPU에 올리려면 `rl.backend: deepspeed`를 사용하고 `deepspeed` launcher로 실행하세요.
- reference 모델은 단일 GPU(`reference_gpu_ids` 첫 번째)만 사용합니다.
- 단, `model.reference_runtime: colocate` + `model.lora.enabled: true`면 별도 reference 모델을 띄우지 않고
  policy LoRA base를 그대로 reference로 재사용합니다. 이 경우 `policy_gpu_ids: [0..7]`로 8 GPU 전체를 policy에 줄 수 있습니다.
- 위 colocate 모드에서는 로컬 8 GPU를 policy가 모두 쓰므로 MetricX/XCOMET은 aux host로 빼거나 비활성화해야 합니다.
- MetricX/XCOMET은 `reward.metricx.device`, `reward.xcomet.device`로 단일 GPU를 직접 지정하세요.
- 예시(6/1/1): `policy=[0,1,2,3,4,5]`, `reference=[6]`, `metricx=cuda:7`.

멀티노드 8+1(정책 8노드 + 보조 1노드):
- DeepSpeed rank는 policy 노드들만 포함하세요 (aux 노드는 hostfile/include에 넣지 않음).
- `model.policy_gpu_ids: [0..7]`는 policy 노드의 local GPU 인덱스 기준입니다.
- reference/MetricX/xCOMET은 aux 노드에서 SSH 서브프로세스로 실행됩니다.
- 예시 config: `configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_metricx_xcomet_multinode8p1aux.yaml`

GEMBA-MQM 프롬프트/스코어링:
- MQM judge 메시지 구성은 아래 구현을 따릅니다.
  - `initial_translation/evalmt/metrics/gemba_mqm_metric.py`
  - `initial_translation/configs/metrics/gemba_mqm.yaml`

토큰 사용 권장 방식:

```bash
export HF_TOKEN=...
python -m gemma27_rl.cli --config configs/train_toy.yaml
```

## 3) 실행

학습:

```bash
python -m gemma27_rl.cli --config configs/train_toy.yaml
```

DeepSpeed 학습(예: 8 GPU):

```bash
deepspeed --num_gpus 8 .venv_train/bin/gemma27_rl --config configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml
```

원격 서버에서 FlashAttention ABI 에러가 날 때(권장 복구):

```bash
# flash_attn_2_cuda.so undefined symbol 류 에러 복구
VENV_BIN=/abs/path/to/.venv_train/bin ./scripts/fix_flash_attn_abi.sh
```

venv 일치 강제 실행(DeepSpeed launcher/entrypoint/python 모두 동일 venv):

```bash
VENV_BIN=/abs/path/to/.venv_train/bin \
INCLUDE=localhost:0,1,2,3,4,5,6,7 \
CONFIG=configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml \
./scripts/run_deepspeed_from_venv.sh
```

멀티노드 8+1 실행 예시:

```bash
HOSTFILE=/path/to/policy_8nodes.hostfile \
INCLUDE='policy-node-01:0,1,2,3,4,5,6,7@policy-node-02:0,1,2,3,4,5,6,7@policy-node-03:0,1,2,3,4,5,6,7@policy-node-04:0,1,2,3,4,5,6,7@policy-node-05:0,1,2,3,4,5,6,7@policy-node-06:0,1,2,3,4,5,6,7@policy-node-07:0,1,2,3,4,5,6,7@policy-node-08:0,1,2,3,4,5,6,7' \
CONFIG=configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_metricx_xcomet_multinode8p1aux.yaml \
./scripts/run_deepspeed_from_venv.sh
```

평가만:

```bash
python -m gemma27_rl.cli --config configs/train_toy.yaml --eval-only
```

로그/체크포인트:
- `logging.output_dir/resolved_config.yaml`
- `logging.output_dir/train_log.jsonl`
- `logging.output_dir/tensorboard` (`logging.tensorboard_enabled: true`일 때)
- `logging.output_dir/train_rollouts.jsonl` (`logging.save_rollouts: true`일 때)
- `logging.output_dir/eval_outputs.jsonl` (`logging.save_eval_outputs: true`일 때)
- `logging.output_dir/checkpoint-*`
- `logging.output_dir/resume_latest` (중단 후 자동 재시작용)
- `logging.output_dir/best` (eval best 모델)
- `logging.output_dir/final`

재시작/저장 관련 옵션:
- `logging.auto_resume: true`면 `resume_latest` 또는 최신 `checkpoint-*`에서 자동 재개
- `logging.resume_from_checkpoint`를 지정하면 해당 체크포인트에서 강제 재개
- `logging.save_only_best: true`면 주기적 `checkpoint-*` 대신 `best` + `resume_latest`만 유지
- `logging.keep_last_n_checkpoints: N`(N>0)이면 주기 저장 시 `checkpoint-*`를 최신 N개만 유지
  (`best` 체크포인트는 별도 유지)

TensorBoard 보기:

```bash
tensorboard --logdir /path/to/output_dir/tensorboard
```

W&B 예시:

```yaml
logging:
  wandb_enabled: true
  wandb_project: gemma27-rl
  wandb_run_name: exp001-mqm
  wandb_mode: offline
```
