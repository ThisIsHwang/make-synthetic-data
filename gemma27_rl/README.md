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
uv pip install -e .
```

분리 환경(학습/MetricX/xCOMET)을 쓰려면:

```bash
./scripts/setup_split_uv_envs.sh
```

## 2) 설정

기본 예시: `configs/train_toy.yaml`
WMT24pp 빠른 테스트 예시: `configs/train_wmt24pp_enko_toy.yaml`
Qwen3.5 + MQM 전용 예시: `configs/qwen35_mqm/`

핵심값:
- `model.policy_name_or_path`: SFT 결과 체크포인트 경로
- `model.policy_gpu_ids`: policy 모델에 할당할 GPU 인덱스 목록 (예: `[0,1,2]`)
- `model.reference_gpu_ids`: reference 모델에 할당할 GPU 인덱스 목록 (예: `[3,4,5]`)
- `data.train_file`: RL 학습용 JSONL/JSON/Parquet
- 또는 `data.hf_dataset_name` + `data.hf_dataset_config_name` + `data.hf_train_split`
- SFT eval set을 쓰려면 `data.eval_file`(권장) 또는 `data.hf_eval_split`을
  train과 다른 값으로 설정
- `data.eval_sampling_count`: `eval_file`이 없을 때 dev(eval) 절대 샘플 수 (우선 적용)
- `data.eval_sampling_ratio`: `eval_sampling_count`가 없을 때 dev(eval) 분할 비율
  (`data.eval_sampling_seed` + `data.id_field` 기반 해시로 고정 분할)
- `generation.num_samples_per_prompt`: GRPO group 크기
- `reward.metricx.*`, `reward.xcomet.*`, `reward.mqm.*`
- `reward.metricx.python_executable`: MetricX를 별도 uv 환경 파이썬으로 실행할 때 지정
- `reward.xcomet.python_executable`: xCOMET을 별도 uv 환경 파이썬으로 실행할 때 지정
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
- MetricX/XCOMET은 `reward.metricx.device`, `reward.xcomet.device`로 단일 GPU를 직접 지정하세요.
- 예시(6/1/1): `policy=[0,1,2,3,4,5]`, `reference=[6]`, `metricx=cuda:7`.

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

평가만:

```bash
python -m gemma27_rl.cli --config configs/train_toy.yaml --eval-only
```

로그/체크포인트:
- `logging.output_dir/resolved_config.yaml`
- `logging.output_dir/train_log.jsonl`
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
