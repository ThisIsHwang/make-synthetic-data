# synth-parallel

TranslateGemma 방식(MADLAD-400 -> prefilter 2-sample -> final 128-sample -> MetricX QE best 선택 -> formatting filter)을 재현하는 데이터 생성 파이프라인입니다.

이 레포는 아래 운영 모델을 전제로 설계되어 있습니다.
- Teacher LLM(Qwen): 원격 OpenAI-compatible API로 호출
- MetricX: 로컬 GPU(H100)에서 실행
- 중단 복구: stage별 재시작 + item-level progress checkpoint
- 캐시: teacher 응답 캐시 + MetricX 점수 캐시

## 1. 핵심 기능

- stage 기반 실행
- `sample_sources`, `prefilter_score`, `select_sources`, `generate_128`, `score_select_best`, `format_filter`, `export`
- 재개(resume)
- `progress.sqlite`로 처리 완료 item 추적
- 캐시
- `teacher_cache.sqlite`, `metricx_cache.sqlite`
- 관측성
- `logs.txt`, `stats.json`에 처리량/실패/재시도/필터 분포 기록
- 샤딩 실행
- `--shard-id`, `--num-shards`

## 2. 디렉터리 구조

- `synth_parallel/cli.py`: CLI 엔트리
- `synth_parallel/pipeline.py`: stage 오케스트레이션
- `synth_parallel/teacher.py`: OpenAI-compatible teacher client
- `synth_parallel/metricx.py`: MetricX CLI wrapper
- `synth_parallel/segmentation.py`: 문서 -> 세그먼트/blob
- `synth_parallel/bucketing.py`: 길이 버킷 샘플러
- `synth_parallel/filters.py`: rule-based + optional LLM judge
- `config/example.yaml`: 실행 설정 예시

## 3. 사전 준비

- OS: Linux 권장(대규모 실행 기준)
- Python: 메인 파이프라인과 MetricX를 별도 env로 운영
- GPU: MetricX용 로컬 H100 1장 이상 권장
- 네트워크: Qwen API endpoint 접근 가능해야 함
- 디스크: 중간 산출물/캐시 용량 충분히 확보
- Hugging Face 토큰: 대용량 MADLAD 접근 시 rate limit 회피를 위해 권장 (`HF_TOKEN`)

## 4. 환경 구성 (uv)

### 4.1 메인 파이프라인 환경

참고: MADLAD-400은 Hugging Face `datasets`의 dataset script(`MADLAD-400.py`)를 사용합니다. `datasets` 4.x에서는 dataset script 로딩이 중단되어 MADLAD 로딩이 실패할 수 있으므로, 이 레포는 `datasets` 2.x(`datasets<3`)를 사용합니다.

```bash
# uv 설치 (없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 루트
cd /path/to/make-data2

# Hugging Face 토큰(권장)
export HF_TOKEN="<your-hf-token>"

# 메인 env 생성
uv venv .venv --python 3.11
source .venv/bin/activate

# 파이프라인 의존성 설치
uv pip install -e .

# 테스트 도구
uv pip install pytest
```

### 4.2 MetricX 전용 별도 환경 (구버전 호환)

MetricX는 PyPI 패키지로 배포되지 않는 경우가 많아서(`metricx24`가 pip에서 안 잡힘), **GitHub `google-research/metricx` 레포 기준**으로 설치합니다.

또한 해당 레포의 `requirements.txt`가 구버전 의존성을 고정합니다:
- `transformers[torch]==4.30.2`
- `sentencepiece==0.1.99`
- `datasets==2.13.1`
- `git+https://github.com/google-research/mt-metrics-eval`

```bash
cd /path/to/make-data2

# MetricX 레포 clone (원하는 위치로 변경 가능)
mkdir -p third_party
git clone https://github.com/google-research/metricx third_party/metricx

# 예시: 구버전 호환용 env (필요 버전에 맞게 변경)
uv venv .venv-metricx --python 3.10
source .venv-metricx/bin/activate

# PyTorch 설치(환경/CUDA에 맞게). 아래는 예시이며, 서버 환경에 맞게 선택하세요.
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# GitHub 레포 requirements 그대로 설치
uv pip install -r third_party/metricx/requirements.txt

# 설치 확인
cd third_party/metricx
python -m metricx24.predict --help
```

메인 env로 다시 복귀:

```bash
source .venv/bin/activate
```

## 5. 설정 파일

기본 템플릿: `config/example.yaml`

핵심 설정:

- `teacher.base_url`
- 원격 Qwen OpenAI-compatible endpoint (`.../v1`)
- `teacher.api_key_env`
- API 키를 읽을 환경변수 이름 (기본: `QWEN_API_KEY`)
- `teacher.model`
- 호출할 model 이름
- `data.madlad_dataset`
- 기본값: `allenai/MADLAD-400`
- `data.src_lang`
- MADLAD config name (예: 한국어는 `ko`)
- `data.hf_token_env`
- Hugging Face 토큰 env 이름 (기본 `HF_TOKEN`)
- `metricx.backend`
- `metricx24_cli` 사용
- `metricx.device`
- `cuda:0` 권장 (H100 1장 고정)
- `metricx.python_bin`
- MetricX 전용 env의 Python 경로
- 예: `./.venv-metricx/bin/python`
- `metricx.module`
- 기본 `metricx24.predict`
- `metricx.repo_dir`
- `google-research/metricx`를 clone한 디렉터리(예: `./third_party/metricx`)
- `metricx.max_input_length`
- MetricX-24 기본 1536 권장(레포 README 예시)

예시(중요 부분):

```yaml
teacher:
  backend: openai_compatible
  base_url: https://your-qwen-endpoint.example.com/v1
  api_key_env: QWEN_API_KEY
  model: Qwen/Qwen3-235B-A22B-Instruct-2507

metricx:
  backend: metricx24_cli
  checkpoint: google/metricx-24-hybrid-large-v2p6
  device: cuda:0
  python_bin: ./.venv-metricx/bin/python
  module: metricx24.predict
  repo_dir: ./third_party/metricx
  tokenizer: google/mt5-xl
  max_input_length: 1536
```

환경변수 설정:

```bash
export QWEN_API_KEY="<your-api-key>"
export HF_TOKEN="<your-hf-token>"
```

## 6. MADLAD 다운로드/캐시 (CPU)

다운로드 자체는 GPU가 필요 없습니다. CPU 머신에서 Hugging Face 캐시를 미리 채워두면, 이후 파이프라인 실행 시 재다운로드를 줄일 수 있습니다.

1) 캐시 위치 지정(권장)

```bash
export HF_HOME=/path/to/hf-cache
```

2) 언어/스플릿 선택해서 다운로드

예: 한국어 clean 전체를 로컬 캐시에 받기

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset('allenai/MADLAD-400', 'ko', split='clean', streaming=False)
print(ds)
PY
```

주의:
- 한국어는 clean만 받아도 약 34GiB(압축) 규모입니다(추가 전처리 캐시까지 감안해 여유 디스크 필요).
- 전체(모든 언어)는 1TiB+ 입니다.
- 완전 다운로드가 아니라면 `streaming: true`로 두고 파이프라인이 스트리밍으로 읽도록 운영하는 편이 안전합니다.

## 7. 실행 방법

### 7.1 먼저 소규모 검증

```bash
python -m synth_parallel.cli run \
  --config config/example.yaml \
  --stage all \
  --dry-run \
  --limit 200 \
  --overwrite
```

### 7.2 본 실행

```bash
python -m synth_parallel.cli run \
  --config config/example.yaml \
  --stage all \
  --resume
```

### 7.3 stage 단독 실행

```bash
python -m synth_parallel.cli run --config config/example.yaml --stage sample_sources
python -m synth_parallel.cli run --config config/example.yaml --stage prefilter_score
python -m synth_parallel.cli run --config config/example.yaml --stage select_sources
python -m synth_parallel.cli run --config config/example.yaml --stage generate_128
python -m synth_parallel.cli run --config config/example.yaml --stage score_select_best
python -m synth_parallel.cli run --config config/example.yaml --stage format_filter
python -m synth_parallel.cli run --config config/example.yaml --stage export
```

### 7.4 공통 옵션

- `--resume`: 진행 DB 기준으로 미처리 항목만 수행
- `--overwrite`: 해당 stage 출력 파일 초기화 후 재실행
- `--limit N`: 디버그용 처리량 제한
- `--dry-run`: 내부 기본 소량 제한 적용
- `--shard-id`, `--num-shards`: 샤드 분산 실행

## 8. 산출물

`run.out_dir` 아래 생성:

- `sampled_sources.jsonl`
- `prefilter_candidates.jsonl`
- `selected_sources.jsonl`
- `generated_candidates.jsonl`
- `scored_best.jsonl`
- `filtered.jsonl`
- `rejected.jsonl`
- `final_dataset.jsonl` (최종)
- `final_candidates_topk.jsonl`
- `resolved_config.yaml`
- `manifest.json`

## 9. 로그, 통계, 캐시, 복구

- `logs.txt`
- stage 시작/종료, 진행률, 에러, 재시도 로그
- `stats.json`
- 처리량, 캐시 hit/miss, 필터 reject 사유 카운트
- `progress.sqlite`
- stage/item 단위 완료 상태
- `teacher_cache.sqlite`
- 동일 요청 teacher 응답 재사용
- `metricx_cache.sqlite`
- 동일 `(source, hypothesis)` 점수 재사용

중단 후 동일 커맨드에 `--resume`를 주면 이어서 처리됩니다.

## 10. 운영 팁

- 대규모 실행 전 반드시 dry-run으로 API/MetricX 연결 확인
- `sample_pool_size`, `target_examples_total`, `num_candidates`를 작게 줄여 비용/시간 예측
- MetricX GPU 고정은 `metricx.device: cuda:0` 사용
- 샤딩 시 모든 워커가 같은 입력/설정 버전을 사용해야 재현성 유지

## 11. 트러블슈팅

- `metricx24.predict` 실행 실패
- `metricx.python_bin` 경로 확인
- MetricX 전용 env에서 `cd third_party/metricx && python -m metricx24.predict --help` 먼저 확인
- Qwen API 인증 실패
- `QWEN_API_KEY` 값과 `teacher.api_key_env` 일치 확인
- 처리 도중 중단됨
- `--resume`로 재개
- 완전 재실행이 필요하면 `--overwrite` 사용
- MetricX가 CPU로 동작함
- `metricx.device` 값(`cuda:0`) 확인
- 드라이버/CUDA/torch 호환성 점검

## 12. 테스트

```bash
source .venv/bin/activate
python -m pytest -q
```
