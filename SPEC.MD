# TranslateGemma 방식 재현: MADLAD-400 → 합성 번역(Parallel) 샘플 생성 요구사항 명세서  
*(Teacher: `Qwen/Qwen3-235B-A22B-Instruct-2507` + 로컬 vLLM(OpenAI-Compatible) / 호출: OpenAI SDK 또는 HTTP)*

> 목적: TranslateGemma Technical Report에서 설명한 “MADLAD-400 기반 합성 번역 병렬 데이터 생성” 절차(길이 버킷 샘플링 → 2-sample prefilter → 128-sample 생성 → MetricX-24-QE로 best 선택 → 포맷/오류 필터)를 **로컬 환경**에서 재현 가능한 파이프라인으로 구현한다.  
> 단, 논문에서는 teacher로 *Gemini 2.5 Flash*를 사용했지만, 본 재현에서는 **로컬에서 vLLM으로 서빙하는 `Qwen/Qwen3-235B-A22B-Instruct-2507`**를 teacher로 사용한다. 결과 수치는 동일하지 않을 수 있으나, *프로세스/데이터플로우/필터링 로직*을 최대한 동일하게 구현한다. (논문 절차 요약: 길이 버킷 샘플링 후 1M 소스 추출, greedy vs temperature=1.0 2샘플을 MetricX-24-QE로 비교해 개선 폭 큰 소스 선택, 선택 소스별 128 샘플 생성 후 MetricX-24-QE로 best 선택, 추가 포맷 필터 적용)  

---

## 0. 산출물(Deliverables)

### 0.1 코드/레포 구조
- Python 패키지 형태의 파이프라인 레포(예: `synth_parallel/`).
- 실행용 CLI(예: `python -m synth_parallel.cli ...`) + YAML 설정 파일.
- 재현 가능한 실행 스크립트 및 실험 로그/메타데이터 저장 구조.

### 0.2 출력 데이터(최소)
- 최종 합성 병렬 데이터: JSONL 또는 Parquet
  - 필수 필드:
    - `pair_id` (예: `en->ko`)
    - `source_lang_code`, `target_lang_code`
    - `source_text`
    - `target_text` (선택된 best 번역)
    - `metricx_qe_score_best` (float)
    - `selection` 메타(아래 “필수 메타데이터” 참조)
- 중간 산출물(재현/디버그용, 옵션)
  - `prefilter_candidates.jsonl` (greedy/sample 2개, 점수 포함)
  - `final_candidates_topk.jsonl` (top-k만 저장, 기본 k=1 또는 k=4)
  - `rejected.jsonl` (필터링 탈락 사유 포함)

### 0.3 필수 메타데이터(재현성)
각 최종 샘플에 아래 정보를 저장(또는 sidecar 파일로 저장)해야 한다.
- `madlad`:
  - `lang` (MADLAD lang id)
  - `split` (`clean`/`noisy`)
  - `doc_id` 또는 `document_key`(가능하면)
  - `segment_index`/`span` 등 원문 추적 가능한 포인터
- `teacher`:
  - `backend`: `"vllm_openai_compatible"`
  - `base_url`
  - `model` (요청에 사용된 `model` 값; `--served-model-name`를 쓰면 그 이름)
  - `sampling_params`:
    - prefilter greedy 설정
    - prefilter sample 설정(temperature=1.0)
    - final 128-sample 설정(temperature=1.0)
- `metricx`:
  - 사용한 모델/체크포인트 식별자
  - 실행 버전(패키지 버전/커밋)
- `filters`:
  - 어떤 필터를 통과/탈락했는지, 탈락 사유 코드

---

## 1. 논문 기반 목표 프로세스(재현 대상)

논문 Section 2.1 요지(구현에 반영해야 하는 핵심):
- MADLAD-400을 monolingual source로 사용.
- 언어쌍별 최대 10K 합성 예시 생성 목표.
- 소스 선택을 위해:
  1) 원본 세그먼트를 길이로 버킷팅  
  2) 버킷별 샘플링으로 언어쌍당 1,000,000개 소스 세그먼트 구성  
  3) 각 소스에 대해 teacher로 **2개 번역** 생성: greedy 1개 + temperature=1.0 샘플 1개  
  4) MetricX-24-QE로 두 번역을 평가하여, 샘플이 greedy 대비 가장 크게 “개선”된 소스를 선택  
- 선택된 소스에 대해:
  5) teacher로 128개의 번역 샘플 생성  
  6) MetricX-24-QE로 best 후보(가장 좋은 점수) 1개 선택  
- 추가로:
  7) 포맷 문제/오류 번역 방지를 위한 formatting filter(논문은 Gemini 2.5 Flash 기반) 적용  
- 길이 다양성을 위해 “개별 문장”과 “최대 512 토큰 블랍(text blob)” 두 유형을 포함.  

> 구현상 애매한 부분(논문에 상세 미기재)은 **config로 조절 가능**하게 만들고, default는 논문 의도에 맞는 합리적 선택으로 둔다.

---

## 2. 시스템 개요(Architecture)

### 2.1 모듈 구성
1. **Data Ingest / Segmenter**
   - MADLAD-400을 로드하고 문서 → 세그먼트(문장/라인)로 변환
2. **Length Bucketer & Sampler**
   - 세그먼트를 길이 기반 버킷으로 나누고 1M 소스를 추출
3. **Teacher Client (vLLM OpenAI-Compatible)**
   - 로컬 vLLM 서버로 chat completions 호출
4. **Prefilter (2-sample + MetricX QE)**
   - greedy vs sampled(temperature=1.0) 점수 비교 → 소스 랭킹/선택
5. **Final Candidate Generation (128-sample)**
   - 선택 소스별 128 후보 생성
6. **MetricX-24-QE Scorer**
   - 후보별 QE 점수 산출, best 선택
7. **Formatting / Error Filter (LLM judge + 규칙 기반)**
   - 번역 포맷/언어/누락/잡음 등을 검사, 탈락 처리
8. **Writer**
   - 최종 JSONL/Parquet 기록 + 메타/통계 저장

### 2.2 실행 방식(Stages)
- 대규모 파이프라인이므로 stage별 실행/재시작 가능해야 한다.
- 예시 stage:
  - `stage=sample_sources`
  - `stage=prefilter_score`
  - `stage=select_sources`
  - `stage=generate_128`
  - `stage=score_128_select_best`
  - `stage=format_filter`
  - `stage=export`

---

## 3. 로컬 Teacher 서빙(vLLM ) 요구사항

### 3.1 기본 전제
- teacher는 로컬에서 vLLM을 사용해 `Qwen/Qwen3-235B-A22B-Instruct-2507`를 서빙한다.
- 파이프라인은 **OpenAI-compatible HTTP API**로 호출한다.
- chat 요청을 쓰므로, vLLM 서버는 Chat Completions를 정상 처리해야 한다(모델에 chat template이 없으면 지정 필요).

### 3.2 vLLM 설치/버전
- Qwen/Qwen3-235B-A22B-Instruct-2507 계열은 vLLM 레시피 문서에서 **vLLM + transformers 최신 소스** 설치를 안내하는 경우가 있다.  
  - 구현자는 설치 실패/모델 로딩 실패 시 해당 가이드를 따라 vLLM/transformers 버전을 맞출 수 있게 문서화할 것.

### 3.4 vLLM 서버 실행 커맨드(권장)
> 아래 커맨드는 “예시”다. 실제 환경(멀티 GPU/포트/메모리)에 맞게 config화한다.

- 최소 실행 예시:
```bash
export VLLM_API_KEY="token-abc123"

vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "${VLLM_API_KEY}" \
  --dtype auto \
  --generation-config vllm
```

필수 의도:
- `--api-key`: 로컬이지만 클라이언트가 key를 보내도록 맞춘다.
- `--dtype auto`: 환경에 맞게 dtype 선택.
- `--generation-config vllm`: HuggingFace `generation_config.json`에 의해 샘플링 기본값이 덮어써지는 것을 방지(재현성/실험통제 목적).


### 3.5 Chat template 요구사항
- vLLM은 chat 요청 처리 시 tokenizer의 chat template을 필요로 한다.
- 모델에 chat template이 없거나 동작이 맞지 않으면, 서버 실행 시 `--chat-template /path/to/template.jinja`를 지원해야 한다.
- 파이프라인은 서버 오류 메시지(“chat template 없음” 등)를 감지하여, 운영자가 chat template 설정을 추가해야 함을 명확히 안내한다.

### 3.6 OpenAI SDK로 호출하는 클라이언트 기본
- Python OpenAI SDK(또는 호환 클라이언트)를 사용해 base_url을 로컬 vLLM로 변경하여 호출한다.
- 예시(파이프라인 내부 구현 예):
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",  # vllm serve의 --api-key와 일치
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[{"role": "user", "content": "Translate: Hello -> Korean"}],
    temperature=1.0,
    top_p=1.0,
    max_tokens=256,
)
text = resp.choices[0].message.content
```

요구사항:
- 네트워크/서버 busy 시 retry(지수 백오프) + idempotency(재시도 시 동일 요청 식별) 지원
- timeout(연결/읽기) 설정 가능
- 동시 요청 수(concurrency) 제한 및 rate limiter 지원

---

## 4. 데이터 소스: MADLAD-400 로딩/세그먼트화 요구사항

### 4.1 데이터셋 로딩
- HuggingFace `datasets`로 MADLAD-400 로드.
- clean/noisy split을 선택 가능해야 한다.
- 언어는 MADLAD lang id(예: `kor` 등) 기준으로 로드한다.

예시(문서화에 포함):
```python
from datasets import load_dataset

ds = load_dataset("allenai/madlad-400", "kor", split="clean")
```

### 4.2 문서 → 세그먼트 추출
MADLAD는 “document-level” 데이터이므로, 파이프라인은 최소 2가지 케이스를 처리해야 한다:
- 케이스 A: 레코드에 이미 문장/라인 리스트가 있는 경우 (예: `text: List[str]`)
- 케이스 B: 레코드가 단일 문자열 문서인 경우 (예: `text: str`)
  - 이 경우 heuristic 분할:
    - 우선 줄바꿈 기준 split
    - 너무 긴 라인은 추가로 문장부호 기반 split(간단 규칙)
    - 너무 짧은 라인은 인접 라인과 병합 가능(옵션)

필수:
- 세그먼트마다 원문 추적 가능 메타 유지(문서 id, 문서 내 offset 등)
- 극단적 노이즈(HTML/코드/URL-only)는 rule-based prefilter로 제거(옵션)

---

## 5. 길이 버킷 & 1M 샘플링 요구사항

### 5.1 길이 정의
- default 길이 측정: “근사 토큰 길이” (공백 토큰 수 + punctuation 가중치)  
- 옵션: teacher tokenizer 기반 길이(비용 큼)도 선택 가능

### 5.2 버킷팅 정책
- config로 버킷 경계(boundaries)를 지정 가능해야 한다.
  - 예: `[0, 10, 20, 40, 80, 120, 200, 400, 800, inf]` (토큰 근사)
- 각 버킷에서 균등하게 샘플링하여 전체 1,000,000개를 만들되,
  - 특정 버킷 데이터가 부족하면 다른 버킷에서 보충

### 5.3 출력
- `sampled_sources.parquet` (또는 jsonl)
  - 필드: `source_text`, `length_bucket_id`, `source_meta...`

---

## 6. Prefilter(2-sample) 요구사항

### 6.1 생성 설정
- **greedy 번역**
  - `temperature = 0`
  - `top_p = 1.0`
  - `max_tokens`: config (예: 512)
- **sample 번역**
  - `temperature = 1.0`
  - `top_p = 1.0`
  - 동일 `max_tokens`

### 6.2 MetricX-24-QE 점수 비교
- 각 소스에 대해:
  - `score_greedy = metricx_qe(source, greedy_translation)`
  - `score_sample = metricx_qe(source, sampled_translation)`
- Metric 방향:
  - MetricX 계열은 “낮을수록 좋음(0 best … 25 worst)” 범위를 사용한다는 문맥이 있으므로,
  - 개선량은 `improvement = score_greedy - score_sample` 로 정의(양수면 sample이 더 좋음).
- 소스 선택:
  - 버킷별 또는 전체 기준으로 `improvement` 내림차순 정렬
  - 상위 N개 선택
  - N 기본값: `target_examples_per_pair_total` (예: 10,000)

### 6.3 저장
- 모든 1M에 대해 2개의 번역을 저장하면 비용 큼. 기본 정책:
  - 최소: `score_greedy`, `score_sample`, `improvement`만 저장
  - optional: 번역 텍스트도 저장(디버그 모드)

---

## 7. Final 128-sample 생성 + MetricX best 선택 요구사항

### 7.1 후보 생성
- 선택된 소스 N개 각각에 대해 128 후보 생성.
- 기본 샘플링:
  - `temperature=1.0`
  - `top_p=1.0`
  - `max_tokens`: config
- 구현 전략:
  - 128회를 개별 요청으로 보내는 방식(확실)
  - (옵션) OpenAI-style `n=128`을 지원하는 서버라면 단일 요청으로 후보를 받는 fast path 제공  
    - 실패 시 자동으로 128회 개별 호출로 fallback

### 7.2 후보 스코어링 & best 선택
- 각 후보에 대해 MetricX-24-QE 계산
- best 정의:
  - `best = argmin(score)` (낮을수록 좋음)
- 저장 정책:
  - default: best 1개만 저장
  - optional: top-k 저장(`k=4` 등) + score 포함

### 7.3 길이 두 유형(sentence vs blob) 지원
- config:
  - `sentence_ratio` / `blob_ratio`
  - `blob_max_tokens` 기본 512
- blob 생성 방식(권장):
  - 동일 문서 내에서 연속 세그먼트를 누적하여 길이가 `<= blob_max_tokens`가 되도록 묶기
  - 문서 경계/문단 경계를 최대한 보존
  - blob도 prefilter/128-sample 과정을 동일 적용할 수 있게 구현

---

## 8. Formatting / Error Filter 요구사항 (로컬 Qwen/Qwen3-235B-A22B-Instruct-2507 기반)

논문은 Gemini 2.5 Flash로 포맷/오류 필터를 추가 적용했다. 본 재현은 외부 의존을 피하기 위해 **동일 vLLM 서버의 Qwen/Qwen3-235B-A22B-Instruct-2507**을 “judge”로 호출하거나, rule-based 필터를 병행한다.

### 8.1 rule-based 필터(필수)
다음 중 하나라도 해당하면 탈락:
- 번역 결과에 메타 문구 포함(예: “Here is the translation”, “번역:”, “I will translate…” 등)
- 역할 토큰/채팅 잔재(예: “assistant:”, “user:”, “system:”)
- 극단적 길이 이상치:
  - `len(target) < min_chars` 또는 `len(target) > max_chars`
  - `len(target)/len(source)`가 너무 작거나 큼 (config)
- 소스가 거의 그대로 복사됨(언어가 다르면 높은 overlap)
- 이상한 태그/마커:
  - `<think>`, `</think>`, ``` 등의 잔재가 있는 경우(허용 리스트로 제어)

### 8.2 LLM judge 필터(옵션이지만 권장)
- 입력: `{source_lang, target_lang, source_text, candidate_translation}`
- 출력(강제 JSON):
  - `{"pass": true/false, "reason_code": "...", "notes": "..."}`
- 판정 기준:
  - 번역이 target 언어로 보이는가?
  - 소스 의미를 크게 훼손/누락하지 않았는가(거친 judge)
  - 불필요한 설명/코멘트가 없는가?
  - 줄바꿈/리스트/번호 등 구조를 지나치게 깨지 않았는가?
- 구현 요건:
  - judge 프롬프트는 “짧고 결정적”이어야 하며, 출력은 JSON만 허용
  - vLLM guided decoding(JSON schema) 또는 regex 기반으로 JSON 강제(가능하면)
  - judge 호출 실패 시: rule-based만으로 통과시키거나, 재시도 후 실패면 탈락 처리(config)

---

## 9. 번역 프롬프트 템플릿 요구사항(Teacher Prompt)

논문 Figure 3의 “preferred prompt”를 참고하되, 본 구현에서는 아래 요구를 만족하는 **템플릿 기반 프롬프트**를 사용한다.

### 9.1 템플릿 변수
- `{source_lang}`: 예 “English”
- `{target_lang}`: 예 “Korean”
- `{src_lang_code}`: 예 “en-US” (또는 “en”)
- `{tgt_lang_code}`: 예 “ko-KR” (또는 “ko”)
- `{text}`: 번역할 원문

### 9.2 프롬프트 요구사항
- 모델에게 “전문 번역가 역할”을 부여
- 의미/뉘앙스 보존, target 언어 문법/자연스러움 준수 지시
- **출력은 번역문만**(설명/해설/코멘트 금지)
- 입력 텍스트를 명확히 구분(예: “Text:” 섹션)

### 9.3 chat messages 구성(권장)
- `system`: “You are a professional translator.” 등 매우 짧게
- `user`: 위 템플릿으로 만든 구체 지시 + 텍스트

---

## 10. MetricX-24-QE 스코어러 요구사항

### 10.1 모드
- QE 모드(Reference-free)로 사용한다.
- 입력 포맷(개념):
  - `source` (원문)
  - `hypothesis` (번역 후보)
  - `reference`는 빈 문자열(또는 null이 아니라 빈 string)

### 10.2 모델 선택(config)
- 기본값 예시:
  - `google/metricx-24-hybrid-large-v2p6` 또는 `google/metricx-24-hybrid-xl-v2p6`
- 구현자는:
  - metric 모델/체크포인트를 쉽게 교체할 수 있게 추상화
  - GPU/CPU에서 실행 가능하게 구현(단, 속도는 GPU 권장)

### 10.3 배치 처리
- 128 후보 스코어링은 배치로 묶어 처리(성능 필수)
- 점수 캐시:
  - 동일 (source, hypothesis) 재평가 방지 (hash key)

---

## 11. 설정 파일(YAML) 스키마 요구사항

> 아래는 예시이며, 실제 구현에서 키 이름은 변경 가능하나 “의미/기능”은 충족해야 한다.

```yaml
run:
  out_dir: ./runs/exp001
  seed: 1234
  log_level: INFO

data:
  madlad_dataset: allenai/madlad-400
  madlad_split: clean          # clean | noisy
  src_lang: kor                # MADLAD lang id
  tgt_lang: eng                # 목적 언어 코드(메타용; 실제 번역 prompt용)
  target_examples_total: 10000
  sample_pool_size: 1000000

segmentation:
  mode: auto                   # auto | newline | list_field
  min_chars: 20
  max_chars: 5000

bucketing:
  boundaries: [0, 10, 20, 40, 80, 120, 200, 400, 800, 999999]
  measure: approx_tokens
  per_bucket_quota: auto       # auto면 균등 분배

teacher:
  backend: vllm_openai_compatible
  base_url: http://localhost:8000/v1
  api_key_env: VLLM_API_KEY
  model: Qwen/Qwen3-235B-A22B-Instruct-2507   # 또는 served-model-name
  request_timeout_s: 120
  max_concurrency: 32
  retry:
    max_attempts: 6
    backoff_s: [1, 2, 4, 8, 16, 32]
  generation:
    max_tokens: 512
    top_p: 1.0
    greedy_temperature: 0.0
    sample_temperature: 1.0
    final_temperature: 1.0
    # optional:
    # seed: 1234

final_generation:
  num_candidates: 128
  store_top_k: 1               # 1이면 best만 저장, 4면 top-4 저장
  strategy: auto               # auto|per_call|n_parameter
  blob:
    enabled: true
    blob_ratio: 0.5
    blob_max_tokens: 512

metricx:
  checkpoint: google/metricx-24-hybrid-large-v2p6
  batch_size: 64
  device: cuda                 # cuda|cpu
  cache_db: ./runs/exp001/metricx_cache.sqlite

filters:
  rule_based: true
  llm_judge:
    enabled: true
    model: Qwen/Qwen3-235B-A22B-Instruct-2507
    temperature: 0.0
    max_tokens: 128
    fail_policy: conservative  # conservative=실패시 탈락, permissive=실패시 통과
```

---

## 12. CLI 요구사항

### 12.1 커맨드
- `synth_parallel run --config path.yaml --stage sample_sources`
- `synth_parallel run --config path.yaml --stage prefilter_score`
- `synth_parallel run --config path.yaml --stage select_sources`
- `synth_parallel run --config path.yaml --stage generate_128`
- `synth_parallel run --config path.yaml --stage score_select_best`
- `synth_parallel run --config path.yaml --stage format_filter`
- `synth_parallel run --config path.yaml --stage export`

### 12.2 공통 옵션
- `--dry-run` (100~1000 샘플만)
- `--resume` (중간 산출물 감지해 이어서)
- `--overwrite`
- `--limit N` (디버그용)

---

## 13. 관측 가능성(Logging/Stats)

필수:
- stage별 처리량(예: examples/s)
- teacher 호출 성공/실패/재시도 통계
- MetricX batch 처리 시간/메모리
- 필터 탈락 분포(reason_code별 카운트)
- 최종 데이터 길이 분포(소스/타겟 토큰/문자)

산출:
- `runs/<exp>/stats.json`
- `runs/<exp>/logs.txt`
- (옵션) Prometheus metric export

---

## 14. 테스트(필수)

### 14.1 단위 테스트
- 세그먼트화:
  - list_field/string_field 케이스 모두 처리
- 길이 버킷/샘플러:
  - 버킷별 quota 충족, 부족 시 보충 로직
- teacher client:
  - timeout/retry 동작
  - 응답 파싱(choices/message/content)
- MetricX wrapper:
  - empty reference로 QE 모드 동작
  - 배치 입력/출력 shape 검증
- best 선택:
  - score 최소값 선택

### 14.2 통합 테스트(로컬)
- 작은 언어쌍(예: 100개 소스, 후보 8개)로 end-to-end 실행
- 산출물 스키마 검증(JSON schema)

---

## 15. 성능/자원 요구사항

- 1M prefilter는 매우 큼:
  - stage별 sharding 지원(예: `--shard-id`, `--num-shards`)
  - teacher 호출 병렬화 + backpressure
- 128-sample 생성은 비용 폭증:
  - 선택 소스를 N=10k로 하면 후보 1.28M개 생성
  - MetricX도 1.28M 스코어링 필요 → 배치/캐시/샤딩 필수
- 디스크:
  - 중간 결과를 모두 저장하면 수백 GB 가능 → 저장 정책 config화

---

## 16. 리스크/주의사항

- **논문 teacher(Gemini) ↔ 재현 teacher(Qwen/Qwen3-235B-A22B-Instruct-2507)** 차이로 인해:
  - prefilter의 improvement 분포, best 선택 결과가 달라질 수 있음
- vLLM 서버의 `generation_config.json` 자동 적용은 샘플링을 예기치 않게 변경할 수 있으므로 `--generation-config vllm`를 권장
- chat template 미설정 시 chat API가 실패할 수 있으므로 서버 설정 점검 필요
- FP8 경로는 하드웨어 의존: 품질/성능이 환경마다 변동

---

## 17. (Codex용) 실행 지시문 템플릿

1) 위 요구사항을 만족하는 Python 프로젝트 스캐폴딩 생성  
2) `teacher`는 OpenAI SDK로 `base_url`을 로컬 vLLM로 설정해 호출  
3) MADLAD-400 로딩 → 세그먼트화 → 길이 버킷 샘플링(1M) 구현  
4) prefilter(2샘플) → MetricX QE 스코어링 → top N 소스 선택 구현  
5) 선택 소스별 128 후보 생성 → MetricX QE로 best 선택 구현  
6) rule-based + LLM judge formatting filter 구현  
7) stage별 저장/재개(resume) 구현  
8) 최소 단위/통합 테스트 작성  
9) `README.md`에 “vLLM 서버 실행법(예시 커맨드), config 작성법, end-to-end 실행 예시” 포함

---

## 18. 참고 링크(개발자용)

- TranslateGemma Technical Report (arXiv 2601.09012)
- MADLAD-400 HF dataset: `allenai/MADLAD-400`
- vLLM OpenAI-compatible server 문서(serve, chat template, served-model-name, generation-config)
- MetricX-24(-QE) 관련: google-research/metricx 및 HF 모델 카드
