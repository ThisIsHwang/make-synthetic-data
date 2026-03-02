# distributed_rl Milestones

## Milestone 1: TRL GRPOTrainer Multi-Node RL (완료)

TRL GRPOTrainer + DeepSpeed ZeRO 기반 Multi-Node GRPO 학습 파이프라인 구축.

- [x] YAML → dataclass config 시스템
- [x] TRL GRPOTrainer 통합 (GRPO/DAPO/DR-GRPO/SAPO/GSPO)
- [x] MetricX / XComet reward 모델 (subprocess GPU 격리)
- [x] DeepSpeed ZeRO-2/3 + Multi-node torchrun
- [x] WMT24pp En→Ko 데이터 파이프라인
- [x] Rank-aware reward architecture (rank 0 only)

## Milestone 2: MiniRL Stability — Phase 1 (완료)

MiniRL(arxiv 2512.01374) 기반 학습 안정성 기능.

- [x] StabilityConfig dataclass
- [x] StabilityMonitorCallback (entropy/KL/reward 추세 분석, collapse 감지)
- [x] Reward clipping (outlier 방지)
- [x] 비대칭 Clipping (epsilon_high=0.27, epsilon_low=0.2)
- [ ] Routing Replay R2/R3 (Phase 2, MoE 전용 — 미완)

---

## Milestone 3: Dynamic Thinking Budget (다음)

### 개요

모델이 문제 난이도에 따라 **thinking token 수를 동적으로 조절**하도록 학습하는 기능.
번역 RL에서는 단순 문장(짧고 명확한 소스)에는 thinking 없이 직접 번역하고,
복잡한 문장(긴 문장, 전문용어, 모호한 구문)에는 thinking을 활성화하여 품질을 높인다.

### 배경 연구

#### GLM-5 (arxiv 2602.15763)

GLM-5는 세 가지 thinking 모드를 도입:

1. **Interleaved Thinking**: 모든 응답/tool call 전에 reasoning 수행
2. **Preserved Thinking**: 멀티턴에서 이전 thinking block을 재활용
3. **Turn-level Thinking**: per-turn으로 thinking 활성화/비활성화 제어

학습 파이프라인은 Progressive Alignment 구조:
- Multi-task SFT (thinking 모드 포함) → Reasoning RL → Agentic RL → General RL
- On-Policy Cross-Stage Distillation으로 catastrophic forgetting 방지
- "Slime" 비동기 RL 인프라: rollout worker pool이 비동기로 trajectory 생성, training worker가 experience buffer에서 소비

**핵심 인사이트**: GLM-5는 thinking을 user-controlled toggle로 구현하지만,
실제 학습에서는 SFT 단계에서 thinking trace가 포함된 데이터로 학습 후
RL 단계에서 reasoning 품질을 강화하는 2단계 접근을 사용.

#### BudgetThinker (arxiv 2508.17196)

Dynamic thinking budget의 가장 구체적인 구현체. **GRPO 기반**이므로 우리 파이프라인과 직접 호환:

1. **Control Token 메커니즘**: K개의 특수 토큰을 thinking 구간에 균등 간격으로 삽입하여 남은 budget을 모델에 알림
   - K=8, budget B일 때 t = k·⌊B/K⌋ 위치에 control token c_{k+1} 삽입
   - 비율 기반(ratio-based) → budget 크기에 관계없이 고정 개수의 토큰으로 budget 인식

2. **2-Stage Training Pipeline**:
   - **Stage 1 — SFT**: 문제-풀이 쌍에 control token 삽입한 데이터로 fine-tuning
   - **Stage 2 — RL (GRPO)**: Length-aware reward로 budget 준수 + 정확도 최적화

3. **Length-Aware Reward Function**:
   ```
   R(y, y_gold, B) = k₁·𝟙{correct} + k₂·𝟙{format} + k₃·max(1 - γ·((B-|y|)/B)², 0)
   ```
   - k₁=0.7 (정확도), k₂=0.15 (포맷), k₃=0.15 (길이)
   - **비대칭 패널티**: γ=1 (budget 이하), γ=16 (budget 초과) → 초과 시 급격한 패널티

4. **Curriculum RL**: Budget을 점진적으로 줄이며 학습
   - B₁=6000 → B₂=4000 → B₃=3000 → B₄=2000
   - 최종 단계에서 모든 budget을 혼합하여 forgetting 방지

#### Budget Forcing (DeepSeek-R1 계열)

- Inference 시 thinking 구간에 hard token limit 적용
- `max_thinking_tokens` 도달 시 end-of-thinking delimiter 강제 삽입하여 답변 생성 유도
- Training 시에는 적용하지 않고 inference-only 기법

### 번역 RL에 대한 적용 설계

번역 태스크에 맞춘 Dynamic Thinking Budget 구현. 수학 문제 풀이와 달리
번역은 "정답 여부"가 binary가 아니라 **연속적 품질 스코어**(MetricX, XComet)이므로
reward 설계를 적절히 변형해야 함.

#### Phase 3-A: Thinking-Aware Data Pipeline

**목표**: thinking trace가 포함된 번역 데이터 구축

```
<think>
The source contains the technical term "gradient checkpointing" which in Korean
is typically translated as "그래디언트 체크포인팅" (transliteration) or
"기울기 검사점 저장" (semantic translation). Given the context is a technical
document, transliteration is more appropriate...
</think>

그래디언트 체크포인팅은 메모리 사용량을 줄이기 위해 ...
```

구현 항목:
- [ ] `ThinkingConfig` dataclass 추가 (`config.py`)
  - `enable_thinking: bool = False`
  - `think_start_token: str = "<think>"`
  - `think_end_token: str = "</think>"`
  - `max_thinking_tokens: int = 512` — thinking 구간 최대 길이
  - `budget_control_tokens: int = 8` — K값 (budget 알림 토큰 수)
  - `budget_levels: list[int] = [512, 256, 128, 64]` — curriculum budget 단계
  - `thinking_reward_weight: float = 0.15` — length reward 가중치
  - `budget_overshoot_penalty: float = 16.0` — 초과 시 비대칭 패널티 γ
- [ ] Thinking-aware prompt template (`prompting.py`)
  - 기존 번역 프롬프트에 thinking 지시 추가
  - Budget 명시: `"Think within {budget} tokens before translating."`
- [ ] Control token 등록 (`tokenizer` 확장)
  - `<think>`, `</think>`, `<budget_1/8>` ~ `<budget_8/8>` special tokens
  - tokenizer.add_special_tokens() 호출 + model embedding resize

#### Phase 3-B: Thinking-Aware Reward Function

**목표**: 번역 품질 + thinking 효율성을 동시에 최적화하는 reward

```python
def thinking_aware_reward(completion, budget, metricx_score, offset=5.0):
    """
    R = w_quality * quality_reward + w_length * length_reward

    quality_reward: MetricX/XComet 기반 (기존과 동일)
    length_reward:  thinking 길이가 budget에 얼마나 부합하는지
    """
    quality_reward = offset - metricx_score  # 기존 MetricX reward

    # thinking 구간 길이 추출
    think_tokens = extract_thinking_length(completion)

    # 정규화된 편차
    delta = abs(think_tokens - budget) / budget

    # 비대칭 패널티: 초과 시 γ=16, 미만 시 γ=1
    gamma = 1.0 if think_tokens <= budget else budget_overshoot_penalty
    length_reward = max(1.0 - gamma * delta**2, 0.0)

    return w_quality * quality_reward + w_length * length_reward
```

구현 항목:
- [ ] `ThinkingLengthReward` 클래스 (`rewards/thinking.py`)
  - completion에서 `<think>...</think>` 구간 파싱
  - thinking token 수 계산
  - budget 대비 length reward 산출
  - 비대칭 패널티 (overshoot γ=16)
- [ ] `make_thinking_reward_func()` — TRL reward function 래퍼
  - 기존 `make_trl_reward_func()`와 병렬로 동작
  - quality reward + length reward 합산은 TRL의 multi-reward 메커니즘 활용
- [ ] Thinking 없는 completion 처리 정책
  - thinking budget이 0이거나 `<think>` 태그 없으면 length_reward = 1.0 (패널티 없음)
  - 모델이 "thinking 안 하기"를 선택할 수 있는 자유도 보장

#### Phase 3-C: Curriculum Budget RL Training

**목표**: 점진적으로 budget을 줄여가며 효율적 thinking 학습

```
Stage 1 (SFT):     Thinking trace 포함 번역 데이터로 fine-tuning
Stage 2 (RL-Easy):  Budget=512 → 여유 있는 thinking으로 품질 최적화
Stage 3 (RL-Mid):   Budget=256 → 핵심만 추리는 thinking 학습
Stage 4 (RL-Hard):  Budget=128 → 최소한의 thinking으로 품질 유지
Stage 5 (RL-Mixed): Budget∈{512,256,128,64} 랜덤 → forgetting 방지
```

구현 항목:
- [ ] `CurriculumBudgetScheduler` 클래스 (`thinking.py`)
  - 학습 step에 따라 budget level 자동 전환
  - TrainerCallback으로 구현, on_step_begin에서 current budget 업데이트
  - Mixed 단계에서는 batch 내 랜덤 budget 할당
- [ ] Budget-conditioned generation
  - `GenerationConfig` 확장: `thinking_budget` 필드 추가
  - Prompt에 현재 budget 삽입
  - Generation 시 thinking 구간에 control token 자동 삽입
- [ ] Control token 삽입 로직
  - Post-processing: 생성된 thinking 구간에 budget fraction 위치마다 control token 삽입
  - 또는 logits processor로 generation 중 실시간 삽입

#### Phase 3-D: Budget Forcing (Inference-Time)

**목표**: Inference 시 thinking token 수 hard limit 적용

구현 항목:
- [ ] `ThinkingBudgetLogitsProcessor` 클래스
  - `transformers.LogitsProcessor` 상속
  - thinking 구간 내 토큰 수 카운팅
  - budget 도달 시 `</think>` 토큰 확률을 1.0으로 강제
- [ ] Generation config 연동
  - `max_thinking_tokens` → LogitsProcessor에 전달
  - `temperature` / `top_p`는 thinking 구간과 translation 구간에서 별도 설정 가능하도록

#### Phase 3-E: 평가 및 분석

- [ ] Thinking 효율성 메트릭
  - `thinking_tokens_per_sample`: 평균 thinking 길이
  - `quality_vs_thinking_length`: thinking 길이별 MetricX score 분포
  - `budget_adherence_rate`: budget 준수율 (BudgetThinker 기준 ~98.6% 목표)
  - `thinking_skip_rate`: thinking을 건너뛴 비율 (간단한 문장)
- [ ] A/B 비교 실험
  - Baseline: thinking 없는 기존 GRPO
  - Variant A: 고정 thinking (항상 생각)
  - Variant B: Dynamic budget thinking (이 milestone)
  - 메트릭: MetricX score, XComet score, 평균 생성 길이, inference 비용

### 파일 변경 예상

| 파일 | 변경 유형 | 설명 |
|------|----------|------|
| `config.py` | Modified | `ThinkingConfig` dataclass 추가, `DistributedRLConfig.thinking` 필드 |
| `prompting.py` | Modified | Thinking-aware prompt template 추가 |
| `rewards/thinking.py` | **New** | `ThinkingLengthReward`, length-aware reward, budget 파싱 |
| `rewards/base.py` | Modified | `make_thinking_reward_func()` 추가 |
| `trainer.py` | Modified | Thinking reward 통합, CurriculumBudgetScheduler callback 주입 |
| `thinking.py` | **New** | `CurriculumBudgetScheduler`, `ThinkingBudgetLogitsProcessor`, control token 유틸 |
| `data.py` | Modified | Thinking trace 포함 데이터 로딩 지원 |
| `tests/test_thinking.py` | **New** | Thinking budget 관련 단위 테스트 |
| `configs/thinking_example.yaml` | **New** | Thinking budget 학습 예시 config |

### 의존성

- 기존 TRL GRPOTrainer 파이프라인 (Milestone 1) 위에 구축
- StabilityMonitor (Milestone 2)와 호환 필요
- 추가 패키지 의존성 없음 (transformers LogitsProcessor 활용)

### 참고 자료

- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) — Interleaved/Preserved/Turn-level Thinking, Slime RL
- [BudgetThinker: Budget-aware LLM Reasoning with Control Tokens](https://arxiv.org/abs/2508.17196) — Control token, GRPO 기반 curriculum RL, length-aware reward
- [Token-Budget-Aware LLM Reasoning](https://arxiv.org/abs/2412.18547) — Dynamic token budget 할당
- [Steering LLM Thinking with Budget Guidance](https://arxiv.org/abs/2506.13752) — Budget guidance 기법
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — Budget forcing, reasoning RL

---

## Milestone 4: Slime-Style Async RL Infrastructure (예정)

GLM-5의 Slime 인프라에서 영감받은 비동기 RL 학습.

현재 구조:
```
[Generation] → [Scoring] → [Training]  (동기, 순차)
```

목표 구조:
```
[Rollout Worker Pool] ──async──▶ [Experience Buffer] ◀──consume── [Training Worker]
```

- [ ] Rollout worker pool (vLLM 기반 inference server)
- [ ] Experience buffer (shared memory 또는 Redis)
- [ ] Training worker의 async consumption
- [ ] Multi-node rollout scaling
- [ ] Fault tolerance (heartbeat, auto-restart)

## Milestone 5: On-Policy Cross-Stage Distillation (예정)

GLM-5의 Cross-Stage Distillation 기법. 다단계 RL 학습 시 이전 단계의 능력 보존.

```
Â(i,t) = sg[log π_teacher(y|x) / π_train(y|x)]
```

- [ ] Teacher model checkpoint 관리
- [ ] Distillation loss 통합 (GRPO loss + distillation loss)
- [ ] Stage 전환 자동화

## Milestone 6: MiniRL Stability — Phase 2: Routing Replay (예정)

MoE 모델(Qwen3.5-35B-A3B) 전용 Routing Replay.

- [ ] R2: Vanilla Routing Replay (rollout expert assignment 재현)
- [ ] R3: Uniform Routing Replay
- [ ] MoE router forward hook 기반 구현
