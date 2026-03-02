# LLM Post-Training 프로젝트

## 프로젝트 개요
번역 모델(Gemma-27B, Qwen3.5-35B MoE)의 Post-Training 파이프라인.
SFT → RL(GRPO/GSPO) → 평가까지의 전체 학습 흐름을 구현.

## 프로젝트 구조
```
hwang-post-training/
├── gemma27_rl/          # Gemma-27B GRPO RL 학습
├── qwen3.5-35b-a3b/     # Qwen3.5-35B MoE GSPO 학습
├── gemma27b_sft/        # Gemma-27B SFT 학습
├── synth_parallel/      # 합성 데이터 생성 파이프라인
├── config/              # 설정 예시
├── tests/               # 통합 테스트
└── .claude/agents/      # 에이전트 지침 파일
```

## 3-Agent System

이 프로젝트는 세 가지 전문 에이전트를 활용한 협업 시스템으로 운영됩니다.

### Agent 1: Research Agent (연구/방향성 에이전트)
- **역할**: 최신 LLM post-training 기법 조사 및 구현 방향성 정립
- **지침 파일**: `.claude/agents/research-agent.md`
- **호출 방법**: `subagent_type=general-purpose` + research-agent.md 내용 참조
- **담당 영역**:
  - 최신 논문/기법 조사 (GRPO, DPO, ORPO, SimPO, KTO, DAPO, Dr.GRPO 등)
  - 알고리즘 비교 분석 및 선택
  - 보상 모델 전략 설계
  - 데이터 전략 및 커리큘럼 학습 설계
  - 구현 우선순위 및 로드맵 수립

### Agent 2: Implementation Agent (구현 에이전트)
- **역할**: Multi-node distributed training 코드 구현
- **지침 파일**: `.claude/agents/implementation-agent.md`
- **호출 방법**: `subagent_type=general-purpose` + implementation-agent.md 내용 참조
- **담당 영역**:
  - PyTorch, Transformers, TRL 기반 학습 코드 작성
  - FSDP / DeepSpeed ZeRO 분산 학습 구현
  - Multi-node torchrun / deepspeed launcher 설정
  - 모델 로딩, 체크포인팅, 데이터 파이프라인 구현
  - Configuration 시스템 설계

### Agent 3: Debug Agent (디버깅 전문가 에이전트)
- **역할**: 로직 정확성, GPU 최적화, 학습 안정화 디버깅
- **지침 파일**: `.claude/agents/debug-agent.md`
- **호출 방법**: `subagent_type=general-purpose` + debug-agent.md 내용 참조
- **담당 영역**:
  - 학습 로직 정확성 검증 (gradient flow, loss 계산, advantage 정규화)
  - GPU 메모리 최적화 (OOM 해결, gradient checkpointing, mixed precision)
  - 분산 학습 버그 수정 (deadlock, hang, collective 불일치)
  - 수치 안정성 (NaN/Inf 감지, gradient clipping, KL divergence 폭발)
  - 성능 프로파일링 및 병목 식별

## Agent 협업 워크플로우
```
[Research Agent] → 기법 조사 & 설계 문서 산출
        ↓
[Implementation Agent] → 설계 기반 코드 구현
        ↓
[Debug Agent] → 코드 검증 & 최적화 & 안정화
        ↓
[Research Agent] → 실험 결과 분석 & 다음 방향 제시
```

## 핵심 기술 스택
- **Framework**: PyTorch 2.6+, Transformers 4.57+, TRL
- **분산 학습**: FSDP, DeepSpeed ZeRO (1/2/3), DDP
- **RL 알고리즘**: GRPO, GSPO (MoE 특화)
- **보상 모델**: MetricX-24-QE, XCOMET-XL, MQM Scorer
- **모델**: Gemma-3-27B-IT, Qwen3.5-35B-A3B (MoE)
- **하드웨어**: H100 8-GPU (단일 노드), Multi-node 확장 예정

## 코딩 컨벤션
- Python 3.10+, type hints 사용
- dataclass 기반 Config 시스템 (YAML 로드)
- 모든 분산 로직에 rank 체크 (`if rank == 0:` for logging)
- 테스트는 pytest, `tests/` 디렉토리에 배치
- 한국어 주석 허용, docstring은 영어
