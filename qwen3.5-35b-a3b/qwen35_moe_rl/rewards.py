from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable

try:
    import torch
except Exception:  # pragma: no cover - optional during lightweight tests
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - optional during lightweight tests
    AutoTokenizer = None  # type: ignore[assignment]

from .config import MetricXConfig, XCometConfig
from .types import RewardOutput, SampleForScoring
from .utils import resolve_device, resolve_torch_dtype


logger = logging.getLogger(__name__)


def metricx_qe_input(src: str, mt: str) -> str:
    return f"source: {src} candidate: {mt}"


def metricx_ref_input(src: str, mt: str, ref: str) -> str:
    return f"source: {src} candidate: {mt} reference: {ref}"


def metricx_score_to_reward(metricx_score: float, offset: float = 5.0) -> float:
    return float(offset) - float(metricx_score)


@dataclass
class MetricXQEScorer:
    cfg: MetricXConfig
    predict_fn: Callable[[list[str]], list[float]] | None = None

    def __post_init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = resolve_device(self.cfg.device)
        self._dtype = resolve_torch_dtype(self.cfg.dtype)
        self._candidate_dtypes: list[Any] = []
        self._active_dtype_idx: int = -1
        self._model_cls: Any = None
        self._model_source: str | None = None

        if self.predict_fn is not None:
            return

        if not self.cfg.enabled:
            return

        if torch is None or AutoTokenizer is None:
            raise RuntimeError(
                "MetricX model loading requires torch and transformers. "
                "Use predict_fn for lightweight testing."
            )
        try:
            from .metricx_model import MT5ForRegression
        except Exception as exc:
            raise RuntimeError(
                "Failed to import MetricX regression model class. "
                "Check transformers installation."
            ) from exc

        self._model_cls = MT5ForRegression
        model_name = self.cfg.model_name
        self._model_source = self._resolve_model_source(model_name)
        self._tokenizer = self._load_tokenizer()
        candidate_dtypes = self._build_candidate_dtypes(model_name=model_name)
        self._candidate_dtypes = list(candidate_dtypes)
        last_error: Exception | None = None
        for idx, cand_dtype in enumerate(candidate_dtypes):
            try:
                logger.info(
                    "Loading MetricX model=%s on device=%s dtype=%s",
                    self._model_source or model_name,
                    self._device,
                    cand_dtype,
                )
                self._model = self._load_metricx_model(self._model_cls, self._model_source or model_name, cand_dtype)
                self._model.to(self._device)
                self._model.eval()
                self._dtype = cand_dtype
                self._active_dtype_idx = idx
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "MetricX load failed for model=%s device=%s dtype=%s: %s",
                    model_name,
                    self._device,
                    cand_dtype,
                    exc,
                )
                if torch is not None and self._device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if self._model is None:
            raise RuntimeError(
                "Failed to load MetricX model after trying multiple dtypes. "
                f"model={model_name} device={self._device} tried_dtypes={candidate_dtypes}."
            ) from last_error

    def _build_candidate_dtypes(self, model_name: str) -> list[Any]:
        if torch is None:
            return [self._dtype]

        candidates: list[Any] = []
        model_name_lc = (model_name or "").lower()
        prefer_fp16_for_xxl = self._device.startswith("cuda") and "metricx-24-hybrid-xxl" in model_name_lc
        if prefer_fp16_for_xxl and torch.float16 not in candidates:
            candidates.append(torch.float16)
        if self._dtype is not None and self._dtype not in candidates:
            candidates.append(self._dtype)

        if self._device.startswith("cuda"):
            if torch.bfloat16 in candidates:
                try:
                    if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
                        logger.warning("CUDA bf16 not supported; removing bfloat16 from MetricX dtype candidates.")
                        candidates = [d for d in candidates if d != torch.bfloat16]
                except Exception:
                    pass
            for fallback_dtype in (torch.float16, torch.float32):
                if fallback_dtype not in candidates:
                    candidates.append(fallback_dtype)
        else:
            if torch.float32 not in candidates:
                candidates.append(torch.float32)

        if not candidates:
            candidates.append(None)
        return candidates

    def _reload_with_next_dtype(self) -> bool:
        if self._model_cls is None:
            return False
        if self._active_dtype_idx < 0:
            return False
        next_idx = self._active_dtype_idx + 1
        if next_idx >= len(self._candidate_dtypes):
            return False

        next_dtype = self._candidate_dtypes[next_idx]
        model_name = self._model_source or self.cfg.model_name
        logger.warning(
            "Retrying MetricX with safer dtype due to non-finite outputs: model=%s old_dtype=%s next_dtype=%s",
            model_name,
            self._dtype,
            next_dtype,
        )
        try:
            self._model = None
            if torch is not None and self._device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model = self._load_metricx_model(self._model_cls, model_name, next_dtype)
            self._model.to(self._device)
            self._model.eval()
            self._dtype = next_dtype
            self._active_dtype_idx = next_idx
            return True
        except Exception as exc:
            logger.warning(
                "MetricX reload failed for model=%s device=%s dtype=%s: %s",
                model_name,
                self._device,
                next_dtype,
                exc,
            )
            if torch is not None and self._device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    def _resolve_model_source(self, model_name: str) -> str:
        text = (model_name or "").strip()
        if not text:
            raise ValueError("MetricX model_name must not be empty.")

        path = Path(text).expanduser()
        if path.exists():
            return str(path)

        try:
            from huggingface_hub import snapshot_download
        except Exception:
            logger.warning(
                "huggingface_hub.snapshot_download unavailable; loading MetricX directly from repo id %s",
                text,
            )
            return text

        cache_dir = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        kwargs: dict[str, Any] = {
            "repo_id": text,
            "cache_dir": cache_dir,
            "max_workers": 1,
            "etag_timeout": 60,
        }

        for attempt in range(2):
            try:
                try:
                    local_path = snapshot_download(resume_download=True, **kwargs)
                except TypeError:
                    kwargs.pop("etag_timeout", None)
                    local_path = snapshot_download(**kwargs)
                logger.info("MetricX snapshot ready: repo=%s local_path=%s", text, local_path)
                return str(local_path)
            except Exception as exc:
                if attempt == 0:
                    self._cleanup_stale_hf_partials(repo_id=text, cache_dir=cache_dir)
                    logger.warning("MetricX snapshot download failed once; retrying: repo=%s err=%s", text, exc)
                    continue
                raise RuntimeError(f"Failed to download MetricX snapshot: repo={text}") from exc

        return text

    @staticmethod
    def _cleanup_stale_hf_partials(repo_id: str, cache_dir: str | None) -> None:
        if not cache_dir:
            return
        repo_cache_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        if not repo_cache_dir.exists():
            return
        for p in repo_cache_dir.glob("blobs/*.incomplete"):
            try:
                p.unlink()
            except Exception:
                pass
        for p in repo_cache_dir.glob(".locks/**/*"):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

    @staticmethod
    def _load_metricx_model(model_cls: Any, model_name: str, dtype: Any):
        kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if dtype is not None:
            kwargs["dtype"] = dtype
        try:
            return model_cls.from_pretrained(model_name, **kwargs)
        except TypeError:
            kwargs.pop("low_cpu_mem_usage", None)
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_cls.from_pretrained(model_name, **kwargs)

    def _load_tokenizer(self):
        candidates: list[str] = []
        if self.cfg.tokenizer_name and self.cfg.tokenizer_name.strip():
            candidates.append(self.cfg.tokenizer_name.strip())
        if self.cfg.model_name and self.cfg.model_name.strip():
            candidates.append(self.cfg.model_name.strip())
        candidates.extend(["google/mt5-xl", "google/mt5-large"])

        tried: list[str] = []
        for tok_name in candidates:
            if tok_name in tried:
                continue
            tried.append(tok_name)
            try:
                return AutoTokenizer.from_pretrained(tok_name, use_fast=False)
            except Exception as exc_slow:
                logger.warning("MetricX tokenizer load failed (%s, slow): %s", tok_name, exc_slow)
                try:
                    return AutoTokenizer.from_pretrained(tok_name, use_fast=True)
                except Exception as exc_fast:
                    logger.warning("MetricX tokenizer load failed (%s, fast): %s", tok_name, exc_fast)

        raise RuntimeError(
            "Failed to load tokenizer for MetricX. "
            f"tried={tried}. Set reward.metricx.tokenizer_name (recommended: google/mt5-xl)."
        )

    def score_batch(self, samples: list[SampleForScoring]) -> RewardOutput:
        if not samples:
            return RewardOutput(sequence_scores=[])

        inputs: list[str] = []
        for sample in samples:
            ref_text = (sample.ref or "").strip()
            if self.cfg.use_reference and ref_text:
                inputs.append(metricx_ref_input(sample.src, sample.mt, ref_text))
            else:
                inputs.append(metricx_qe_input(sample.src, sample.mt))
        if self.predict_fn is not None:
            scores = [float(v) for v in self.predict_fn(inputs)]
            return RewardOutput(sequence_scores=scores, metadata={"inputs": inputs})

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MetricXQEScorer is not initialized with a model.")

        all_scores: list[float] = []
        skipped_samples = 0
        for i in range(0, len(inputs), self.cfg.batch_size):
            batch = inputs[i : i + self.cfg.batch_size]

            if self.cfg.overflow_policy == "skip":
                raw_ids = self._tokenizer(batch, truncation=False, padding=False)["input_ids"]
                lengths = [len(ids) for ids in raw_ids]
                kept_batch = []
                filtered_idx = []
                for j, length in enumerate(lengths):
                    if int(length) <= self.cfg.max_input_length:
                        kept_batch.append(batch[j])
                        filtered_idx.append(j)
                batch_scores = [math.nan] * len(batch)
                if kept_batch:
                    pred_scores = self._predict_scores(kept_batch)
                    for pos, score in zip(filtered_idx, pred_scores):
                        batch_scores[pos] = score
                skipped_samples += sum(1 for score in batch_scores if math.isnan(score))
                all_scores.extend(batch_scores)
                continue

            all_scores.extend(self._predict_scores(batch))

        return RewardOutput(
            sequence_scores=all_scores,
            metadata={
                "inputs": inputs,
                "skipped_count": skipped_samples,
                "use_reference": bool(self.cfg.use_reference),
            },
        )

    def _predict_scores(self, batch_inputs: list[str]) -> list[float]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MetricXQEScorer is not initialized with a model.")

        tokenized = self._tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_input_length,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)

        for _ in range(max(1, len(self._candidate_dtypes) + 1)):
            with torch.no_grad():
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.predictions
            if torch.isfinite(preds).all():
                return [float(v) for v in preds.detach().float().cpu().tolist()]

            bad_count = int((~torch.isfinite(preds)).sum().item())
            logger.warning(
                "MetricX produced non-finite predictions (count=%s) model=%s dtype=%s device=%s.",
                bad_count,
                self.cfg.model_name,
                self._dtype,
                self._device,
            )
            if not self._reload_with_next_dtype():
                break

        raise RuntimeError(
            "MetricX produced non-finite predictions and recovery failed. "
            f"model={self.cfg.model_name} device={self._device} dtype={self._dtype}"
        )

    def _strip_final_eos(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._tokenizer is None:
            return input_ids, attention_mask
        eos_token_id = self._tokenizer.eos_token_id
        pad_token_id = self._tokenizer.pad_token_id or 0
        if eos_token_id is None:
            return input_ids, attention_mask

        ids = input_ids.clone()
        mask = attention_mask.clone()
        for row_idx in range(ids.size(0)):
            length = int(mask[row_idx].sum().item())
            if length <= 1:
                continue
            last_idx = length - 1
            if int(ids[row_idx, last_idx].item()) == int(eos_token_id):
                mask[row_idx, last_idx] = 0
                ids[row_idx, last_idx] = int(pad_token_id)
        return ids, mask


@dataclass
class XCometXLScorer:
    cfg: XCometConfig
    predict_fn: Callable[[list[dict[str, str]]], Any] | None = None

    def __post_init__(self) -> None:
        self._model = None
        self._device = resolve_device(self.cfg.device)
        if self.predict_fn is not None or not self.cfg.enabled:
            return

        try:
            from comet import download_model, load_from_checkpoint
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "XCometXLScorer requires unbabel-comet>=2.2.0. Install it before running RL."
            ) from exc

        model_path = download_model(self.cfg.model_name)
        self._model = load_from_checkpoint(model_path)
        if self._device.startswith("cuda") and torch is not None:
            self._model.to(torch.device(self._device))
        self._model.eval()

    def score_batch(self, samples: list[SampleForScoring]) -> RewardOutput:
        if not samples:
            return RewardOutput(sequence_scores=[], metadata={"error_spans": []})

        payload: list[dict[str, str]] = []
        for sample in samples:
            record = {"src": sample.src, "mt": sample.mt}
            if self.cfg.use_reference and sample.ref:
                record["ref"] = sample.ref
            payload.append(record)

        if self.predict_fn is not None:
            result = self.predict_fn(payload)
            scores, spans = self._parse_prediction(result, expected=len(samples))
            return RewardOutput(sequence_scores=scores, metadata={"error_spans": spans})

        if self._model is None:
            raise RuntimeError("XCometXLScorer is not initialized with a model.")

        scores, spans = self._predict_with_loaded_model(payload)
        return RewardOutput(sequence_scores=scores, metadata={"error_spans": spans})

    def _predict_with_loaded_model(self, payload: list[dict[str, str]]) -> tuple[list[float], list[list[dict[str, Any]]]]:
        if self._model is None:
            raise RuntimeError("XCometXLScorer is not initialized with a model.")
        if torch is None:
            raise RuntimeError("XCometXLScorer requires torch for model inference.")

        all_scores: list[float] = []
        all_spans: list[list[dict[str, Any]]] = []
        for i in range(0, len(payload), self.cfg.batch_size):
            batch_payload = payload[i : i + self.cfg.batch_size]
            batch_inputs = self._model.prepare_for_inference(batch_payload)
            batch_inputs = self._move_to_device(batch_inputs, self._device)
            with torch.no_grad():
                pred = self._model.predict_step(batch_inputs)

            batch_scores_tensor = pred.get("scores") if isinstance(pred, dict) else getattr(pred, "scores", None)
            if batch_scores_tensor is None:
                raise ValueError("xCOMET predict_step returned no scores.")
            batch_scores = torch.as_tensor(batch_scores_tensor).detach().float().cpu().tolist()
            batch_size = len(batch_scores)

            if isinstance(pred, dict):
                metadata = pred.get("metadata")
            else:
                metadata = getattr(pred, "metadata", None)
            batch_spans = extract_error_spans(metadata, expected=batch_size)

            all_scores.extend(float(v) for v in batch_scores)
            all_spans.extend(batch_spans)

        return all_scores, all_spans

    @staticmethod
    def _move_to_device(batch: Any, device: str) -> Any:
        if torch is None:
            return batch
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: XCometXLScorer._move_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, tuple):
            return tuple(XCometXLScorer._move_to_device(v, device) for v in batch)
        if isinstance(batch, list):
            return [XCometXLScorer._move_to_device(v, device) for v in batch]
        return batch

    @staticmethod
    def _parse_prediction(result: Any, expected: int) -> tuple[list[float], list[list[dict[str, Any]]]]:
        scores: list[float] | None = None
        metadata: Any = None

        if hasattr(result, "scores"):
            scores = [float(v) for v in list(result.scores)]
            metadata = getattr(result, "metadata", None)
        elif isinstance(result, dict):
            if "scores" in result:
                scores = [float(v) for v in list(result["scores"])]
            metadata = result.get("metadata", result)
        elif isinstance(result, tuple):
            if len(result) >= 1:
                scores = [float(v) for v in list(result[0])]
            if len(result) >= 2:
                metadata = result[1]

        if scores is None:
            raise ValueError("Unsupported xCOMET prediction output format.")

        spans = extract_error_spans(metadata, expected=expected)
        if len(scores) != expected:
            raise ValueError(f"xCOMET score length mismatch expected={expected} got={len(scores)}")
        return scores, spans


def extract_error_spans(metadata: Any, expected: int) -> list[list[dict[str, Any]]]:
    if metadata is None:
        return [[] for _ in range(expected)]

    def _normalize_one(item: Any) -> list[dict[str, Any]]:
        if item is None:
            return []
        if isinstance(item, dict):
            spans = item.get("error_spans")
            if isinstance(spans, list):
                return [span for span in spans if isinstance(span, dict)]
            return []
        if isinstance(item, list):
            return [span for span in item if isinstance(span, dict)]
        return []

    if isinstance(metadata, list):
        out = [_normalize_one(item) for item in metadata]
        if len(out) < expected:
            out.extend([[] for _ in range(expected - len(out))])
        return out[:expected]

    if isinstance(metadata, dict):
        spans = metadata.get("error_spans")
        if isinstance(spans, list):
            if spans and isinstance(spans[0], list):
                out = [_normalize_one(item) for item in spans]
            elif spans and isinstance(spans[0], dict):
                out = [_normalize_one(spans)] if expected == 1 else [[*spans]] + [[] for _ in range(expected - 1)]
            else:
                out = [[] for _ in range(expected)]
            if len(out) < expected:
                out.extend([[] for _ in range(expected - len(out))])
            return out[:expected]

        if "samples" in metadata and isinstance(metadata["samples"], list):
            out = [_normalize_one(item) for item in metadata["samples"]]
            if len(out) < expected:
                out.extend([[] for _ in range(expected - len(out))])
            return out[:expected]

    return [[] for _ in range(expected)]


def spans_to_token_rewards(
    mt_text: str,
    token_char_offsets: list[tuple[int, int]],
    error_spans: list[dict[str, Any]],
    severity_weights: dict[str, float],
    overlap_policy: str = "any_overlap",
    majority_threshold: float = 0.5,
    use_confidence: bool = False,
    combine_policy: str = "sum",
) -> list[float]:
    del mt_text  # Reserved for debug extension.

    if overlap_policy not in {"any_overlap", "majority_overlap"}:
        raise ValueError("overlap_policy must be any_overlap or majority_overlap")
    if combine_policy not in {"sum", "min", "max"}:
        raise ValueError("combine_policy must be sum|min|max")

    rewards = [0.0 for _ in token_char_offsets]
    initialized = [False for _ in token_char_offsets]

    for span in error_spans:
        try:
            start = int(span.get("start", 0))
            end = int(span.get("end", 0))
        except Exception:
            continue
        if end <= start:
            continue

        severity = str(span.get("severity", "")).strip().upper()
        penalty = float(severity_weights.get(severity, 0.0))
        if use_confidence and "confidence" in span:
            try:
                penalty *= float(span.get("confidence", 1.0))
            except Exception:
                pass
        if penalty == 0.0:
            continue

        for token_idx, (tok_s, tok_e) in enumerate(token_char_offsets):
            if tok_e <= tok_s:
                continue
            overlap = max(0, min(tok_e, end) - max(tok_s, start))
            if overlap <= 0:
                continue

            apply = False
            if overlap_policy == "any_overlap":
                apply = True
            else:
                ratio = overlap / max(1, tok_e - tok_s)
                apply = ratio >= majority_threshold
            if not apply:
                continue

            if combine_policy == "sum":
                rewards[token_idx] += penalty
            elif combine_policy == "min":
                rewards[token_idx] = penalty if not initialized[token_idx] else min(rewards[token_idx], penalty)
            elif combine_policy == "max":
                rewards[token_idx] = penalty if not initialized[token_idx] else max(rewards[token_idx], penalty)
            initialized[token_idx] = True

    return rewards
