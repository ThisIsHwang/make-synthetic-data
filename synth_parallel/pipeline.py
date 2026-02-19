from __future__ import annotations

import json
import heapq
import logging
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .bucketing import BalancedBucketSampler, assign_bucket
from .caches import StageProgressStore
from .config import PipelineConfig, dump_config, resolve_metricx_cache_path
from .filters import apply_llm_judge_filter, apply_rule_based_filters
from .io_utils import ensure_dir, read_jsonl, write_jsonl
from .metricx import MetricXScorer
from .prompts import build_translation_messages
from .segmentation import Segment, build_blobs_from_doc_segments, extract_segments_from_record
from .stats import StatsCollector
from .teacher import CompletionTask, TeacherClient
from .utils import chunks, stable_hash

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        cfg: PipelineConfig,
        dry_run: bool = False,
        resume: bool = False,
        overwrite: bool = False,
        limit: int | None = None,
        shard_id: int | None = None,
        num_shards: int | None = None,
    ):
        self.cfg = cfg
        self.dry_run = dry_run
        self.resume = resume
        self.overwrite = overwrite
        self.limit = limit
        self.shard_id = shard_id
        self.num_shards = num_shards

        if self.cfg.run.randomize_seed_each_run:
            runtime_seed = random.SystemRandom().randint(0, 2_147_483_647)
            logger.info(
                "Using randomized run seed=%s (run.randomize_seed_each_run=true).",
                runtime_seed,
            )
            self.cfg.run.seed = runtime_seed

        self.out_dir = ensure_dir(cfg.run.out_dir)
        dump_config(cfg, self.out_dir / "resolved_config.yaml")

        self.paths = {
            "sampled_sources": self.out_dir / "sampled_sources.jsonl",
            "prefilter": self.out_dir / "prefilter_candidates.jsonl",
            "selected": self.out_dir / "selected_sources.jsonl",
            "generated": self.out_dir / "generated_candidates.jsonl",
            "scored_best": self.out_dir / "scored_best.jsonl",
            "filtered": self.out_dir / "filtered.jsonl",
            "rejected": self.out_dir / "rejected.jsonl",
            "final": self.out_dir / "final_dataset.jsonl",
            "topk": self.out_dir / "final_candidates_topk.jsonl",
        }

        self.stats = StatsCollector(self.out_dir / "stats.json")
        self.progress = StageProgressStore(self.out_dir / "progress.sqlite")
        self._teacher: TeacherClient | None = None
        self._metricx: MetricXScorer | None = None
        self._rng = random.Random(cfg.run.seed)

    @property
    def teacher(self) -> TeacherClient:
        if self._teacher is None:
            self._teacher = TeacherClient(
                cfg=self.cfg.teacher,
                cache_db_path=str(self.out_dir / "teacher_cache.sqlite"),
                stats=self.stats,
                logger=logger,
            )
        return self._teacher

    @property
    def metricx(self) -> MetricXScorer:
        if self._metricx is None:
            metricx_cache = resolve_metricx_cache_path(self.cfg)
            self._metricx = MetricXScorer(
                cfg=self.cfg.metricx,
                cache_db_path=str(metricx_cache),
                stats=self.stats,
                logger=logger,
            )
        return self._metricx

    def close(self) -> None:
        if self._teacher is not None:
            self._teacher.close()
        if self._metricx is not None:
            self._metricx.close()
        self.progress.close()
        self.stats.flush()

    def run(self, stage: str) -> None:
        if stage == "all":
            for step in [
                "sample_sources",
                "prefilter_score",
                "select_sources",
                "generate_128",
                "score_select_best",
                "format_filter",
                "export",
            ]:
                self._run_stage(step)
            return
        self._run_stage(stage)

    def _run_stage(self, stage: str) -> None:
        logger.info("========== Stage start: %s =========", stage)
        self.stats.stage_start(stage)
        try:
            getattr(self, f"stage_{stage}")()
        finally:
            self.stats.stage_end(stage)
            self.stats.flush()
            logger.info("========== Stage end: %s =========", stage)

    def _in_shard(self, key: str) -> bool:
        if self.shard_id is None or self.num_shards is None:
            return True
        if self.num_shards <= 0:
            return True
        shard_key = int(stable_hash({"key": key})[:12], 16)
        return (shard_key % self.num_shards) == self.shard_id

    def _effective_limit(self, default: int | None = None) -> int | None:
        if self.limit is not None:
            return self.limit
        if self.dry_run:
            return default or 1000
        return default

    def _maybe_reset_file(self, path: Path) -> None:
        if path.exists() and self.overwrite:
            path.unlink()

    def _sampling_temperature(self, default_temperature: float) -> float:
        override = self.cfg.teacher.generation.sampling_temperature
        if override is None:
            return default_temperature
        return float(override)

    def _sampling_top_p(self) -> float:
        override = self.cfg.teacher.generation.sampling_top_p
        if override is None:
            return float(self.cfg.teacher.generation.top_p)
        return float(override)

    def _sampling_presence_penalty(self) -> float | None:
        return self.cfg.teacher.generation.sampling_presence_penalty

    def _sampling_extra_body(self) -> dict[str, Any]:
        generation_cfg = self.cfg.teacher.generation
        payload: dict[str, Any] = {}
        if generation_cfg.sampling_top_k is not None:
            payload["top_k"] = int(generation_cfg.sampling_top_k)
        if generation_cfg.sampling_min_p is not None:
            payload["min_p"] = float(generation_cfg.sampling_min_p)
        if generation_cfg.sampling_repetition_penalty is not None:
            payload["repetition_penalty"] = float(generation_cfg.sampling_repetition_penalty)
        return payload

    def stage_sample_sources(self) -> None:
        out_path = self.paths["sampled_sources"]
        self._maybe_reset_file(out_path)

        if out_path.exists() and self.resume and not self.overwrite:
            logger.info("sample_sources output already exists, skip due to --resume: %s", out_path)
            return

        # In full runs, cap scan by data.max_scan_docs to avoid scanning the whole corpus.
        # In dry-run, _effective_limit() returns a small default (1000).
        limit = self._effective_limit()
        scan_doc_cap = limit if limit is not None else self.cfg.data.max_scan_docs
        pool_size = min(self.cfg.data.sample_pool_size, limit) if limit else self.cfg.data.sample_pool_size

        sentence_quota = int(pool_size * self.cfg.data.sentence_ratio)
        blob_quota = pool_size - sentence_quota

        if not self.cfg.final_generation.blob.enabled:
            sentence_quota = pool_size
            blob_quota = 0

        sentence_sampler = BalancedBucketSampler(
            boundaries=self.cfg.bucketing.boundaries,
            sample_size=max(sentence_quota, 0),
            oversample_factor=self.cfg.bucketing.bucket_oversample_factor,
            seed=self.cfg.run.seed,
        )
        blob_sampler = (
            BalancedBucketSampler(
                boundaries=self.cfg.bucketing.boundaries,
                sample_size=max(blob_quota, 0),
                oversample_factor=self.cfg.bucketing.bucket_oversample_factor,
                seed=self.cfg.run.seed + 1,
            )
            if blob_quota > 0
            else None
        )

        logger.info(
            "Loading dataset=%s lang=%s split=%s streaming=%s",
            self.cfg.data.madlad_dataset,
            self.cfg.data.src_lang,
            self.cfg.data.madlad_split,
            self.cfg.data.streaming,
        )
        if self.cfg.data.local_data_glob:
            logger.info("Using local MADLAD files from glob=%s", self.cfg.data.local_data_glob)
            dataset = load_dataset(
                "json",
                data_files=self.cfg.data.local_data_glob,
                split="train",
                streaming=self.cfg.data.streaming,
            )
        else:
            token = os.getenv(self.cfg.data.hf_token_env, "").strip()
            load_kwargs: dict[str, Any] = {
                "split": self.cfg.data.madlad_split,
                "streaming": self.cfg.data.streaming,
                "trust_remote_code": self.cfg.data.trust_remote_code,
            }
            if token:
                load_kwargs["token"] = token
            if self.cfg.data.madlad_revision:
                load_kwargs["revision"] = self.cfg.data.madlad_revision

            try:
                dataset = load_dataset(
                    self.cfg.data.madlad_dataset,
                    self.cfg.data.src_lang,
                    **load_kwargs,
                )
            except Exception as exc:  # pylint: disable=broad-except
                msg = str(exc).lower()
                if "429" in msg or "rate limit" in msg:
                    raise RuntimeError(
                        "Failed to access MADLAD due to Hugging Face rate limit. "
                        f"Set {self.cfg.data.hf_token_env} and retry."
                    ) from exc
                if "dataset scripts are no longer supported" in msg:
                    raise RuntimeError(
                        "Your datasets package is too new for MADLAD dataset script loading. "
                        "Use datasets<3 (e.g. 2.21.x) in your runtime environment."
                    ) from exc
                if "datafilesnotfounderror" in type(exc).__name__.lower() or "no (supported) data files found" in msg:
                    raise RuntimeError(
                        "MADLAD files could not be resolved. "
                        "Check data.madlad_dataset='allenai/MADLAD-400', "
                        "data.src_lang (e.g. en/ko), and ensure HF token is configured. "
                        "If this persists, predownload shards and set data.local_data_glob. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc
                raise

        processed_docs = 0
        processed_segments = 0

        for doc_index, record in enumerate(dataset):
            doc_key = str(record.get("id") or record.get("doc_id") or doc_index)
            if not self._in_shard(doc_key):
                continue

            segments = extract_segments_from_record(
                record=record,
                doc_index=doc_index,
                cfg=self.cfg.segmentation,
                text_field=self.cfg.data.text_field,
            )
            processed_docs += 1
            processed_segments += len(segments)

            for seg in segments:
                row = self._segment_to_source_row(seg)
                sentence_sampler.add(row, row["length_approx"])

            if blob_sampler is not None:
                blobs = build_blobs_from_doc_segments(
                    segments=segments,
                    blob_max_tokens=self.cfg.final_generation.blob.blob_max_tokens,
                )
                for blob in blobs:
                    row = self._segment_to_source_row(blob)
                    blob_sampler.add(row, row["length_approx"])

            if processed_docs % 500 == 0:
                sentence_buffered = sentence_sampler.buffered_count()
                blob_buffered = blob_sampler.buffered_count() if blob_sampler is not None else 0
                logger.info(
                    "sample_sources progress docs=%s segments=%s sentence_buffered=%s/%s blob_buffered=%s/%s",
                    processed_docs,
                    processed_segments,
                    sentence_buffered,
                    sentence_quota,
                    blob_buffered,
                    blob_quota,
                )

            if (
                self.cfg.data.stop_when_pool_ready
                and sentence_sampler.can_fill_target()
                and (blob_sampler is None or blob_sampler.can_fill_target())
            ):
                logger.info(
                    "sample_sources early stop: pool ready docs=%s segments=%s sentence_buffered=%s/%s blob_buffered=%s/%s",
                    processed_docs,
                    processed_segments,
                    sentence_sampler.buffered_count(),
                    sentence_quota,
                    (blob_sampler.buffered_count() if blob_sampler is not None else 0),
                    blob_quota,
                )
                break

            if scan_doc_cap is not None and processed_docs >= scan_doc_cap:
                logger.warning(
                    "sample_sources stopped by max scan cap docs=%s (data.max_scan_docs or --limit).",
                    scan_doc_cap,
                )
                break

        sampled_sentences, sentence_stats = sentence_sampler.finalize()
        sampled_blobs: list[dict[str, Any]] = []
        blob_stats: dict[int, Any] = {}
        if blob_sampler is not None:
            sampled_blobs, blob_stats = blob_sampler.finalize()

        sampled = sampled_sentences + sampled_blobs
        self._rng.shuffle(sampled)
        if len(sampled) > pool_size:
            sampled = self._rng.sample(sampled, k=pool_size)
        if len(sampled) < pool_size:
            logger.warning(
                "sample_sources produced fewer rows than requested sampled=%s requested=%s. "
                "Increase data.max_scan_docs or relax segmentation/bucketing constraints.",
                len(sampled),
                pool_size,
            )

        logger.info(
            "sample_sources collection done docs=%s segments=%s sampled=%s (sentence=%s blob=%s). Writing output...",
            processed_docs,
            processed_segments,
            len(sampled),
            len(sampled_sentences),
            len(sampled_blobs),
        )

        write_start = time.perf_counter()
        written = write_jsonl(out_path, sampled, append=False)
        logger.info(
            "sample_sources write done rows=%s path=%s elapsed=%.2fs",
            written,
            out_path,
            time.perf_counter() - write_start,
        )

        for bid, st in sentence_stats.items():
            self.stats.inc(f"sample.bucket_sentence.{bid}.seen", st.seen)
            self.stats.inc(f"sample.bucket_sentence.{bid}.kept", st.kept)
        for bid, st in blob_stats.items():
            self.stats.inc(f"sample.bucket_blob.{bid}.seen", st.seen)
            self.stats.inc(f"sample.bucket_blob.{bid}.kept", st.kept)
        self.stats.inc("sample.docs", processed_docs)
        self.stats.inc("sample.segments", processed_segments)
        self.stats.inc("sample.output", written)

    def _segment_to_source_row(self, seg: Segment) -> dict[str, Any]:
        bucket_id = assign_bucket(seg.length_approx, self.cfg.bucketing.boundaries)
        return {
            "pair_id": f"{self.cfg.data.src_lang}->{self.cfg.data.tgt_lang}",
            "source_lang_code": self.cfg.data.src_lang,
            "target_lang_code": self.cfg.data.tgt_lang,
            "source_id": seg.source_id,
            "source_text": seg.source_text,
            "kind": seg.kind,
            "length_approx": seg.length_approx,
            "length_bucket_id": bucket_id,
            "madlad": {
                "lang": self.cfg.data.src_lang,
                "split": self.cfg.data.madlad_split,
                **seg.meta,
            },
        }

    def stage_prefilter_score(self) -> None:
        in_path = self.paths["sampled_sources"]
        out_path = self.paths["prefilter"]
        self._maybe_reset_file(out_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        stage_name = "prefilter_score"
        max_batch = max(1, self.cfg.teacher.max_concurrency)
        rows_processed = 0

        chunk_rows: list[dict[str, Any]] = []
        for row in read_jsonl(in_path):
            source_id = row["source_id"]
            if self.resume and out_path.exists() and self.progress.has(stage_name, source_id):
                continue
            if not self._in_shard(source_id):
                continue
            chunk_rows.append(row)
            if len(chunk_rows) >= max_batch:
                rows_processed += self._process_prefilter_chunk(chunk_rows, out_path, stage_name)
                chunk_rows = []

        if chunk_rows:
            rows_processed += self._process_prefilter_chunk(chunk_rows, out_path, stage_name)

        logger.info("prefilter_score done rows=%s", rows_processed)
        self.stats.inc("prefilter.output", rows_processed)

    def _process_prefilter_chunk(
        self,
        chunk_rows: list[dict[str, Any]],
        out_path: Path,
        stage_name: str,
    ) -> int:
        if not chunk_rows:
            return 0

        t0 = time.perf_counter()
        sample_temperature = self._sampling_temperature(self.cfg.teacher.generation.sample_temperature)
        sample_top_p = self._sampling_top_p()
        sample_presence_penalty = self._sampling_presence_penalty()
        sample_extra_body = self._sampling_extra_body()
        tasks: list[CompletionTask] = []
        for row in chunk_rows:
            messages = build_translation_messages(
                source_lang=self.cfg.data.src_lang_name,
                target_lang=self.cfg.data.tgt_lang_name,
                src_lang_code=self.cfg.data.src_lang,
                tgt_lang_code=self.cfg.data.tgt_lang,
                text=row["source_text"],
            )
            tasks.append(
                CompletionTask(
                    item_id=f"{row['source_id']}::greedy",
                    messages=messages,
                    temperature=self.cfg.teacher.generation.greedy_temperature,
                    top_p=self.cfg.teacher.generation.top_p,
                    max_tokens=self.cfg.teacher.generation.max_tokens,
                )
            )
            tasks.append(
                CompletionTask(
                    item_id=f"{row['source_id']}::sample",
                    messages=messages,
                    temperature=sample_temperature,
                    top_p=sample_top_p,
                    max_tokens=self.cfg.teacher.generation.max_tokens,
                    presence_penalty=sample_presence_penalty,
                    extra_body=sample_extra_body,
                )
            )

        logger.info(
            "prefilter chunk start rows=%s teacher_tasks=%s (concurrency=%s)",
            len(chunk_rows),
            len(tasks),
            self.cfg.teacher.max_concurrency,
        )
        row_map = {row["source_id"]: row for row in chunk_rows}
        scores_by_source: dict[str, dict[str, float]] = {}
        by_source: dict[str, dict[str, str]] = {}
        ready_sources: list[str] = []
        dispatched_sources: set[str] = set()
        source_batch_for_metric = max(
            1,
            min(len(chunk_rows), max(8, self.cfg.teacher.max_concurrency // 2)),
        )

        def flush_metric_ready(force: bool = False) -> None:
            nonlocal ready_sources
            if not ready_sources:
                return
            if not force and len(ready_sources) < source_batch_for_metric:
                return

            batch_source_ids = ready_sources[:source_batch_for_metric]
            ready_sources = ready_sources[source_batch_for_metric:]
            pairs: list[tuple[str, str]] = []
            score_index: list[tuple[str, str]] = []
            for source_id in batch_source_ids:
                texts = by_source[source_id]
                source_text = row_map[source_id]["source_text"]
                pairs.append((source_text, texts.get("greedy", "")))
                score_index.append((source_id, "greedy"))
                pairs.append((source_text, texts.get("sample", "")))
                score_index.append((source_id, "sample"))

            score_start = time.perf_counter()
            logger.info(
                "prefilter chunk metricx start pairs=%s ready_sources=%s",
                len(pairs),
                len(batch_source_ids),
            )
            scores = self.metricx.score_batch(pairs)
            logger.info(
                "prefilter chunk metricx done pairs=%s elapsed=%.2fs",
                len(pairs),
                time.perf_counter() - score_start,
            )
            for (source_id, mode), score in zip(score_index, scores):
                scores_by_source.setdefault(source_id, {})[mode] = float(score)

        workers = max(1, self.cfg.teacher.max_concurrency)
        future_map: dict[Future[str], str] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for task in tasks:
                future = executor.submit(
                    self.teacher.complete,
                    messages=task.messages,
                    temperature=task.temperature,
                    top_p=task.top_p,
                    max_tokens=task.max_tokens,
                    model=task.model,
                    presence_penalty=task.presence_penalty,
                    extra_body=task.extra_body,
                )
                future_map[future] = task.item_id

            for future in as_completed(future_map):
                item_id = future_map[future]
                text = future.result()
                source_id, mode = item_id.split("::", maxsplit=1)
                by_source.setdefault(source_id, {})[mode] = text
                source_modes = by_source[source_id]
                if "greedy" in source_modes and "sample" in source_modes and source_id not in dispatched_sources:
                    ready_sources.append(source_id)
                    dispatched_sources.add(source_id)
                flush_metric_ready(force=False)

        while ready_sources:
            flush_metric_ready(force=True)

        t_teacher = time.perf_counter()
        logger.info(
            "prefilter chunk teacher+metricx overlapped done rows=%s tasks=%s elapsed=%.2fs",
            len(chunk_rows),
            len(tasks),
            t_teacher - t0,
        )

        written = 0
        processed_ids: list[str] = []
        out_rows: list[dict[str, Any]] = []
        for source_id, row in row_map.items():
            greedy_text = by_source.get(source_id, {}).get("greedy", "")
            sample_text = by_source.get(source_id, {}).get("sample", "")
            score_g = float(scores_by_source.get(source_id, {}).get("greedy", 25.0))
            score_s = float(scores_by_source.get(source_id, {}).get("sample", 25.0))
            improvement = score_g - score_s

            output = {
                **row,
                "prefilter": {
                    "greedy_text": greedy_text,
                    "sample_text": sample_text,
                    "score_greedy": score_g,
                    "score_sample": score_s,
                    "improvement": improvement,
                },
                "teacher": {
                    "backend": self.cfg.teacher.backend,
                    "base_url": self.cfg.teacher.base_url,
                    "model": self.cfg.teacher.model,
                    "sampling_params": {
                        "prefilter_greedy": {
                            "temperature": self.cfg.teacher.generation.greedy_temperature,
                            "top_p": self.cfg.teacher.generation.top_p,
                            "max_tokens": self.cfg.teacher.generation.max_tokens,
                        },
                        "prefilter_sample": {
                            "temperature": sample_temperature,
                            "top_p": sample_top_p,
                            "max_tokens": self.cfg.teacher.generation.max_tokens,
                            "presence_penalty": sample_presence_penalty,
                            "extra_body": sample_extra_body,
                        },
                    },
                },
                "metricx": {
                    "checkpoint": self.cfg.metricx.checkpoint,
                    "backend": self.cfg.metricx.backend,
                },
            }
            out_rows.append(output)
            processed_ids.append(source_id)
            written += 1

        write_jsonl(out_path, out_rows, append=True)
        self.progress.add_many(stage_name, processed_ids)
        logger.info(
            "prefilter chunk done size=%s written=%s progress=%s total_elapsed=%.2fs",
            len(chunk_rows),
            written,
            self.progress.count_stage(stage_name),
            time.perf_counter() - t0,
        )
        return written

    def stage_select_sources(self) -> None:
        in_path = self.paths["prefilter"]
        out_path = self.paths["selected"]
        self._maybe_reset_file(out_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        if out_path.exists() and self.resume and not self.overwrite:
            logger.info("select_sources output already exists, skip due to --resume")
            return

        target_n = self.cfg.data.target_examples_total
        if self.limit is not None:
            target_n = min(target_n, self.limit)

        heap: list[tuple[float, int, dict[str, Any]]] = []
        seen = 0

        for row in read_jsonl(in_path):
            source_id = row["source_id"]
            if not self._in_shard(source_id):
                continue
            seen += 1
            imp = float(row["prefilter"]["improvement"])
            payload = (imp, seen, row)
            if len(heap) < target_n:
                heapq.heappush(heap, payload)
            elif imp > heap[0][0]:
                heapq.heapreplace(heap, payload)

            if seen % 5000 == 0:
                logger.info("select_sources progress seen=%s heap=%s", seen, len(heap))

        selected = [heapq.heappop(heap)[2] for _ in range(len(heap))]
        selected.reverse()

        output_rows: list[dict[str, Any]] = []
        for rank, row in enumerate(selected, start=1):
            row["selection"] = {
                "selected": True,
                "selection_rank": rank,
                "selection_score": row["prefilter"]["improvement"],
                "selection_stage": "prefilter_improvement",
            }
            output_rows.append(row)
        write_jsonl(out_path, output_rows, append=False)

        logger.info("select_sources done seen=%s selected=%s", seen, len(selected))
        self.stats.inc("select.seen", seen)
        self.stats.inc("select.selected", len(selected))

    def stage_generate_128(self) -> None:
        in_path = self.paths["selected"]
        out_path = self.paths["generated"]
        self._maybe_reset_file(out_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        stage_name = "generate_128"
        total_written = 0
        final_temperature = self._sampling_temperature(self.cfg.teacher.generation.final_temperature)
        final_top_p = self._sampling_top_p()
        final_presence_penalty = self._sampling_presence_penalty()
        final_extra_body = self._sampling_extra_body()

        for source_idx, row in enumerate(read_jsonl(in_path)):
            source_id = row["source_id"]
            if not self._in_shard(source_id):
                continue

            messages = build_translation_messages(
                source_lang=self.cfg.data.src_lang_name,
                target_lang=self.cfg.data.tgt_lang_name,
                src_lang_code=self.cfg.data.src_lang,
                tgt_lang_code=self.cfg.data.tgt_lang,
                text=row["source_text"],
            )

            missing_indices: list[int] = []
            for candidate_idx in range(self.cfg.final_generation.num_candidates):
                item_id = f"{source_id}::{candidate_idx}"
                if self.resume and out_path.exists() and self.progress.has(stage_name, item_id):
                    continue
                missing_indices.append(candidate_idx)

            if not missing_indices:
                continue

            for candidate_chunk in chunks(missing_indices, max(1, self.cfg.teacher.max_concurrency)):
                tasks = [
                    CompletionTask(
                        item_id=f"{source_id}::{candidate_idx}",
                        messages=messages,
                        temperature=final_temperature,
                        top_p=final_top_p,
                        max_tokens=self.cfg.teacher.generation.max_tokens,
                        presence_penalty=final_presence_penalty,
                        extra_body=final_extra_body,
                    )
                    for candidate_idx in candidate_chunk
                ]

                results = self.teacher.run_tasks(
                    tasks=tasks,
                    worker_fn=lambda task, text: {"item_id": task.item_id, "text": text},
                )

                processed_ids: list[str] = []
                out_rows: list[dict[str, Any]] = []
                for result in results:
                    item_id = result["item_id"]
                    candidate_idx = int(item_id.split("::", maxsplit=1)[1])
                    out_row = {
                        "source_id": source_id,
                        "source_text": row["source_text"],
                        "candidate_index": candidate_idx,
                        "candidate_text": result["text"],
                        "pair_id": row["pair_id"],
                        "source_lang_code": row["source_lang_code"],
                        "target_lang_code": row["target_lang_code"],
                        "kind": row.get("kind", "sentence"),
                        "selection": row.get("selection", {}),
                        "madlad": row.get("madlad", {}),
                        "teacher": {
                            "backend": self.cfg.teacher.backend,
                            "base_url": self.cfg.teacher.base_url,
                            "model": self.cfg.teacher.model,
                            "sampling_params": {
                                "final_128": {
                                    "temperature": final_temperature,
                                    "top_p": final_top_p,
                                    "max_tokens": self.cfg.teacher.generation.max_tokens,
                                    "presence_penalty": final_presence_penalty,
                                    "extra_body": final_extra_body,
                                }
                            },
                        },
                    }
                    out_rows.append(out_row)
                    processed_ids.append(item_id)
                    total_written += 1

                write_jsonl(out_path, out_rows, append=True)
                self.progress.add_many(stage_name, processed_ids)

            if source_idx % 50 == 0:
                logger.info(
                    "generate_128 progress source_idx=%s total_written=%s",
                    source_idx,
                    total_written,
                )

        logger.info("generate_128 done total_written=%s", total_written)
        self.stats.inc("generate.output", total_written)

    def stage_score_select_best(self) -> None:
        in_path = self.paths["generated"]
        out_path = self.paths["scored_best"]
        topk_path = self.paths["topk"]
        self._maybe_reset_file(out_path)
        self._maybe_reset_file(topk_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        stage_name = "score_select_best"
        source_count = 0
        flush_every = 200
        out_rows_buffer: list[dict[str, Any]] = []
        topk_rows_buffer: list[dict[str, Any]] = []
        progress_ids: list[str] = []

        for source_id, group in self._iter_candidate_groups(in_path):
            if self.resume and out_path.exists() and self.progress.has(stage_name, source_id):
                continue
            if not self._in_shard(source_id):
                continue

            pairs = [(row["source_text"], row["candidate_text"]) for row in group]
            scores = self.metricx.score_batch(pairs)

            ranked = []
            for row, score in zip(group, scores):
                ranked.append({**row, "metricx_qe_score": float(score)})
            ranked.sort(key=lambda x: x["metricx_qe_score"])  # lower is better

            best = ranked[0]
            top_k = ranked[: max(1, self.cfg.final_generation.store_top_k)]

            out_row = {
                "pair_id": best["pair_id"],
                "source_lang_code": best["source_lang_code"],
                "target_lang_code": best["target_lang_code"],
                "source_id": source_id,
                "source_text": best["source_text"],
                "target_text": best["candidate_text"],
                "metricx_qe_score_best": best["metricx_qe_score"],
                "selection": best.get("selection", {}),
                "madlad": best.get("madlad", {}),
                "teacher": best.get("teacher", {}),
                "metricx": {
                    "checkpoint": self.cfg.metricx.checkpoint,
                    "backend": self.cfg.metricx.backend,
                },
            }

            out_rows_buffer.append(out_row)
            for row in top_k:
                topk_rows_buffer.append(
                    {
                        "source_id": source_id,
                        "candidate_index": row["candidate_index"],
                        "candidate_text": row["candidate_text"],
                        "metricx_qe_score": row["metricx_qe_score"],
                    },
                )

            progress_ids.append(source_id)
            source_count += 1

            if source_count % flush_every == 0:
                write_jsonl(out_path, out_rows_buffer, append=True)
                out_rows_buffer = []
                write_jsonl(topk_path, topk_rows_buffer, append=True)
                topk_rows_buffer = []
                self.progress.add_many(stage_name, progress_ids)
                progress_ids = []

            if source_count % 100 == 0:
                logger.info("score_select_best progress sources=%s", source_count)

        write_jsonl(out_path, out_rows_buffer, append=True)
        write_jsonl(topk_path, topk_rows_buffer, append=True)
        self.progress.add_many(stage_name, progress_ids)
        logger.info("score_select_best done sources=%s", source_count)
        self.stats.inc("score_select_best.sources", source_count)

    def stage_score_128_select_best(self) -> None:
        self.stage_score_select_best()

    def _iter_candidate_groups(self, path: Path):
        current_source = None
        current_rows: list[dict[str, Any]] = []
        for row in read_jsonl(path):
            source_id = row["source_id"]
            if current_source is None:
                current_source = source_id
            if source_id != current_source:
                if current_rows:
                    yield current_source, current_rows
                current_source = source_id
                current_rows = []
            current_rows.append(row)
        if current_source is not None and current_rows:
            yield current_source, current_rows

    def stage_format_filter(self) -> None:
        in_path = self.paths["scored_best"]
        out_path = self.paths["filtered"]
        reject_path = self.paths["rejected"]
        self._maybe_reset_file(out_path)
        self._maybe_reset_file(reject_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        stage_name = "format_filter"
        passed = 0
        rejected = 0
        flush_every = 200
        pass_buffer: list[dict[str, Any]] = []
        reject_buffer: list[dict[str, Any]] = []
        progress_ids: list[str] = []

        for row in read_jsonl(in_path):
            source_id = row["source_id"]
            if self.resume and out_path.exists() and self.progress.has(stage_name, source_id):
                continue
            if not self._in_shard(source_id):
                continue

            decision = apply_rule_based_filters(row["source_text"], row["target_text"], self.cfg.filters)
            if not decision.passed:
                rejected += 1
                reject_buffer.append(
                    {
                        **row,
                        "filters": {
                            "passed": False,
                            "reason_code": decision.reason_code,
                            "notes": decision.notes,
                            "type": "rule_based",
                        },
                    }
                )
                self.stats.inc(f"filter.reject.{decision.reason_code}")
                progress_ids.append(source_id)
                if len(progress_ids) >= flush_every:
                    write_jsonl(out_path, pass_buffer, append=True)
                    pass_buffer = []
                    write_jsonl(reject_path, reject_buffer, append=True)
                    reject_buffer = []
                    self.progress.add_many(stage_name, progress_ids)
                    progress_ids = []
                continue

            if self.cfg.filters.llm_judge.enabled:
                judge_decision = apply_llm_judge_filter(
                    teacher=self.teacher,
                    source_lang=self.cfg.data.src_lang_name,
                    target_lang=self.cfg.data.tgt_lang_name,
                    source_text=row["source_text"],
                    target_text=row["target_text"],
                    cfg=self.cfg.filters,
                )
                if not judge_decision.passed:
                    rejected += 1
                    reject_buffer.append(
                        {
                            **row,
                            "filters": {
                                "passed": False,
                                "reason_code": judge_decision.reason_code,
                                "notes": judge_decision.notes,
                                "type": "llm_judge",
                            },
                        }
                    )
                    self.stats.inc(f"filter.reject.{judge_decision.reason_code}")
                    progress_ids.append(source_id)
                    if len(progress_ids) >= flush_every:
                        write_jsonl(out_path, pass_buffer, append=True)
                        pass_buffer = []
                        write_jsonl(reject_path, reject_buffer, append=True)
                        reject_buffer = []
                        self.progress.add_many(stage_name, progress_ids)
                        progress_ids = []
                    continue

            passed += 1
            out_row = {
                **row,
                "filters": {
                    "passed": True,
                    "reason_code": "pass",
                    "notes": "",
                    "type": "rule_based+llm_judge" if self.cfg.filters.llm_judge.enabled else "rule_based",
                },
            }
            pass_buffer.append(out_row)
            self.stats.inc("filter.pass")
            progress_ids.append(source_id)

            if len(progress_ids) >= flush_every:
                write_jsonl(out_path, pass_buffer, append=True)
                pass_buffer = []
                write_jsonl(reject_path, reject_buffer, append=True)
                reject_buffer = []
                self.progress.add_many(stage_name, progress_ids)
                progress_ids = []

            if (passed + rejected) % 200 == 0:
                logger.info("format_filter progress processed=%s pass=%s reject=%s", passed + rejected, passed, rejected)

        write_jsonl(out_path, pass_buffer, append=True)
        write_jsonl(reject_path, reject_buffer, append=True)
        self.progress.add_many(stage_name, progress_ids)
        logger.info("format_filter done pass=%s reject=%s", passed, rejected)
        self.stats.inc("filter.passed", passed)
        self.stats.inc("filter.rejected", rejected)

    def stage_export(self) -> None:
        in_path = self.paths["filtered"]
        out_path = self.paths["final"]
        self._maybe_reset_file(out_path)

        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        if out_path.exists() and self.resume and not self.overwrite:
            logger.info("export output already exists, skip due to --resume")
            return

        total = 0
        len_source = []
        len_target = []
        out_buffer: list[dict[str, Any]] = []
        flush_every = 1000

        for row in read_jsonl(in_path):
            output = {
                "pair_id": row["pair_id"],
                "source_lang_code": row["source_lang_code"],
                "target_lang_code": row["target_lang_code"],
                "source_text": row["source_text"],
                "target_text": row["target_text"],
                "metricx_qe_score_best": row["metricx_qe_score_best"],
                "selection": row.get("selection", {}),
                "madlad": row.get("madlad", {}),
                "teacher": row.get("teacher", {}),
                "metricx": row.get("metricx", {}),
                "filters": row.get("filters", {}),
            }
            out_buffer.append(output)
            total += 1
            len_source.append(len(row["source_text"]))
            len_target.append(len(row["target_text"]))

            if len(out_buffer) >= flush_every:
                write_jsonl(out_path, out_buffer, append=True)
                out_buffer = []
            if total % 1000 == 0:
                logger.info("export progress total=%s", total)

        write_jsonl(out_path, out_buffer, append=True)
        logger.info("export done total=%s", total)
        self.stats.inc("export.total", total)
        if len_source:
            self.stats.inc("length.source.min", int(min(len_source)))
            self.stats.inc("length.source.max", int(max(len_source)))
            self.stats.inc("length.source.avg_x100", int(sum(len_source) * 100 / len(len_source)))
        if len_target:
            self.stats.inc("length.target.min", int(min(len_target)))
            self.stats.inc("length.target.max", int(max(len_target)))
            self.stats.inc("length.target.avg_x100", int(sum(len_target) * 100 / len(len_target)))

        manifest = {
            "paths": {k: str(v) for k, v in self.paths.items()},
            "config": asdict(self.cfg),
            "exported_rows": total,
        }
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
