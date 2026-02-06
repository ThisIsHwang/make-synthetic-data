from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import SegmentationConfig
from .utils import approx_token_length

_URL_ONLY_RE = re.compile(r"^(https?://\S+|www\.\S+)$", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CODE_HINT_RE = re.compile(r"[{};]{2,}|</?\w+>|\b(class|def|function|import|return)\b")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Segment:
    source_id: str
    source_text: str
    kind: str  # sentence | blob
    length_approx: int
    meta: dict[str, Any]


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if _URL_ONLY_RE.match(stripped):
        return True
    if len(_HTML_TAG_RE.findall(stripped)) >= 3 and len(stripped) < 200:
        return True
    if _CODE_HINT_RE.search(stripped) and len(stripped) < 300:
        return True
    return False


def _split_long_line(line: str, punctuation_regex: str) -> list[str]:
    pieces = re.split(punctuation_regex, line)
    out: list[str] = []
    for piece in pieces:
        clean = normalize_text(piece)
        if clean:
            out.append(clean)
    return out


def _merge_short_lines(lines: list[str], threshold: int) -> list[str]:
    if not lines:
        return []
    merged: list[str] = []
    buffer = ""
    for line in lines:
        if len(line) < threshold:
            buffer = f"{buffer} {line}".strip()
            continue
        if buffer:
            line = f"{buffer} {line}".strip()
            buffer = ""
        merged.append(line)
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)
    return merged


def extract_segments_from_record(
    record: dict[str, Any],
    doc_index: int,
    cfg: SegmentationConfig,
    text_field: str = "text",
) -> list[Segment]:
    doc_id = str(
        record.get("id")
        or record.get("doc_id")
        or record.get("document_id")
        or f"doc_{doc_index}"
    )
    raw = record.get(text_field)

    lines: list[str]
    if isinstance(raw, list):
        lines = [normalize_text(str(x)) for x in raw if normalize_text(str(x))]
    elif isinstance(raw, str):
        parts = raw.splitlines() if cfg.mode in {"auto", "newline"} else [raw]
        lines = []
        for part in parts:
            part = normalize_text(part)
            if not part:
                continue
            if len(part) > cfg.max_chars:
                lines.extend(_split_long_line(part, cfg.split_punctuation_regex))
            else:
                lines.append(part)
    else:
        return []

    if cfg.merge_short_lines:
        lines = _merge_short_lines(lines, cfg.short_line_threshold)

    segments: list[Segment] = []
    for idx, line in enumerate(lines):
        if len(line) < cfg.min_chars or len(line) > cfg.max_chars:
            continue
        if cfg.drop_noise and is_noise(line):
            continue
        source_id = f"{doc_id}:s:{idx}"
        segments.append(
            Segment(
                source_id=source_id,
                source_text=line,
                kind="sentence",
                length_approx=approx_token_length(line),
                meta={
                    "doc_id": doc_id,
                    "segment_index": idx,
                    "char_len": len(line),
                },
            )
        )
    return segments


def build_blobs_from_doc_segments(
    segments: list[Segment],
    blob_max_tokens: int,
) -> list[Segment]:
    if not segments:
        return []

    doc_id = str(segments[0].meta.get("doc_id", "doc"))
    blobs: list[Segment] = []
    current_parts: list[str] = []
    current_indices: list[int] = []
    current_len = 0

    for segment in segments:
        seg_len = segment.length_approx
        if current_parts and current_len + seg_len > blob_max_tokens:
            blob_text = "\n".join(current_parts)
            blob_idx = len(blobs)
            blobs.append(
                Segment(
                    source_id=f"{doc_id}:b:{blob_idx}",
                    source_text=blob_text,
                    kind="blob",
                    length_approx=approx_token_length(blob_text),
                    meta={
                        "doc_id": doc_id,
                        "segment_span": [min(current_indices), max(current_indices)],
                        "num_segments": len(current_parts),
                    },
                )
            )
            current_parts = []
            current_indices = []
            current_len = 0

        current_parts.append(segment.source_text)
        current_indices.append(int(segment.meta.get("segment_index", len(current_indices))))
        current_len += seg_len

    if current_parts:
        blob_text = "\n".join(current_parts)
        blob_idx = len(blobs)
        blobs.append(
            Segment(
                source_id=f"{doc_id}:b:{blob_idx}",
                source_text=blob_text,
                kind="blob",
                length_approx=approx_token_length(blob_text),
                meta={
                    "doc_id": doc_id,
                    "segment_span": [min(current_indices), max(current_indices)],
                    "num_segments": len(current_parts),
                },
            )
        )

    return blobs
