from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from typing import Any

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT_RE = re.compile(r"[.,!?;:。！？、，；：()\[\]{}\"'`-]", re.UNICODE)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def approx_token_length(text: str) -> int:
    words = len(_WORD_RE.findall(text))
    punct = len(_PUNCT_RE.findall(text))
    return words + max(1, punct // 2)


def chunks(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def jaccard_overlap(a: str, b: str) -> float:
    tokens_a = set(_WORD_RE.findall(a.lower()))
    tokens_b = set(_WORD_RE.findall(b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return inter / union if union else 0.0
