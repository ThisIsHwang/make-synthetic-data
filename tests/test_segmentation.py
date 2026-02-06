from synth_parallel.config import SegmentationConfig
from synth_parallel.segmentation import build_blobs_from_doc_segments, extract_segments_from_record


def test_extract_segments_from_list_field():
    cfg = SegmentationConfig(min_chars=3, max_chars=200, merge_short_lines=False)
    record = {"id": "doc1", "text": ["hello world", "second sentence"]}

    segments = extract_segments_from_record(record, doc_index=0, cfg=cfg, text_field="text")

    assert len(segments) == 2
    assert segments[0].source_id == "doc1:s:0"
    assert segments[0].source_text == "hello world"


def test_extract_segments_from_string_field_and_split():
    cfg = SegmentationConfig(min_chars=3, max_chars=60, merge_short_lines=False)
    record = {"doc_id": "x", "text": "Line one. Line two!\nLine three?"}

    segments = extract_segments_from_record(record, doc_index=1, cfg=cfg, text_field="text")

    assert len(segments) >= 2
    assert all(seg.meta["doc_id"] == "x" for seg in segments)


def test_build_blobs_from_segments():
    cfg = SegmentationConfig(min_chars=1, max_chars=200, merge_short_lines=False)
    record = {
        "id": "docb",
        "text": ["a short sentence", "another short sentence", "one more line"],
    }
    segments = extract_segments_from_record(record, doc_index=0, cfg=cfg)
    blobs = build_blobs_from_doc_segments(segments, blob_max_tokens=8)

    assert len(blobs) >= 2
    assert all(blob.kind == "blob" for blob in blobs)
    assert all(blob.meta["doc_id"] == "docb" for blob in blobs)
