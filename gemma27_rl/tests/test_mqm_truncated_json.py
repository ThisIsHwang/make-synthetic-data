from __future__ import annotations

import unittest

from gemma27_rl.rewards import gemba_mqm_parse_structured_errors, gemba_mqm_score


class MQMTruncatedJsonParseTests(unittest.TestCase):
    def test_recovers_truncated_non_translation_json(self) -> None:
        raw = """{
  "errors": [
    {
      "severity": "critical",
      "type": "non-translation",
      "target_span": "garbled output that was cut off mid-string
"""

        parsed = gemba_mqm_parse_structured_errors(raw)

        self.assertEqual(
            parsed,
            [
                {
                    "severity": "critical",
                    "type": "non-translation",
                    "target_span": "garbled output that was cut off mid-string",
                    "source_span": None,
                    "confidence": 1.0,
                }
            ],
        )
        self.assertEqual(gemba_mqm_score(raw), -25)

    def test_keeps_complete_items_before_truncated_tail(self) -> None:
        raw = """{
  "errors": [
    {
      "severity": "major",
      "type": "accuracy/mistranslation",
      "target_span": "wrong term",
      "source_span": "source term",
      "confidence": 0.91
    },
    {
      "severity": "minor",
      "type": "non-translation",
      "target_span": "repeated repeated repeated
"""

        parsed = gemba_mqm_parse_structured_errors(raw)

        self.assertEqual(
            parsed,
            [
                {
                    "severity": "major",
                    "type": "accuracy/mistranslation",
                    "target_span": "wrong term",
                    "source_span": "source term",
                    "confidence": 0.91,
                },
                {
                    "severity": "minor",
                    "type": "non-translation",
                    "target_span": "repeated repeated repeated",
                    "source_span": None,
                    "confidence": 1.0,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
