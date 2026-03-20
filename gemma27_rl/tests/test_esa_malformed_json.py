from __future__ import annotations

import unittest

from gemma27_rl.rewards import gemba_esa_format_error_spans, gemba_esa_parse_structured_errors


class ESAMalformedJsonParseTests(unittest.TestCase):
    def test_repairs_unescaped_inner_quotes_in_target_span(self) -> None:
        raw = """{
  "errors": [
    {
      "severity": "minor",
      "type": "style/awkward",
      "target_span": "asserting that the European Wine Association is focused on presenting policymakers with information that is"relevant and appropriate"to the industry",
      "source_span": "맥주 제조업체들의 법적 주장의 근거에 대한 의문을 제기하며 유럽 와인 기업 협회는 정책 입안자들에게 해당 업계와 \\"관련성 있고 적절한\\" 정보를 제시하는데 중점을 두고 있다고 주장했습니다",
      "confidence": 0.75
    }
  ]
}"""

        parsed = gemba_esa_parse_structured_errors(raw)

        self.assertEqual(
            parsed,
            [
                {
                    "severity": "minor",
                    "type": "style/awkward",
                    "target_span": (
                        'asserting that the European Wine Association is focused on presenting '
                        'policymakers with information that is"relevant and appropriate"to the industry'
                    ),
                    "source_span": (
                        '맥주 제조업체들의 법적 주장의 근거에 대한 의문을 제기하며 유럽 와인 기업 협회는 정책 입안자들에게 해당 업계와 '
                        '"관련성 있고 적절한" 정보를 제시하는데 중점을 두고 있다고 주장했습니다'
                    ),
                    "confidence": 0.75,
                }
            ],
        )

    def test_repairs_doubled_quotes_in_target_and_source_span(self) -> None:
        raw = """{
  "errors": [
    {
      "severity": "major",
      "type": "accuracy/mistranslation",
      "target_span": ""Appendix Forms 1 and 2"",
      "source_span": ""별지 제1호 내지 제2호 서식"",
      "confidence": 0.95
    }
  ]
}"""

        formatted = gemba_esa_format_error_spans(raw)

        self.assertEqual(
            formatted,
            '{\n  "errors": [\n    {\n      "severity": "major",\n      "type": "accuracy/mistranslation",\n      "target_span": "\\"Appendix Forms 1 and 2\\"",\n      "source_span": "\\"별지 제1호 내지 제2호 서식\\"",\n      "confidence": 0.95\n    }\n  ]\n}',
        )

    def test_repairs_unescaped_quotes_in_both_spans(self) -> None:
        raw = """{
  "errors": [
    {
      "severity": "major",
      "type": "accuracy/mistranslation",
      "target_span": "Council members of the Gangdong-gu Council, Seoul Metropolitan City (hereinafter referred to as "members") shall appoint auditors",
      "source_span": "서울특별시 강동구의회 의원(이하 "의원"이라 한다) 이외의 결산검사위원(이하 "검사위원"이라 한다)은 다음 각 호의 어느 하나에 해당하는 사람을 선임한다.",
      "confidence": 0.98
    }
  ]
}"""

        parsed = gemba_esa_parse_structured_errors(raw)

        self.assertEqual(
            parsed,
            [
                {
                    "severity": "major",
                    "type": "accuracy/mistranslation",
                    "target_span": (
                        'Council members of the Gangdong-gu Council, Seoul Metropolitan City '
                        '(hereinafter referred to as "members") shall appoint auditors'
                    ),
                    "source_span": (
                        '서울특별시 강동구의회 의원(이하 "의원"이라 한다) 이외의 결산검사위원(이하 "검사위원"이라 한다)은 '
                        "다음 각 호의 어느 하나에 해당하는 사람을 선임한다."
                    ),
                    "confidence": 0.98,
                }
            ],
        )

    def test_repairs_trailing_escaped_quote_in_target_span(self) -> None:
        raw = """{
  "errors": [
    {
      "severity": "minor",
      "type": "fluency/punctuation",
      "target_span": "만들었습니다."\\"",
      "source_span": "acceptable.\\"\\"",
      "confidence": 0.98
    }
  ]
}"""

        parsed = gemba_esa_parse_structured_errors(raw)

        self.assertEqual(
            parsed,
            [
                {
                    "severity": "minor",
                    "type": "fluency/punctuation",
                    "target_span": '만들었습니다.""',
                    "source_span": 'acceptable.""',
                    "confidence": 0.98,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
