"""M2 produces reasoning + a final answer in one generation. parse_cot splits them:
text (after the last 'Final:') feeds CER/WER; the reasoning becomes the log. When the
model ignores the format (no marker), we fall back so the comparison stays honest."""
from htr_sp2 import cot


def test_parse_cot_splits_reasoning_and_final():
    raw = "Reasoning: word 4 has a loop 'd'\nFinal: medical"
    text, log = cot.parse_cot(raw)
    assert text == "medical"
    assert log == "Reasoning: word 4 has a loop 'd'"


def test_parse_cot_takes_last_final_marker():
    raw = "Final: draft\nFinal: medical"
    text, log = cot.parse_cot(raw)
    assert text == "medical"
    assert log == "Final: draft"


def test_parse_cot_strips_whitespace():
    text, log = cot.parse_cot("  Final:   hello  ")
    assert text == "hello"
    assert log == ""


def test_parse_cot_fallback_when_no_marker():
    raw = "the quick brown fox"
    text, log = cot.parse_cot(raw)
    assert text == "the quick brown fox"
    assert cot.NO_MARKER_NOTE in log
