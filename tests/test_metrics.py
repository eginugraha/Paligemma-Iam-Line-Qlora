"""CER/WER are the headline numbers of the thesis, so we test exact, hand-computed cases."""
from htr_sp1 import metrics


def test_cer_perfect_match_is_zero():
    assert metrics.cer("the quick brown fox", "the quick brown fox") == 0.0


def test_cer_single_substitution_percentage():
    # "fox" -> "fux": 1 wrong char out of 19 reference chars = 1/19 * 100 ≈ 5.263.
    value = metrics.cer("the quick brown fox", "the quick brown fux")
    assert round(value, 2) == 5.26


def test_wer_one_wrong_word_of_four():
    # 1 wrong word / 4 reference words = 25%.
    value = metrics.wer("the quick brown fox", "the quick brown fux")
    assert round(value, 2) == 25.0


def test_metrics_return_percentages_not_fractions():
    # We standardize on PERCENT (0–100) so UI/report numbers match the PRD example.
    assert metrics.cer("ab", "xy") == 100.0


def test_wer_return_percentages_not_fractions():
    # Guard the WER path against accidental loss of the *100 conversion (2 words, both wrong).
    assert metrics.wer("alpha beta", "gamma delta") == 100.0
