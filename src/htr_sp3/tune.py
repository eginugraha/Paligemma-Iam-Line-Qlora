"""Pick the correction threshold that minimizes validation CER.

Thesis integrity: tune on VALIDATION predictions, never test. `tune_threshold` is pure logic —
it takes (prediction, ground_truth) pairs and a factory that builds a corrector for a given
threshold — so it is unit-testable with an in-memory corrector and no model/DB. The CLI
(scripts/tune_sp3.py) supplies real M1 predictions on the validation split.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from htr_sp1.metrics import cer as cer_metric

# A corrector factory: threshold -> object with .correct(text) -> (text, log).
CorrectorFactory = Callable[[float], object]


def tune_threshold(
    pairs: List[Tuple[str, str]],
    make_corrector: CorrectorFactory,
    thresholds: List[float],
) -> Dict[str, object]:
    """Return the best threshold and the full CER-vs-threshold curve.

    Args:
        pairs:          list of (prediction, ground_truth) on the validation split.
        make_corrector: builds a corrector for a given threshold.
        thresholds:     candidate thresholds to scan (e.g. [0.10, 0.15, ... 0.50]).

    Returns:
        {"best_threshold": float, "best_cer": float, "curve": {threshold: mean_cer}}.
    """
    curve: Dict[float, float] = {}
    for t in thresholds:
        corrector = make_corrector(t)
        total = 0.0
        for prediction, truth in pairs:
            corrected, _log = corrector.correct(prediction)
            total += cer_metric(truth, corrected)
        curve[t] = total / len(pairs) if pairs else 0.0

    best_threshold = min(curve, key=curve.get)
    return {"best_threshold": best_threshold, "best_cer": curve[best_threshold], "curve": curve}
