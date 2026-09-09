"""Offline evaluation entry points."""

from .corpus import CASES, CORPUS_VERSION, case_by_id, cases_by_family
from .harness import LAYERS, compare_models, run_case, run_corpus
from .metrics import empty_metrics, score_brief_case, summarise

__all__ = [
    "CASES",
    "CORPUS_VERSION",
    "LAYERS",
    "case_by_id",
    "cases_by_family",
    "compare_models",
    "empty_metrics",
    "run_case",
    "run_corpus",
    "score_brief_case",
    "summarise",
]
