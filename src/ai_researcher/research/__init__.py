"""Karpathy-style deep research: gated multi-turn wiki briefs."""

from .schema import SCHEMA, TURNS, adapt_complete, index_markdown
from .wiki import DeepResearcher, decision_to_verdict, parse_decision, parse_scores

__all__ = [
    "SCHEMA",
    "TURNS",
    "DeepResearcher",
    "adapt_complete",
    "decision_to_verdict",
    "parse_decision",
    "parse_scores",
]
