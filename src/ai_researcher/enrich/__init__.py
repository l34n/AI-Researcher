"""Enrichment: local-model summarisation, classification, embeddings, judgment."""

from .analyze import Enricher
from .chat import ChatRouter
from .embed import Embedder, load_vectors
from .judge import Judge, judge_text, readiness_of, verdict_of
from .ollama import OllamaClient
from .unslop import UNSLOP_RULE, unslop_text

__all__ = [
    "ChatRouter",
    "Enricher",
    "Embedder",
    "Judge",
    "OllamaClient",
    "UNSLOP_RULE",
    "judge_text",
    "load_vectors",
    "readiness_of",
    "unslop_text",
    "verdict_of",
]
