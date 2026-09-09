"""Practitioner judgment: quality, practicality, feasibility, usefulness."""

from ai_researcher.enrich.judge import (
    blend,
    extract_artifacts,
    judge_text,
    readiness_of,
    verdict_of,
)


class TestJudgeText:
    def test_scores_are_bounded(self):
        judged = judge_text("x", "", category="opinion-analysis", tier="news")
        for key in ("quality", "practicality", "feasibility", "usefulness", "readiness"):
            assert 0.0 <= judged[key] <= 1.0
        assert judged["verdict"] in ("skip", "watch", "research", "adopt")

    def test_open_repo_outranks_a_hot_take(self):
        tool = judge_text(
            "vLLM v0.8 released with OpenAI-compatible server",
            "Release notes and pip install vllm. Code at https://github.com/vllm-project/vllm "
            "runs locally with llama.cpp-style GGUF and a 4-bit quant.",
            category="tooling-oss",
            tier="vendor",
            url="https://github.com/vllm-project/vllm",
            importance=0.7,
        )
        rant = judge_text(
            "What do you think about the latest AI hype? Unpopular opinion thread",
            "Just a weekly digest hot take. How to get started as a beginner.",
            category="opinion-analysis",
            tier="community",
            importance=0.3,
        )
        assert tool["readiness"] > rant["readiness"]
        assert tool["practicality"] > rant["practicality"]
        assert tool["verdict"] in ("research", "adopt")
        assert rant["verdict"] in ("skip", "watch")

    def test_closed_frontier_is_not_adoptable(self):
        judged = judge_text(
            "Lab announces a closed 405B API-only model, waitlist now open",
            "Weights not released. Proprietary API-only access. Cluster of H100s. "
            "Not yet available to the public.",
            category="model-release",
            tier="lab",
            importance=0.95,
        )
        assert judged["feasibility"] < 0.45
        assert judged["verdict"] != "adopt"

    def test_arxiv_without_code_is_less_practical(self):
        paper = judge_text(
            "We propose a new attention variant",
            "In this paper we introduce a method. Empirical study and ablation. arxiv:2401.12345",
            category="research",
            tier="research",
            url="https://arxiv.org/abs/2401.12345",
            importance=0.6,
        )
        coded = judge_text(
            "We propose a new attention variant",
            "In this paper we introduce a method. Code at https://github.com/lab/attn "
            "and open weights on huggingface.co/lab/attn. Runs locally on RTX.",
            category="research",
            tier="research",
            url="https://arxiv.org/abs/2401.12345",
            importance=0.6,
        )
        assert coded["practicality"] > paper["practicality"]
        assert coded["feasibility"] > paper["feasibility"]

    def test_corroboration_lifts_quality(self):
        one = judge_text("Lab ships weights", "open weights on huggingface",
                         category="model-release", tier="lab", source_count=1)
        many = judge_text("Lab ships weights", "open weights on huggingface",
                          category="model-release", tier="lab", source_count=5)
        assert many["quality"] > one["quality"]


class TestVerdict:
    def test_adopt_needs_brakes(self):
        assert verdict_of(0.90, practicality=0.4, feasibility=0.8) == "research"
        assert verdict_of(0.90, practicality=0.8, feasibility=0.4) == "research"
        assert verdict_of(0.90, practicality=0.8, feasibility=0.8) == "adopt"

    def test_bands(self):
        assert verdict_of(0.20) == "skip"
        assert verdict_of(0.50) == "watch"
        assert verdict_of(0.65) == "research"

    def test_readiness_weights_sum_to_one(self):
        # A unit vector on each axis should recover that axis's weight.
        assert readiness_of(1, 0, 0, 0) == 0.28
        assert readiness_of(0, 1, 0, 0) == 0.22
        assert abs(readiness_of(1, 1, 1, 1) - 1.0) < 1e-9


class TestArtifacts:
    def test_extracts_github_hf_arxiv(self):
        found = extract_artifacts(
            "New model",
            "Code https://github.com/acme/thing and weights "
            "https://huggingface.co/acme/thing — see arxiv:2403.01234",
            "https://acme.example/blog",
        )
        assert any("github.com/acme/thing" in a for a in found)
        assert any("huggingface.co/acme/thing" in a for a in found)
        assert "arxiv:2403.01234" in found

    def test_strips_trailing_punctuation(self):
        found = extract_artifacts("x", "See https://github.com/acme/thing).", "")
        assert any(a.endswith("acme/thing") for a in found)
        assert not any(a.endswith("thing).") for a in found)

    def test_item_url_counts_when_it_is_the_artifact(self):
        found = extract_artifacts("Weights", "now up", "https://huggingface.co/acme/mod")
        assert "https://huggingface.co/acme/mod" in found

    def test_blog_url_is_not_an_artifact(self):
        found = extract_artifacts("Announcement", "we shipped", "https://acme.example/blog")
        assert found == []

    def test_dedupes_and_caps(self):
        body = " ".join(
            f"https://github.com/acme/r{i}" for i in range(8)
        )
        found = extract_artifacts("many", body + " https://github.com/acme/r0", "")
        assert found[0].endswith("acme/r0")
        assert found.count(found[0]) == 1
        assert len(found) <= 6

    def test_arxiv_abs_and_colon_forms(self):
        found = extract_artifacts("p", "https://arxiv.org/abs/2401.12345 and arxiv:2401.99999", "")
        assert "arxiv:2401.12345" in found
        assert "arxiv:2401.99999" in found


class TestBlend:
    def test_model_cannot_force_adopt_on_a_skip(self):
        heuristic = judge_text("weekly digest hot take", "what do you think",
                               category="opinion-analysis", tier="news", importance=0.2)
        model = {
            "quality": 0.95, "practicality": 0.95, "feasibility": 0.95,
            "usefulness": 0.95, "verdict": "adopt",
            "reasons": ["amazing"], "artifacts": [],
        }
        out = blend(heuristic, model)
        assert out["verdict"] != "adopt"
        assert out["readiness"] < 0.85

    def test_keeps_heuristic_artifacts(self):
        heuristic = {
            "quality": 0.6, "practicality": 0.6, "feasibility": 0.6,
            "usefulness": 0.6, "readiness": 0.6, "verdict": "research",
            "reasons": ["names a fetchable artifact"],
            "artifacts": ["https://github.com/acme/x"],
        }
        out = blend(heuristic, {"quality": 0.7, "practicality": 0.7,
                                "feasibility": 0.7, "usefulness": 0.7})
        assert "https://github.com/acme/x" in out["artifacts"]
