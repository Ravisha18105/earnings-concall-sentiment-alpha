"""
test_sentiment.py — unit tests for sentiment.py.
Tests helpers without loading the full FinBERT model.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.sentiment import lm_uncertainty_score, score_section  # noqa: E402


def test_lm_uncertainty_detects_hedging():
    sents = [
        "Revenue may increase in the next quarter.",
        "We cannot predict the outcome with certainty.",
    ]
    score = lm_uncertainty_score(sents)
    assert score > 0.05, f"Expected >0.05, got {score}"


def test_lm_uncertainty_zero_for_confident():
    sents = ["Revenue grew 15 percent. Margins expanded strongly."]
    score = lm_uncertainty_score(sents)
    assert score < 0.02, f"Expected near 0, got {score}"


def test_score_section_empty_returns_zeros():
    class MockPipe:
        def __call__(self, sents, **kw):
            return []

    result = score_section(MockPipe(), [])
    assert result["net"] == 0.0
    assert result["n_sents"] == 0


def test_score_section_net_is_pos_minus_neg():
    class MockPipe:
        def __call__(self, sents, **kw):
            return [
                [
                    {"label": "positive", "score": 0.7},
                    {"label": "negative", "score": 0.1},
                    {"label": "neutral", "score": 0.2},
                ]
                for _ in sents
            ]

    result = score_section(MockPipe(), ["Revenue grew.", "Margins expanded."])
    assert abs(result["net"] - 0.6) < 0.01
    assert result["n_sents"] == 2


def test_lm_uncertainty_case_insensitive():
    sents = ["We MAY see volatility. Risks remain UNCERTAIN."]
    score = lm_uncertainty_score(sents)
    assert score > 0.0
