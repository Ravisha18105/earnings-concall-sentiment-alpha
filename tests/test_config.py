"""
Smoke tests: verify config loaded correctly and all directories exist.
Run with: pytest tests/test_config.py -v
"""

from src import config


def test_directories_exist():
    """All data directories should be created on import."""
    assert config.DATA_RAW.exists()
    assert config.DATA_TRANSCRIPTS.exists()
    assert config.DATA_PRICES.exists()
    assert config.DATA_PROCESSED.exists()
    assert config.REPORTS_DIR.exists()


def test_tickers_not_empty():
    assert len(config.TICKERS) > 0


def test_signal_weights_sum_to_one():
    total = sum(abs(v) for v in config.SIGNAL_WEIGHTS.values())
    assert abs(total - 1.0) < 0.01, f"Weights should sum to 1, got {total}"


def test_finbert_model_name():
    assert config.FINBERT_MODEL == "ProsusAI/finbert"
