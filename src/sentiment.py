"""
sentiment.py — runs FinBERT and LM word-list scoring on parsed transcripts.
Input : data/transcripts/*.json  (from parser.py)
Output: data/processed/scores.parquet  (one row per transcript)

Key design decisions:
  - Load model ONCE per session (expensive: ~8s on CPU)
  - Batch sentences (16-32 at a time) for speed
  - Weight sentence scores by word count (longer sentences matter more)
  - Skip already-scored transcripts (idempotent)
  - Score 6 sub-signals independently (not just one aggregate)
"""
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from src.config import DATA_PROCESSED, DATA_TRANSCRIPTS, FINBERT_BATCH_SIZE, FINBERT_MODEL
from src.parser import parse_transcript

logger = logging.getLogger(__name__)

SCORES_PATH = DATA_PROCESSED / "scores.parquet"

LM_UNCERTAINTY_WORDS = {
    "approximately",
    "uncertain",
    "uncertainty",
    "unclear",
    "ambiguous",
    "ambiguity",
    "doubt",
    "doubtful",
    "indefinite",
    "imprecise",
    "may",
    "might",
    "could",
    "possible",
    "possibly",
    "perhaps",
    "contingent",
    "depend",
    "depends",
    "depending",
    "variable",
    "fluctuate",
    "fluctuating",
    "unpredictable",
    "volatile",
    "volatility",
    "risk",
    "risks",
    "risky",
    "exposure",
    "concern",
    "concerns",
    "subject to",
    "no assurance",
    "no guarantee",
    "cannot predict",
}


def _get_device() -> str:
    """Auto-detect best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon GPU)")
        return "mps"
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return "cuda"
    logger.info("Using CPU — expect ~45-90 min for full run")
    return "cpu"


def load_finbert():
    """
    Load FinBERT pipeline. Call ONCE per session — not per transcript.
    Returns a callable that scores batches of sentences.
    """
    device = _get_device()
    logger.info(f"Loading FinBERT ({FINBERT_MODEL})...")
    pipe = pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        return_all_scores=True,
        truncation=True,
        max_length=512,
        device=device,
    )
    logger.info("FinBERT loaded")
    return pipe


def score_batch(pipe, sentences: list[str]) -> list[dict]:
    """
    Score a batch of sentences with FinBERT.
    Returns list of {positive, negative, neutral} dicts.
    Handles empty input and inference errors gracefully.
    """
    if not sentences:
        return []
    try:
        results = pipe(sentences, batch_size=FINBERT_BATCH_SIZE)
        parsed = []
        for result in results:
            scores = {r["label"]: r["score"] for r in result}
            parsed.append(
                {
                    "positive": scores.get("positive", 0.0),
                    "negative": scores.get("negative", 0.0),
                    "neutral": scores.get("neutral", 1.0),
                }
            )
        return parsed
    except Exception as e:
        logger.error(f"FinBERT batch failed: {e}")
        return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}] * len(sentences)


def score_section(pipe, sentences: list[str]) -> dict:
    """
    Score a full section (list of sentences) and return aggregate metrics.

    Returns:
        pos      : weighted mean positive score
        neg      : weighted mean negative score
        net      : pos - neg  (primary signal, range -1 to +1)
        n_sents  : number of sentences scored
    """
    if not sentences:
        return {"pos": 0.0, "neg": 0.0, "net": 0.0, "n_sents": 0}

    # Score in chunks to show progress and avoid OOM
    all_scores = []
    chunk_size = FINBERT_BATCH_SIZE
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i : i + chunk_size]
        all_scores.extend(score_batch(pipe, chunk))

    pos_scores = np.array([s["positive"] for s in all_scores])
    neg_scores = np.array([s["negative"] for s in all_scores])

    # Weight by sentence length (longer sentences carry more information)
    weights = np.array([max(len(s.split()), 1) for s in sentences])

    weighted_pos = float(np.average(pos_scores, weights=weights))
    weighted_neg = float(np.average(neg_scores, weights=weights))

    return {
        "pos": round(weighted_pos, 5),
        "neg": round(weighted_neg, 5),
        "net": round(weighted_pos - weighted_neg, 5),
        "n_sents": len(sentences),
    }


def lm_uncertainty_score(sentences: list[str]) -> float:
    """
    Loughran-McDonald uncertainty score.
    = count of uncertainty words / total word count.
    Range: 0.0 (no uncertainty) to ~0.15 (very hedged).
    """
    if not sentences:
        return 0.0
    all_words = " ".join(sentences).lower().split()
    if not all_words:
        return 0.0
    uncertain_count = sum(1 for w in all_words if w in LM_UNCERTAINTY_WORDS)
    return round(uncertain_count / len(all_words), 5)


def _load_transcript_input(transcript_input: Union[dict, str, Path]) -> dict:
    """Load transcript data from a parsed dict, JSON file, or raw HTML/meta path."""
    if isinstance(transcript_input, dict):
        return transcript_input

    path = Path(transcript_input)
    if path.suffix == ".json":
        if path.name.endswith("_meta.json"):
            html_path = Path(str(path).replace("_meta.json", ".html"))
            parsed = parse_transcript(html_path)
            if parsed is None:
                raise ValueError(f"Unable to parse transcript HTML: {html_path}")
            return asdict(parsed)
        return json.loads(path.read_text(encoding="utf-8"))

    if path.suffix == ".html":
        parsed = parse_transcript(path)
        if parsed is None:
            raise ValueError(f"Unable to parse transcript HTML: {path}")
        return asdict(parsed)

    raise ValueError(f"Unsupported transcript input: {path}")


def score_transcript(
    pipe_or_transcript, transcript: Optional[Union[dict, str, Path]] = None
) -> dict:
    """
    Score all sections of one parsed transcript.
    Returns a flat dict of scores — one row in scores.parquet.
    """
    if transcript is None:
        pipe = load_finbert()
        transcript = pipe_or_transcript
    else:
        pipe = pipe_or_transcript

    transcript = _load_transcript_input(transcript)
    ticker = transcript["ticker"]
    date = transcript["date"]
    quarter = transcript["quarter"]

    prep_sents = transcript.get("prepared_remarks", [])
    qa_ceo_sents = transcript.get("qa_ceo", [])
    qa_anal_sents = transcript.get("qa_analyst", [])
    guidance_sents = transcript.get("guidance_sentences", [])

    logger.info(
        f"Scoring {ticker} {quarter}: "
        f"{len(prep_sents)} prep / {len(qa_ceo_sents)} qa_ceo / "
        f"{len(guidance_sents)} guidance sentences"
    )

    prep = score_section(pipe, prep_sents)
    qa_ceo = score_section(pipe, qa_ceo_sents)
    qa_anal = score_section(pipe, qa_anal_sents)
    guidance = score_section(pipe, guidance_sents)

    lm_unc = lm_uncertainty_score(prep_sents + qa_ceo_sents)

    # qa_delta: how much does CEO tone shift from prepared remarks to Q&A?
    # Negative delta = CEO becomes less positive under analyst questioning
    qa_delta = round(qa_ceo["net"] - prep["net"], 5)

    return {
        "ticker": ticker,
        "date": date,
        "quarter": quarter,
        "source": transcript.get("source", ""),
        "word_count": transcript.get("word_count", 0),
        "sentence_count": transcript.get("sentence_count", 0),
        # Prepared remarks
        "prep_pos": prep["pos"],
        "prep_neg": prep["neg"],
        "prep_net": prep["net"],
        "prep_n_sents": prep["n_sents"],
        # Q&A — CEO
        "qa_ceo_pos": qa_ceo["pos"],
        "qa_ceo_neg": qa_ceo["neg"],
        "qa_ceo_net": qa_ceo["net"],
        "qa_ceo_n_sents": qa_ceo["n_sents"],
        # Q&A — analysts
        "qa_anal_pos": qa_anal["pos"],
        "qa_anal_neg": qa_anal["neg"],
        "qa_anal_net": qa_anal["net"],
        "qa_anal_n_sents": qa_anal["n_sents"],
        # Guidance sentences
        "guidance_pos": guidance["pos"],
        "guidance_neg": guidance["neg"],
        "guidance_net": guidance["net"],
        "guidance_n_sents": guidance["n_sents"],
        # Derived signals
        "lm_uncertainty": lm_unc,
        "qa_delta": qa_delta,
    }


def score_all(overwrite: bool = False) -> pd.DataFrame:
    """
    Score all parsed transcripts in data/transcripts/.
    Skips already-scored rows unless overwrite=True.
    Saves incrementally to scores.parquet after each ticker.
    Returns the full scores DataFrame.
    """
    json_files = sorted(DATA_TRANSCRIPTS.glob("*.json"))
    logger.info(f"Found {len(json_files)} parsed transcripts to score")

    # Load existing scores to check what's already done
    if SCORES_PATH.exists() and not overwrite:
        existing = pd.read_parquet(SCORES_PATH)
        done_keys = set(zip(existing["ticker"], existing["date"]))
        logger.info(f"{len(done_keys)} transcripts already scored — will skip")
    else:
        existing = pd.DataFrame()
        done_keys = set()

    pipe = load_finbert()
    new_rows = []

    for json_path in tqdm(json_files, desc="Scoring transcripts"):
        transcript = json.loads(json_path.read_text(encoding="utf-8"))
        key = (transcript["ticker"], transcript["date"])

        if key in done_keys:
            logger.info(f"Skipping {key} — already scored")
            continue

        try:
            row = score_transcript(pipe, transcript)
            new_rows.append(row)
        except Exception as e:
            logger.error(f"Failed scoring {key}: {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = (
            pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        )
        combined.to_parquet(SCORES_PATH, index=False)
        logger.info(f"Saved {len(combined)} total rows to {SCORES_PATH}")
        return combined

    logger.info("No new transcripts scored")
    return existing
