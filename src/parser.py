"""
parser.py — parses raw Motley Fool HTML transcripts into structured JSON.
Input : data/raw/{TICKER}_{DATE}_{QUARTER}_motleyfool.html
Output: data/transcripts/{TICKER}_{DATE}_{QUARTER}.json

Never fetches from network — reads from disk only.
"""
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import nltk
from bs4 import BeautifulSoup

from src.config import DATA_RAW, DATA_TRANSCRIPTS

logger = logging.getLogger(__name__)

# Download NLTK data once on import (no-op if already downloaded)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

GUIDANCE_KEYWORDS = {
    "expect",
    "expects",
    "expected",
    "expecting",
    "guidance",
    "outlook",
    "forecast",
    "forecasting",
    "anticipate",
    "anticipates",
    "anticipated",
    "next quarter",
    "full year",
    "fiscal year",
    "going forward",
    "remainder of",
    "second half",
}

EXECUTIVE_TITLES = {
    "chief executive",
    "ceo",
    "chief financial",
    "cfo",
    "chief operating",
    "coo",
    "president",
    "chairman",
    "executive vice president",
    "evp",
    "senior vice president",
}


# ── data schema ───────────────────────────────────────────────────────────────


@dataclass
class SpeakerTurn:
    speaker_raw: str  # "Tim Cook -- Apple -- CEO"
    speaker_name: str  # "Tim Cook"
    speaker_title: str  # "CEO"
    role: str  # "executive" | "analyst" | "operator"
    text: str  # full paragraph text
    sentences: list[str]  # sentence-tokenised text
    section: str  # "prepared_remarks" | "qa"


@dataclass
class ParsedTranscript:
    ticker: str
    date: str
    quarter: str
    source: str
    source_url: str
    word_count: int
    sentence_count: int
    prepared_remarks: list[str]  # sentences (executive turns only)
    qa_ceo: list[str]  # sentences (executive turns in Q&A)
    qa_analyst: list[str]  # sentences (analyst turns in Q&A)
    guidance_sentences: list[str]  # forward-looking sentences
    speaker_turns: list[dict]  # full turn-level data
    call_participants: list[str]  # names listed in participants section
    parse_errors: list[str] = field(default_factory=list)


# ── helpers ───────────────────────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Strip extra whitespace and non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


def _parse_speaker(raw: str) -> tuple[str, str, str]:
    """
    Parse 'Tim Cook -- Apple -- CEO' into (name, company, title).
    Falls back gracefully when format is unexpected.
    Returns (name, title, role).
    """
    parts = [p.strip() for p in raw.split("--")]
    name = parts[0] if parts else raw
    title = parts[-1] if len(parts) > 1 else ""
    title_lower = title.lower()

    if any(t in title_lower for t in EXECUTIVE_TITLES):
        role = "executive"
    elif "analyst" in title_lower or any(
        bank in raw.lower()
        for bank in [
            "morgan",
            "goldman",
            "barclays",
            "citigroup",
            "ubs",
            "jefferies",
            "bernstein",
            "baird",
            "cowen",
            "wells",
            "bank of america",
            "deutsche",
            "credit suisse",
        ]
    ):
        role = "analyst"
    elif "operator" in name.lower():
        role = "operator"
    else:
        role = "unknown"

    return name.strip(), title.strip(), role


def _is_guidance_sentence(sentence: str) -> bool:
    """True if sentence contains forward-looking language."""
    s = sentence.lower()
    return any(kw in s for kw in GUIDANCE_KEYWORDS)


def _tokenise(text: str) -> list[str]:
    """Split text into sentences, filtering very short ones."""
    sentences = nltk.sent_tokenize(_clean_text(text))
    return [s.strip() for s in sentences]


# ── section splitter ──────────────────────────────────────────────────────────


def _split_sections(soup: BeautifulSoup) -> dict[str, list]:
    """
        Split Motley Fool HTML into sections using h2 tags.
        Returns dict: {section_name: [BeautifulSoup elements]}

        Motley Fool h2 structure:

    Prepared Remarks:

          ... speaker turns ...

    Questions and Answers:

          ... Q&A turns ...

    Call Participants:

          ... participant list ...
    """
    sections = {
        "prepared_remarks": [],
        "qa": [],
        "participants": [],
    }

    SECTION_MAP = {
        "prepared": "prepared_remarks",
        "question": "qa",
        "call participant": "participants",
        "participants": "participants",
    }

    current = None
    for element in soup.find_all(["h2", "h3", "strong", "p", "ul", "li"]):
        tag = element.name
        text = element.get_text(strip=True).lower()

        if tag in ("h2", "h3"):
            for key, section_name in SECTION_MAP.items():
                if key in text:
                    current = section_name
                    break
            continue

        if current and tag in ("strong", "p", "ul", "li"):
            sections[current].append(element)

    return sections


# ── turn extractor ────────────────────────────────────────────────────────────


def _extract_turns(elements: list, section: str) -> list[SpeakerTurn]:
    """
       Extract speaker turns from a list of BeautifulSoup elements.
       Motley Fool:  = speaker name, following
    = their text.
    """
    turns = []
    current_speaker_raw = None
    current_texts = []

    def flush():
        if current_speaker_raw and current_texts:
            combined = " ".join(current_texts)
            name, title, role = _parse_speaker(current_speaker_raw)
            turns.append(
                SpeakerTurn(
                    speaker_raw=current_speaker_raw,
                    speaker_name=name,
                    speaker_title=title,
                    role=role,
                    text=_clean_text(combined),
                    sentences=_tokenise(combined),
                    section=section,
                )
            )

    for el in elements:
        if el.name != "p":
            continue

        label = el.find("strong")
        if label:
            flush()
            speaker_name = label.get_text(strip=True)
            title = ""
            title_el = el.find("em")
            if title_el:
                title = title_el.get_text(" ", strip=True)

            current_speaker_raw = f"{speaker_name} -- {title}" if title else speaker_name
            current_texts = []
            continue

        if current_speaker_raw:
            t = el.get_text(" ", strip=True)
            if t:
                current_texts.append(t)

    flush()
    return turns


# ── main parser ───────────────────────────────────────────────────────────────


def _load_meta_for_html(html_path: Path) -> dict:
    """Load transcript metadata from the adjacent _meta.json file or filename."""
    meta_path = Path(str(html_path).replace(".html", "_meta.json"))
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    parts = html_path.stem.split("_")
    meta = {}
    if len(parts) >= 3:
        meta = {"ticker": parts[0], "date": parts[1], "quarter": parts[2]}
    return meta


def parse_transcript(
    html_path: Union[Path, str],
    meta: Optional[dict] = None,
) -> Optional[ParsedTranscript]:
    """
    Parse one raw HTML file into a ParsedTranscript.
    Returns None and logs error if parsing fails critically.
    """
    html_path = Path(html_path)
    if meta is None:
        meta = _load_meta_for_html(html_path)

    ticker = meta.get("ticker", "UNKN")
    date = meta.get("date", "")
    quarter = meta.get("quarter", "")
    errors = []

    try:
        html = html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:
        logger.error(f"Cannot read {html_path.name}: {e}")
        return None

    sections = _split_sections(soup)

    if not sections["prepared_remarks"] and not sections["qa"]:
        logger.warning(f"{ticker} {quarter} — no sections found. Check HTML structure.")
        errors.append("no_sections_found")

    # Extract speaker turns per section
    prep_turns = _extract_turns(sections["prepared_remarks"], "prepared_remarks")
    qa_turns = _extract_turns(sections["qa"], "qa")
    all_turns = prep_turns + qa_turns

    if not all_turns:
        logger.warning(f"{ticker} {quarter} — zero speaker turns extracted")
        errors.append("no_speaker_turns")

    # Aggregate sentences by role and section
    prepared_sents = [s for t in prep_turns if t.role == "executive" for s in t.sentences]
    qa_ceo_sents = [s for t in qa_turns if t.role == "executive" for s in t.sentences]
    qa_analyst_sents = [s for t in qa_turns if t.role == "analyst" for s in t.sentences]
    guidance_sents = [s for s in prepared_sents + qa_ceo_sents if _is_guidance_sentence(s)]

    # Call participants
    participants = [
        li.get_text(strip=True)
        for li in sections["participants"]
        if li.name in ("li", "p") and li.get_text(strip=True)
    ]

    all_sentences = prepared_sents + qa_ceo_sents + qa_analyst_sents
    word_count = sum(len(s.split()) for s in all_sentences)

    result = ParsedTranscript(
        ticker=ticker,
        date=date,
        quarter=quarter,
        source="motley_fool",
        source_url=meta.get("url", ""),
        word_count=word_count,
        sentence_count=len(all_sentences),
        prepared_remarks=prepared_sents,
        qa_ceo=qa_ceo_sents,
        qa_analyst=qa_analyst_sents,
        guidance_sentences=guidance_sents,
        speaker_turns=[asdict(t) for t in all_turns],
        call_participants=participants,
        parse_errors=errors,
    )

    logger.info(
        f"{ticker} {quarter}: {len(prep_turns)} prep turns, "
        f"{len(qa_turns)} Q&A turns, "
        f"{len(guidance_sents)} guidance sentences, "
        f"{word_count:,} words"
    )
    return result


def save_transcript(parsed: ParsedTranscript) -> Path:
    """Save ParsedTranscript as JSON to data/transcripts/."""
    fname = f"{parsed.ticker}_{parsed.date}_{parsed.quarter}.json"
    out_path = DATA_TRANSCRIPTS / fname
    out_path.write_text(json.dumps(asdict(parsed), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def parse_all(overwrite: bool = False) -> list[Path]:
    """
    Parse all HTML files in data/raw/ that have a matching _meta.json.
    Skips files already parsed unless overwrite=True.
    """
    html_files = sorted(DATA_RAW.glob("*_motleyfool.html"))
    logger.info(f"Found {len(html_files)} HTML files to parse")

    saved = []
    for html_path in html_files:
        out_name = html_path.stem.replace("_motleyfool", "") + ".json"
        out_path = DATA_TRANSCRIPTS / out_name

        if out_path.exists() and not overwrite:
            logger.info(f"Already parsed: {out_name} — skipping")
            saved.append(out_path)
            continue

        parsed = parse_transcript(html_path)
        if parsed:
            out = save_transcript(parsed)
            saved.append(out)
            logger.info(f"Saved: {out.name}")

    logger.info(f"Parsing complete: {len(saved)} / {len(html_files)} transcripts saved")
    return saved
