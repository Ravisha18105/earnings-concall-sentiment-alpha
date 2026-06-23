"""
parser.py — v3. Rewrites role detection and quarter extraction.
"""
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import nltk
from bs4 import BeautifulSoup, NavigableString

from src.config import DATA_RAW, DATA_TRANSCRIPTS

logger = logging.getLogger(__name__)

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

# Keywords that indicate executive role in the participants section
EXECUTIVE_KEYWORDS = {
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
    "svp",
    "director of investor",
    "investor relations",
    "head of",
    "vice chairman",
}

# Keywords that indicate analyst role
ANALYST_KEYWORDS = {
    "analyst",
    "research",
    "securities",
    "capital",
    "partners",
    "morgan stanley",
    "goldman sachs",
    "barclays",
    "citigroup",
    "ubs",
    "jefferies",
    "bernstein",
    "baird",
    "cowen",
    "wells fargo",
    "bank of america",
    "deutsche bank",
    "evercore",
    "raymond james",
    "piper sandler",
    "oppenheimer",
    "rbc",
    "bmo",
    "truist",
    "mizuho",
    "hsbc",
    "td cowen",
    "wolfe research",
    "melius",
    "bofa",
}


# ── data schema ───────────────────────────────────────────────────────────────


@dataclass
class SpeakerTurn:
    speaker_raw: str
    speaker_name: str
    speaker_title: str
    role: str
    text: str
    sentences: list
    section: str


@dataclass
class ParsedTranscript:
    ticker: str
    date: str
    quarter: str
    source: str
    source_url: str
    word_count: int
    sentence_count: int
    prepared_remarks: list
    qa_ceo: list
    qa_analyst: list
    guidance_sentences: list
    speaker_turns: list
    call_participants: list
    parse_errors: list = field(default_factory=list)


# ── helpers ───────────────────────────────────────────────────────────────────


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenise(text: str) -> list:
    sents = nltk.sent_tokenize(_clean(text))
    return [s.strip() for s in sents if len(s.strip()) > 20]


def _is_guidance(s: str) -> bool:
    sl = s.lower()
    return any(kw in sl for kw in GUIDANCE_KEYWORDS)


def _infer_role_from_description(description: str) -> str:
    """
    Given a participant description line like:
      'Tim Cook -- Apple -- CEO'
      'Tejas Gala -- Apple -- Director of Investor Relations'
      'Wamsi Mohan -- Bank of America -- Analyst'
    infer the role.
    """
    dl = description.lower()
    if "operator" in dl:
        return "operator"
    if any(k in dl for k in EXECUTIVE_KEYWORDS):
        return "executive"
    if any(k in dl for k in ANALYST_KEYWORDS):
        return "analyst"
    return "unknown"


def _build_role_lookup(participants_text: list) -> dict:
    """
    Build a dict mapping first-name or full-name → role,
    built from the Call Participants section.

    Motley Fool participants look like:
      'Tim Cook -- Apple -- CEO'
      'Wamsi Mohan -- Bank of America -- Analyst'
    Returns e.g. {'Tim Cook': 'executive', 'Wamsi Mohan': 'analyst', ...}
    Also stores first-name only as a fallback key.
    """
    lookup = {}
    for line in participants_text:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        role = _infer_role_from_description(line)
        # Full name = everything before the first ' -- '
        name = line.split("--")[0].strip()
        if name:
            lookup[name.lower()] = role
            # Also store first name as fallback
            first = name.split()[0].lower()
            if first not in lookup:
                lookup[first] = role
    return lookup


def _resolve_role(speaker_raw: str, role_lookup: dict) -> tuple:
    """
    Try to resolve role from the pre-built lookup table.
    Falls back to keyword matching on the raw string.
    Returns (name, title, role).
    """
    # Parse name and title from raw string
    parts = [p.strip() for p in speaker_raw.split("--")]
    name = parts[0].strip()
    title = parts[-1].strip() if len(parts) > 1 else ""

    if name.lower() == "operator":
        return name, "Operator", "operator"

    # Try full name lookup
    role = role_lookup.get(name.lower())
    if role:
        return name, title, role

    # Try first name lookup
    first = name.split()[0].lower() if name else ""
    role = role_lookup.get(first)
    if role:
        return name, title, role

    # Fall back to keyword matching on the full raw string
    rl = speaker_raw.lower()
    if any(k in rl for k in EXECUTIVE_KEYWORDS):
        return name, title, "executive"
    if any(k in rl for k in ANALYST_KEYWORDS):
        return name, title, "analyst"

    return name, title, "unknown"


# ── section extraction ────────────────────────────────────────────────────────


def _get_section_elements(soup: BeautifulSoup, header_text: str) -> list:
    """
    Find the h2 containing header_text and return all sibling elements
    until the next h2. Handles any nesting depth.
    """
    h2 = None
    for tag in soup.find_all("h2"):
        if header_text.lower() in tag.get_text(strip=True).lower():
            h2 = tag
            break
    if h2 is None:
        return []

    elements = []
    for sib in h2.next_siblings:
        if isinstance(sib, NavigableString):
            continue
        if sib.name == "h2":
            break
        elements.append(sib)
    return elements


def _get_participants_raw(soup: BeautifulSoup) -> list:
    """
    Extract raw participant lines from the Call Participants section.
    These are used to build the role lookup table.
    """
    elements = _get_section_elements(soup, "Call participant")
    lines = []
    for el in elements:
        if not hasattr(el, "get_text"):
            continue
        # Each participant is in a <p> or <li>
        # Sometimes they're direct children, sometimes nested
        for child in el.find_all(["p", "li", "strong"]) or [el]:
            t = child.get_text(strip=True)
            if 3 < len(t) < 150:
                lines.append(t)
        # Also try the element itself
        t = el.get_text(strip=True)
        if 3 < len(t) < 150 and t not in lines:
            lines.append(t)
    return lines


# ── turn extraction ───────────────────────────────────────────────────────────


def _extract_turns(elements: list, section: str, role_lookup: dict) -> list:
    """
    Walk elements and extract speaker turns.
    Uses role_lookup to assign roles since Motley Fool often omits titles
    in the transcript body (only listing them in Call Participants).
    """
    turns = []
    current_speaker_raw = None
    current_texts = []

    def flush():
        nonlocal current_speaker_raw, current_texts
        if current_speaker_raw and current_texts:
            combined = " ".join(current_texts)
            name, title, role = _resolve_role(current_speaker_raw, role_lookup)
            if role != "operator" and combined.strip():
                turns.append(
                    SpeakerTurn(
                        speaker_raw=current_speaker_raw,
                        speaker_name=name,
                        speaker_title=title,
                        role=role,
                        text=_clean(combined),
                        sentences=_tokenise(combined),
                        section=section,
                    )
                )
        current_speaker_raw = None
        current_texts = []

    def process(el):
        nonlocal current_speaker_raw, current_texts
        if not hasattr(el, "name") or not el.name:
            return

        full_text = el.get_text(strip=True)
        if not full_text:
            return

        strongs = el.find_all("strong")

        # A speaker label: element contains a <strong> and is short
        # OR the element IS a <strong> tag
        is_speaker_label = (
            strongs and len(full_text) < 200 and el.name in ("p", "div")
        ) or el.name == "strong"

        if is_speaker_label:
            label_text = strongs[0].get_text(strip=True) if strongs else full_text
            # Validate: speaker labels don't contain sentence-ending punctuation
            # and are reasonably short
            if label_text and len(label_text) < 120 and label_text.count(".") <= 1:
                flush()
                current_speaker_raw = label_text
                # Any text after the <strong> in same <p> = first sentence
                if strongs:
                    remainder = full_text.replace(strongs[0].get_text(strip=True), "").strip()
                    if remainder and len(remainder) > 10:
                        current_texts.append(remainder)
                return

        # Regular speech paragraph
        if current_speaker_raw and len(full_text) > 10:
            current_texts.append(full_text)

    for el in elements:
        if not hasattr(el, "name"):
            continue
        # Unwrap wrapper divs
        if el.name in ("div", "section", "article"):
            for child in el.children:
                if hasattr(child, "name") and child.name:
                    process(child)
        else:
            process(el)

    flush()
    return turns


# ── filename utils ────────────────────────────────────────────────────────────


def _parse_filename(html_path: Path) -> dict:
    """
    Parse metadata from filename like:
      AAPL_2022-01-27_Q1-2022_motleyfool.html
      AAPL_01-28-apple-aapl-q1-2022-earnings-call-transcript_motleyfool.html

    Returns dict with ticker, date, quarter (best effort).
    """
    stem = html_path.stem  # strip .html

    # Remove the _motleyfool suffix
    stem = re.sub(r"_motleyfool$", "", stem)

    # Pattern 1: TICKER_YYYY-MM-DD_QN-YYYY  (our clean format)
    m = re.match(r"^([A-Z]+)_(\d{4}-\d{2}-\d{2})_(Q\d-\d{4})$", stem)
    if m:
        return {"ticker": m.group(1), "date": m.group(2), "quarter": m.group(3)}

    # Pattern 2: TICKER_MM-DD-slug (Motley Fool raw download format)
    # e.g. AAPL_01-28-apple-aapl-q1-2022-earnings-call-transcript
    m2 = re.match(r"^([A-Z]+)_(\d{2}-\d{2})-(.+)$", stem)
    if m2:
        ticker = m2.group(1)
        slug = m2.group(3)  # apple-aapl-q1-2022-earnings-call-transcript

        # Extract quarter and year from slug
        qm = re.search(r"(q\d)-(\d{4})", slug)
        if qm:
            quarter = f"Q{qm.group(1)[1]}-{qm.group(2)}"
            year = qm.group(2)
            month = m2.group(2).split("-")[0]  # "01"
            day = m2.group(2).split("-")[1]  # "28"
            date = f"{year}-{month}-{day}"
            return {"ticker": ticker, "date": date, "quarter": quarter}

        return {"ticker": ticker, "date": "", "quarter": ""}

    # Fallback: split by underscore
    parts = stem.split("_")
    return {
        "ticker": parts[0] if parts else "UNKN",
        "date": parts[1] if len(parts) > 1 else "",
        "quarter": parts[2] if len(parts) > 2 else "",
    }


# ── main ──────────────────────────────────────────────────────────────────────


def parse_transcript(html_path: Path, meta: dict) -> Optional[ParsedTranscript]:
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

    # Build role lookup from participants section FIRST
    participants_raw = _get_participants_raw(soup)
    role_lookup = _build_role_lookup(participants_raw)

    if not role_lookup:
        logger.warning(f"{ticker} {quarter} — empty role lookup, role detection may be poor")
        errors.append("empty_role_lookup")

    # Extract section elements
    prep_elements = _get_section_elements(soup, "Prepared Remarks")
    qa_elements = _get_section_elements(soup, "Question")

    if not prep_elements:
        logger.warning(f"{ticker} {quarter} — no Prepared Remarks section found")
        errors.append("no_prepared_remarks")
    if not qa_elements:
        logger.warning(f"{ticker} {quarter} — no Q&A section found")
        errors.append("no_qa_section")

    # Extract turns
    prep_turns = _extract_turns(prep_elements, "prepared_remarks", role_lookup)
    qa_turns = _extract_turns(qa_elements, "qa", role_lookup)

    # Log role breakdown for debugging
    prep_roles = {t.role for t in prep_turns}
    qa_roles = {t.role for t in qa_turns}
    logger.info(
        f"{ticker} {quarter}: "
        f"{len(prep_turns)} prep turns {prep_roles}, "
        f"{len(qa_turns)} Q&A turns {qa_roles}"
    )

    # Aggregate by role
    prepared_sents = [s for t in prep_turns if t.role == "executive" for s in t.sentences]
    qa_ceo_sents = [s for t in qa_turns if t.role == "executive" for s in t.sentences]
    qa_analyst_sents = [s for t in qa_turns if t.role == "analyst" for s in t.sentences]
    guidance_sents = [s for s in prepared_sents + qa_ceo_sents if _is_guidance(s)]

    if not prepared_sents and not qa_ceo_sents:
        logger.warning(
            f"{ticker} {quarter} — 0 executive sentences after role filtering. "
            f"Role lookup had {len(role_lookup)} entries: {dict(list(role_lookup.items())[:5])}"
        )
        errors.append("no_executive_sentences")

    all_sents = prepared_sents + qa_ceo_sents + qa_analyst_sents
    word_count = sum(len(s.split()) for s in all_sents)

    logger.info(
        f"{ticker} {quarter}: "
        f"prep={len(prepared_sents)} qa_ceo={len(qa_ceo_sents)} "
        f"analyst={len(qa_analyst_sents)} guidance={len(guidance_sents)} "
        f"words={word_count:,}"
    )

    return ParsedTranscript(
        ticker=ticker,
        date=date,
        quarter=quarter,
        source="motley_fool",
        source_url=meta.get("url", ""),
        word_count=word_count,
        sentence_count=len(all_sents),
        prepared_remarks=prepared_sents,
        qa_ceo=qa_ceo_sents,
        qa_analyst=qa_analyst_sents,
        guidance_sentences=guidance_sents,
        speaker_turns=[asdict(t) for t in prep_turns + qa_turns],
        call_participants=participants_raw,
        parse_errors=errors,
    )


def save_transcript(parsed: ParsedTranscript) -> Path:
    fname = f"{parsed.ticker}_{parsed.date}_{parsed.quarter}.json"
    out_path = DATA_TRANSCRIPTS / fname
    out_path.write_text(json.dumps(asdict(parsed), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def parse_all(overwrite: bool = False) -> list:
    html_files = sorted(DATA_RAW.glob("*_motleyfool.html"))
    logger.info(f"Found {len(html_files)} HTML files to parse")
    saved = []

    for html_path in html_files:
        # Get metadata from filename
        file_meta = _parse_filename(html_path)

        # Try _meta.json for url field
        meta_path = Path(str(html_path).replace(".html", "_meta.json"))
        if meta_path.exists():
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # Merge: filename parsing takes priority for ticker/date/quarter
            meta = {**saved_meta, **file_meta}
        else:
            meta = file_meta

        ticker = meta.get("ticker", "UNKN")
        date = meta.get("date", "")
        quarter = meta.get("quarter", "")

        out_name = f"{ticker}_{date}_{quarter}.json"
        out_path = DATA_TRANSCRIPTS / out_name

        if out_path.exists() and not overwrite:
            logger.info(f"Already parsed: {out_name} — skipping")
            saved.append(out_path)
            continue

        parsed = parse_transcript(html_path, meta)
        if parsed:
            out = save_transcript(parsed)
            saved.append(out)
            logger.info(f"Saved: {out.name}")

    logger.info(f"Done: {len(saved)} / {len(html_files)} transcripts parsed")
    return saved
