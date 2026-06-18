"""
test_parser.py — unit tests for parser.py.
Uses minimal fixture HTML — no real files needed.
"""

from bs4 import BeautifulSoup

from src.parser import (
    _extract_turns,
    _is_guidance_sentence,
    _parse_speaker,
    _split_sections,
    _tokenise,
)

SAMPLE_HTML = """

<html>

<body>

<h2>Prepared Remarks</h2>

<strong>Tim Cook -- Apple -- Chief Executive Officer</strong>

<p>

Good afternoon everyone. Revenue grew 8 percent year over year.

</p>

<p>

We expect strong growth in the next quarter driven by iPhone demand.

</p>

<h2>Questions and Answers</h2>

<p>

    <strong>Kyle McNealy</strong>

    <em>Analyst</em>

</p>

<p>

    Hi Tim, can you talk about services growth?

</p>

<p>

    <strong>Tim Cook</strong>

    <em>Chief Executive Officer</em>

</p>

<p>

    Sure, services grew 16 percent. We anticipate continued momentum.

</p>

<h2>Call Participants</h2>

<ul>

    <li>Tim Cook -- CEO</li>

    <li>Kyle McNealy -- Jefferies</li>

</ul>

</body>

</html>

"""


def test_parse_speaker_executive():
    name, title, role = _parse_speaker("Tim Cook -- Apple -- Chief Executive Officer")
    assert name == "Tim Cook"
    assert role == "executive"


def test_parse_speaker_analyst():
    name, title, role = _parse_speaker("Kyle McNealy -- Jefferies -- Analyst")
    assert name == "Kyle McNealy"
    assert role == "analyst"


def test_parse_speaker_operator():
    name, title, role = _parse_speaker("Operator")
    assert role == "operator"


def test_guidance_detection():
    assert _is_guidance_sentence("We expect strong growth next quarter.")
    assert _is_guidance_sentence("Our outlook remains positive.")
    assert not _is_guidance_sentence("Revenue was 10 billion last quarter.")


def test_tokenise_splits_sentences():
    text = "Revenue grew. Margins expanded. We are optimistic."
    sents = _tokenise(text)
    assert len(sents) == 3


def test_split_sections_finds_all():
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")
    sections = _split_sections(soup)

    assert len(sections["prepared_remarks"]) > 0
    assert len(sections["qa"]) > 0
    assert len(sections["participants"]) > 0


def test_extract_turns_roles():
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")
    sections = _split_sections(soup)

    qa_turns = _extract_turns(sections["qa"], "qa")
    roles = {t.role for t in qa_turns}

    assert "executive" in roles
    assert "analyst" in roles


def test_ceo_turns_have_guidance():
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")
    sections = _split_sections(soup)
    qa_turns = _extract_turns(sections["qa"], "qa")
    ceo_sents = [s for t in qa_turns if t.role == "executive" for s in t.sentences]
    guidance = [s for s in ceo_sents if _is_guidance_sentence(s)]
    assert len(guidance) >= 1
