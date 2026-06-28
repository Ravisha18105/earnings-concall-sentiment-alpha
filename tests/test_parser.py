"""
test_parser.py — unit tests for parser.py (v3)
"""

from bs4 import BeautifulSoup

from src.parser import (
    _build_role_lookup,
    _extract_turns,
    _get_participants_raw,
    _get_section_elements,
    _is_guidance,
    _resolve_role,
    _tokenise,
)

SAMPLE_HTML = """
<html>
<body>

<h2>Prepared Remarks</h2>

<strong>Tim Cook -- Apple -- Chief Executive Officer</strong>

<p>
Good afternoon everyone. Revenue grew eight percent
year over year and margins expanded substantially.
</p>

<p>
We expect strong growth in the next quarter driven by iPhone demand and continued services momentum.
</p>

<h2>Questions and Answers</h2>

<p>
<strong>Kyle McNealy</strong>
<em>Analyst</em>
</p>

<p>
Hi Tim, can you talk about services growth and your expectations for the next quarter?
</p>

<p>
<strong>Tim Cook</strong>
<em>Chief Executive Officer</em>
</p>

<p>
Sure. Services grew sixteen percent this quarter.
We anticipate continued momentum going forward
and expect another strong quarter.
</p>

<h2>Call Participants</h2>

<ul>
<li>Tim Cook -- Apple -- Chief Executive Officer</li>
<li>Kyle McNealy -- Jefferies -- Analyst</li>
</ul>

</body>
</html>
"""


def make_lookup(soup):
    participants = _get_participants_raw(soup)
    return _build_role_lookup(participants)


def test_resolve_role_executive():
    lookup = {"tim cook": "executive"}

    name, title, role = _resolve_role(
        "Tim Cook -- Apple -- Chief Executive Officer",
        lookup,
    )

    assert name == "Tim Cook"
    assert role == "executive"


def test_resolve_role_analyst():
    lookup = {"kyle mcnealy": "analyst"}

    name, title, role = _resolve_role(
        "Kyle McNealy -- Jefferies -- Analyst",
        lookup,
    )

    assert name == "Kyle McNealy"
    assert role == "analyst"


def test_resolve_role_operator():
    name, title, role = _resolve_role("Operator", {})

    assert role == "operator"


def test_guidance_detection():
    assert _is_guidance("We expect strong growth next quarter.")
    assert _is_guidance("Our outlook remains positive.")
    assert not _is_guidance("Revenue was ten billion last quarter.")


def test_tokenise_splits_sentences():
    text = (
        "Revenue increased significantly during the quarter. "
        "Operating margins expanded substantially because of lower costs. "
        "We remain optimistic about next quarter performance."
    )

    sents = _tokenise(text)

    assert len(sents) == 3


def test_get_section_elements():
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")

    prepared = _get_section_elements(soup, "Prepared Remarks")
    qa = _get_section_elements(soup, "Question")
    participants = _get_section_elements(soup, "Call participant")

    assert len(prepared) > 0
    assert len(qa) > 0
    assert len(participants) > 0


def test_role_lookup():
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")

    lookup = make_lookup(soup)

    assert lookup["tim cook"] == "executive"
    assert lookup["kyle mcnealy"] == "analyst"


def test_extract_turns_roles():
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")

    lookup = make_lookup(soup)

    qa = _get_section_elements(soup, "Question")

    turns = _extract_turns(
        qa,
        "qa",
        lookup,
    )

    roles = {t.role for t in turns}

    assert "executive" in roles
    assert "analyst" in roles


def test_ceo_turns_have_guidance():
    soup = BeautifulSoup(SAMPLE_HTML, "lxml")

    lookup = make_lookup(soup)

    qa = _get_section_elements(soup, "Question")

    turns = _extract_turns(
        qa,
        "qa",
        lookup,
    )

    ceo_sentences = [s for t in turns if t.role == "executive" for s in t.sentences]

    guidance = [s for s in ceo_sentences if _is_guidance(s)]

    assert len(guidance) >= 1
