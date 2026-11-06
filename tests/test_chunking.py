from src.hospital_rag.chunking import chunk_sops
from src.hospital_rag.parsing import ParsedPage

SAMPLE = """Intro text before any SOP.
SOP-TST-001: Test Procedure Title
that wraps
Department: Nursing
Version: 2.1
1. Purpose
To test things.
2. Procedure
1. Do the first thing.
2. Verify the identity of the patient using a photo ID and record the details in the
register.
3. Check the label.
3. Escalation
Call the supervisor.
SOP-TST-002: Second
Department: Lab
1. Purpose
Second purpose."""


def make_chunks():
    pages = [ParsedPage(page_number=1, text=SAMPLE)]
    return chunk_sops(pages, "doc")


def test_chunk_count_ignores_steps():
    assert len(make_chunks()) == 4


def test_wrapped_title_and_metadata():
    first = make_chunks()[0]

    assert first.metadata["sop_title"] == "Test Procedure Title that wraps"
    assert first.metadata["department"] == "Nursing"
    assert first.metadata["version"] == "2.1"


def test_steps_stay_inside_their_section():
    procedure = make_chunks()[1]

    assert procedure.metadata["section"] == "Procedure"
    assert "Do the first thing." in procedure.text
    assert "Check the label." in procedure.text
    assert "register." in procedure.text


def test_numbering_resets_for_new_sop():
    last = make_chunks()[3]

    assert last.metadata["sop_id"] == "SOP-TST-002"
    assert last.metadata["section"] == "Purpose"
    assert last.text.startswith("SOP-TST-002 Second > Purpose")


def test_trailing_non_sop_text_is_not_added_to_last_section():
    text = (
        "SOP-TST-001: T\n1. Purpose\nReal content.\nPage 1 of 2\n"
        "Suggested Test Questions for Your RAG System\nWhat is this?\n"
        "SOP-TST-002: U\n1. Purpose\nMore."
    )

    chunks = chunk_sops([ParsedPage(page_number=1, text=text)], "doc")

    assert len(chunks) == 2
    assert "Real content." in chunks[0].text
    assert "Suggested" not in chunks[0].text
    assert "Page 1 of 2" not in chunks[0].text
