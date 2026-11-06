from src.hospital_rag.interfaces import Chunk
from src.hospital_rag.refine import merge_short, refine_chunks, split_long


def mk(sop: str, section: str, body: str) -> Chunk:
    return Chunk(
        id="x",
        document_id="d",
        text=f"{sop} T > {section}\n{body}",
        metadata={"sop_id": sop, "section": section, "page": 1},
    )


def test_short_section_merges_into_next():
    chunks = [mk("A", "Purpose", "short"), mk("A", "Procedure", "x" * 300)]

    result = merge_short(chunks, min_chars=100)

    assert len(result) == 1
    assert result[0].metadata["section"] == "Purpose + Procedure"


def test_short_last_section_merges_into_previous():
    chunks = [mk("A", "Procedure", "x" * 300), mk("A", "Records", "short")]

    result = merge_short(chunks, min_chars=100)

    assert len(result) == 1
    assert result[0].metadata["section"] == "Procedure + Records"


def test_merge_never_crosses_sops():
    chunks = [mk("A", "Purpose", "short"), mk("B", "Purpose", "short")]

    assert len(merge_short(chunks, min_chars=100)) == 2


def test_split_keeps_header_and_overlaps_one_line():
    lines = [f"{i}. " + "w" * 36 for i in range(1, 7)]
    chunk = mk("A", "Procedure", "\n".join(lines))
    header = chunk.text.split("\n")[0]

    parts = split_long([chunk], max_chars=120)

    assert len(parts) > 1
    assert all(len(p.text) <= 120 for p in parts)
    assert all(p.text.startswith(header) for p in parts)
    assert parts[0].text.split("\n")[-1] == parts[1].text.split("\n")[1]


def test_refine_renumbers_ids():
    chunks = [mk("A", "Purpose", "short"), mk("A", "Procedure", "x" * 300)]

    result = refine_chunks(chunks, min_chars=100, max_chars=500)

    assert [c.id for c in result] == ["d-001"]