import re
from collections import Counter
from collections.abc import Iterator

from src.hospital_rag.interfaces import Chunk
from src.hospital_rag.parsing import ParsedPage

SOP_RE = re.compile(r"^(SOP-[A-Z]{3}-\d{3}):\s*(.+)$")
META_RE = re.compile(
    r"^(Department|Version|Effective Date|Owner|Review Cycle):\s*(.+)$"
)
NUMBERED_RE = re.compile(r"^(\d+)\.\s+(.+)$")
MAX_HEADING_LENGTH = 60

FOOTER_RE = re.compile(r"^Page \d+ of \d+$")
END_MARKERS = (
    "Suggested Test Questions",
    "Ingestion Tips",
    "Additional Sample SOPs",
)

def find_repeated_lines(
    pages: list[ParsedPage], min_fraction: float = 0.9
) -> set[str]:
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({ln.strip() for ln in page.text.splitlines() if ln.strip()})
    threshold = len(pages) * min_fraction
    return {line for line, n in counts.items() if n >= threshold}


def iter_lines(
    pages: list[ParsedPage], skip: set[str] | frozenset[str] = frozenset()
) -> Iterator[tuple[int, str]]:
    for page in pages:
        for line in page.text.splitlines():
            line = line.strip()
            if line and line not in skip and not FOOTER_RE.match(line):
                yield page.page_number, line


def parse_heading(line: str, expected: int) -> str | None:
    match = NUMBERED_RE.match(line)
    if not match:
        return None
    number, title = int(match.group(1)), match.group(2).strip()
    if number != expected:
        return None
    if len(title) > MAX_HEADING_LENGTH or title.endswith("."):
        return None
    return title


def chunk_sops(pages: list[ParsedPage], document_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    sop_id: str | None = None
    sop_title = ""
    meta: dict[str, str] = {}
    reading_title = False
    expected = 1
    section: str | None = None
    section_page = 0
    body: list[str] = []

    def flush() -> None:
        if sop_id is None or section is None:
            return
        header = f"{sop_id} {sop_title} > {section}"
        chunks.append(
            Chunk(
                id=f"{document_id}-{len(chunks) + 1:03d}",
                document_id=document_id,
                text=header + "\n" + "\n".join(body),
                metadata={
                    "sop_id": sop_id,
                    "sop_title": sop_title,
                    "section": section,
                    "page": section_page,
                    **meta,
                },
            )
        )

    noise = find_repeated_lines(pages)
    for page_number, line in iter_lines(pages, noise):
        if line.startswith(END_MARKERS):
            flush()
            sop_id, section, body = None, None, []
            continue

        sop_match = SOP_RE.match(line)
        if sop_match:
            flush()
            sop_id, sop_title = sop_match.group(1), sop_match.group(2)
            meta, body = {}, []
            section = None
            reading_title = True
            expected = 1
            continue

        if sop_id is None:
            continue

        meta_match = META_RE.match(line)
        if meta_match and section is None:
            reading_title = False
            key = meta_match.group(1).lower().replace(" ", "_")
            meta[key] = meta_match.group(2)
            continue

        heading = parse_heading(line, expected)
        if heading:
            flush()
            reading_title = False
            section, section_page, body = heading, page_number, []
            expected += 1
            continue

        if reading_title:
            sop_title += " " + line
            continue

        if section is not None:
            body.append(line)

    flush()
    return chunks