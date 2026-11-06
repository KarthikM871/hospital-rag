from src.hospital_rag.interfaces import Chunk


def _merge(first: Chunk, second: Chunk) -> Chunk:
    section = f"{first.metadata['section']} + {second.metadata['section']}"
    return first.model_copy(
        update={
            "text": first.text + "\n" + second.text,
            "metadata": {**first.metadata, "section": section},
        }
    )


def _close(result: list[Chunk], current: Chunk, min_chars: int) -> None:
    previous = result[-1] if result else None
    same_sop = previous is not None and (
        previous.metadata["sop_id"] == current.metadata["sop_id"]
    )
    if len(current.text) < min_chars and previous is not None and same_sop:
        result[-1] = _merge(previous, current)
    else:
        result.append(current)


def merge_short(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    result: list[Chunk] = []
    current: Chunk | None = None
    for chunk in chunks:
        if current is not None and (
            current.metadata["sop_id"] != chunk.metadata["sop_id"]
        ):
            _close(result, current, min_chars)
            current = None
        if current is None:
            current = chunk
        elif len(current.text) < min_chars:
            current = _merge(current, chunk)
        else:
            result.append(current)
            current = chunk
    if current is not None:
        _close(result, current, min_chars)
    return result


def split_long(
    chunks: list[Chunk], max_chars: int, overlap_lines: int = 1
) -> list[Chunk]:
    result: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) <= max_chars:
            result.append(chunk)
            continue
        header, *body = chunk.text.split("\n")
        parts: list[list[str]] = []
        current: list[str] = []
        for line in body:
            size = len(header) + sum(len(x) + 1 for x in current) + len(line) + 1
            if current and size > max_chars:
                parts.append(current)
                current = current[-overlap_lines:] if overlap_lines else []
            current.append(line)
        parts.append(current)
        for number, lines in enumerate(parts, start=1):
            result.append(
                chunk.model_copy(
                    update={
                        "text": header + "\n" + "\n".join(lines),
                        "metadata": {**chunk.metadata, "part": number},
                    }
                )
            )
    return result


def refine_chunks(
    chunks: list[Chunk], min_chars: int, max_chars: int
) -> list[Chunk]:
    refined = split_long(merge_short(chunks, min_chars), max_chars)
    return [
        c.model_copy(update={"id": f"{c.document_id}-{i:03d}"})
        for i, c in enumerate(refined, start=1)
    ]